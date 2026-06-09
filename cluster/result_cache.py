from __future__ import annotations

from .common import *


CLUSTER_CALC_CACHE_SCHEMA_VERSION = 2
CLUSTER_CALC_CACHE_GZIP_PREFIX = "gzjson:"


def normalize_cluster_calculation_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Keeps only settings that can affect the active CALC result."""
    source = config or {}
    clean = dict(source.get("clean") or {})
    normalized_clean = {
        "use_non_finite": bool(clean.get("use_non_finite")),
        "use_variance_threshold": bool(clean.get("use_variance_threshold")),
        "use_correlation_filter": bool(clean.get("use_correlation_filter")),
    }
    if normalized_clean["use_non_finite"]:
        normalized_clean["non_finite_mode"] = str(clean.get("non_finite_mode") or "impute")

    pca = dict(source.get("pca") or {})
    pca_enabled = bool(pca.get("enabled"))
    normalized_pca: dict[str, Any] = {"enabled": pca_enabled}
    if pca_enabled:
        pca_mode = str(pca.get("mode") or "variance_ratio")
        normalized_pca["mode"] = pca_mode
        if pca_mode == "fixed_components":
            normalized_pca["value"] = int(pca.get("fixed_components", pca.get("value", 2)))
        else:
            normalized_pca["value"] = float(pca.get("variance_ratio", pca.get("value", 0.95)))

    method = str(source.get("method") or "kmeans")
    params = dict(source.get("method_params") or {})
    if method == "hdbscan":
        active_method_params = {
            "hdbscan_min_cluster_size": int(params.get("hdbscan_min_cluster_size", 5)),
            "hdbscan_min_samples": int(params.get("hdbscan_min_samples", 5)),
            "hdbscan_metric": str(params.get("hdbscan_metric") or "euclidean"),
        }
    elif method == "gmm":
        active_method_params = {
            "gmm_n_components": int(params.get("gmm_n_components", 2)),
            "gmm_covariance_type": str(params.get("gmm_covariance_type") or "full"),
        }
    else:
        method = "kmeans"
        active_method_params = {
            "kmeans_n_clusters": int(params.get("kmeans_n_clusters", 2)),
            "kmeans_n_init": int(params.get("kmeans_n_init", 10)),
        }

    metrics = dict(source.get("metrics") or {})
    smoothing = dict(source.get("smoothing") or {})
    smoothing_enabled = bool(smoothing.get("enabled"))
    normalized_smoothing: dict[str, Any] = {"enabled": smoothing_enabled}
    if smoothing_enabled:
        normalized_smoothing.update({
            "method": str(smoothing.get("method") or "maj"),
            "window": int(smoothing.get("window", 3)),
        })

    return {
        "clean": normalized_clean,
        "preprocess_mode": str(source.get("preprocess_mode") or "none"),
        "pca": normalized_pca,
        "method": method,
        "method_params": active_method_params,
        "metrics": {
            "use_silhouette": bool(metrics.get("use_silhouette")),
            "use_db": bool(metrics.get("use_db")),
            "use_ch": bool(metrics.get("use_ch")),
        },
        "smoothing": normalized_smoothing,
    }


def build_cluster_calculation_cache_key(
        *,
        source_type: str,
        dataset_id: int,
        data_hash: str,
        config: dict[str, Any]
) -> str:
    """Builds a stable key for a completed CALC result."""
    payload = {
        "schema_version": CLUSTER_CALC_CACHE_SCHEMA_VERSION,
        "source_type": str(source_type or "").strip().lower(),
        "dataset_id": int(dataset_id),
        "data_hash": str(data_hash or ""),
        "config": normalize_cluster_calculation_config(config),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def encode_cluster_calculation_payload(payload: dict[str, Any]) -> str:
    """Serializes and compresses a CALC result for a Text database column."""
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    return CLUSTER_CALC_CACHE_GZIP_PREFIX + base64.b64encode(gzip.compress(raw)).decode("ascii")


def decode_cluster_calculation_payload(value: str) -> dict[str, Any] | None:
    """Deserializes a CALC cache payload, including uncompressed legacy JSON."""
    if not value:
        return None
    try:
        if value.startswith(CLUSTER_CALC_CACHE_GZIP_PREFIX):
            compressed = base64.b64decode(value[len(CLUSTER_CALC_CACHE_GZIP_PREFIX):].encode("ascii"))
            decoded = gzip.decompress(compressed).decode("utf-8")
        else:
            decoded = value
        payload = json.loads(decoded)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def get_cached_cluster_labels(
        payload: dict[str, Any] | None,
        *,
        result_type: str,
        expected_count: int | None = None
) -> list[int] | None:
    """Validates the cache kind and returns normalized labels for a cache hit."""
    if not payload or payload.get("result_type") != result_type:
        return None
    try:
        labels = [int(value) for value in payload.get("labels", [])]
    except (TypeError, ValueError):
        return None
    if not labels:
        return None
    if expected_count is not None and len(labels) != int(expected_count):
        return None
    return labels


def get_cached_gpr_calculation(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Validates that a GPR cache entry can skip the complete preparation pipeline."""
    labels = get_cached_cluster_labels(payload, result_type="gpr")
    if labels is None:
        return None
    try:
        kept_row_indices = [int(value) for value in payload.get("kept_row_indices", [])]
        data_for_diagnostics = list(payload.get("data_for_diagnostics", []))
    except (TypeError, ValueError):
        return None
    if len(labels) != len(kept_row_indices) or len(labels) != len(data_for_diagnostics):
        return None
    return {
        "labels": labels,
        "kept_row_indices": kept_row_indices,
        "data_for_diagnostics": data_for_diagnostics,
        "cluster_info": dict(payload.get("cluster_info") or {}),
        "pca_info_report": dict(payload.get("pca_info_report") or {}),
    }


def _cluster_calculation_cache_model(source_type: str):
    if str(source_type or "").strip().lower() == "well_log":
        return WellLogClusterCalculationCache
    return ClusterCalculationCache


def _ensure_cluster_calculation_cache_table(source_type: str) -> None:
    _cluster_calculation_cache_model(source_type).__table__.create(bind=engine, checkfirst=True)


def load_cluster_calculation_cache(
        *,
        source_type: str,
        dataset_id: int,
        cache_key: str
) -> dict[str, Any] | None:
    """Loads a completed CALC result from the persistent cache."""
    try:
        _ensure_cluster_calculation_cache_table(source_type)
        cache_model = _cluster_calculation_cache_model(source_type)
        id_field = "dataset_id" if cache_model is WellLogClusterCalculationCache else "object_set_id"
        row = session.query(cache_model).filter_by(
            cache_key=str(cache_key),
            **{id_field: int(dataset_id)},
        ).first()
        if row is None:
            return None
        return decode_cluster_calculation_payload(row.result_payload)
    except Exception as exc:
        set_info(f"CALC: ошибка чтения сохраненного результата: {exc}", "brown")
        return None


def save_cluster_calculation_cache(
        *,
        source_type: str,
        dataset_id: int,
        cache_key: str,
        data_hash: str,
        config: dict[str, Any],
        result_payload: dict[str, Any]
) -> None:
    """Creates or updates a persistent completed-CALC cache record."""
    try:
        _ensure_cluster_calculation_cache_table(source_type)
        cache_model = _cluster_calculation_cache_model(source_type)
        id_field = "dataset_id" if cache_model is WellLogClusterCalculationCache else "object_set_id"
        id_filter = {id_field: int(dataset_id)}
        row = session.query(cache_model).filter_by(cache_key=str(cache_key), **id_filter).first()
        if row is None:
            row = cache_model(cache_key=str(cache_key), **id_filter)
            session.add(row)
        row.created_at = dt.datetime.utcnow()
        row.data_hash = str(data_hash or "")
        row.config_json = json.dumps(
            normalize_cluster_calculation_config(config),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        row.result_payload = encode_cluster_calculation_payload(result_payload or {})
        session.commit()
    except Exception as exc:
        session.rollback()
        set_info(f"CALC: ошибка сохранения результата: {exc}", "brown")
