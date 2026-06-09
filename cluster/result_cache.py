from __future__ import annotations

from .common import *


CLUSTER_CALC_CACHE_SCHEMA_VERSION = 3
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

    return {
        "clean": normalized_clean,
        "preprocess_mode": str(source.get("preprocess_mode") or "none"),
        "pca": normalized_pca,
        "method": method,
        "method_params": active_method_params,
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


def build_cluster_postprocess_cache_key(config: dict[str, Any] | None) -> str:
    """Builds a key for smoothing and quality metrics applied to base labels."""
    source = config or {}
    smoothing = dict(source.get("smoothing") or {})
    metrics = dict(source.get("metrics") or {})
    payload = {
        "smoothing": {
            "enabled": bool(smoothing.get("enabled")),
            "method": str(smoothing.get("method") or "maj"),
            "window": int(smoothing.get("window", 3)),
        },
        "metrics": {
            "use_silhouette": bool(metrics.get("use_silhouette")),
            "use_db": bool(metrics.get("use_db")),
            "use_ch": bool(metrics.get("use_ch")),
        },
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_cached_postprocess_result(
        payload: dict[str, Any] | None,
        config: dict[str, Any] | None
) -> dict[str, Any] | None:
    if not payload:
        return None
    variants = payload.get("postprocess_results") or {}
    result = variants.get(build_cluster_postprocess_cache_key(config))
    return dict(result) if isinstance(result, dict) else None


def put_cached_postprocess_result(
        payload: dict[str, Any],
        config: dict[str, Any] | None,
        result: dict[str, Any]
) -> dict[str, Any]:
    updated = dict(payload or {})
    variants = dict(updated.get("postprocess_results") or {})
    variants[build_cluster_postprocess_cache_key(config)] = result
    updated["postprocess_results"] = variants
    return updated


def encode_cluster_calculation_payload(payload: dict[str, Any]) -> str:
    """Serializes and compresses a CALC result for a Text database column."""
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    return CLUSTER_CALC_CACHE_GZIP_PREFIX + base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii")


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


def get_cached_base_calculation(
        payload: dict[str, Any] | None,
        *,
        result_type: str
) -> dict[str, Any] | None:
    """Validates a pre-smoothing calculation payload for either data source."""
    labels = get_cached_cluster_labels(payload, result_type=result_type)
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


def get_cached_gpr_calculation(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Backward-compatible wrapper for GPR pre-smoothing cache validation."""
    return get_cached_base_calculation(payload, result_type="gpr")


def _cluster_calculation_cache_model(source_type: str):
    if str(source_type or "").strip().lower() == "well_log":
        return WellLogClusterCalculationCache
    return ClusterCalculationCache


def _ensure_cluster_calculation_cache_table(source_type: str) -> None:
    cache_model = _cluster_calculation_cache_model(source_type)
    cache_model.__table__.create(bind=engine, checkfirst=True)
    # Existing installations may already have the first cache-table version.
    # Add explicit result columns in-place so saving labels does not depend on
    # whether Alembic was run before the application starts.
    table_name = cache_model.__tablename__
    with engine.begin() as connection:
        columns = {
            str(row[1])
            for row in connection.exec_driver_sql(f"PRAGMA table_info({table_name})").fetchall()
        }
        if "labels_json" not in columns:
            connection.exec_driver_sql(
                f"ALTER TABLE {table_name} ADD COLUMN labels_json TEXT NOT NULL DEFAULT '[]'"
            )
        if "kept_row_indices_json" not in columns:
            connection.exec_driver_sql(
                f"ALTER TABLE {table_name} ADD COLUMN kept_row_indices_json TEXT NOT NULL DEFAULT '[]'"
            )
        if "assignments_json" not in columns:
            connection.exec_driver_sql(
                f"ALTER TABLE {table_name} ADD COLUMN assignments_json TEXT NOT NULL DEFAULT '[]'"
            )
        if "postprocess_results_json" not in columns:
            connection.exec_driver_sql(
                f"ALTER TABLE {table_name} ADD COLUMN postprocess_results_json TEXT NOT NULL DEFAULT '{{}}'"
            )


def load_cluster_calculation_cache(
        *,
        source_type: str,
        dataset_id: int,
        cache_key: str,
        data_hash: str = "",
        config: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Loads a completed CALC result from the persistent cache."""
    try:
        _ensure_cluster_calculation_cache_table(source_type)
        cache_model = _cluster_calculation_cache_model(source_type)
        id_field = "dataset_id" if cache_model is WellLogClusterCalculationCache else "object_set_id"
        id_filter = {id_field: int(dataset_id)}
        row = session.query(cache_model).filter_by(
            cache_key=str(cache_key),
            **id_filter,
        ).first()
        lookup_mode = "cache_key"
        if row is None:
            # Be tolerant to cache-key implementation/version changes. The
            # persisted normalized config and data hash are the authoritative
            # identity of the expensive base calculation.
            expected_config_json = json.dumps(
                normalize_cluster_calculation_config(config),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ) if config is not None else None
            if expected_config_json is not None:
                row = (
                    session.query(cache_model)
                    .filter_by(
                        data_hash=str(data_hash or ""),
                        config_json=expected_config_json,
                        **id_filter,
                    )
                    .order_by(cache_model.created_at.desc())
                    .first()
                )
                lookup_mode = "data_hash+config"
        if row is None:
            return None
        payload = decode_cluster_calculation_payload(row.result_payload)
        if payload is None:
            return None
        try:
            payload["labels"] = json.loads(row.labels_json or "[]")
            payload["kept_row_indices"] = json.loads(row.kept_row_indices_json or "[]")
            payload["assignments"] = json.loads(row.assignments_json or "[]")
            payload["postprocess_results"] = json.loads(row.postprocess_results_json or "{}")
            payload["_cache_lookup_mode"] = lookup_mode
            payload["_cache_row_id"] = int(row.id)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return payload
    except Exception as exc:
        set_info(f"CALC: ошибка чтения сохраненного результата: {exc}", "brown")
        return None


def _persistent_cluster_cache_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Removes runtime-only lookup metadata before comparing or persisting."""
    return {
        str(key): value
        for key, value in dict(payload or {}).items()
        if not str(key).startswith("_cache_")
    }


def _cluster_cache_row_matches(
        row: Any,
        *,
        data_hash: str,
        config_json: str,
        labels_json: str,
        kept_row_indices_json: str,
        assignments_json: str,
        postprocess_results_json: str,
        result_payload: str
) -> bool:
    """Returns True when a DB row already contains the exact same cache data."""
    return all((
        str(row.data_hash or "") == str(data_hash or ""),
        str(row.config_json or "") == config_json,
        str(row.labels_json or "[]") == labels_json,
        str(row.kept_row_indices_json or "[]") == kept_row_indices_json,
        str(row.assignments_json or "[]") == assignments_json,
        str(row.postprocess_results_json or "{}") == postprocess_results_json,
        str(row.result_payload or "") == result_payload,
    ))


def save_cluster_calculation_cache(
        *,
        source_type: str,
        dataset_id: int,
        cache_key: str,
        data_hash: str,
        config: dict[str, Any],
        result_payload: dict[str, Any]
) -> bool:
    """Creates or updates a persistent completed-CALC cache record."""
    try:
        _ensure_cluster_calculation_cache_table(source_type)
        cache_model = _cluster_calculation_cache_model(source_type)
        id_field = "dataset_id" if cache_model is WellLogClusterCalculationCache else "object_set_id"
        id_filter = {id_field: int(dataset_id)}

        persistent_payload = _persistent_cluster_cache_payload(result_payload)
        existing_row_id = (result_payload or {}).get("_cache_row_id")
        row = None
        if existing_row_id is not None:
            try:
                row = session.query(cache_model).filter_by(
                    id=int(existing_row_id),
                    **id_filter,
                ).first()
            except (TypeError, ValueError):
                row = None
        if row is None:
            row = session.query(cache_model).filter_by(cache_key=str(cache_key), **id_filter).first()
        if row is None:
            row = cache_model(cache_key=str(cache_key), **id_filter)
            session.add(row)
        elif str(row.cache_key) != str(cache_key):
            conflicting_row = session.query(cache_model).filter_by(cache_key=str(cache_key)).first()
            if conflicting_row is None:
                row.cache_key = str(cache_key)

        config_json = json.dumps(
            normalize_cluster_calculation_config(config),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        labels = [int(value) for value in persistent_payload.get("labels", [])]
        kept_row_indices = [int(value) for value in persistent_payload.get("kept_row_indices", [])]
        labels_json = json.dumps(labels, ensure_ascii=False)
        kept_row_indices_json = json.dumps(kept_row_indices, ensure_ascii=False)
        assignments_json = json.dumps(
            list(persistent_payload.get("assignments") or []),
            ensure_ascii=False,
            default=str,
        )
        postprocess_results_json = json.dumps(
            dict(persistent_payload.get("postprocess_results") or {}),
            ensure_ascii=False,
            default=str,
        )
        encoded_payload = encode_cluster_calculation_payload(persistent_payload)

        if getattr(row, "id", None) is not None and _cluster_cache_row_matches(
                row,
                data_hash=str(data_hash or ""),
                config_json=config_json,
                labels_json=labels_json,
                kept_row_indices_json=kept_row_indices_json,
                assignments_json=assignments_json,
                postprocess_results_json=postprocess_results_json,
                result_payload=encoded_payload,
        ):
            return True

        # created_at is creation time. It must not change when metrics or
        # diagnostics are appended to an existing calculation cache row.
        row.data_hash = str(data_hash or "")
        row.config_json = config_json
        row.labels_json = labels_json
        row.kept_row_indices_json = kept_row_indices_json
        row.assignments_json = assignments_json
        row.postprocess_results_json = postprocess_results_json
        row.result_payload = encoded_payload
        session.commit()
        return True
    except Exception as exc:
        session.rollback()
        set_info(f"CALC: ошибка сохранения результата: {exc}", "brown")
        return False
