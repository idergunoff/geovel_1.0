from __future__ import annotations

from .common import *


CLUSTER_CALC_CACHE_SCHEMA_VERSION = 1
CLUSTER_CALC_CACHE_GZIP_PREFIX = "gzjson:"


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
        "config": config or {},
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
        row.config_json = json.dumps(config or {}, ensure_ascii=False, sort_keys=True, default=str)
        row.result_payload = encode_cluster_calculation_payload(result_payload or {})
        session.commit()
    except Exception as exc:
        session.rollback()
        set_info(f"CALC: ошибка сохранения результата: {exc}", "brown")
