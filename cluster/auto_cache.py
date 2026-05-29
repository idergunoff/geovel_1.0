from __future__ import annotations

from .common import *

def _normalize_cache_payload(data: Any) -> Any:
    """
    Нормализует структуру данных для стабильного JSON/hash.
    """
    if isinstance(data, dict):
        return {str(k): _normalize_cache_payload(v) for k, v in sorted(data.items(), key=lambda item: str(item[0]))}
    if isinstance(data, (list, tuple)):
        return [_normalize_cache_payload(v) for v in data]
    return data


def build_cluster_auto_tuning_cache_key(
        *,
        clust_object_id: int,
        auto_mode: str,
        max_candidates: int,
        top_k: int,
        constraints: Optional[Dict[str, Any]],
        weights: Dict[str, float],
        clean_kwargs: Dict[str, Any]
) -> str:
    """
    Формирует hash-ключ для сохранения/поиска результата AUTO-подбора.
    """
    payload = {
        "clust_object_id": int(clust_object_id),
        "auto_mode": str(auto_mode).upper(),
        "max_candidates": int(max_candidates),
        "top_k": int(top_k),
        "weights": {
            "silhouette": float(weights.get("silhouette", 0.4)),
            "davies_bouldin": float(weights.get("davies_bouldin", 0.3)),
            "calinski_harabasz": float(weights.get("calinski_harabasz", 0.3))
        },
        "constraints": _normalize_cache_payload(constraints or {}),
        "clean_kwargs": _normalize_cache_payload(clean_kwargs or {})
    }
    payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def _ensure_cluster_auto_tuning_run_state_table() -> None:
    """
    Создает таблицу checkpoint-состояния AUTO-подбора (если ее еще нет).
    """
    ClusterAutoTuningRunState.__table__.create(bind=engine, checkfirst=True)


def _save_auto_tuning_run_state(
        *,
        run_key: str,
        object_set_id: int,
        random_seed: int,
        sampled_indices: np.ndarray,
        completed_candidate_ids: set[str],
        coarse_results: list[CandidateResult],
        fine_results: list[CandidateResult]
) -> None:
    _ensure_cluster_auto_tuning_run_state_table()
    row = session.query(ClusterAutoTuningRunState).filter_by(run_key=str(run_key), object_set_id=int(object_set_id)).first()
    if row is None:
        row = ClusterAutoTuningRunState(run_key=str(run_key), object_set_id=int(object_set_id), random_seed=int(random_seed), sampled_indices_json='[]')
        session.add(row)
    row.random_seed = int(random_seed)
    row.sampled_indices_json = json.dumps([int(i) for i in np.asarray(sampled_indices, dtype=int).tolist()], ensure_ascii=False)
    row.completed_candidate_ids_json = json.dumps(sorted(str(x) for x in completed_candidate_ids), ensure_ascii=False)
    row.raw_results_json = json.dumps((coarse_results or []) + (fine_results or []), ensure_ascii=False)
    row.coarse_count = int(len(coarse_results or []))
    row.fine_count = int(len(fine_results or []))
    row.updated_at = dt.datetime.utcnow()
    session.commit()


def _load_auto_tuning_run_state(*, run_key: str, object_set_id: int) -> Optional[dict]:
    _ensure_cluster_auto_tuning_run_state_table()
    row = session.query(ClusterAutoTuningRunState).filter_by(run_key=str(run_key), object_set_id=int(object_set_id)).first()
    if row is None:
        return None
    try:
        sampled_indices = np.asarray(json.loads(row.sampled_indices_json or '[]'), dtype=int)
        completed_ids = set(str(x) for x in json.loads(row.completed_candidate_ids_json or '[]'))
        raw_results = json.loads(row.raw_results_json or '[]')
    except Exception:
        return None
    return {
        'random_seed': int(row.random_seed),
        'sampled_indices': sampled_indices,
        'completed_ids': completed_ids,
        'raw_results': raw_results,
        'coarse_count': int(row.coarse_count or 0),
        'fine_count': int(row.fine_count or 0)
    }


def _clear_auto_tuning_run_state(*, run_key: str, object_set_id: int) -> None:
    _ensure_cluster_auto_tuning_run_state_table()
    session.query(ClusterAutoTuningRunState).filter_by(run_key=str(run_key), object_set_id=int(object_set_id)).delete(synchronize_session=False)
    session.commit()


def _ensure_cluster_auto_tuning_cache_table() -> None:
    """
    Создает таблицу cache автоподбора (если ее еще нет).
    """
    ClusterAutoTuningCache.__table__.create(bind=engine, checkfirst=True)


def load_cluster_auto_tuning_cache(
        *,
        cache_key: str,
        clust_object_id: int,
        top_k: int = 5
) -> list[CandidateResult]:
    """
    Загружает top-K настроек AUTO-подбора из persistent-cache.
    """
    try:
        _ensure_cluster_auto_tuning_cache_table()
        row = (
            session.query(ClusterAutoTuningCache)
            .filter_by(cache_key=str(cache_key), object_set_id=int(clust_object_id))
            .first()
        )
        if row is None or not row.top_results:
            return []
        payload = json.loads(row.top_results)
        if not isinstance(payload, list):
            return []
        return payload[:max(1, int(top_k))]
    except Exception as exc:
        set_info(f"AUTO: ошибка чтения cache top-K: {exc}", "brown")
        return []


def save_cluster_auto_tuning_cache(
        *,
        cache_key: str,
        clust_object_id: int,
        top_results: list[CandidateResult],
        top_k: int = 5
) -> None:
    """
    Сохраняет top-K настроек AUTO-подбора (только конфигурации кандидатов).
    """
    try:
        _ensure_cluster_auto_tuning_cache_table()
        compact_top_results: list[dict] = []
        for idx, result in enumerate((top_results or [])[:max(1, int(top_k))], start=1):
            result_row = result or {}
            cfg = result_row.get("candidate_config")
            if not cfg:
                continue
            metrics = result_row.get("metrics") or {}
            stats = result_row.get("stats") or {}
            score = result_row.get("score")

            def _safe_float(value):
                if value is None:
                    return None
                try:
                    return float(value)
                except Exception:
                    return None

            def _safe_int(value):
                if value is None:
                    return None
                try:
                    return int(value)
                except Exception:
                    return None

            compact_top_results.append(
                {
                    "candidate_id": str(result_row.get("candidate_id") or f"T{idx:02d}"),
                    "candidate_config": cfg,
                    "metrics": {
                        "silhouette": _safe_float(metrics.get("silhouette")),
                        "davies_bouldin": _safe_float(metrics.get("davies_bouldin")),
                        "calinski_harabasz": _safe_float(metrics.get("calinski_harabasz"))
                    },
                    "stats": {
                        "n_clusters": _safe_int(stats.get("n_clusters")),
                        "noise_fraction": _safe_float(stats.get("noise_fraction")),
                        "n_samples_eval": _safe_int(stats.get("n_samples_eval")),
                        "partition_hash": str(stats.get("partition_hash") or "")
                    },
                    "score": _safe_float(score),
                    "status": str(result_row.get("status") or "ok"),
                    "error_text": str(result_row.get("error_text") or "")
                }
            )
        if not compact_top_results:
            return
        existing_row = (
            session.query(ClusterAutoTuningCache)
            .filter_by(cache_key=str(cache_key), object_set_id=int(clust_object_id))
            .first()
        )
        if existing_row is None:
            existing_row = ClusterAutoTuningCache(
                object_set_id=int(clust_object_id),
                cache_key=str(cache_key)
            )
            session.add(existing_row)

        existing_row.created_at = dt.datetime.utcnow()
        existing_row.top_results = json.dumps(compact_top_results, ensure_ascii=False)
        session.commit()
    except Exception as exc:
        set_info(f"AUTO: ошибка сохранения cache top-5: {exc}", "brown")


