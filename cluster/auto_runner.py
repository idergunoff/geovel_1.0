from __future__ import annotations

from .common import *

def run_auto_cluster_tuning(
        base_data,
        *,
        auto_mode: str = "COARSE",
        top_k: int = 5,
        max_candidates: Optional[int] = 200,
        max_clusters: int = 8,
        hdbscan_metric: str = "euclidean",
        hdbscan_metrics: Optional[list[str]] = None,
        scaler_only: bool = False,
        pca_only: bool = False,
        soft_timeout_sec: Optional[float] = None,
        candidate_soft_timeout_sec: Optional[float] = None,
        min_pca_components: int = 2,
        weights: Optional[Dict[str, float]] = None,
        clean_kwargs: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
        min_cluster_samples: int = 1,
        run_key: Optional[str] = None,
        object_set_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Orchestrator AUTO-подбора.
    В режиме FINE выполняет coarse + fine и объединяет leaderboard.
    """
    mode = (auto_mode or "COARSE").strip().upper()
    if mode not in {"COARSE", "FINE"}:
        raise ValueError(f"Unsupported auto_mode='{auto_mode}'. Expected 'COARSE' or 'FINE'.")

    completed_candidate_ids: set[str] = set()
    restored_raw_results: list[CandidateResult] = []
    if run_key and object_set_id is not None:
        restored_state = _load_auto_tuning_run_state(run_key=str(run_key), object_set_id=int(object_set_id))
        if restored_state is not None:
            completed_candidate_ids = set(restored_state.get("completed_ids", set()))
            restored_raw_results = list(restored_state.get("raw_results", []))
            if random_seed is None:
                random_seed = int(restored_state.get("random_seed", random_seed or 42))
            set_info(f"AUTO {mode}: восстановлен checkpoint, выполнено {len(completed_candidate_ids)} кандидатов.", "brown")

    coarse_candidates = build_auto_search_space(
        "COARSE",
        max_candidates=max_candidates,
        max_clusters=max_clusters,
        hdbscan_metric=hdbscan_metric,
        hdbscan_metrics=hdbscan_metrics,
        scaler_only=scaler_only,
        pca_only=pca_only,
        random_seed=random_seed
    )
    coarse_results: list[CandidateResult] = []
    coarse_silhouette_samples = _compute_auto_silhouette_sample_size(
        len(base_data),
        stage="coarse",
        estimated_n_clusters=max_clusters
    )
    auto_cache_enabled = len(base_data) <= int(AUTO_TRANSFORM_CACHE_MAX_ROWS)
    if not auto_cache_enabled:
        _set_auto_info(
            f"AUTO {mode}: transform-cache отключен для большого набора ({len(base_data)} строк).",
            "brown"
        )
    transform_cache: Optional["OrderedDict[tuple, Any]"] = OrderedDict() if auto_cache_enabled else None
    preprocess_cache: Optional["OrderedDict[tuple, Any]"] = OrderedDict() if auto_cache_enabled else None
    preprocess_rank_cache: Dict[tuple, int] = {}
    metrics_cache: Dict[str, CandidateMetrics] = {}
    transform_cache_sizes: "OrderedDict[tuple, int]" = OrderedDict()
    transform_cache_total_bytes = 0
    run_start_ts = monotonic()
    for idx, candidate in enumerate(coarse_candidates, start=1):
        candidate_id = f"C{idx:03d}"
        if candidate_id in completed_candidate_ids:
            continue
        if soft_timeout_sec is not None and (monotonic() - run_start_ts) > float(soft_timeout_sec):
            _set_auto_info(
                f"AUTO {mode}: достигнут soft-timeout {soft_timeout_sec:.1f}s, "
                f"coarse остановлен на кандидате {idx}.",
                "brown"
            )
            break
        _set_auto_info(f"Coarse: {idx}/{len(coarse_candidates)}", "blue")
        QApplication.processEvents()
        candidate_start_ts = monotonic()
        try:
            result = run_cluster_candidate_isolated(
                base_data=base_data,
                candidate=candidate,
                candidate_id=candidate_id,
                clean_kwargs=clean_kwargs,
                transform_cache=transform_cache,
                preprocess_cache=preprocess_cache,
                preprocess_rank_cache=preprocess_rank_cache,
                metrics_cache=metrics_cache,
                min_pca_components=min_pca_components,
                max_silhouette_samples=coarse_silhouette_samples,
                min_cluster_samples=min_cluster_samples,
                max_clusters=max_clusters,
                hard_timeout_sec=candidate_soft_timeout_sec
            )
        except Exception as exc:
            _set_auto_info(f"AUTO Coarse C{idx:03d}: исключение {exc}", "red")
            result = make_candidate_result(
                candidate_id=candidate_id,
                candidate_config=candidate,
                status="error",
                error_text=f"unexpected candidate exception: {exc}"
            )

        elapsed_candidate = monotonic() - candidate_start_ts
        if candidate_soft_timeout_sec is not None and elapsed_candidate > float(candidate_soft_timeout_sec):
            _set_auto_info(
                f"AUTO Coarse C{idx:03d}: превышен soft-timeout {candidate_soft_timeout_sec:.1f}s "
                f"({elapsed_candidate:.2f}s).",
                "brown"
            )
            result["status"] = "invalid"
            result["score"] = None
            result["error_text"] = (
                (result.get("error_text", "") + "; ").strip("; ")
                + f"candidate soft-timeout {elapsed_candidate:.2f}s > {candidate_soft_timeout_sec:.2f}s"
            ).strip()
        _log_candidate_rejection(
            result,
            phase="coarse",
            min_cluster_samples=min_cluster_samples,
            n_samples=len(base_data)
        )
        coarse_results.append(result)
        completed_candidate_ids.add(str(result.get("candidate_id") or candidate_id))
        if run_key and object_set_id is not None and (len(coarse_results) % int(AUTO_CHECKPOINT_SAVE_EVERY) == 0):
            _save_auto_tuning_run_state(run_key=str(run_key), object_set_id=int(object_set_id), random_seed=int(random_seed or 42), sampled_indices=np.arange(len(base_data), dtype=int), completed_candidate_ids=completed_candidate_ids, coarse_results=coarse_results, fine_results=[])
        if transform_cache is not None and len(transform_cache_sizes) != len(transform_cache):
            stale_keys = [key for key in list(transform_cache_sizes.keys()) if key not in transform_cache]
            for key in stale_keys:
                transform_cache_total_bytes -= int(transform_cache_sizes.pop(key, 0))
            for key, value in list(transform_cache.items()):
                if key in transform_cache_sizes:
                    continue
                cached_size = _estimate_transform_cache_item_nbytes(value)
                if cached_size > AUTO_TRANSFORM_CACHE_MAX_ITEM_BYTES:
                    transform_cache.pop(key, None)
                    continue
                if cached_size > 0:
                    transform_cache_sizes[key] = cached_size
                    transform_cache_total_bytes += cached_size
            transform_cache_total_bytes = _trim_transform_cache(
                transform_cache,
                transform_cache_sizes,
                transform_cache_total_bytes,
                max_cache_bytes=AUTO_TRANSFORM_CACHE_MAX_BYTES
            )
        gc.collect()

    has_valid_coarse = any(row.get("status") == "ok" for row in (restored_raw_results + coarse_results))
    if not has_valid_coarse:
        _set_auto_info(
            "AUTO: в основном наборе не найдено валидных конфигураций, запускаю резервный mini-grid.",
            "brown"
        )
        rescue_candidates = _build_auto_rescue_candidates(max_clusters=max_clusters)
        for idx, candidate in enumerate(rescue_candidates, start=1):
            try:
                result = run_cluster_candidate_isolated(
                    base_data=base_data,
                    candidate=candidate,
                    candidate_id=f"R{idx:03d}",
                    clean_kwargs=clean_kwargs,
                    transform_cache=transform_cache,
                    preprocess_cache=preprocess_cache,
                    preprocess_rank_cache=preprocess_rank_cache,
                    metrics_cache=metrics_cache,
                    min_pca_components=min_pca_components,
                    max_silhouette_samples=coarse_silhouette_samples,
                    min_cluster_samples=min_cluster_samples,
                    max_clusters=max_clusters,
                    hard_timeout_sec=candidate_soft_timeout_sec
                )
            except Exception as exc:
                result = make_candidate_result(
                    candidate_id=f"R{idx:03d}",
                    candidate_config=candidate,
                    status="error",
                    error_text=f"rescue candidate exception: {exc}"
                )
            _log_candidate_rejection(
                result,
                phase="rescue",
                min_cluster_samples=min_cluster_samples,
                n_samples=len(base_data)
            )
            coarse_results.append(result)
        completed_candidate_ids.add(str(result.get("candidate_id") or candidate_id))
        if run_key and object_set_id is not None:
            _save_auto_tuning_run_state(run_key=str(run_key), object_set_id=int(object_set_id), random_seed=int(random_seed or 42), sampled_indices=np.arange(len(base_data), dtype=int), completed_candidate_ids=completed_candidate_ids, coarse_results=coarse_results, fine_results=[])
        has_valid_after_rescue = any(row.get("status") == "ok" for row in coarse_results)
        if not has_valid_after_rescue:
            _set_auto_info(
                "AUTO: rescue mini-grid не дал валидных конфигураций, повторяю с упрощенной очисткой.",
                "brown"
            )
            relaxed_clean_kwargs = {
                "use_non_finite": True,
                "non_finite_mode": "impute",
                "use_variance_threshold": False,
                "use_correlation_filter": False
            }
            for idx, candidate in enumerate(rescue_candidates, start=1):
                try:
                    result = run_cluster_candidate_isolated(
                        base_data=base_data,
                        candidate=candidate,
                        candidate_id=f"RR{idx:03d}",
                        clean_kwargs=relaxed_clean_kwargs,
                        transform_cache=transform_cache,
                        preprocess_cache=preprocess_cache,
                        preprocess_rank_cache=preprocess_rank_cache,
                        metrics_cache=metrics_cache,
                        min_pca_components=min_pca_components,
                        max_silhouette_samples=coarse_silhouette_samples,
                        min_cluster_samples=min_cluster_samples,
                        max_clusters=max_clusters,
                        hard_timeout_sec=candidate_soft_timeout_sec
                    )
                except Exception as exc:
                    result = make_candidate_result(
                        candidate_id=f"RR{idx:03d}",
                        candidate_config=candidate,
                        status="error",
                        error_text=f"relaxed rescue exception: {exc}"
                    )
                _log_candidate_rejection(
                    result,
                    phase="rescue_relaxed",
                    min_cluster_samples=min_cluster_samples,
                    n_samples=len(base_data)
                )
                coarse_results.append(result)
        completed_candidate_ids.add(str(result.get("candidate_id") or candidate_id))
        if run_key and object_set_id is not None:
            _save_auto_tuning_run_state(run_key=str(run_key), object_set_id=int(object_set_id), random_seed=int(random_seed or 42), sampled_indices=np.arange(len(base_data), dtype=int), completed_candidate_ids=completed_candidate_ids, coarse_results=coarse_results, fine_results=[])
        if not any(row.get("status") == "ok" for row in coarse_results):
            top_failures = _summarize_candidate_failures(coarse_results, top_n=3)
            if top_failures:
                _set_auto_info(
                    "AUTO: топ причин невалидности: "
                    + "; ".join(f"{reason} ({count})" for reason, count in top_failures),
                    "brown"
                )

    if restored_raw_results:
        coarse_results = list(restored_raw_results) + coarse_results
    ranked_coarse = rank_candidates(coarse_results, weights=weights, max_clusters=max_clusters)
    coarse_best_result = ranked_coarse[0] if ranked_coarse else None
    coarse_top_diverse = select_diverse_top_results(
        ranked_coarse,
        top_k=top_k,
        diversity_key="partition_hash"
    )

    if mode == "COARSE":
        if run_key and object_set_id is not None and coarse_top_diverse:
            _clear_auto_tuning_run_state(run_key=str(run_key), object_set_id=int(object_set_id))
        return {
            "mode": mode,
            "best_result": coarse_best_result,
            "top_results": coarse_top_diverse,
            "raw_results": coarse_results,
            "coarse_results": ranked_coarse,
            "fine_results": []
        }

    fine_seed_count = max(int(top_k), min(max(3, int(top_k)), 10))
    fine_seed_results = select_diverse_top_results(
        ranked_coarse,
        top_k=fine_seed_count,
        diversity_key="partition_hash"
    )
    best_coarse_score = coarse_best_result.get("score") if coarse_best_result else None
    if best_coarse_score is not None:
        score_delta = float(AUTO_FINE_SEED_DELTA_FROM_BEST)
        filtered_seed_results = [
            row for row in fine_seed_results
            if row.get("score") is not None and float(row.get("score")) >= float(best_coarse_score) - score_delta
        ]
        if filtered_seed_results:
            removed_count = len(fine_seed_results) - len(filtered_seed_results)
            if removed_count > 0:
                _set_auto_info(
                    f"AUTO FINE: отфильтровано {removed_count} seed-кандидатов по delta_score>{score_delta:.3f}.",
                    "blue"
                )
            fine_seed_results = filtered_seed_results
    fine_candidates = build_fine_search_space(
        fine_seed_results,
        top_k=fine_seed_count,
        max_candidates=max_candidates,
        hdbscan_metrics=hdbscan_metrics,
        max_clusters=max_clusters
    )
    fine_results: list[CandidateResult] = []
    for idx, candidate in enumerate(fine_candidates, start=1):
        candidate_id = f"F{idx:03d}"
        if candidate_id in completed_candidate_ids:
            continue
        if soft_timeout_sec is not None and (monotonic() - run_start_ts) > float(soft_timeout_sec):
            _set_auto_info(
                f"AUTO {mode}: достигнут soft-timeout {soft_timeout_sec:.1f}s, "
                f"fine остановлен на кандидате {idx}.",
                "brown"
            )
            break
        _set_auto_info(f"Fine: {idx}/{len(fine_candidates)}", "blue")
        QApplication.processEvents()
        candidate_start_ts = monotonic()
        try:
            result = run_cluster_candidate_isolated(
                base_data=base_data,
                candidate=candidate,
                candidate_id=candidate_id,
                clean_kwargs=clean_kwargs,
                transform_cache=transform_cache,
                preprocess_cache=preprocess_cache,
                preprocess_rank_cache=preprocess_rank_cache,
                metrics_cache=metrics_cache,
                min_pca_components=min_pca_components,
                max_silhouette_samples=_compute_auto_silhouette_sample_size(len(base_data), stage="fine", estimated_n_clusters=max_clusters),
                min_cluster_samples=min_cluster_samples,
                max_clusters=max_clusters,
                hard_timeout_sec=candidate_soft_timeout_sec
            )
        except Exception as exc:
            _set_auto_info(f"AUTO Fine F{idx:03d}: исключение {exc}", "red")
            result = make_candidate_result(
                candidate_id=candidate_id,
                candidate_config=candidate,
                status="error",
                error_text=f"unexpected candidate exception: {exc}"
            )

        elapsed_candidate = monotonic() - candidate_start_ts
        if candidate_soft_timeout_sec is not None and elapsed_candidate > float(candidate_soft_timeout_sec):
            _set_auto_info(
                f"AUTO Fine F{idx:03d}: превышен soft-timeout {candidate_soft_timeout_sec:.1f}s "
                f"({elapsed_candidate:.2f}s).",
                "brown"
            )
            result["status"] = "invalid"
            result["score"] = None
            result["error_text"] = (
                (result.get("error_text", "") + "; ").strip("; ")
                + f"candidate soft-timeout {elapsed_candidate:.2f}s > {candidate_soft_timeout_sec:.2f}s"
            ).strip()
        _log_candidate_rejection(
            result,
            phase="fine",
            min_cluster_samples=min_cluster_samples,
            n_samples=len(base_data)
        )
        fine_results.append(result)
        completed_candidate_ids.add(str(result.get("candidate_id") or candidate_id))
        if run_key and object_set_id is not None and (len(coarse_results) + len(fine_results)) % int(AUTO_CHECKPOINT_SAVE_EVERY) == 0:
            _save_auto_tuning_run_state(run_key=str(run_key), object_set_id=int(object_set_id), random_seed=int(random_seed or 42), sampled_indices=np.arange(len(base_data), dtype=int), completed_candidate_ids=completed_candidate_ids, coarse_results=coarse_results, fine_results=fine_results)
        if transform_cache is not None and len(transform_cache_sizes) != len(transform_cache):
            stale_keys = [key for key in list(transform_cache_sizes.keys()) if key not in transform_cache]
            for key in stale_keys:
                transform_cache_total_bytes -= int(transform_cache_sizes.pop(key, 0))
            for key, value in list(transform_cache.items()):
                if key in transform_cache_sizes:
                    continue
                cached_size = _estimate_transform_cache_item_nbytes(value)
                if cached_size > AUTO_TRANSFORM_CACHE_MAX_ITEM_BYTES:
                    transform_cache.pop(key, None)
                    continue
                if cached_size > 0:
                    transform_cache_sizes[key] = cached_size
                    transform_cache_total_bytes += cached_size
            transform_cache_total_bytes = _trim_transform_cache(
                transform_cache,
                transform_cache_sizes,
                transform_cache_total_bytes,
                max_cache_bytes=AUTO_TRANSFORM_CACHE_MAX_BYTES
            )
        gc.collect()

    if run_key and object_set_id is not None and (coarse_results or fine_results):
        _save_auto_tuning_run_state(run_key=str(run_key), object_set_id=int(object_set_id), random_seed=int(random_seed or 42), sampled_indices=np.arange(len(base_data), dtype=int), completed_candidate_ids=completed_candidate_ids, coarse_results=coarse_results, fine_results=fine_results)

    combined_ranked = rank_candidates(coarse_results + fine_results, weights=weights, max_clusters=max_clusters)
    best_result = combined_ranked[0] if combined_ranked else coarse_best_result
    combined_top_diverse = select_diverse_top_results(
        combined_ranked,
        top_k=top_k,
        diversity_key="partition_hash"
    )

    if run_key and object_set_id is not None and combined_top_diverse:
        _clear_auto_tuning_run_state(run_key=str(run_key), object_set_id=int(object_set_id))
    return {
        "mode": mode,
        "best_result": best_result,
        "top_results": combined_top_diverse,
        "raw_results": coarse_results + fine_results,
        "coarse_results": ranked_coarse,
        "fine_results": rank_candidates(fine_results, weights=weights, max_clusters=max_clusters),
        "coarse_best_result": coarse_best_result
    }


def _collect_top_signatures(results: list[CandidateResult], top_n: int = 5) -> list[tuple]:
    """
    Возвращает подписи top-N валидных результатов для проверки стабильности ранжирования.
    """
    signatures: list[tuple] = []
    for row in results:
        if row.get("status") != "ok":
            continue
        cfg = row.get("candidate_config")
        if not cfg:
            continue
        signatures.append(_candidate_signature(cfg))
        if len(signatures) >= max(1, int(top_n)):
            break
    return signatures


def validate_auto_tuning_run(
        tuning_result: Dict[str, Any],
        *,
        expected_mode: Optional[str] = None,
        require_non_empty_leaderboard: bool = True,
        require_fine_stage: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Валидирует один запуск AUTO-подбора и формирует структурированный отчёт.

    Проверки:
    - режим запуска совпадает с ожидаемым;
    - leaderboard не пустой (если требуется);
    - в FINE-режиме есть coarse + fine результаты.
    """
    issues: list[str] = []
    summary: Dict[str, Any] = {
        "mode": tuning_result.get("mode"),
        "passed": True,
        "issues": issues,
        "stats": {}
    }

    mode = str(tuning_result.get("mode", "")).upper()
    if expected_mode is not None and mode != str(expected_mode).upper():
        issues.append(f"mode mismatch: expected={expected_mode}, got={mode}")

    top_results = tuning_result.get("top_results") or []
    if require_non_empty_leaderboard and len(top_results) == 0:
        issues.append("top_results is empty")

    coarse_results = tuning_result.get("coarse_results") or []
    fine_results = tuning_result.get("fine_results") or []
    inferred_require_fine = (mode == "FINE") if require_fine_stage is None else bool(require_fine_stage)
    if inferred_require_fine:
        if len(coarse_results) == 0:
            issues.append("coarse_results is empty in FINE run")
        if len(fine_results) == 0:
            issues.append("fine_results is empty in FINE run")

    best_result = tuning_result.get("best_result")
    best_is_ok = bool(best_result and best_result.get("status") == "ok")
    summary["stats"] = {
        "n_top": len(top_results),
        "n_coarse": len(coarse_results),
        "n_fine": len(fine_results),
        "best_ok": best_is_ok
    }
    summary["passed"] = len(issues) == 0
    return summary


def _log_candidate_rejection(result: CandidateResult, *, phase: str, min_cluster_samples: int, n_samples: int) -> None:
    """
    Структурированный лог отбраковки кандидата для диагностики AUTO-режима.
    """
    if result.get("status") == "ok":
        return
    payload = {
        "event": "cluster_auto_candidate_rejected",
        "phase": phase,
        "candidate_id": result.get("candidate_id"),
        "status": result.get("status"),
        "reason": result.get("error_text", ""),
        "min_cluster_samples": int(min_cluster_samples),
        "n_samples": int(n_samples)
    }
    set_info(f"AUTO_REJECT {json.dumps(payload, ensure_ascii=False, sort_keys=True)}", "brown")


def validate_auto_tuning_quality(
        *,
        coarse_result: Dict[str, Any],
        fine_result: Dict[str, Any],
        repeated_results: Optional[list[Dict[str, Any]]] = None,
        stability_top_n: int = 5
) -> Dict[str, Any]:
    """
    Комплексная валидация качества AUTO (этап 8):
    1) COARSE возвращает непустой leaderboard.
    2) FINE содержит coarse + fine этапы.
    3) Повторные запуски дают близкий топ ранга (по пересечению сигнатур).

    Возвращает отчёт с флагом passed и списком issues.
    """
    report: Dict[str, Any] = {
        "passed": True,
        "issues": [],
        "checks": {}
    }
    issues: list[str] = report["issues"]

    coarse_check = validate_auto_tuning_run(
        coarse_result,
        expected_mode="COARSE",
        require_non_empty_leaderboard=True,
        require_fine_stage=False
    )
    fine_check = validate_auto_tuning_run(
        fine_result,
        expected_mode="FINE",
        require_non_empty_leaderboard=True,
        require_fine_stage=True
    )
    report["checks"]["coarse"] = coarse_check
    report["checks"]["fine"] = fine_check
    if not coarse_check["passed"]:
        issues.extend(f"coarse: {msg}" for msg in coarse_check["issues"])
    if not fine_check["passed"]:
        issues.extend(f"fine: {msg}" for msg in fine_check["issues"])

    if repeated_results:
        base_top = _collect_top_signatures(repeated_results[0].get("top_results", []), top_n=stability_top_n)
        overlaps: list[float] = []
        for run in repeated_results[1:]:
            curr_top = _collect_top_signatures(run.get("top_results", []), top_n=stability_top_n)
            if not base_top:
                overlap = 0.0
            else:
                overlap = len(set(base_top) & set(curr_top)) / float(len(set(base_top)))
            overlaps.append(overlap)

        min_overlap = min(overlaps) if overlaps else 1.0
        report["checks"]["stability"] = {
            "top_n": stability_top_n,
            "overlaps": overlaps,
            "min_overlap": min_overlap
        }
        if min_overlap < 0.6:
            issues.append(
                f"stability overlap is too low: min_overlap={min_overlap:.2f}, required>=0.60"
            )
    else:
        report["checks"]["stability"] = {
            "skipped": True,
            "reason": "repeated_results not provided"
        }

    report["passed"] = len(issues) == 0
    return report


def _safe_num(value: Any, precision: int = 4, fallback: str = "—") -> str:
    val = _to_finite_float(value)
    if val is None:
        return fallback
    return f"{val:.{precision}f}"


def _build_partition_hash(labels: np.ndarray) -> str:
    """
    Строит хэш разбиения, инвариантный к переименованию меток кластеров.
    """
    if labels is None or len(labels) == 0:
        return "empty"

    labels_arr = np.asarray(labels, dtype=int).ravel()
    groups: dict[int, list[int]] = {}
    for row_idx, label in enumerate(labels_arr.tolist()):
        groups.setdefault(int(label), []).append(int(row_idx))

    canonical_groups: list[list[int]] = []
    for _, indices in groups.items():
        canonical_groups.append(sorted(indices))

    canonical_groups.sort(key=lambda idxs: (len(idxs), idxs[0] if idxs else -1, idxs))
    payload = json.dumps(canonical_groups, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _candidate_method_short(candidate: CandidateConfig) -> str:
    method = candidate.get("method", "")
    params = candidate.get("method_params", {}) or {}
    if method == "kmeans":
        return f"kmeans(k={params.get('kmeans_n_clusters', '?')})"
    if method == "hdbscan":
        return (
            f"hdbscan(mcs={params.get('hdbscan_min_cluster_size', '?')},"
            f" ms={params.get('hdbscan_min_samples', '?')})"
        )
    if method == "gmm":
        return (
            f"gmm(n={params.get('gmm_n_components', '?')},"
            f" cov={params.get('gmm_covariance_type', '?')})"
        )
    return str(method)


def _candidate_pca_short(candidate: CandidateConfig) -> str:
    if not candidate.get("pca_enabled"):
        return "off"
    pca_mode = candidate.get("pca_mode")
    pca_value = candidate.get("pca_value")
    if pca_mode == "variance_ratio":
        return f"var={_safe_num(pca_value, precision=2)}"
    if pca_mode == "fixed_components":
        try:
            return f"fix={int(float(pca_value))}"
        except (TypeError, ValueError):
            return "fix=?"
    return "on"


def _auto_context_cache_key(run_context: ClusterRunContext) -> tuple[str, int, str]:
    return (
        str(run_context.get("source_type", "gpr")),
        int(run_context.get("dataset_id", 0)),
        str(run_context.get("data_hash", "")),
    )


def _format_well_log_auto_diagnostics(diagnostics: dict[str, Any]) -> str:
    if not diagnostics:
        return "—"
    valid_wells = diagnostics.get("valid_well_count", "—")
    valid_rows = diagnostics.get("valid_row_count", "—")
    excluded = diagnostics.get("excluded_well_count", 0)
    invalid = diagnostics.get("invalid_row_count", 0)
    return f"wells={valid_wells}, rows={valid_rows}, excl={excluded}, invalid={invalid}"


def enrich_auto_results_for_context(
        results: list[CandidateResult],
        run_context: ClusterRunContext,
        *,
        auto_mode: str
) -> list[CandidateResult]:
    """Добавляет к AUTO-кандидатам источник/dataset для общей таблицы результатов."""
    enriched_results: list[CandidateResult] = []
    source_type = str(run_context.get("source_type", "gpr"))
    diagnostics = dict(run_context.get("diagnostics", {}) or {})
    for result in results or []:
        enriched = dict(result)
        enriched["source_type"] = source_type
        enriched["dataset_id"] = int(run_context.get("dataset_id", 0))
        enriched["dataset_title"] = str(run_context.get("dataset_title", ""))
        enriched["data_hash"] = str(run_context.get("data_hash", ""))
        enriched["auto_mode"] = str(auto_mode).upper()
        if source_type == "well_log":
            enriched["well_log_diagnostics"] = diagnostics
            enriched["well_log_diagnostics_text"] = _format_well_log_auto_diagnostics(diagnostics)
        enriched_results.append(enriched)
    return enriched_results


def remember_auto_results_for_context(run_context: ClusterRunContext, results: list[CandidateResult]) -> None:
    cluster_auto_results_by_context[_auto_context_cache_key(run_context)] = list(results or [])


def _cache_row_matches_source_type(cache_row: Any, source_type: str) -> bool:
    try:
        payload = json.loads(cache_row.top_results or "[]")
    except Exception:
        return False
    if not isinstance(payload, list) or not payload:
        return False
    row_source_type = str((payload[0] or {}).get("source_type") or "gpr").strip().lower()
    return row_source_type == str(source_type or "gpr").strip().lower()


def load_saved_auto_results_for_context(run_context: ClusterRunContext) -> bool:
    """Загружает последнюю сохраненную AUTO-таблицу для текущего GPR/Well Log контекста."""
    source_type = str(run_context.get("source_type") or "gpr").strip().lower()
    dataset_id = int(run_context.get("dataset_id", 0) or 0)
    if dataset_id <= 0:
        render_auto_results_table([])
        return False
    try:
        _ensure_cluster_auto_tuning_cache_table(source_type)
        cache_model = _auto_tuning_cache_model(source_type)
        id_filter = {"dataset_id" if cache_model is WellLogClusterAutoTuningCache else "object_set_id": dataset_id}
        cache_rows = (
            session.query(cache_model)
            .filter_by(**id_filter)
            .order_by(cache_model.created_at.desc())
            .all()
        )
    except Exception as exc:
        render_auto_results_table([])
        set_info(f"AUTO: ошибка чтения сохраненного результата: {exc}", "brown")
        return False

    for cache_row in cache_rows:
        if not _cache_row_matches_source_type(cache_row, source_type):
            continue
        try:
            cached_results = json.loads(cache_row.top_results or "[]")
        except Exception as exc:
            render_auto_results_table([])
            set_info(f"AUTO: ошибка распаковки сохраненного результата: {exc}", "brown")
            return False
        if not isinstance(cached_results, list):
            continue

        if source_type == "well_log":
            cached_results = enrich_auto_results_for_context(
                cached_results,
                run_context,
                auto_mode=str((cached_results[0] or {}).get("auto_mode") or "")
            )
        render_auto_results_table(cached_results)
        remember_auto_results_for_context(run_context, cached_results)
        set_info(
            f"AUTO: загружены сохраненные top-{len(cached_results)} настройки для выбранного набора.",
            "green"
        )
        return True

    render_auto_results_table([])
    return False


def refresh_cluster_auto_results_for_active_context() -> None:
    """Перерисовывает общую AUTO-таблицу для текущей вкладки/dataset из runtime/persistent cache."""
    run_context = build_cluster_run_context(show_errors=False)
    if run_context is None:
        render_auto_results_table([])
        return
    cached_results = cluster_auto_results_by_context.get(_auto_context_cache_key(run_context))
    if cached_results is not None:
        render_auto_results_table(cached_results)
        return
    load_saved_auto_results_for_context(run_context)


def render_auto_results_table(results: list[CandidateResult]) -> None:
    """
    Заполняет общую таблицу результатов AUTO-подбора для GPR и Well Log.
    """
    global cluster_auto_results_cache
    cluster_auto_results_cache = list(results or [])

    table = ui.tableWidget_cluster_auto_result
    headers = [
        "Rank", "Score", "Source", "Dataset", "Clusters", "Method", "Scaler", "PCA",
        "PCA comps", "Silhouette", "DB", "CH", "Noise %", "Well diagnostics", "PartHash", "Status"
    ]
    table.clear()
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setRowCount(len(results))

    center_columns = {0, 1, 2, 4, 8, 9, 10, 11, 12, 14, 15}
    status_col = headers.index("Status")

    for row_idx, result in enumerate(results):
        cfg = result.get("candidate_config", {})
        metrics = result.get("metrics", {})
        stats = result.get("stats", {})
        score_val = result.get("score")
        noise = _to_finite_float(stats.get("noise_fraction"))
        source_type = str(result.get("source_type") or "gpr")
        dataset_title = str(result.get("dataset_title") or result.get("dataset_id") or "—")
        well_diagnostics_text = str(result.get("well_log_diagnostics_text") or ("—" if source_type != "well_log" else "no diagnostics"))

        status_raw = str(result.get("status", "—"))
        status_view = {
            "ok": "OK",
            "invalid": "INVALID",
            "error": "ERROR"
        }.get(status_raw, status_raw.upper() if status_raw else "—")

        row_values = [
            str(row_idx + 1),
            _safe_num(score_val, precision=4),
            source_type,
            dataset_title,
            str(stats.get("n_clusters", "—")),
            _candidate_method_short(cfg),
            str(cfg.get("scaler_mode", "—")),
            _candidate_pca_short(cfg),
            str(stats.get("pca_components_after", "—")),
            _safe_num(metrics.get("silhouette"), precision=4),
            _safe_num(metrics.get("davies_bouldin"), precision=4),
            _safe_num(metrics.get("calinski_harabasz"), precision=2),
            (f"{(noise * 100.0):.1f}%" if noise is not None else "—"),
            well_diagnostics_text,
            str(stats.get("partition_hash", "—")),
            status_view
        ]

        tooltip_payload = {
            "candidate_id": result.get("candidate_id"),
            "source_type": source_type,
            "dataset_id": result.get("dataset_id"),
            "dataset_title": result.get("dataset_title"),
            "data_hash": result.get("data_hash"),
            "auto_mode": result.get("auto_mode"),
            "candidate_config": cfg,
            "well_log_diagnostics": result.get("well_log_diagnostics", {}),
            "error_text": result.get("error_text", "")
        }
        tooltip = json.dumps(tooltip_payload, ensure_ascii=False, default=str)

        for col_idx, value in enumerate(row_values):
            item = QTableWidgetItem(str(value))
            if col_idx in center_columns:
                item.setTextAlignment(Qt.AlignCenter)
            if col_idx == status_col:
                if status_raw == "ok":
                    item.setForeground(QBrush(QColor("darkgreen")))
                elif status_raw == "invalid":
                    item.setForeground(QBrush(QColor("darkorange")))
                else:
                    item.setForeground(QBrush(QColor("darkred")))
            item.setToolTip(tooltip)
            table.setItem(row_idx, col_idx, item)

    table.resizeColumnsToContents()



def clear_cluster_auto_tune_results_with_confirm() -> None:
    """
    Очищает таблицу результатов AUTO-подбора после подтверждения пользователя.
    """
    table = getattr(ui, "tableWidget_cluster_auto_result", None)
    if table is None:
        return

    global cluster_auto_results_cache
    active_context = build_cluster_run_context(show_errors=False)
    active_context_key = _auto_context_cache_key(active_context) if active_context is not None else None
    has_cached_results = len(cluster_auto_results_cache) > 0
    has_table_results = table.rowCount() > 0

    if not (has_cached_results or has_table_results):
        set_info("AUTO: таблица результатов уже пустая.", "brown")
        return

    reply = QMessageBox.warning(
        MainWindow,
        "Очистка результатов AUTO",
        "Вы действительно хотите очистить результаты автоподбора?\nЭто действие удалит строки из таблицы, runtime-cache и сохраненный кэш текущего набора.",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
    )

    if reply != QMessageBox.Yes:
        set_info("AUTO: очистка результатов отменена пользователем.", "brown")
        return

    cluster_auto_results_cache = []
    if active_context_key is not None:
        cluster_auto_results_by_context.pop(active_context_key, None)
    table.clear()
    table.setRowCount(0)
    table.setColumnCount(0)

    if active_context is not None:
        dataset_id = int(active_context.get("dataset_id", 0) or 0)
        source_type = str(active_context.get("source_type") or "gpr").strip().lower()
        if dataset_id > 0:
            try:
                _ensure_cluster_auto_tuning_cache_table(source_type)
                cache_model = _auto_tuning_cache_model(source_type)
                id_filter = {"dataset_id" if cache_model is WellLogClusterAutoTuningCache else "object_set_id": dataset_id}
                cache_rows = (
                    session.query(cache_model)
                    .filter_by(**id_filter)
                    .all()
                )
                for cache_row in cache_rows:
                    if _cache_row_matches_source_type(cache_row, source_type):
                        session.delete(cache_row)
                session.commit()
            except Exception as exc:
                session.rollback()
                set_info(f"AUTO: таблица очищена, но не удалось удалить сохраненный кэш: {exc}", "brown")
                return

    set_info("AUTO: результаты автоподбора очищены.", "green")

def apply_selected_auto_result_from_table(row_idx: int, _column_idx: int) -> None:
    """
    По двойному клику в таблице AUTO применяет выбранный вариант к настройкам UI.
    """
    if row_idx < 0 or row_idx >= len(cluster_auto_results_cache):
        set_info("AUTO: выбранная строка вне диапазона результатов.", "brown")
        return

    selected_result = cluster_auto_results_cache[row_idx]
    if selected_result.get("status") != "ok":
        set_info("AUTO: выбранный вариант невалиден и не может быть применен.", "brown")
        return

    apply_auto_result_to_ui(selected_result)
    selected_hash = str((selected_result.get("stats") or {}).get("partition_hash") or "—")
    set_info(
        f"AUTO: применен вариант #{row_idx + 1} "
        f"(score={_safe_num(selected_result.get('score'), precision=4)}, partition={selected_hash}).",
        "green"
    )


def apply_auto_result_to_ui(best_result: CandidateResult) -> None:
    """
    Применяет лучший найденный конфиг к контролам UI.
    """
    if not best_result or best_result.get("status") != "ok":
        return

    cfg = best_result.get("candidate_config", {})
    method_params = cfg.get("method_params", {}) or {}

    scaler_mode = cfg.get("scaler_mode")
    ui.radioButton_clust_scaler_none.setChecked(scaler_mode == "none")
    ui.radioButton_clust_scaler_stnd.setChecked(scaler_mode == "standard")
    ui.radioButton_clust_scaler_rob.setChecked(scaler_mode == "robust")
    ui.radioButton_clust_scaler_l2.setChecked(scaler_mode == "l2_norm")
    ui.radioButton_clust_scaler_row.setChecked(scaler_mode == "row_center")

    pca_enabled = bool(cfg.get("pca_enabled"))
    pca_mode = cfg.get("pca_mode")
    pca_value = cfg.get("pca_value")
    ui.checkBox_cluster_pca.setChecked(pca_enabled)
    if pca_enabled:
        ui.radioButton_clust_pca_fix.setChecked(pca_mode == "fixed_components")
        ui.radioButton_clust_pca_disp.setChecked(pca_mode != "fixed_components")
        if pca_mode == "fixed_components" and pca_value is not None:
            ui.spinBox_clust_pca_fix.setValue(max(2, int(float(pca_value))))
        elif pca_mode == "variance_ratio" and pca_value is not None:
            ui.doubleSpinBox_clust_pca_disp.setValue(float(pca_value))

    method = cfg.get("method")
    ui.radioButton_clust_kmean.setChecked(method == "kmeans")
    ui.radioButton_clust_hdbscan.setChecked(method == "hdbscan")
    ui.radioButton_clust_gaussmix.setChecked(method == "gmm")

    if method == "kmeans":
        if "kmeans_n_clusters" in method_params:
            ui.spinBox_clust_kmeans_n.setValue(int(method_params["kmeans_n_clusters"]))
        if "kmeans_n_init" in method_params:
            ui.spinBox_clust_kmean_ninint.setValue(int(method_params["kmeans_n_init"]))
    elif method == "hdbscan":
        if "hdbscan_min_cluster_size" in method_params:
            ui.spinBox_clust_hdbsc_minsize.setValue(int(method_params["hdbscan_min_cluster_size"]))
        if "hdbscan_min_samples" in method_params:
            ui.spinBox_clust_hdbsc_minsamp.setValue(int(method_params["hdbscan_min_samples"]))
        hdbscan_metric = method_params.get("hdbscan_metric")
        if hdbscan_metric is not None:
            idx = ui.comboBox_clust_hdbsc_type.findText(str(hdbscan_metric))
            if idx >= 0:
                ui.comboBox_clust_hdbsc_type.setCurrentIndex(idx)
    elif method == "gmm":
        if "gmm_n_components" in method_params:
            ui.spinBox_clust_gmm_n.setValue(int(method_params["gmm_n_components"]))
        gmm_cov_type = method_params.get("gmm_covariance_type")
        if gmm_cov_type is not None:
            idx = ui.comboBox_clust_gmm_type.findText(str(gmm_cov_type))
            if idx >= 0:
                ui.comboBox_clust_gmm_type.setCurrentIndex(idx)


def _read_auto_float_setting(control_name: str, fallback: float) -> float:
    """
    Безопасно читает float-настройку из UI-контрола (если он существует).
    """
    control = getattr(ui, control_name, None)
    if control is None:
        return float(fallback)
    try:
        return float(control.value())
    except Exception:
        return float(fallback)


def _read_auto_int_setting(control_name: str, fallback: int, minimum: int = 1) -> int:
    """
    Безопасно читает int-настройку из UI-контрола (если он существует).
    """
    control = getattr(ui, control_name, None)
    if control is None:
        return max(minimum, int(fallback))
    try:
        return max(minimum, int(control.value()))
    except Exception:
        return max(minimum, int(fallback))


def _read_auto_min_pca_components_setting(fallback: int = 2) -> int:
    """
    Читает минимальное число PCA-компонент для AUTO-подбора.
    """
    control = _get_ui_control_by_names("spinBox_cluster_pca_min_comp")
    if control is not None:
        try:
            value = int(control.value())
            return max(2, value)
        except Exception:
            return max(2, int(fallback))
    return max(2, int(fallback))


def calculate_cluster_auto():
    """
    Запускает AUTO-подбор параметров кластеризации из UI.
    """
    run_context = build_cluster_run_context(show_errors=True)
    if run_context is None:
        return

    source_type = str(run_context["source_type"])
    dataset_id = int(run_context["dataset_id"])
    use_persistent_auto_cache = source_type in {"gpr", "well_log"}
    if source_type == "gpr":
        clust_object = session.query(ObjectSet).filter_by(id=dataset_id).first()
        if clust_object is None:
            set_info(f"AUTO: объект id={dataset_id} не найден.", "brown")
            return
    elif source_type == "well_log":
        dataset = session.query(WellLogClusterDataset).filter_by(id=dataset_id).first()
        if dataset is None:
            set_info(f"AUTO: Well Log dataset id={dataset_id} не найден.", "brown")
            return

    base_data = run_context["raw_rows"]
    if not base_data:
        set_info("AUTO: пустой набор данных для подбора.", "brown")
        return

    auto_random_seed = random.SystemRandom().randrange(1, 2_147_483_647)
    original_rows_count = len(base_data)
    sample_idx = _build_sample_indices_for_auto_tuning(original_rows_count, AUTO_TUNING_MAX_ROWS, seed=auto_random_seed)
    sampled_base_data = _apply_sample_indices(base_data, sample_idx)
    sampled_rows_count = len(sampled_base_data) if sampled_base_data is not None else 0
    if sampled_rows_count < original_rows_count:
        set_info(
            f"AUTO: для устойчивости расчетов использована подвыборка {sampled_rows_count}/{original_rows_count} строк.",
            "brown"
        )
    base_data = sampled_base_data

    sample_limits = calculate_auto_min_cluster_sample_limits(len(base_data), min_value=1)

    auto_mode = "COARSE" if ui.radioButton_cluster_coarse_auto.isChecked() else "FINE"
    selected_button = ui.buttonGroup_3.checkedButton()
    text_method_nan = selected_button.text() if selected_button else "impute"

    total_timeout_toggle = getattr(ui, "checkBox_cluster_auto_total_time", None)
    candidate_timeout_toggle = getattr(ui, "checkBox_cluster_auto_candidate_time", None)
    use_total_timeout = bool(total_timeout_toggle.isChecked()) if total_timeout_toggle is not None else False
    use_candidate_timeout = bool(candidate_timeout_toggle.isChecked()) if candidate_timeout_toggle is not None else False

    total_timeout_ctrl = getattr(ui, "spinBox_cluster_auto_timeout_all", None)
    per_candidate_timeout_ctrl = getattr(ui, "spinBox_cluster_auto_timeout_candidate", None)

    total_timeout_sec = (
        float(total_timeout_ctrl.value())
        if (use_total_timeout and total_timeout_ctrl is not None)
        else None
    )
    candidate_timeout_sec = (
        float(per_candidate_timeout_ctrl.value())
        if (use_candidate_timeout and per_candidate_timeout_ctrl is not None)
        else None
    )

    auto_apply_toggle = getattr(ui, "checkBox_cluster_auto_apply_best", None)
    auto_apply_best = bool(auto_apply_toggle.isChecked()) if auto_apply_toggle is not None else True
    force_recompute_toggle = getattr(ui, "checkBox_cluster_auto_recalc", None)
    force_recompute = bool(force_recompute_toggle.isChecked()) if force_recompute_toggle is not None else False

    limit_candidates_toggle = _get_ui_control_by_names(
        "checkBox_cluster_auto_limit200",
        "checkBox_cluster_auto_limit_candidates",
        "checkBox_cluster_auto_limit_200"
    )
    use_candidates_limit = bool(limit_candidates_toggle.isChecked()) if limit_candidates_toggle is not None else True
    max_candidates_value = _read_auto_int_setting("spinBox_cluster_auto_max_candidates", fallback=200, minimum=1)
    max_candidates = max_candidates_value if use_candidates_limit else None
    top_k = _read_auto_int_setting("spinBox_cluster_auto_top_results", fallback=5, minimum=1)
    max_clusters = _read_auto_int_setting("spinBox_cluster_auto_max_cluster", fallback=8, minimum=2)
    min_pca_components = _read_auto_min_pca_components_setting(fallback=2)
    min_cluster_samples = _read_auto_int_setting(
        "spinBox_cluster_auto_min_n_cluster",
        fallback=sample_limits["recommended_default_value"],
        minimum=sample_limits["min_cluster_samples"]
    )
    min_cluster_samples = min(min_cluster_samples, sample_limits["max_spinbox_value"])
    hdbscan_metric = str(ui.comboBox_clust_hdbsc_type.currentText() or "euclidean")
    hdbscan_metrics = [hdbscan_metric]
    for idx in range(ui.comboBox_clust_hdbsc_type.count()):
        metric_name = str(ui.comboBox_clust_hdbsc_type.itemText(idx)).strip()
        if metric_name and metric_name not in hdbscan_metrics:
            hdbscan_metrics.append(metric_name)
    scaler_only_toggle = getattr(ui, "checkBox_cluster_auto_scaler_only", None)
    pca_only_toggle = getattr(ui, "checkBox_cluster_auto_pca_only", None)
    scaler_only = bool(scaler_only_toggle.isChecked()) if scaler_only_toggle is not None else False
    pca_only = bool(pca_only_toggle.isChecked()) if pca_only_toggle is not None else False
    metric_weights = {
        "silhouette": _read_auto_float_setting("doubleSpinBox_cluster_auto_w_sil", fallback=0.4),
        "davies_bouldin": _read_auto_float_setting("doubleSpinBox_cluster_auto_w_db", fallback=0.3),
        "calinski_harabasz": _read_auto_float_setting("doubleSpinBox_cluster_auto_w_ch", fallback=0.3)
    }
    cache_key = build_cluster_auto_tuning_cache_key(
        clust_object_id=dataset_id,
        auto_mode=auto_mode,
        max_candidates=max_candidates_value if use_candidates_limit else 0,
        top_k=top_k,
        constraints={
            "use_candidates_limit": bool(use_candidates_limit),
            "max_candidates_value": max_candidates_value,
            "max_clusters": max_clusters,
            "hdbscan_metric": hdbscan_metric,
            "hdbscan_metrics": hdbscan_metrics,
            "min_pca_components": min_pca_components,
            "scaler_only": scaler_only,
            "pca_only": pca_only,
            "use_total_timeout": bool(use_total_timeout),
            "use_candidate_timeout": bool(use_candidate_timeout),
            "total_timeout_sec": total_timeout_sec,
            "candidate_timeout_sec": candidate_timeout_sec,
            "min_cluster_samples": min_cluster_samples,
            "recommended_min_cluster_samples": sample_limits["recommended_default_value"],
            "max_min_cluster_samples": sample_limits["max_spinbox_value"]
        },
        weights=metric_weights,
        clean_kwargs={
            "use_non_finite": ui.checkBox_clust_clean_nan.isChecked(),
            "non_finite_mode": text_method_nan,
            "use_variance_threshold": ui.checkBox_clust_clear_vartresh.isChecked(),
            "use_correlation_filter": ui.checkBox_clust_clear_corr.isChecked()
        },
        source_type=source_type
    )

    render_auto_results_table([])
    set_info(f"AUTO: запуск подбора ({auto_mode})...", "blue")
    if not use_total_timeout and not use_candidate_timeout:
        set_info("AUTO: лимиты времени отключены (будут рассчитаны все кандидаты).", "blue")
    set_info(
        (
            f"AUTO: настройки подбора max_candidates={max_candidates if max_candidates is not None else 'OFF'}, "
            f"top_k={top_k}, "
            f"max_clusters={max_clusters}, min_pca_components={min_pca_components}, "
            f"min_cluster_samples={min_cluster_samples} (recommended={sample_limits['recommended_default_value']}, "
            f"max={sample_limits['max_spinbox_value']}), "
            f"hdbscan_metrics={hdbscan_metrics}, "
            f"scaler_only={scaler_only}, pca_only={pca_only}, "
            f"weights(sil/db/ch)=({metric_weights['silhouette']:.2f}/"
            f"{metric_weights['davies_bouldin']:.2f}/{metric_weights['calinski_harabasz']:.2f})."
        ),
        "blue"
    )
    QApplication.processEvents()

    if not force_recompute and use_persistent_auto_cache:
        cached_top_results = load_cluster_auto_tuning_cache(
            cache_key=cache_key,
            clust_object_id=dataset_id,
            top_k=top_k,
            source_type=source_type
        )
        if cached_top_results:
            cached_top_results = select_diverse_top_results(
                rank_candidates(cached_top_results, weights=metric_weights, max_clusters=max_clusters),
                top_k=top_k,
                diversity_key="partition_hash"
            )
        if cached_top_results:
            enriched_cached_results = enrich_auto_results_for_context(cached_top_results, run_context, auto_mode=auto_mode)
            remember_auto_results_for_context(run_context, enriched_cached_results)
            render_auto_results_table(enriched_cached_results)
            best_result = enriched_cached_results[0]
            if auto_apply_best:
                apply_auto_result_to_ui(best_result)
            set_info(
                f"AUTO {auto_mode}: использованы сохраненные top-{len(enriched_cached_results)} настройки.",
                "green"
            )
            return

    tuning_result = run_auto_cluster_tuning(
        base_data=base_data,
        auto_mode=auto_mode,
        max_candidates=max_candidates,
        top_k=top_k,
        max_clusters=max_clusters,
        hdbscan_metric=hdbscan_metric,
        hdbscan_metrics=hdbscan_metrics,
        scaler_only=scaler_only,
        pca_only=pca_only,
        min_pca_components=min_pca_components,
        weights=metric_weights,
        soft_timeout_sec=total_timeout_sec,
        candidate_soft_timeout_sec=candidate_timeout_sec,
        random_seed=auto_random_seed,
        min_cluster_samples=min_cluster_samples,
        run_key=cache_key if source_type == "gpr" else None,
        object_set_id=dataset_id if source_type == "gpr" else None,
        clean_kwargs={
            "use_non_finite": ui.checkBox_clust_clean_nan.isChecked(),
            "non_finite_mode": text_method_nan,
            "use_variance_threshold": ui.checkBox_clust_clear_vartresh.isChecked(),
            "use_correlation_filter": ui.checkBox_clust_clear_corr.isChecked()
        }
    )

    top_results = tuning_result.get("top_results", [])
    enriched_top_results = enrich_auto_results_for_context(top_results, run_context, auto_mode=auto_mode)
    if use_persistent_auto_cache:
        save_cluster_auto_tuning_cache(
            cache_key=cache_key,
            clust_object_id=dataset_id,
            top_results=top_results,
            top_k=top_k,
            source_type=source_type,
            dataset_title=str(run_context.get("dataset_title", "")),
            data_hash=str(run_context.get("data_hash", "")),
            auto_mode=auto_mode,
            diagnostics=dict(run_context.get("diagnostics", {}) or {})
        )
    remember_auto_results_for_context(run_context, enriched_top_results)
    render_auto_results_table(enriched_top_results)
    best_result = enriched_top_results[0] if enriched_top_results else tuning_result.get("best_result")
    raw_results = tuning_result.get("raw_results", []) or []
    dropped_candidates = sum(1 for row in raw_results if row.get("status") != "ok")
    total_candidates = len(raw_results)
    if total_candidates > 0:
        set_info(
            f"AUTO {auto_mode}: отброшено кандидатов {dropped_candidates}/{total_candidates}.",
            "blue"
        )

    if not best_result or best_result.get("status") != "ok":
        set_info(
            "AUTO: не найдено валидных конфигураций. Уменьшите порог минимального размера кластера или используйте 'Сброс к 5%'.",
            "brown"
        )
        return

    if auto_apply_best:
        apply_auto_result_to_ui(best_result)

    best_cfg = best_result.get("candidate_config", {})
    best_metrics = best_result.get("metrics", {})
    apply_message = (
        "Параметры применены в UI, нажмите CALC для расчета."
        if auto_apply_best
        else "Авто-применение отключено: проверьте top в таблице и примените вручную."
    )
    set_info(
        (
            f"AUTO {auto_mode}: лучший score={_safe_num(best_result.get('score'), 4)} | "
            f"method={_candidate_method_short(best_cfg)} | "
            f"sil={_safe_num(best_metrics.get('silhouette'), 3)} | "
            f"db={_safe_num(best_metrics.get('davies_bouldin'), 3)} | "
            f"ch={_safe_num(best_metrics.get('calinski_harabasz'), 1)}. "
            f"{apply_message}"
        ),
        "green"
    )


def calculate_cluster_auto_batch() -> None:
    """
    Запускает AUTO-подбор в batch-режиме для активного источника Cluster.

    Как CALC/AUTO, сначала определяет активную вкладку Cluster:
    - GPR/георадар: обрабатывает все ObjectSet текущего анализа;
    - Well Log/каротаж: обрабатывает все WellLogClusterDataset с собранными data.
    """
    source_type = get_active_cluster_source_type()
    if source_type == "well_log":
        _calculate_cluster_auto_batch_well_log()
        return
    _calculate_cluster_auto_batch_gpr()


def _read_auto_batch_settings() -> dict[str, Any]:
    """
    Считывает общие настройки AUTO BATCH из UI.
    """
    auto_mode = "COARSE" if ui.radioButton_cluster_coarse_auto.isChecked() else "FINE"
    selected_button = ui.buttonGroup_3.checkedButton()
    text_method_nan = selected_button.text() if selected_button else "impute"
    force_recompute_toggle = getattr(ui, "checkBox_cluster_auto_recalc", None)
    force_recompute = bool(force_recompute_toggle.isChecked()) if force_recompute_toggle is not None else False

    total_timeout_toggle = getattr(ui, "checkBox_cluster_auto_total_time", None)
    candidate_timeout_toggle = getattr(ui, "checkBox_cluster_auto_candidate_time", None)
    use_total_timeout = bool(total_timeout_toggle.isChecked()) if total_timeout_toggle is not None else False
    use_candidate_timeout = bool(candidate_timeout_toggle.isChecked()) if candidate_timeout_toggle is not None else False
    total_timeout_ctrl = getattr(ui, "spinBox_cluster_auto_timeout_all", None)
    per_candidate_timeout_ctrl = getattr(ui, "spinBox_cluster_auto_timeout_candidate", None)
    total_timeout_sec = float(total_timeout_ctrl.value()) if (use_total_timeout and total_timeout_ctrl is not None) else None
    candidate_timeout_sec = (
        float(per_candidate_timeout_ctrl.value())
        if (use_candidate_timeout and per_candidate_timeout_ctrl is not None)
        else None
    )

    limit_candidates_toggle = _get_ui_control_by_names(
        "checkBox_cluster_auto_limit200",
        "checkBox_cluster_auto_limit_candidates",
        "checkBox_cluster_auto_limit_200"
    )
    use_candidates_limit = bool(limit_candidates_toggle.isChecked()) if limit_candidates_toggle is not None else True
    max_candidates_value = _read_auto_int_setting("spinBox_cluster_auto_max_candidates", fallback=200, minimum=1)
    max_candidates = max_candidates_value if use_candidates_limit else None
    top_k = _read_auto_int_setting("spinBox_cluster_auto_top_results", fallback=5, minimum=1)
    max_clusters = _read_auto_int_setting("spinBox_cluster_auto_max_cluster", fallback=8, minimum=2)
    min_pca_components = _read_auto_min_pca_components_setting(fallback=2)
    hdbscan_metric = str(ui.comboBox_clust_hdbsc_type.currentText() or "euclidean")
    hdbscan_metrics = [hdbscan_metric]
    for idx in range(ui.comboBox_clust_hdbsc_type.count()):
        metric_name = str(ui.comboBox_clust_hdbsc_type.itemText(idx)).strip()
        if metric_name and metric_name not in hdbscan_metrics:
            hdbscan_metrics.append(metric_name)
    scaler_only_toggle = getattr(ui, "checkBox_cluster_auto_scaler_only", None)
    pca_only_toggle = getattr(ui, "checkBox_cluster_auto_pca_only", None)
    scaler_only = bool(scaler_only_toggle.isChecked()) if scaler_only_toggle is not None else False
    pca_only = bool(pca_only_toggle.isChecked()) if pca_only_toggle is not None else False
    metric_weights = {
        "silhouette": _read_auto_float_setting("doubleSpinBox_cluster_auto_w_sil", fallback=0.4),
        "davies_bouldin": _read_auto_float_setting("doubleSpinBox_cluster_auto_w_db", fallback=0.3),
        "calinski_harabasz": _read_auto_float_setting("doubleSpinBox_cluster_auto_w_ch", fallback=0.3)
    }
    clean_kwargs = {
        "use_non_finite": ui.checkBox_clust_clean_nan.isChecked(),
        "non_finite_mode": text_method_nan,
        "use_variance_threshold": ui.checkBox_clust_clear_vartresh.isChecked(),
        "use_correlation_filter": ui.checkBox_clust_clear_corr.isChecked()
    }
    return {
        "auto_mode": auto_mode,
        "text_method_nan": text_method_nan,
        "force_recompute": force_recompute,
        "use_total_timeout": use_total_timeout,
        "use_candidate_timeout": use_candidate_timeout,
        "total_timeout_sec": total_timeout_sec,
        "candidate_timeout_sec": candidate_timeout_sec,
        "use_candidates_limit": use_candidates_limit,
        "max_candidates_value": max_candidates_value,
        "max_candidates": max_candidates,
        "top_k": top_k,
        "max_clusters": max_clusters,
        "min_pca_components": min_pca_components,
        "hdbscan_metric": hdbscan_metric,
        "hdbscan_metrics": hdbscan_metrics,
        "scaler_only": scaler_only,
        "pca_only": pca_only,
        "metric_weights": metric_weights,
        "clean_kwargs": clean_kwargs,
    }


def _build_auto_batch_cache_key(
        *,
        dataset_id: int,
        settings: dict[str, Any],
        min_cluster_samples: int,
        sample_limits: dict[str, int],
        source_type: str,
) -> str:
    return build_cluster_auto_tuning_cache_key(
        clust_object_id=dataset_id,
        auto_mode=settings["auto_mode"],
        max_candidates=settings["max_candidates_value"] if settings["use_candidates_limit"] else 0,
        top_k=settings["top_k"],
        constraints={
            "use_candidates_limit": bool(settings["use_candidates_limit"]),
            "max_candidates_value": settings["max_candidates_value"],
            "max_clusters": settings["max_clusters"],
            "hdbscan_metric": settings["hdbscan_metric"],
            "hdbscan_metrics": settings["hdbscan_metrics"],
            "min_pca_components": settings["min_pca_components"],
            "scaler_only": settings["scaler_only"],
            "pca_only": settings["pca_only"],
            "use_total_timeout": bool(settings["use_total_timeout"]),
            "use_candidate_timeout": bool(settings["use_candidate_timeout"]),
            "total_timeout_sec": settings["total_timeout_sec"],
            "candidate_timeout_sec": settings["candidate_timeout_sec"],
            "min_cluster_samples": min_cluster_samples,
            "recommended_min_cluster_samples": sample_limits["recommended_default_value"],
            "max_min_cluster_samples": sample_limits["max_spinbox_value"]
        },
        weights=settings["metric_weights"],
        clean_kwargs=settings["clean_kwargs"],
        source_type=source_type
    )


def _calculate_cluster_auto_batch_gpr() -> None:
    """
    Запускает AUTO-подбор последовательно для всех ObjectSet в текущем анализе.
    """
    clust_analys_id = get_curr_clust_analys_id()
    if not str(clust_analys_id).isdigit():
        set_info("AUTO BATCH: не выбран набор кластерного анализа.", "brown")
        return

    clust_objects = (
        session.query(ObjectSet)
        .filter_by(analysis_id=int(clust_analys_id))
        .order_by(ObjectSet.id)
        .all()
    )
    if not clust_objects:
        set_info("AUTO BATCH: нет добавленных наборов объектов для обработки.", "brown")
        return

    settings = _read_auto_batch_settings()
    _run_auto_batch_for_contexts(
        source_type="gpr",
        items=[(int(clust_obj.id), f"object_set_id={int(clust_obj.id)}", clust_obj) for clust_obj in clust_objects],
        settings=settings,
        context_loader=lambda clust_obj: {
            "dataset_id": int(clust_obj.id),
            "dataset_title": f"object_set_id={int(clust_obj.id)}",
            "raw_rows": _deserialize_cluster_dataset(clust_obj.data),
            "data_hash": "",
            "diagnostics": {},
        },
        empty_message="пустой набор данных",
        use_checkpoint=True,
    )


def _calculate_cluster_auto_batch_well_log() -> None:
    """
    Запускает AUTO-подбор последовательно для всех Well Log datasets.
    """
    datasets = (
        session.query(WellLogClusterDataset)
        .order_by(WellLogClusterDataset.created_at, WellLogClusterDataset.id)
        .all()
    )
    if not datasets:
        set_info("AUTO BATCH Well Log: нет наборов каротажа для обработки.", "brown")
        return

    settings = _read_auto_batch_settings()
    _run_auto_batch_for_contexts(
        source_type="well_log",
        items=[(int(dataset.id), str(dataset.name or f"dataset_id={int(dataset.id)}"), dataset) for dataset in datasets],
        settings=settings,
        context_loader=lambda dataset: build_well_log_cluster_context(int(dataset.id)),
        empty_message="пустой Well Log data",
        use_checkpoint=False,
    )


def _run_auto_batch_for_contexts(
        *,
        source_type: str,
        items: list[tuple[int, str, Any]],
        settings: dict[str, Any],
        context_loader,
        empty_message: str,
        use_checkpoint: bool,
) -> None:
    total_objects = len(items)
    started_at = monotonic()
    skipped_cached = 0
    calculated = 0
    failed = 0
    auto_mode = settings["auto_mode"]
    source_label = "Well Log" if source_type == "well_log" else "GPR"

    set_info(
        f"AUTO BATCH {source_label} {auto_mode}: старт по {total_objects} datasets | "
        f"retune={settings['force_recompute']} | apply_auto_best=OFF.",
        "blue"
    )

    for idx, (dataset_id, dataset_name, payload) in enumerate(items, start=1):
        object_name = f"{dataset_name} (id={dataset_id})"
        try:
            run_context = context_loader(payload)
            base_data = run_context.get("raw_rows", []) if isinstance(run_context, dict) else []
        except Exception as exc:
            failed += 1
            set_info(f"AUTO BATCH {source_label} {auto_mode} [{idx}/{total_objects}] {object_name}: FAILED (чтение данных: {exc}).", "red")
            continue

        if not base_data:
            failed += 1
            set_info(f"AUTO BATCH {source_label} {auto_mode} [{idx}/{total_objects}] {object_name}: FAILED ({empty_message}).", "brown")
            continue

        auto_random_seed = random.SystemRandom().randrange(1, 2_147_483_647)
        original_rows_count = len(base_data)
        sample_idx = _build_sample_indices_for_auto_tuning(original_rows_count, AUTO_TUNING_MAX_ROWS, seed=auto_random_seed)
        base_data = _apply_sample_indices(base_data, sample_idx)
        sampled_rows_count = len(base_data) if base_data is not None else 0
        if sampled_rows_count < original_rows_count:
            set_info(
                f"AUTO BATCH {source_label} {auto_mode} [{idx}/{total_objects}] {object_name}: "
                f"использована подвыборка {sampled_rows_count}/{original_rows_count} строк.",
                "brown"
            )

        sample_limits = calculate_auto_min_cluster_sample_limits(len(base_data), min_value=1)
        min_cluster_samples = int(sample_limits["recommended_default_value"])
        cache_key = _build_auto_batch_cache_key(
            dataset_id=dataset_id,
            settings=settings,
            min_cluster_samples=min_cluster_samples,
            sample_limits=sample_limits,
            source_type=source_type,
        )

        if not settings["force_recompute"]:
            cached_top_results = load_cluster_auto_tuning_cache(
                cache_key=cache_key,
                clust_object_id=dataset_id,
                top_k=settings["top_k"],
                source_type=source_type,
            )
            if cached_top_results:
                cached_top_results = select_diverse_top_results(
                    rank_candidates(cached_top_results, weights=settings["metric_weights"], max_clusters=settings["max_clusters"]),
                    top_k=settings["top_k"],
                    diversity_key="partition_hash"
                )
            if cached_top_results:
                skipped_cached += 1
                best_cached = cached_top_results[0] if cached_top_results else {}
                set_info(
                    f"AUTO BATCH {source_label} {auto_mode} [{idx}/{total_objects}] {object_name}: "
                    f"SKIPPED_CACHED top={len(cached_top_results)} score={_safe_num(best_cached.get('score'), 4)} "
                    f"min_cluster_samples={min_cluster_samples}.",
                    "green"
                )
                continue

        tuning_result = run_auto_cluster_tuning(
            base_data=base_data,
            auto_mode=auto_mode,
            max_candidates=settings["max_candidates"],
            top_k=settings["top_k"],
            max_clusters=settings["max_clusters"],
            hdbscan_metric=settings["hdbscan_metric"],
            hdbscan_metrics=settings["hdbscan_metrics"],
            scaler_only=settings["scaler_only"],
            pca_only=settings["pca_only"],
            min_pca_components=settings["min_pca_components"],
            weights=settings["metric_weights"],
            soft_timeout_sec=settings["total_timeout_sec"],
            candidate_soft_timeout_sec=settings["candidate_timeout_sec"],
            random_seed=auto_random_seed,
            min_cluster_samples=min_cluster_samples,
            run_key=cache_key if use_checkpoint else None,
            object_set_id=dataset_id if use_checkpoint else None,
            clean_kwargs=settings["clean_kwargs"]
        )
        top_results = tuning_result.get("top_results", [])
        save_cluster_auto_tuning_cache(
            cache_key=cache_key,
            clust_object_id=dataset_id,
            top_results=top_results,
            top_k=settings["top_k"],
            source_type=source_type,
            dataset_title=str(run_context.get("dataset_title", object_name)) if isinstance(run_context, dict) else object_name,
            data_hash=str(run_context.get("data_hash", "")) if isinstance(run_context, dict) else "",
            auto_mode=auto_mode,
            diagnostics=dict(run_context.get("diagnostics", {}) or {}) if isinstance(run_context, dict) else {},
        )

        best_result = tuning_result.get("best_result")
        if not best_result or best_result.get("status") != "ok":
            failed += 1
            set_info(
                f"AUTO BATCH {source_label} {auto_mode} [{idx}/{total_objects}] {object_name}: FAILED "
                f"(нет валидных конфигураций) min_cluster_samples={min_cluster_samples}.",
                "brown"
            )
            continue

        calculated += 1
        best_metrics = best_result.get("metrics", {})
        best_cfg = best_result.get("candidate_config", {})
        set_info(
            f"AUTO BATCH {source_label} {auto_mode} [{idx}/{total_objects}] {object_name}: CALCULATED "
            f"score={_safe_num(best_result.get('score'), 4)} "
            f"method={_candidate_method_short(best_cfg)} "
            f"sil={_safe_num(best_metrics.get('silhouette'), 3)} "
            f"db={_safe_num(best_metrics.get('davies_bouldin'), 3)} "
            f"ch={_safe_num(best_metrics.get('calinski_harabasz'), 1)} "
            f"min_cluster_samples={min_cluster_samples}.",
            "blue"
        )
        QApplication.processEvents()

    elapsed_sec = monotonic() - started_at
    set_info(
        f"AUTO BATCH {source_label} {auto_mode}: завершено за {elapsed_sec:.1f} сек | total={total_objects}, "
        f"calculated={calculated}, skipped_cached={skipped_cached}, failed={failed}.",
        "green" if failed == 0 else "brown"
    )


