from __future__ import annotations

from .common import *

def make_candidate_config(
        *,
        scaler_mode: str,
        pca_enabled: bool,
        pca_mode: Optional[str],
        pca_value: Optional[float],
        method: Literal["kmeans", "hdbscan", "gmm"],
        method_params: Optional[Dict[str, Any]] = None
) -> CandidateConfig:
    """
    Собирает и валидирует словарь candidate_config.
    """
    if pca_enabled and pca_mode not in ("fixed_components", "variance_ratio"):
        raise ValueError("pca_mode must be 'fixed_components' or 'variance_ratio' when pca_enabled=True")
    if pca_enabled and pca_value is None:
        raise ValueError("pca_value is required when pca_enabled=True")
    if not pca_enabled:
        pca_mode = None
        pca_value = None

    return CandidateConfig(
        scaler_mode=scaler_mode,
        pca_enabled=bool(pca_enabled),
        pca_mode=pca_mode,
        pca_value=pca_value,
        method=method,
        method_params=dict(method_params or {})
    )


def make_candidate_result(
        *,
        candidate_id: str,
        candidate_config: CandidateConfig,
        metrics: Optional[CandidateMetrics] = None,
        stats: Optional[CandidateStats] = None,
        score: Optional[float] = None,
        status: Literal["ok", "invalid", "error"] = "ok",
        error_text: str = ""
) -> CandidateResult:
    """
    Собирает унифицированный словарь candidate_result.
    """
    return CandidateResult(
        candidate_id=str(candidate_id),
        candidate_config=candidate_config,
        metrics=metrics or CandidateMetrics(),
        stats=stats or CandidateStats(),
        score=score,
        status=status,
        error_text=error_text
    )


def _set_auto_info(message: str, color: str = "blue") -> None:
    """
    Безопасный логгер для этапов AUTO-подбора.
    """
    try:
        set_info(message, color)
    except Exception:
        print(message)


def _apply_candidate_limit(
        candidates: list[CandidateConfig],
        *,
        max_candidates: Optional[int],
        scope_label: str,
        random_seed: Optional[int] = None
) -> list[CandidateConfig]:
    """
    Применяет лимит кандидатов.
    Если лимит активен — выбирает случайную подвыборку без повторений.
    """
    if max_candidates is None:
        _set_auto_info(f"{scope_label}: лимит кандидатов отключен ({len(candidates)}).", "blue")
        return candidates

    max_candidates = max(1, int(max_candidates))
    total_candidates = len(candidates)
    if total_candidates <= max_candidates:
        _set_auto_info(f"{scope_label}: размер search space = {total_candidates}.", "blue")
        return candidates

    # Стратифицированная выборка: сохраняем представительство методов,
    # чтобы случайный сэмпл не состоял только из "тяжелых"/неудачных кандидатов.
    method_groups: Dict[str, list[CandidateConfig]] = {}
    for candidate in candidates:
        method_name = str(candidate.get("method", "unknown"))
        method_groups.setdefault(method_name, []).append(candidate)

    sampled_candidates: list[CandidateConfig] = []
    methods = sorted(method_groups.keys())
    if methods:
        per_method_quota = max(1, max_candidates // len(methods))
        if random_seed is None:
            random_seed = random.SystemRandom().randrange(1, 2_147_483_647)
        rng = random.Random(int(random_seed))
        for method_name in methods:
            method_candidates = method_groups.get(method_name, [])
            take = min(len(method_candidates), per_method_quota)
            if take > 0:
                sampled_candidates.extend(rng.sample(method_candidates, take))
        if len(sampled_candidates) < max_candidates:
            sampled_signatures = {
                _candidate_signature(candidate)
                for candidate in sampled_candidates
            }
            remainder_pool = [
                candidate for candidate in candidates
                if _candidate_signature(candidate) not in sampled_signatures
            ]
            need = max_candidates - len(sampled_candidates)
            if need > 0 and remainder_pool:
                sampled_candidates.extend(rng.sample(remainder_pool, min(need, len(remainder_pool))))
    else:
        sampled_candidates = random.sample(candidates, max_candidates)

    _set_auto_info(
        f"{scope_label}: сгенерировано {total_candidates} кандидатов, "
        f"случайно выбрано {max_candidates} (seed={random_seed}).",
        "brown"
    )
    return sampled_candidates


def _build_auto_rescue_candidates(max_clusters: int) -> list[CandidateConfig]:
    """
    Резервный небольшой набор "надежных" кандидатов, если основной сэмпл
    не дал ни одной валидной конфигурации.
    """
    max_clusters = max(2, int(max_clusters))
    rescue_candidates: list[CandidateConfig] = []
    for scaler_mode, pca_enabled, k in product(
            ("none", "standard", "robust"),
            (False, True),
            range(2, min(8, max_clusters) + 1)
    ):
        candidate = make_candidate_config(
            scaler_mode=scaler_mode,
            pca_enabled=pca_enabled,
            pca_mode="variance_ratio" if pca_enabled else None,
            pca_value=0.9 if pca_enabled else None,
            method="kmeans",
            method_params={
                "kmeans_n_clusters": int(k),
                "kmeans_n_init": 20
            }
        )
        rescue_candidates.append(candidate)
    return rescue_candidates


def _summarize_candidate_failures(results: list[CandidateResult], top_n: int = 5) -> list[tuple[str, int]]:
    """
    Возвращает топ причин невалидности/ошибок кандидатов.
    """
    counter: Counter[str] = Counter()
    for row in results:
        status = str(row.get("status", ""))
        if status == "ok":
            continue
        reason = str(row.get("error_text", "")).strip() or f"status={status}"
        counter[reason] += 1
    return counter.most_common(max(1, int(top_n)))


def build_auto_search_space(
        auto_mode: str,
        max_candidates: Optional[int] = 200,
        *,
        max_clusters: int = 8,
        hdbscan_metric: str = "euclidean",
        hdbscan_metrics: Optional[list[str]] = None,
        scaler_only: bool = False,
        pca_only: bool = False,
        random_seed: Optional[int] = None
) -> list[CandidateConfig]:
    """
    Формирует coarse/fine пространство кандидатов для AUTO-подбора.

    На текущем этапе реализована coarse-сетка из плана.
    Для режима FINE возвращается эта же coarse-сетка (точная локальная
    окрестность будет добавлена на следующем этапе).
    """
    mode = (auto_mode or "").strip().upper()
    if mode not in {"COARSE", "FINE"}:
        raise ValueError(f"Unsupported auto_mode='{auto_mode}'. Expected 'COARSE' or 'FINE'.")

    scaler_modes = (
        ("standard", "robust", "l2_norm", "row_center")
        if scaler_only
        else ("none", "standard", "robust", "l2_norm", "row_center")
    )
    pca_variance_values = (0.85, 0.90, 0.95)
    max_clusters = max(2, int(max_clusters))
    # Для coarse-режима используем более широкий перебор fixed_components,
    # но ограничиваем размер сетки, чтобы не раздувать search space.
    fixed_upper = max(6, min(24, max_clusters * 2))
    fixed_step = 2 if fixed_upper <= 12 else 3
    pca_fixed_values = tuple(range(2, fixed_upper + 1, fixed_step))
    pca_variants = []
    if not pca_only:
        pca_variants.append({"pca_enabled": False, "pca_mode": None, "pca_value": None})
    pca_variants.extend(
        {"pca_enabled": True, "pca_mode": "variance_ratio", "pca_value": pca_value}
        for pca_value in pca_variance_values
    )
    pca_variants.extend(
        {"pca_enabled": True, "pca_mode": "fixed_components", "pca_value": float(n_comp)}
        for n_comp in pca_fixed_values
    )
    candidates: list[CandidateConfig] = []

    # KMeans: k=2..max_clusters, n_init={10,20,50}.
    for scaler_mode, pca_variant, k, n_init in product(
            scaler_modes,
            pca_variants,
            range(2, max_clusters + 1),
            (10, 20, 50)
    ):
        candidates.append(
            make_candidate_config(
                scaler_mode=scaler_mode,
                pca_enabled=pca_variant["pca_enabled"],
                pca_mode=pca_variant["pca_mode"],
                pca_value=pca_variant["pca_value"],
                method="kmeans",
                method_params={
                    "kmeans_n_clusters": k,
                    "kmeans_n_init": int(n_init)
                }
            )
        )

    hdbscan_metric_values = [
        str(metric).strip()
        for metric in (hdbscan_metrics or [hdbscan_metric])
        if str(metric).strip()
    ]
    if not hdbscan_metric_values:
        hdbscan_metric_values = ["euclidean"]
    hdbscan_metric_values = list(dict.fromkeys(hdbscan_metric_values))

    # HDBSCAN: min_cluster_size={10,20,40}, min_samples={3,5,10}, metric={...}.
    for scaler_mode, pca_variant, min_cluster_size, min_samples, metric in product(
            scaler_modes, pca_variants, (10, 20, 40), (3, 5, 10), hdbscan_metric_values
    ):
        candidates.append(
            make_candidate_config(
                scaler_mode=scaler_mode,
                pca_enabled=pca_variant["pca_enabled"],
                pca_mode=pca_variant["pca_mode"],
                pca_value=pca_variant["pca_value"],
                method="hdbscan",
                method_params={
                    "hdbscan_min_cluster_size": min_cluster_size,
                    "hdbscan_min_samples": min_samples,
                    "hdbscan_metric": str(metric or "euclidean")
                }
            )
        )

    # GMM: n_components=2..max_clusters, covariance_type={full,diag,tied,spherical}.
    for scaler_mode, pca_variant, n_components, covariance_type in product(
            scaler_modes, pca_variants, range(2, max_clusters + 1), GMM_COVARIANCE_TYPES
    ):
        candidates.append(
            make_candidate_config(
                scaler_mode=scaler_mode,
                pca_enabled=pca_variant["pca_enabled"],
                pca_mode=pca_variant["pca_mode"],
                pca_value=pca_variant["pca_value"],
                method="gmm",
                method_params={
                    "gmm_n_components": n_components,
                    "gmm_covariance_type": covariance_type
                }
            )
        )

    return _apply_candidate_limit(
        candidates,
        max_candidates=max_candidates,
        scope_label=f"AUTO {mode}",
        random_seed=random_seed
    )


def run_cluster_candidate(
        base_data,
        candidate: CandidateConfig,
        *,
        candidate_id: str = "",
        clean_kwargs: Optional[Dict[str, Any]] = None,
        transform_cache: Optional[Dict[tuple, Any]] = None,
        preprocess_cache: Optional[Dict[tuple, Any]] = None,
        preprocess_rank_cache: Optional[Dict[tuple, int]] = None,
        metrics_cache: Optional[Dict[str, CandidateMetrics]] = None,
        min_pca_components: int = 2,
        max_silhouette_samples: int = AUTO_SILHOUETTE_MAX_SAMPLES,
        min_cluster_samples: int = 1
) -> CandidateResult:
    """
    Прогоняет одного кандидата AUTO-подбора без UI-зависимостей.

    Этапы:
    1) clean_features
    2) preprocess_features
    3) apply_pca (опционально)
    4) cluster_data
    5) evaluate_clustering
    """
    clean_params = {
        "use_non_finite": True,
        "non_finite_mode": "impute",
        "use_variance_threshold": False,
        "use_correlation_filter": False
    }
    if clean_kwargs:
        clean_params.update(clean_kwargs)

    try:
        base_data = _sample_rows_for_auto_tuning(base_data, AUTO_TUNING_MAX_ROWS)
        clear_data, _ = clean_features(data=base_data, **clean_params)
        clear_data = _reduce_feature_space_for_auto_tuning(clear_data, AUTO_TUNING_MAX_FEATURES)
    except Exception as exc:
        return make_candidate_result(
            candidate_id=candidate_id,
            candidate_config=candidate,
            status="error",
            error_text=f"clean_features failed: {exc}"
        )

    if clear_data is None or len(clear_data) == 0:
        return make_candidate_result(
            candidate_id=candidate_id,
            candidate_config=candidate,
            status="invalid",
            error_text="empty sample after cleaning"
        )

    clean_key = tuple(sorted(clean_params.items()))
    try:
        min_pca_components_value = max(2, int(min_pca_components))
    except (TypeError, ValueError):
        min_pca_components_value = 2
    preprocess_key = (
        clean_key,
        candidate["scaler_mode"]
    )
    transform_key = (
        preprocess_key,
        bool(candidate["pca_enabled"]),
        candidate.get("pca_mode"),
        candidate.get("pca_value"),
        min_pca_components_value
    )
    data_for_cluster = None
    pca_components_after: Optional[int] = None
    if transform_cache is not None:
        cached_value = transform_cache.get(transform_key)
        if cached_value is not None:
            data_for_cluster, pca_components_after = cached_value

    if data_for_cluster is None:
        preprocess_data = None
        if preprocess_cache is not None:
            preprocess_data = preprocess_cache.get(preprocess_key)
        if preprocess_data is None:
            try:
                preprocess_data = preprocess_features(clear_data, mode=candidate["scaler_mode"])
            except Exception as exc:
                return make_candidate_result(
                    candidate_id=candidate_id,
                    candidate_config=candidate,
                    status="error",
                    error_text=f"preprocess_features failed: {exc}"
                )
            if preprocess_cache is not None:
                preprocess_cache[preprocess_key] = preprocess_data
                if hasattr(preprocess_cache, "move_to_end"):
                    preprocess_cache.move_to_end(preprocess_key, last=True)  # type: ignore[attr-defined]

        try:
            if candidate["pca_enabled"]:
                rank_precheck = None
                if preprocess_rank_cache is not None:
                    rank_precheck = preprocess_rank_cache.get(preprocess_key)
                pca_mode = candidate["pca_mode"] or "variance_ratio"
                pca_raw_value = candidate.get("pca_value")
                pca_n_components = 20
                pca_variance_ratio = 0.9

                n_rows_pre = int(len(preprocess_data)) if preprocess_data is not None else 0
                n_features_pre = int(len(preprocess_data[0])) if n_rows_pre > 0 else 0
                max_possible_components_geom = max(0, min(n_features_pre, max(0, n_rows_pre - 1)))

                pilot_data = _build_pilot_sample_for_pca(
                    preprocess_data,
                    max_rows=int(AUTO_PCA_PILOT_MAX_ROWS),
                    seed=42
                )
                pilot_rows = int(len(pilot_data)) if pilot_data is not None else 0
                rank_cache_key = (preprocess_key, int(AUTO_PCA_PILOT_MAX_ROWS))
                rank_precheck = None
                if preprocess_rank_cache is not None:
                    rank_precheck = preprocess_rank_cache.get(rank_cache_key)
                if rank_precheck is None:
                    pilot_arr = np.asarray(pilot_data) if pilot_rows > 0 else np.asarray([])
                    rank_precheck = int(np.linalg.matrix_rank(pilot_arr)) if pilot_arr.size > 0 else 0
                    if preprocess_rank_cache is not None:
                        preprocess_rank_cache[rank_cache_key] = int(rank_precheck)
                max_possible_components = int(min(max_possible_components_geom, max(0, int(rank_precheck))))
                if max_possible_components < min_pca_components_value:
                    _set_auto_info(
                        f"AUTO PRECHECK {candidate_id or 'candidate'}: PCA skipped "
                        f"(pilot_rank={int(rank_precheck or 0)}, max_comp={max_possible_components}, "
                        f"min_required={min_pca_components_value}, pilot_rows={pilot_rows}).",
                        "brown"
                    )
                    return make_candidate_result(
                        candidate_id=candidate_id,
                        candidate_config=candidate,
                        stats=CandidateStats(
                            pca_components_after=max_possible_components
                        ),
                        status="invalid",
                        error_text=(
                            f"pca skipped: max possible components {max_possible_components} "
                            f"< min_pca_components {min_pca_components_value}"
                        )
                    )
                if pca_mode == "fixed_components":
                    pca_n_components = int(float(pca_raw_value))
                    if pca_n_components < 1 or pca_n_components > max_possible_components:
                        return make_candidate_result(
                            candidate_id=candidate_id,
                            candidate_config=candidate,
                            status="invalid",
                            error_text=(
                                f"pca n_components {pca_n_components} out of range [1, {max_possible_components}]"
                            )
                        )
                else:
                    pca_variance_ratio = float(pca_raw_value)
                    if not (0.0 < pca_variance_ratio <= 1.0):
                        return make_candidate_result(
                            candidate_id=candidate_id,
                            candidate_config=candidate,
                            status="invalid",
                            error_text=(
                                f"pca variance_ratio {pca_variance_ratio} out of range (0, 1]"
                            )
                        )
                    if bool(AUTO_PCA_PILOT_ENABLED):
                        try:
                            _, pilot_info = apply_pca(
                                pilot_data,
                                mode="variance_ratio",
                                variance_ratio=pca_variance_ratio
                            )
                            pilot_components = int(pilot_info.get("components_after_pca", 0) or 0)
                        except Exception as exc:
                            return make_candidate_result(
                                candidate_id=candidate_id,
                                candidate_config=candidate,
                                status="invalid",
                                error_text=f"pca pilot failed: {exc}"
                            )
                        if pilot_components < min_pca_components_value:
                            _set_auto_info(
                                f"AUTO PRECHECK {candidate_id or 'candidate'}: PCA pilot skipped "
                                f"(pilot_comp={pilot_components}, min_required={min_pca_components_value}, "
                                f"pilot_rows={pilot_rows}).",
                                "brown"
                            )
                            return make_candidate_result(
                                candidate_id=candidate_id,
                                candidate_config=candidate,
                                stats=CandidateStats(
                                    pca_components_after=pilot_components
                                ),
                                status="invalid",
                                error_text=(
                                    f"pca pilot components {pilot_components} "
                                    f"< min_pca_components {min_pca_components_value}"
                                )
                            )
                data_for_cluster, pca_info = apply_pca(
                    preprocess_data,
                    mode=pca_mode,
                    n_components=pca_n_components,
                    variance_ratio=pca_variance_ratio
                )
                pca_components_after = int(pca_info.get("components_after_pca", 0) or 0)
                if pca_components_after < min_pca_components_value:
                    return make_candidate_result(
                        candidate_id=candidate_id,
                        candidate_config=candidate,
                        stats=CandidateStats(
                            pca_components_after=pca_components_after
                        ),
                        status="invalid",
                        error_text=(
                            f"pca components {pca_components_after} "
                            f"< min_pca_components {min_pca_components_value}"
                        )
                    )
            else:
                data_for_cluster = preprocess_data
        except Exception as exc:
            return make_candidate_result(
                candidate_id=candidate_id,
                candidate_config=candidate,
                status="error",
                error_text=f"apply_pca failed: {exc}"
            )

        if transform_cache is not None:
            transform_cache[transform_key] = (data_for_cluster, pca_components_after)
            if hasattr(transform_cache, "move_to_end"):
                transform_cache.move_to_end(transform_key, last=True)  # type: ignore[attr-defined]
    pca_stats: Dict[str, int] = {}
    if pca_components_after is not None:
        pca_stats["pca_components_after"] = int(pca_components_after)

    try:
        labels, cluster_info = cluster_data(
            data=data_for_cluster,
            method=candidate["method"],
            **candidate["method_params"]
        )
    except Exception as exc:
        return make_candidate_result(
            candidate_id=candidate_id,
            candidate_config=candidate,
            status="error",
            error_text=f"cluster_data failed: {exc}"
        )

    labels_np = np.asarray(labels, dtype=int)
    partition_hash = _build_partition_hash(labels_np)
    mask_eval = labels_np != -1
    labels_eval = labels_np[mask_eval]
    unique_clusters_eval = np.unique(labels_eval)
    n_samples_eval = int(np.count_nonzero(mask_eval))

    try:
        min_cluster_samples_value = max(1, int(min_cluster_samples))
    except (TypeError, ValueError):
        min_cluster_samples_value = 1

    if n_samples_eval == 0:
        return make_candidate_result(
            candidate_id=candidate_id,
            candidate_config=candidate,
            stats=CandidateStats(
                n_clusters=int(cluster_info.get("n_clusters", 0)),
                noise_fraction=float(cluster_info.get("noise_fraction", 0.0)),
                n_samples_eval=0,
                partition_hash=partition_hash,
                **pca_stats
            ),
            status="invalid",
            error_text="empty sample after noise filtering"
        )

    cluster_size_min = int(min((labels_eval == label).sum() for label in unique_clusters_eval)) if len(unique_clusters_eval) > 0 else 0
    if cluster_size_min < min_cluster_samples_value:
        return make_candidate_result(
            candidate_id=candidate_id,
            candidate_config=candidate,
            stats=CandidateStats(
                n_clusters=int(cluster_info.get("n_clusters", 0)),
                noise_fraction=float(cluster_info.get("noise_fraction", 0.0)),
                n_samples_eval=n_samples_eval,
                partition_hash=partition_hash,
                **pca_stats
            ),
            status="invalid",
            error_text=(
                f"cluster min size {cluster_size_min} < min_cluster_samples {min_cluster_samples_value}"
            )
        )

    if len(unique_clusters_eval) < 2:
        return make_candidate_result(
            candidate_id=candidate_id,
            candidate_config=candidate,
            stats=CandidateStats(
                n_clusters=int(cluster_info.get("n_clusters", 0)),
                noise_fraction=float(cluster_info.get("noise_fraction", 0.0)),
                n_samples_eval=n_samples_eval,
                partition_hash=partition_hash,
                **pca_stats
            ),
            status="invalid",
            error_text="less than 2 clusters after noise filtering"
        )

    if metrics_cache is not None:
        cached_metrics = metrics_cache.get(partition_hash)
        if cached_metrics is not None:
            return make_candidate_result(
                candidate_id=candidate_id,
                candidate_config=candidate,
                metrics=CandidateMetrics(
                    silhouette=float(cached_metrics.get("silhouette")),
                    davies_bouldin=float(cached_metrics.get("davies_bouldin")),
                    calinski_harabasz=float(cached_metrics.get("calinski_harabasz"))
                ),
                stats=CandidateStats(
                    n_clusters=int(cluster_info.get("n_clusters", 0)),
                    noise_fraction=float(cluster_info.get("noise_fraction", 0.0)),
                    n_samples_eval=n_samples_eval,
                    partition_hash=partition_hash,
                    **pca_stats
                ),
                status="ok",
                error_text=""
            )

    try:
        eval_info = evaluate_clustering(
            data_for_cluster,
            labels,
            use_silhouette=True,
            use_db=True,
            use_ch=True,
            max_silhouette_samples=max_silhouette_samples
        )
    except Exception as exc:
        return make_candidate_result(
            candidate_id=candidate_id,
            candidate_config=candidate,
            status="error",
            error_text=f"evaluate_clustering failed: {exc}"
        )

    metrics = eval_info.get("metrics", {})
    if not metrics:
        return make_candidate_result(
            candidate_id=candidate_id,
            candidate_config=candidate,
            stats=CandidateStats(
                n_clusters=int(cluster_info.get("n_clusters", 0)),
                noise_fraction=float(cluster_info.get("noise_fraction", 0.0)),
                n_samples_eval=n_samples_eval,
                partition_hash=partition_hash,
                **pca_stats
            ),
            status="invalid",
            error_text="metrics unavailable for candidate"
        )

    metric_payload = CandidateMetrics(
        silhouette=float(metrics.get("silhouette")),
        davies_bouldin=float(metrics.get("davies_bouldin")),
        calinski_harabasz=float(metrics.get("calinski_harabasz"))
    )
    if metrics_cache is not None:
        metrics_cache[partition_hash] = metric_payload

    return make_candidate_result(
        candidate_id=candidate_id,
        candidate_config=candidate,
        metrics=metric_payload,
        stats=CandidateStats(
            n_clusters=int(cluster_info.get("n_clusters", 0)),
            noise_fraction=float(cluster_info.get("noise_fraction", 0.0)),
            n_samples_eval=n_samples_eval,
            partition_hash=partition_hash,
            **pca_stats
        ),
        status="ok",
        error_text=""
    )


def _to_finite_float(value: Any) -> Optional[float]:
    """
    Приводит значение к float, если оно конечно. Иначе возвращает None.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def score_candidate(
        metrics: CandidateMetrics,
        stats: CandidateStats,
        weights: Optional[Dict[str, float]] = None
) -> Optional[float]:
    """
    Рассчитывает суммарный score кандидата.

    Нормализация:
    - silhouette: линейно из [-1, 1] в [0, 1]
    - davies_bouldin: инверсия 1 / (1 + db)
    - calinski_harabasz: rank-like насыщение ch / (ch + 1000)

    Штрафы:
    - высокий noise_fraction
    - слишком мало/слишком много кластеров
    """
    w = {"silhouette": 0.4, "davies_bouldin": 0.3, "calinski_harabasz": 0.3}
    if weights:
        w.update(weights)

    sil = _to_finite_float(metrics.get("silhouette"))
    db = _to_finite_float(metrics.get("davies_bouldin"))
    ch = _to_finite_float(metrics.get("calinski_harabasz"))
    if sil is None or db is None or ch is None:
        return None

    sil_norm = _clip01((sil + 1.0) / 2.0)
    db_norm = _clip01(1.0 / (1.0 + max(db, 0.0)))
    ch_norm = _clip01(ch / (ch + 1000.0)) if ch > 0 else 0.0
    score = (
        float(w["silhouette"]) * sil_norm
        + float(w["davies_bouldin"]) * db_norm
        + float(w["calinski_harabasz"]) * ch_norm
    )

    noise_fraction = _to_finite_float(stats.get("noise_fraction")) or 0.0
    noise_fraction = _clip01(noise_fraction)
    n_clusters = int(stats.get("n_clusters", 0) or 0)

    penalty = 0.0
    if noise_fraction > 0.30:
        penalty += min(0.20, (noise_fraction - 0.30) * 0.40)
    if n_clusters < 2:
        penalty += 0.30
    elif n_clusters == 2:
        penalty += 0.05
    elif n_clusters > 12:
        penalty += min(0.20, (n_clusters - 12) * 0.02)

    return max(0.0, float(score - penalty))


def rank_candidates(
        results: list[CandidateResult],
        weights: Optional[Dict[str, float]] = None
) -> list[CandidateResult]:
    """
    Выставляет score и возвращает отсортированный список кандидатов.

    Правила tie-break:
    1) выше score
    2) выше silhouette
    3) ниже davies_bouldin
    4) ниже noise_fraction
    """
    ranked: list[CandidateResult] = []
    for result in results:
        result_copy = CandidateResult(
            candidate_id=result["candidate_id"],
            candidate_config=result["candidate_config"],
            metrics=result.get("metrics", CandidateMetrics()),
            stats=result.get("stats", CandidateStats()),
            score=result.get("score"),
            status=result.get("status", "error"),
            error_text=result.get("error_text", "")
        )

        if result_copy["status"] == "ok":
            result_copy["score"] = score_candidate(
                result_copy.get("metrics", CandidateMetrics()),
                result_copy.get("stats", CandidateStats()),
                weights=weights
            )
            if result_copy["score"] is None:
                result_copy["status"] = "invalid"
                result_copy["error_text"] = (
                    (result_copy.get("error_text", "") + "; ").strip("; ")
                    + "score is unavailable"
                ).strip()
        else:
            result_copy["score"] = None

        ranked.append(result_copy)

    def _sort_key(row: CandidateResult):
        is_ok = row.get("status") == "ok" and row.get("score") is not None
        metrics = row.get("metrics", {})
        stats = row.get("stats", {})

        score_val = float(row["score"]) if row.get("score") is not None else float("-inf")
        sil = _to_finite_float(metrics.get("silhouette"))
        db = _to_finite_float(metrics.get("davies_bouldin"))
        noise = _to_finite_float(stats.get("noise_fraction"))

        return (
            1 if is_ok else 0,
            score_val,
            sil if sil is not None else float("-inf"),
            -(db if db is not None else float("inf")),
            -(noise if noise is not None else float("inf"))
        )

    ranked.sort(key=_sort_key, reverse=True)
    return ranked


def select_diverse_top_results(
        ranked_results: list[CandidateResult],
        *,
        top_k: int,
        diversity_key: str = "partition_hash"
) -> list[CandidateResult]:
    """
    Выбирает top-K с дедупликацией по ключу разнообразия (по умолчанию partition_hash).

    Логика:
    1) сначала добавляются валидные уникальные результаты (status=ok);
    2) затем, если уникальных меньше top_k, список добивается следующими лучшими
       (включая дубликаты/невалидные), чтобы сохранить длину leaderboard.
    """
    target_k = max(1, int(top_k))
    selected: list[CandidateResult] = []
    selected_ids: set[str] = set()
    seen_diversity_values: set[str] = set()

    for row in ranked_results:
        if len(selected) >= target_k:
            break
        if row.get("status") != "ok":
            continue

        stats = row.get("stats") or {}
        diversity_value = str(stats.get(diversity_key) or "").strip()
        if not diversity_value:
            diversity_value = f"candidate:{_candidate_signature(row.get('candidate_config', {}))}"
        if diversity_value in seen_diversity_values:
            continue

        candidate_id = str(row.get("candidate_id"))
        selected.append(row)
        selected_ids.add(candidate_id)
        seen_diversity_values.add(diversity_value)

    if len(selected) < target_k:
        for row in ranked_results:
            if len(selected) >= target_k:
                break
            candidate_id = str(row.get("candidate_id"))
            if candidate_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(candidate_id)

    return selected


def _candidate_signature(candidate: CandidateConfig) -> tuple:
    """
    Возвращает hashable-подпись кандидата для дедупликации.
    """
    method_params_tuple = tuple(sorted((candidate.get("method_params") or {}).items()))
    return (
        candidate.get("scaler_mode"),
        bool(candidate.get("pca_enabled")),
        candidate.get("pca_mode"),
        candidate.get("pca_value"),
        candidate.get("method"),
        method_params_tuple
    )


def build_fine_search_space(
        top_results: list[CandidateResult],
        *,
        top_k: int = 5,
        max_candidates: Optional[int] = 200,
        hdbscan_metrics: Optional[list[str]] = None
) -> list[CandidateConfig]:
    """
    Строит локальное fine-пространство вокруг top-K coarse-кандидатов.
    """
    fine_candidates: list[CandidateConfig] = []
    seen_signatures = set()

    for result in top_results[:max(1, int(top_k))]:
        if result.get("status") != "ok":
            continue
        cfg = result.get("candidate_config")
        if not cfg:
            continue

        base_scaler = cfg["scaler_mode"]
        base_method = cfg["method"]
        base_method_params = dict(cfg.get("method_params", {}))

        # Локальные варианты PCA.
        pca_variants = [{
            "pca_enabled": bool(cfg.get("pca_enabled")),
            "pca_mode": cfg.get("pca_mode"),
            "pca_value": cfg.get("pca_value")
        }]
        if cfg.get("pca_enabled") and cfg.get("pca_mode") == "variance_ratio":
            try:
                base_vr = float(cfg.get("pca_value"))
                for delta in (-0.02, 0.02):
                    vr = round(base_vr + delta, 3)
                    if 0.50 <= vr <= 0.99:
                        pca_variants.append({
                            "pca_enabled": True,
                            "pca_mode": "variance_ratio",
                            "pca_value": vr
                        })
            except (TypeError, ValueError):
                pass
        elif cfg.get("pca_enabled") and cfg.get("pca_mode") == "fixed_components":
            try:
                base_n_comp = int(float(cfg.get("pca_value")))
                for delta in (-2, -1, 1, 2):
                    n_comp = max(2, base_n_comp + delta)
                    pca_variants.append({
                        "pca_enabled": True,
                        "pca_mode": "fixed_components",
                        "pca_value": float(n_comp)
                    })
            except (TypeError, ValueError):
                pass

        method_variants: list[Dict[str, Any]] = []
        if base_method == "kmeans":
            base_k = int(base_method_params.get("kmeans_n_clusters", 4))
            base_n_init = int(base_method_params.get("kmeans_n_init", 10))
            for k in range(max(2, base_k - 2), base_k + 3):
                for n_init in sorted({base_n_init, 20, base_n_init + 10}):
                    method_variants.append({
                        "kmeans_n_clusters": int(k),
                        "kmeans_n_init": int(max(10, n_init))
                    })
        elif base_method == "hdbscan":
            base_mcs = int(base_method_params.get("hdbscan_min_cluster_size", 20))
            base_ms = int(base_method_params.get("hdbscan_min_samples", 5))
            base_metric = str(base_method_params.get("hdbscan_metric", "euclidean"))
            for delta_mcs in (-10, -5, 0, 5, 10):
                for delta_ms in (-2, -1, 0, 1, 2):
                    method_variants.append({
                        "hdbscan_min_cluster_size": int(max(5, base_mcs + delta_mcs)),
                        "hdbscan_min_samples": int(max(1, base_ms + delta_ms)),
                        "hdbscan_metric": base_metric
                    })
        elif base_method == "gmm":
            base_n = int(base_method_params.get("gmm_n_components", 4))
            base_cov = str(base_method_params.get("gmm_covariance_type", "full"))
            for n_components in range(max(2, base_n - 2), base_n + 3):
                method_variants.append({
                    "gmm_n_components": int(n_components),
                    "gmm_covariance_type": base_cov
                })
        else:
            continue

        for pca_variant in pca_variants:
            for method_params in method_variants:
                candidate = make_candidate_config(
                    scaler_mode=base_scaler,
                    pca_enabled=pca_variant["pca_enabled"],
                    pca_mode=pca_variant["pca_mode"],
                    pca_value=pca_variant["pca_value"],
                    method=base_method,
                    method_params=method_params
                )
                sig = _candidate_signature(candidate)
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)
                fine_candidates.append(candidate)

    return _apply_candidate_limit(
        fine_candidates,
        max_candidates=max_candidates,
        scope_label="AUTO FINE"
    )


def _cluster_candidate_worker(payload: dict, out_queue) -> None:
    try:
        result = run_cluster_candidate(**payload)
        out_queue.put({"ok": True, "result": result})
    except Exception as exc:
        out_queue.put({"ok": False, "error": f"subprocess exception: {exc}"})


def run_cluster_candidate_isolated(
        *,
        hard_timeout_sec: Optional[float] = AUTO_CANDIDATE_HARD_TIMEOUT_SEC,
        **kwargs
) -> CandidateResult:
    start_methods = mp.get_all_start_methods()
    if "fork" not in start_methods:
        # Windows fallback: избегаем spawn, чтобы дочерний процесс не инициализировал GUI.
        return run_cluster_candidate(**kwargs)
    ctx = mp.get_context("fork")
    q = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_cluster_candidate_worker, args=(kwargs, q), daemon=True)
    proc.start()
    if hard_timeout_sec is None:
        proc.join()
    else:
        proc.join(timeout=max(1.0, float(hard_timeout_sec)))
    candidate_id = str(kwargs.get("candidate_id") or "")
    candidate = kwargs.get("candidate")
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=1.0)
        return make_candidate_result(candidate_id=candidate_id, candidate_config=candidate, status="invalid", error_text=f"hard-timeout>{hard_timeout_sec:.1f}s (isolated worker killed)")
    if proc.exitcode not in (0, None):
        return make_candidate_result(candidate_id=candidate_id, candidate_config=candidate, status="error", error_text=f"isolated worker exitcode={proc.exitcode}")
    try:
        msg = q.get_nowait()
    except Exception:
        return make_candidate_result(candidate_id=candidate_id, candidate_config=candidate, status="error", error_text="isolated worker: empty result")
    if not msg.get("ok"):
        return make_candidate_result(candidate_id=candidate_id, candidate_config=candidate, status="error", error_text=str(msg.get("error") or "isolated worker unknown error"))
    return msg.get("result")


