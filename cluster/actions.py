from __future__ import annotations

from .common import *

def calculate_well_log_cluster(
        run_context: ClusterRunContext,
        config: dict[str, Any] | None = None
) -> WellLogClusterVisualizationData | None:
    """
    Выполняет ручной CALC для Well Log через общий pipeline clean → scale → PCA → cluster → metrics.
    """
    data = run_context["raw_rows"]
    config = config or _read_manual_cluster_ui_config()
    cache_key = build_cluster_calculation_cache_key(
        source_type="well_log",
        dataset_id=int(run_context["dataset_id"]),
        data_hash=str(run_context.get("data_hash", "")),
        config=config,
    )
    cached_result = load_cluster_calculation_cache(
        source_type="well_log",
        dataset_id=int(run_context["dataset_id"]),
        cache_key=cache_key,
        data_hash=str(run_context.get("data_hash", "")),
        config=config,
    )
    cached_base = get_cached_base_calculation(cached_result, result_type="well_log")
    params = config["method_params"]

    if cached_base is not None:
        base_labels = cached_base["labels"]
        kept_row_indices = cached_base["kept_row_indices"]
        data_pca = cached_base["data_for_diagnostics"]
        clust_info = cached_base["cluster_info"]
        pca_info_report = cached_base["pca_info_report"]
        set_info(
            f"CALC Well Log: CACHE HIT ({cached_result.get('_cache_lookup_mode', 'cache_key')}), "
            "базовые метки загружены из базы; очистка, PCA и кластеризация пропущены.",
            "green",
        )
    else:
        set_info(
            f"CALC Well Log: CACHE MISS, сохраненная база не найдена или неполна "
            f"(key={cache_key[:12]}, data={str(run_context.get('data_hash', ''))[:12]}).",
            "brown",
        )
        clear_data, kept_row_indices = clean_features(data=data, **config["clean"])
        if not clear_data:
            set_info("CALC Well Log: после очистки не осталось строк для кластеризации.", "brown")
            return None

        set_info(
            f"CALC Well Log: очистка данных {len(data)} → {len(clear_data)} строк, "
            f"признаков после очистки={len(clear_data[0]) if clear_data else 0}.",
            "blue"
        )
        preprocess_data = preprocess_features(clear_data, mode=config["preprocess_mode"])

        pca_info_report: dict[str, Any] = {}
        if config["pca"]["enabled"]:
            if config["pca"]["mode"] == "fixed_components":
                data_pca, pca_info_report = apply_pca(
                    preprocess_data,
                    mode="fixed_components",
                    n_components=config["pca"]["fixed_components"],
                    variance_ratio=config["pca"]["variance_ratio"],
                )
            else:
                data_pca, pca_info_report = apply_pca(
                    preprocess_data,
                    mode="variance_ratio",
                    n_components=config["pca"]["fixed_components"],
                    variance_ratio=config["pca"]["variance_ratio"],
                )
        else:
            data_pca = preprocess_data

        label_list, clust_info = cluster_data(
            data=data_pca,
            method=config["method"],
            kmeans_n_clusters=params["kmeans_n_clusters"],
            kmeans_n_init=params["kmeans_n_init"],
            hdbscan_min_cluster_size=params["hdbscan_min_cluster_size"],
            hdbscan_min_samples=params["hdbscan_min_samples"],
            hdbscan_metric=params["hdbscan_metric"],
            gmm_n_components=params["gmm_n_components"],
            gmm_covariance_type=params["gmm_covariance_type"],
        )
        base_labels = [int(label) for label in label_list]
        meta_rows = run_context.get("meta", [])
        assignments = []
        for clean_row_idx, cluster_label in enumerate(base_labels):
            if clean_row_idx >= len(kept_row_indices):
                break
            source_row_idx = int(kept_row_indices[clean_row_idx])
            if source_row_idx < 0 or source_row_idx >= len(meta_rows):
                continue
            meta = meta_rows[source_row_idx]
            assignments.append({
                "source_row_index": source_row_idx,
                "well_id": int(meta["well_id"]),
                "depth_md": float(meta["depth_md"]),
                "cluster_label": int(cluster_label),
            })
        cached_result = {
            "result_type": "well_log",
            "labels": base_labels,
            "kept_row_indices": [int(value) for value in kept_row_indices],
            "assignments": assignments,
            "data_for_diagnostics": np.asarray(data_pca).tolist(),
            "cluster_info": clust_info,
            "pca_info_report": pca_info_report,
        }
        cache_saved = save_cluster_calculation_cache(
            source_type="well_log",
            dataset_id=int(run_context["dataset_id"]),
            cache_key=cache_key,
            data_hash=str(run_context.get("data_hash", "")),
            config=config,
            result_payload=cached_result,
        )
        if cache_saved:
            set_info(
                f"CALC Well Log: сохранены базовые метки до сглаживания: {len(base_labels)} строк.",
                "blue",
            )
        else:
            set_info("CALC Well Log: базовые метки не удалось сохранить в БД.", "red")

    # Smoothing must never mutate the labels stored in the database.
    labels_for_output = list(base_labels)

    well_trace_rows: dict[int, dict[int, int]] = {}
    meta_rows = run_context.get("meta", [])
    for clean_row_idx, _ in enumerate(labels_for_output):
        if clean_row_idx >= len(kept_row_indices):
            break
        source_row_idx = int(kept_row_indices[clean_row_idx])
        if source_row_idx < 0 or source_row_idx >= len(meta_rows):
            continue
        meta = meta_rows[source_row_idx]
        well_id = int(meta["well_id"])
        row_index = int(meta.get("row_index_in_well", clean_row_idx))
        well_trace_rows.setdefault(well_id, {})[row_index] = int(clean_row_idx)

    smoothing_changes = 0
    smooth_window = _normalize_smoothing_window(config["smoothing"]["window"])
    smoothing_applied = bool(config["smoothing"]["enabled"] and smooth_window >= 3)
    if smoothing_applied:
        labels_for_output, smoothing_changes = _smooth_labels_by_profile_trace(
            labels_for_output,
            well_trace_rows,
            method=config["smoothing"]["method"],
            window=smooth_window,
            preserve_noise=True,
        )
        set_info(
            f"CALC Well Log: smoothing применен ({config['smoothing']['method']}, window={smooth_window}), "
            f"изменено меток: {smoothing_changes}.",
            "blue"
        )
    elif config["smoothing"]["enabled"]:
        set_info("CALC Well Log: smoothing включен, но окно слишком маленькое. Постобработка пропущена.", "brown")

    cached_postprocess = get_cached_postprocess_result(cached_result, config)
    cached_output_labels = (
        [int(value) for value in cached_postprocess.get("labels", [])]
        if cached_postprocess else []
    )
    if cached_postprocess and cached_output_labels == labels_for_output:
        result_eval = dict(cached_postprocess.get("evaluation") or {})
        set_info("CALC Well Log: оценочные показатели загружены из базы данных.", "green")
    else:
        result_eval = evaluate_clustering(
            data_pca,
            labels_for_output,
            use_silhouette=config["metrics"]["use_silhouette"],
            use_db=config["metrics"]["use_db"],
            use_ch=config["metrics"]["use_ch"],
        )
    report_text = build_clustering_report(
        preprocess_mode=config["preprocess_mode"],
        pca_mode=config["pca"]["mode"],
        pca_info=pca_info_report,
        cluster_info={
            "method": config["method"],
            "kmeans_n": params["kmeans_n_clusters"],
            "kmeans_n_init": params["kmeans_n_init"],
            "min_size": params["hdbscan_min_cluster_size"],
            "min_sample": params["hdbscan_min_samples"],
            "hdbscan_type": params["hdbscan_metric"],
            "n": params["gmm_n_components"],
            "gmm_type": params["gmm_covariance_type"],
            "smoothing": (
                f"{config['smoothing']['method']}(window={smooth_window})" if smoothing_applied else "off"
            ),
        },
        result_info=clust_info,
        evaluation=result_eval,
    )
    set_info(report_text, "blue")

    cached_visualization = cached_postprocess.get("visualization_data") if cached_postprocess else None
    if isinstance(cached_visualization, dict) and cached_output_labels == labels_for_output:
        visualization_data = cached_visualization
        set_info("CALC Well Log: данные визуализации загружены из базы данных.", "green")
    else:
        visualization_data = build_well_log_cluster_visualization_data(
            run_context=run_context,
            labels=labels_for_output,
            kept_row_indices=kept_row_indices,
            metrics=result_eval,
            config=config,
            pca_info=pca_info_report,
            cluster_info=clust_info,
            smoothing_changes=smoothing_changes,
        )
    postprocess_payload = dict(cached_postprocess or {})
    postprocess_payload.update({
        "labels": [int(value) for value in labels_for_output],
        "smoothing_changes": int(smoothing_changes),
        "evaluation": result_eval,
        "visualization_data": visualization_data,
    })
    if not cached_postprocess or cached_postprocess != postprocess_payload:
        cached_result = put_cached_postprocess_result(cached_result or {}, config, postprocess_payload)
        save_cluster_calculation_cache(
            source_type="well_log",
            dataset_id=int(run_context["dataset_id"]),
            cache_key=cache_key,
            data_hash=str(run_context.get("data_hash", "")),
            config=config,
            result_payload=cached_result,
        )
    well_log_cluster_result_cache[int(run_context["dataset_id"])] = visualization_data
    try:
        cached_diagnostics_image = (
            str(cached_postprocess.get("diagnostics_image_base64") or "")
            if cached_postprocess else ""
        )
        diagnostics_image = show_cluster_diagnostics(
            data_for_clustering=data_pca,
            labels=labels_for_output,
            method_name=config["method"],
            cached_image_base64=cached_diagnostics_image or None,
        )
        if cached_diagnostics_image:
            set_info("CALC Well Log: диагностические графики загружены из базы данных.", "green")
        else:
            set_info("CALC Well Log: обновлены диагностические графики качества кластеризации.", "blue")
        if diagnostics_image and diagnostics_image != cached_diagnostics_image:
            postprocess_payload["diagnostics_image_base64"] = diagnostics_image
            cached_result = put_cached_postprocess_result(cached_result or {}, config, postprocess_payload)
            save_cluster_calculation_cache(
                source_type="well_log",
                dataset_id=int(run_context["dataset_id"]),
                cache_key=cache_key,
                data_hash=str(run_context.get("data_hash", "")),
                config=config,
                result_payload=cached_result,
            )
    except Exception as exc:
        set_info(f"CALC Well Log: не удалось построить диагностические графики: {exc}", "brown")

    show_well_log_cluster_visualization_stub(visualization_data)
    set_info(
        f"CALC Well Log: расчет завершен, labels={len(labels_for_output)}, "
        f"clusters={visualization_data['summary'].get('cluster_count', 0)}.",
        "green"
    )
    return visualization_data

def calculate_cluster():
    run_context = build_cluster_run_context(show_errors=True)
    if run_context is None:
        return

    manual_config = _read_manual_cluster_ui_config()

    if run_context["source_type"] == "well_log":
        try:
            calculate_well_log_cluster(run_context, manual_config)
        except Exception as exc:
            set_info(f"CALC Well Log: ошибка расчета: {exc}", "red")
            try:
                QMessageBox.critical(MainWindow, "WELL LOG CLUSTER", f"Ошибка расчета Well Log: {exc}")
            except Exception:
                pass
        return

    clust_object_id = int(run_context["dataset_id"])
    clust_analys_id = get_curr_clust_analys_id()
    clust_object = session.query(ObjectSet).filter_by(id=clust_object_id).first()
    data = run_context["raw_rows"]
    raw_meta = np.array(data, dtype=object)[:, 0] if data else np.array([])
    selected_button = ui.buttonGroup_3.checkedButton()
    text_method_nan = selected_button.text() if selected_button else 'impute'
    preprocess_mode = str(manual_config["preprocess_mode"])
    pca_enabled = bool(manual_config["pca"]["enabled"])
    mode_pca = manual_config["pca"]["mode"] if pca_enabled else None
    n_comp_pca = int(manual_config["pca"]["fixed_components"])
    disp_pca = float(manual_config["pca"]["variance_ratio"])

    kmeans_n = ui.spinBox_clust_kmeans_n.value()
    kmeans_n_init = ui.spinBox_clust_kmean_ninint.value()

    hdbsc_min_size = ui.spinBox_clust_hdbsc_minsize.value()
    hdbsc_min_sample = ui.spinBox_clust_hdbsc_minsamp.value()
    hdbsc_type = ui.comboBox_clust_hdbsc_type.currentText()

    gmm_n = ui.spinBox_clust_gmm_n.value()
    gmm_type = ui.comboBox_clust_gmm_type.currentText()

    if ui.radioButton_clust_kmean.isChecked():
        clust_method_analys = "kmeans"
    elif ui.radioButton_clust_hdbscan.isChecked():
        clust_method_analys = "hdbscan"
    elif ui.radioButton_clust_gaussmix.isChecked():
        clust_method_analys = "gmm"
    else:
        clust_method_analys = "kmeans"

    cache_key = build_cluster_calculation_cache_key(
        source_type="gpr",
        dataset_id=clust_object_id,
        data_hash=str(run_context.get("data_hash", "")),
        config=manual_config,
    )
    cached_result = load_cluster_calculation_cache(
        source_type="gpr",
        dataset_id=clust_object_id,
        cache_key=cache_key,
        data_hash=str(run_context.get("data_hash", "")),
        config=manual_config,
    )
    cached_gpr = get_cached_gpr_calculation(cached_result)

    if cached_gpr is not None:
        label_list = cached_gpr["labels"]
        kept_row_indices = cached_gpr["kept_row_indices"]
        data_pca = cached_gpr["data_for_diagnostics"]
        clust_info = cached_gpr["cluster_info"]
        pca_info_report = cached_gpr["pca_info_report"]
        set_info(
            f"CALC: CACHE HIT ({cached_result.get('_cache_lookup_mode', 'cache_key')}), результат загружен "
            "из базы без повторной очистки, PCA и кластеризации.",
            "green",
        )
    else:
        set_info(
            f"CALC: CACHE MISS, сохраненная база не найдена или неполна "
            f"(key={cache_key[:12]}, data={str(run_context.get('data_hash', ''))[:12]}).",
            "brown",
        )
        clear_data, kept_row_indices = clean_features(
            data=data,
            use_non_finite=ui.checkBox_clust_clean_nan.isChecked(),
            non_finite_mode=text_method_nan,
            use_variance_threshold=ui.checkBox_clust_clear_vartresh.isChecked(),
            use_correlation_filter=ui.checkBox_clust_clear_corr.isChecked()
        )
        if not clear_data:
            return
        print('Before: ', len(data), len(data[0]))
        print('After: ', len(clear_data), len(clear_data[0]))
        print('Rows kept after cleaning: ', len(kept_row_indices))

        preprocess_data = preprocess_features(clear_data, mode=preprocess_mode)
        if pca_enabled:
            data_pca, pca_info = apply_pca(
                preprocess_data,
                mode=mode_pca,
                n_components=n_comp_pca,
                variance_ratio=disp_pca,
            )
            print("PCA info: ", pca_info)
            pca_info_report = {
                "components_after_pca": n_comp_pca,
                "explained_variance": disp_pca,
            }
        else:
            data_pca = preprocess_data
            pca_info_report = {}

        label_list, clust_info = cluster_data(
            data=data_pca,
            method=clust_method_analys,
            kmeans_n_clusters=kmeans_n,
            kmeans_n_init=kmeans_n_init,
            hdbscan_min_cluster_size=hdbsc_min_size,
            hdbscan_min_samples=hdbsc_min_sample,
            hdbscan_metric=hdbsc_type,
            gmm_n_components=gmm_n,
            gmm_covariance_type=gmm_type
        )
        assignments = []
        for clean_row_idx, cluster_label in enumerate(label_list):
            if clean_row_idx >= len(kept_row_indices):
                break
            source_row_idx = int(kept_row_indices[clean_row_idx])
            if source_row_idx < 0 or source_row_idx >= len(data):
                continue
            assignments.append({
                "source_row_index": source_row_idx,
                "measurement_key": str(data[source_row_idx][0]),
                "cluster_label": int(cluster_label),
            })
        cached_result = {
            "result_type": "gpr",
            "labels": [int(value) for value in label_list],
            "kept_row_indices": [int(value) for value in kept_row_indices],
            "assignments": assignments,
            "data_for_diagnostics": np.asarray(data_pca).tolist(),
            "cluster_info": clust_info,
            "pca_info_report": pca_info_report,
        }
        cache_saved = save_cluster_calculation_cache(
            source_type="gpr",
            dataset_id=clust_object_id,
            cache_key=cache_key,
            data_hash=str(run_context.get("data_hash", "")),
            config=manual_config,
            result_payload=cached_result,
        )
        if cache_saved:
            set_info(f"CALC: сохранены базовые метки до сглаживания: {len(label_list)} строк.", "blue")
        else:
            set_info("CALC: базовые метки не удалось сохранить в БД.", "red")

    labels_for_output = list(label_list)
    profile_trace_rows: dict[int, dict[int, int]] = {}
    invalid_prof_index_count = 0
    duplicate_prof_index_count = 0

    if len(label_list) != len(kept_row_indices):
        set_info(
            f'Внимание: размер labels ({len(label_list)}) не совпадает с числом сохраненных строк '
            f'({len(kept_row_indices)}). Построение профилей выполнено частично.',
            'brown'
        )

    for clean_row_idx, _ in enumerate(label_list):
        if clean_row_idx >= len(kept_row_indices):
            break

        source_row_idx = kept_row_indices[clean_row_idx]
        if source_row_idx >= len(raw_meta):
            invalid_prof_index_count += 1
            continue

        prof_index_value = str(raw_meta[source_row_idx])
        if "_" not in prof_index_value:
            invalid_prof_index_count += 1
            continue

        profile_part, trace_part = prof_index_value.rsplit("_", 1)
        try:
            profile_id = int(profile_part)
            trace_index = int(trace_part)
        except ValueError:
            invalid_prof_index_count += 1
            continue

        if profile_id not in profile_trace_rows:
            profile_trace_rows[profile_id] = {}
        if trace_index in profile_trace_rows[profile_id]:
            duplicate_prof_index_count += 1

        profile_trace_rows[profile_id][trace_index] = int(clean_row_idx)

    smooth_enabled = ui.checkBox_cluster_smooth.isChecked()
    smooth_method = "maj" if ui.radioButton_cluster_smooth_maj.isChecked() else "med"
    smooth_window_raw = ui.spinBox_cluster_smooth_window.value()
    smooth_window = _normalize_smoothing_window(smooth_window_raw)
    smoothing_applied = bool(smooth_enabled and smooth_window >= 3)
    smoothing_changes = 0

    if smooth_enabled and smooth_window_raw != smooth_window and smooth_window >= 3:
        set_info(
            f"Smoothing: окно {smooth_window_raw} скорректировано до нечетного {smooth_window}.",
            "brown"
        )

    if smoothing_applied:
        labels_for_output, smoothing_changes = _smooth_labels_by_profile_trace(
            labels_for_output,
            profile_trace_rows,
            method=smooth_method,
            window=smooth_window,
            preserve_noise=True
        )
        set_info(
            f"Smoothing применен ({smooth_method}, window={smooth_window}), изменено меток: {smoothing_changes}.",
            "blue"
        )
    elif smooth_enabled:
        set_info("Smoothing включен, но окно слишком маленькое. Постобработка пропущена.", "brown")

    profile_labels = {}
    for profile_id, trace_rows in profile_trace_rows.items():
        profile_labels[profile_id] = {
            int(trace_idx): int(labels_for_output[row_idx])
            for trace_idx, row_idx in trace_rows.items()
        }

    if invalid_prof_index_count:
        set_info(
            f'Внимание: пропущено строк с некорректным prof_index: {invalid_prof_index_count}.',
            'brown'
        )
    if duplicate_prof_index_count:
        set_info(
            f'Внимание: обнаружены дубликаты (profile_id, trace_index): {duplicate_prof_index_count}. '
            f'Использованы последние метки.',
            'brown'
        )

    map_data = [data[row_idx] for row_idx in kept_row_indices if 0 <= row_idx < len(data)]
    if len(map_data) != len(labels_for_output):
        set_info(
            f"Внимание: карта кластеров построена по усеченному набору точек "
            f"(labels={len(labels_for_output)}, map_data={len(map_data)}).",
            "brown"
        )
        map_len = min(len(map_data), len(labels_for_output))
        map_data = map_data[:map_len]
        labels_for_map = labels_for_output[:map_len]
    else:
        labels_for_map = labels_for_output

    map_x = [float(row[1]) for row in map_data]
    map_y = [float(row[2]) for row in map_data]
    map_title = f"Cluster {ui.comboBox_clust_obj.currentText().split(' id')[0]}"

    print(labels_for_output)
    print(clust_info)

    cached_postprocess = get_cached_postprocess_result(cached_result, manual_config)
    cached_output_labels = (
        [int(value) for value in cached_postprocess.get("labels", [])]
        if cached_postprocess else []
    )
    if cached_postprocess and cached_output_labels == labels_for_output:
        result_eval = dict(cached_postprocess.get("evaluation") or {})
        set_info("CALC: оценочные показатели загружены из базы данных.", "green")
    else:
        result_eval = evaluate_clustering(
            data_pca,
            labels_for_output,
            use_silhouette=ui.checkBox_cluster_silhoutte.isChecked(),
            use_db=ui.checkBox_cluster_dav_boul.isChecked(),
            use_ch=ui.checkBox_cluster_calin_har.isChecked()
        )
        cached_result = put_cached_postprocess_result(
            cached_result or {},
            manual_config,
            {
                "labels": [int(value) for value in labels_for_output],
                "smoothing_changes": int(smoothing_changes),
                "evaluation": result_eval,
            },
        )
        save_cluster_calculation_cache(
            source_type="gpr",
            dataset_id=clust_object_id,
            cache_key=cache_key,
            data_hash=str(run_context.get("data_hash", "")),
            config=manual_config,
            result_payload=cached_result,
        )

    print(result_eval)

    text = build_clustering_report(
        preprocess_mode=preprocess_mode,
        pca_mode=mode_pca,
        pca_info=pca_info_report,
        cluster_info={
            "method": clust_method_analys,
            "kmeans_n": kmeans_n,
            "kmeans_n_init": kmeans_n_init,
            "min_size": hdbsc_min_size,
            "min_sample": hdbsc_min_sample,
            "hdbscan_type": hdbsc_type,
            "n": gmm_n,
            "gmm_type": gmm_type,
            "smoothing": (
                f"{smooth_method}(window={smooth_window})"
                if smoothing_applied else "off"
            )
        },
        result_info=clust_info,
        evaluation=result_eval
    )

    set_info(text, 'blue')

    analysis_key = build_cluster_analysis_key(
        clust_object_id=clust_object_id,
        clust_analys_id=clust_analys_id,
        method=clust_method_analys,
        preprocess_mode=preprocess_mode,
        pca_enabled=ui.checkBox_cluster_pca.isChecked(),
        pca_mode=mode_pca,
        pca_value=(n_comp_pca if mode_pca == "fixed_components" else disp_pca if mode_pca else None),
        extra_params={
            "kmeans_n": kmeans_n,
            "kmeans_n_init": kmeans_n_init,
            "hdbscan_min_cluster_size": hdbsc_min_size,
            "hdbscan_min_samples": hdbsc_min_sample,
            "hdbscan_metric": hdbsc_type,
            "gmm_n_components": gmm_n,
            "gmm_covariance_type": gmm_type,
        }
    )
    save_cluster_profile_cache(
        analysis_key=analysis_key,
        profile_labels=profile_labels,
        meta={
            "method": clust_method_analys,
            "n_points": len(labels_for_output),
            "smoothing_enabled": bool(smoothing_applied),
            "smoothing_method": (smooth_method if smoothing_applied else None),
            "smoothing_window": (int(smooth_window) if smoothing_applied else None),
            "smoothing_changes": int(smoothing_changes),
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "clust_object_id": int(clust_object_id),
            "clust_analys_id": int(clust_analys_id),
        }
    )
    set_info(
        "Кластеризация рассчитана. Теперь доступна мгновенная перерисовка по профилям без повторного CALC.",
        "green"
    )
    set_info(build_cluster_legend(profile_labels), "blue")

    if sync_ui_to_cluster_object_research(clust_object) and ui.comboBox_profile.count() > 0:
        ui.comboBox_profile.setCurrentIndex(0)
        draw_radarogram()
        first_profile_id = get_profile_id()
        if first_profile_id:
            update_formation_combobox()
            draw_cluster_profile_result(
                profile_id=first_profile_id,
                profile_labels=profile_labels,
                use_relief=True
            )
        else:
            set_info("Профиль не выбран: результаты кластеров на радарограмме не обновлены.", "brown")
    else:
        set_info("Не удалось автоматически выбрать исследование/профиль для отрисовки кластеров.", "brown")

    try:
        cached_diagnostics_image = (
            str(cached_postprocess.get("diagnostics_image_base64") or "")
            if cached_postprocess else ""
        )
        diagnostics_image = show_cluster_diagnostics(
            data_for_clustering=data_pca,
            labels=labels_for_output,
            method_name=clust_method_analys,
            cached_image_base64=cached_diagnostics_image or None,
        )
        if cached_diagnostics_image:
            set_info("Диагностические графики загружены из базы данных.", "green")
        else:
            set_info("Открыты диагностические графики кластеризации (PCA 2D/3D, distance matrix, silhouette и спец-графики метода).", "blue")
        if diagnostics_image and diagnostics_image != cached_diagnostics_image:
            updated_postprocess = dict(cached_postprocess or {})
            updated_postprocess.update({
                "labels": [int(value) for value in labels_for_output],
                "smoothing_changes": int(smoothing_changes),
                "evaluation": result_eval,
                "diagnostics_image_base64": diagnostics_image,
            })
            cached_result = put_cached_postprocess_result(cached_result or {}, manual_config, updated_postprocess)
            save_cluster_calculation_cache(
                source_type="gpr",
                dataset_id=clust_object_id,
                cache_key=cache_key,
                data_hash=str(run_context.get("data_hash", "")),
                config=manual_config,
                result_payload=cached_result,
            )
    except Exception as exc:
        set_info(f"Не удалось построить диагностические графики: {exc}", "brown")

    set_info(
        "Открыто окно настроек карты кластеров. Выберите параметры и нажмите DRAW.",
        "blue"
    )
    draw_map(
        map_x,
        map_y,
        labels_for_map,
        map_title,
        color_marker=False,
        initial_map_mode="categorical"
    )
