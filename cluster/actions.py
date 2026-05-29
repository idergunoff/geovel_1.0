from __future__ import annotations

from .common import *

def calculate_well_log_cluster(run_context: ClusterRunContext) -> WellLogClusterVisualizationData | None:
    """
    Выполняет ручной CALC для Well Log через общий pipeline clean → scale → PCA → cluster → metrics.
    """
    data = run_context["raw_rows"]
    config = _read_manual_cluster_ui_config()
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

    params = config["method_params"]
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
    labels_for_output = [int(label) for label in label_list]

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
    well_log_cluster_result_cache[int(run_context["dataset_id"])] = visualization_data

    try:
        show_cluster_diagnostics(
            data_for_clustering=data_pca,
            labels=labels_for_output,
            method_name=config["method"],
        )
        set_info("CALC Well Log: обновлены диагностические графики качества кластеризации.", "blue")
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

    if run_context["source_type"] == "well_log":
        try:
            calculate_well_log_cluster(run_context)
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
    clear_data, kept_row_indices = clean_features(
        data=data,
        use_non_finite=ui.checkBox_clust_clean_nan.isChecked(),
        non_finite_mode=text_method_nan,
        use_variance_threshold=ui.checkBox_clust_clear_vartresh.isChecked(),
        use_correlation_filter=ui.checkBox_clust_clear_corr.isChecked()
    )
    if clear_data:
        print('Before: ', len(data), len(data[0]))
        print('After: ', len(clear_data), len(clear_data[0]))
        print('Rows kept after cleaning: ', len(kept_row_indices))
    else:
        return

    if ui.radioButton_clust_scaler_none.isChecked():
        preprocess_mode = 'none'
    elif ui.radioButton_clust_scaler_stnd.isChecked():
        preprocess_mode = 'standard'
    elif ui.radioButton_clust_scaler_rob.isChecked():
        preprocess_mode = 'robust'
    elif ui.radioButton_clust_scaler_l2.isChecked():
        preprocess_mode = 'l2_norm'
    else:
        preprocess_mode = 'row_center'

    preprocess_data = preprocess_features(clear_data, mode=preprocess_mode)

    if ui.checkBox_cluster_pca.isChecked():
        mode_pca = "fixed_components" if ui.radioButton_clust_pca_fix.isChecked() else "variance_ratio"
        n_comp_pca = ui.spinBox_clust_pca_fix.value()
        disp_pca = ui.doubleSpinBox_clust_pca_disp.value()

        data_pca, pca_info = apply_pca(preprocess_data, mode=mode_pca, n_components=n_comp_pca, variance_ratio=disp_pca)
        print("PCA info: ", pca_info)

        pca_info_report = {
            "components_after_pca": n_comp_pca,
            "explained_variance": disp_pca
        }
    else:
        data_pca = preprocess_data
        mode_pca = None
        pca_info_report = {}

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

    result_eval = evaluate_clustering(
        data_pca,
        labels_for_output,
        use_silhouette=ui.checkBox_cluster_silhoutte.isChecked(),
        use_db=ui.checkBox_cluster_dav_boul.isChecked(),
        use_ch=ui.checkBox_cluster_calin_har.isChecked()
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
        show_cluster_diagnostics(
            data_for_clustering=data_pca,
            labels=labels_for_output,
            method_name=clust_method_analys
        )
        set_info("Открыты диагностические графики кластеризации (PCA 2D/3D, distance matrix, silhouette и спец-графики метода).", "blue")
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
