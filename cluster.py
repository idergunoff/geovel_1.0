import pandas as pd
from datetime import datetime, timezone

import build_table
from draw import draw_radarogram
from func import *

# Runtime-cache результатов кластеризации для быстрой перерисовки по профилям без пересчета.
# Этап 1 (MVP): ключ анализа = id ObjectSet (clust_object_id).
cluster_profile_cache = {}
is_cluster_redraw_in_progress = False


def build_cluster_analysis_key(
        clust_object_id=None,
        clust_analys_id=None,
        *,
        method=None,
        preprocess_mode=None,
        pca_enabled=None,
        pca_mode=None,
        pca_value=None,
        extra_params=None
):
    """
    Формирует ключ cache-записи.
    MVP-режим: используем только clust_object_id (перезапись по повторному CALC допустима).
    """
    if clust_object_id is None:
        raise ValueError("clust_object_id is required for cluster cache key")
    return str(clust_object_id)


def save_cluster_profile_cache(analysis_key, profile_labels, meta=None):
    """
    Сохраняет результаты кластеризации в runtime-cache.
    """
    cluster_profile_cache[analysis_key] = {
        "profile_labels": profile_labels or {},
        "meta": meta or {}
    }


def get_cluster_profile_cache(analysis_key):
    """
    Возвращает cache-запись по ключу анализа.
    """
    return cluster_profile_cache.get(analysis_key)


def get_last_cluster_profile_cache():
    """
    Возвращает последнюю добавленную cache-запись (если есть).
    """
    if not cluster_profile_cache:
        return None
    last_key = next(reversed(cluster_profile_cache))
    return cluster_profile_cache[last_key]


def redraw_cluster_for_current_profile_from_cache():
    """
    Перерисовывает кластерную заливку для текущего профиля из runtime-cache без пересчета cluster_data.
    """
    global is_cluster_redraw_in_progress
    if is_cluster_redraw_in_progress:
        return

    is_cluster_redraw_in_progress = True
    try:
        current_profile_id = get_profile_id()
        if not current_profile_id:
            return

        cache_entry = None
        clust_object_id = get_curr_clust_object_id()
        if clust_object_id:
            analysis_key = build_cluster_analysis_key(clust_object_id=clust_object_id)
            cache_entry = get_cluster_profile_cache(analysis_key)
            if cache_entry is None and cluster_profile_cache:
                set_info(
                    "Кэш кластеризации устарел для текущего ObjectSet. "
                    "Выберите актуальный набор и нажмите CALC.",
                    "brown"
                )
                return

        if cache_entry is None and not clust_object_id:
            cache_entry = get_last_cluster_profile_cache()

        if cache_entry is None:
            set_info("Нет кэша кластеризации. Сначала выполните CALC в модуле Cluster.", "brown")
            return

        meta = cache_entry.get("meta", {})
        if clust_object_id and meta.get("clust_object_id") not in (None, int(clust_object_id)):
            set_info(
                "Кэш кластеризации относится к другому ObjectSet. Нажмите CALC для пересчета.",
                "brown"
            )
            return

        profile_labels = cache_entry.get("profile_labels", {})
        if current_profile_id not in profile_labels:
            set_info("Для текущего профиля нет кластерных меток в кэше.", "brown")
            return

        draw_cluster_profile_result(
            profile_id=current_profile_id,
            profile_labels=profile_labels,
            use_relief=True
        )
    finally:
        is_cluster_redraw_in_progress = False


def get_cluster_color(label):
    """
    Возвращает цвет кластера:
    -1 -> серый (шум), остальные -> циклическая палитра.
    """
    if int(label) == -1:
        return "#808080"

    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173"
    ]
    return palette[int(label) % len(palette)]


def build_cluster_legend(profile_labels, max_items=10):
    """
    Возвращает короткую легенду по цветам кластеров для set_info.
    """
    labels = set()
    for trace_to_label in (profile_labels or {}).values():
        labels.update(trace_to_label.values())

    labels_sorted = sorted(labels, key=lambda x: (int(x) == -1, int(x)))
    if not labels_sorted:
        return "Легенда кластеров недоступна: метки отсутствуют."

    legend_items = []
    for label in labels_sorted[:max_items]:
        label_int = int(label)
        title = "noise" if label_int == -1 else f"cluster {label_int}"
        legend_items.append(f"{title} = {get_cluster_color(label_int)}")

    if len(labels_sorted) > max_items:
        legend_items.append(f"... +{len(labels_sorted) - max_items} кластер(ов)")
    return "Легенда: " + ", ".join(legend_items)


def draw_cluster_profile_result(profile_id: int, profile_labels: dict, *, use_relief=True):
    """
    Отрисовывает кластерные сегменты на радарограмме выбранного профиля.
    profile_labels: {profile_id: {trace_index: label}}
    """
    from draw import draw_fill_result, remove_poly_item

    profile = session.query(Profile).filter_by(id=profile_id).first()
    if not profile:
        set_info(f"Профиль id{profile_id} не найден. Отрисовка кластеров отменена.", "brown")
        return

    formation_id = get_formation_id()
    current_formation = None
    if formation_id:
        current_formation = session.query(Formation).filter_by(id=formation_id, profile_id=profile_id).first()
    if not current_formation and profile.formations:
        current_formation = profile.formations[0]

    if not current_formation:
        set_info("Не выбрана формация для отрисовки кластеров.", "brown")
        return
    if not current_formation.layer_up or not current_formation.layer_down:
        set_info("Для формации отсутствуют границы (layer_up/layer_down).", "brown")
        return
    if not current_formation.layer_up.layer_line or not current_formation.layer_down.layer_line:
        set_info("Для формации отсутствуют линии границ. Отрисовка кластеров пропущена.", "brown")
        return

    list_up = json.loads(current_formation.layer_up.layer_line)
    list_down = json.loads(current_formation.layer_down.layer_line)

    if use_relief and ui.checkBox_relief.isChecked() and profile.depth_relief:
        depth = [i * 100 / 40 for i in json.loads(profile.depth_relief)]
        coeff = 512 / (512 + np.max(depth))
        list_up = [int((x + y) * coeff) for x, y in zip(list_up, depth)]
        list_down = [int((x + y) * coeff) for x, y in zip(list_down, depth)]

    trace_to_label = profile_labels.get(profile_id, {})
    if not trace_to_label:
        set_info("Для текущего профиля нет кластерных меток в кэше.", "brown")
        return

    count_measure = len(json.loads(profile.signal)) if profile.signal else 0
    max_len = min(count_measure, len(list_up), len(list_down))
    if max_len <= 0:
        set_info("Недостаточно данных для отрисовки кластеров по профилю.", "brown")
        return
    if len(trace_to_label) > max_len:
        set_info(
            f"Размер меток ({len(trace_to_label)}) превышает ожидаемое число трасс ({max_len}) "
            f"для профиля {profile.title}. Перерисовка отменена.",
            "brown"
        )
        return

    labels_sequence = [trace_to_label.get(i) for i in range(max_len)]

    remove_poly_item()

    segment_indices = []
    segment_label = None

    def flush_segment(indices, label):
        if not indices or label is None:
            return
        x_seg = list(indices)
        if x_seg[-1] + 1 < max_len:
            x_seg.append(x_seg[-1] + 1)
        y_up = [list_up[i] for i in x_seg]
        y_down = [list_down[i] for i in x_seg]
        draw_fill_result(x_seg, y_up, y_down, get_cluster_color(label))

    for idx, label in enumerate(labels_sequence):
        if label is None:
            flush_segment(segment_indices, segment_label)
            segment_indices = []
            segment_label = None
            continue

        if segment_label is None:
            segment_indices = [idx]
            segment_label = label
            continue

        if label == segment_label:
            segment_indices.append(idx)
        else:
            flush_segment(segment_indices, segment_label)
            segment_indices = [idx]
            segment_label = label

    flush_segment(segment_indices, segment_label)

    set_info(
        f"Кластеры отрисованы на профиле {profile.title}: "
        f"{len([i for i in labels_sequence if i is not None])} трасс с метками.",
        "blue"
    )


def update_clust_clear_nan():
    is_checked = ui.checkBox_clust_clean_nan.isChecked()

    # Блокируем или разблокируем радиокнопки
    ui.radioButton_clust_clean_nan_impute.setEnabled(is_checked)
    ui.radioButton_clust_clean_nan_col.setEnabled(is_checked)
    ui.radioButton_clust_clean_nan_row.setEnabled(is_checked)

    # Если включили чекбокс, по умолчанию выбираем первый вариант
    if is_checked:
        ui.radioButton_clust_clean_nan_impute.setChecked(True)

def add_clust_analys_from_cls():
    cls_analys = session.query(AnalysisMLP).options(joinedload(AnalysisMLP.parameters)).filter_by(id=get_MLP_id()).first()
    list_param = [i.parameter for i in cls_analys.parameters]
    new_clust_analys = AnalysisCluster(title=f'CLS {cls_analys.title}', parameter=json.dumps(list_param))
    session.add(new_clust_analys)
    session.commit()
    update_list_clust_analys()

def add_clust_analys_from_reg():
    reg_analys = session.query(AnalysisReg).filter_by(id=get_regmod_id()).first()
    list_param = [i.parameter for i in reg_analys.parameters]
    new_clust_analys = AnalysisCluster(title=f'REG {reg_analys.title}', parameter=json.dumps(list_param))
    session.add(new_clust_analys)
    session.commit()
    update_list_clust_analys()


def remove_clust_analys():
    session.query(AnalysisCluster).filter_by(id=get_curr_clust_analys_id()).delete()
    session.commit()
    update_list_clust_analys()


def collect_clust_object():
    global flag_break
    working_data_result = pd.DataFrame()
    list_formation = []
    profiles = session.query(Profile).filter(Profile.research_id == get_research_id()).all()
    flag_break = []
    for n, prof in enumerate(profiles):
        if flag_break:
            if flag_break[0] == 'stop':
                break
            else:
                set_info(f'Нет пласта с названием {flag_break[1]} для профиля {flag_break[0]}', 'red')
                QMessageBox.critical(MainWindow, 'Ошибка', f'Нет пласта с названием {flag_break[1]} для профиля '
                                                           f'{flag_break[0]}, выберите пласты для каждого профиля.')
                return
        count_measure = len(json.loads(session.query(Profile.signal).filter(Profile.id == prof.id).first()[0]))
        ui.comboBox_profile.setCurrentText(f'{prof.title} ({count_measure} измерений) id{prof.id}')
        set_info(f'Профиль {prof.title} ({count_measure} измерений)', 'blue')
        update_formation_combobox()
        if len(prof.formations) == 1:
            # ui.comboBox_plast.setCurrentText(f'{prof.formations[0].title} id{prof.formations[0].id}')
            list_formation.append(f'{prof.formations[0].title} id{prof.formations[0].id}')
        elif len(prof.formations) > 1:
            Choose_Formation = QtWidgets.QDialog()
            ui_cf = Ui_FormationLDA()
            ui_cf.setupUi(Choose_Formation)
            Choose_Formation.show()
            Choose_Formation.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия
            for f in prof.formations:
                ui_cf.listWidget_form_lda.addItem(f'{f.title} id{f.id}')
            ui_cf.listWidget_form_lda.setCurrentRow(0)

            def form_mlp_ok():
                global flag_break
                # ui.comboBox_plast.setCurrentText(ui_cf.listWidget_form_lda.currentItem().text())
                if ui_cf.checkBox_to_all.isChecked():
                    title_form = ui_cf.listWidget_form_lda.currentItem().text().split(' id')[0]
                    for prof in profiles:
                        prof_form = session.query(Formation).filter_by(
                            profile_id=prof.id,
                            title=title_form
                        ).first()
                        if prof_form:
                            list_formation.append(f'{prof_form.title} id{prof_form.id}')
                        else:
                            flag_break = [prof.title, title_form]
                            Choose_Formation.close()
                            return
                    flag_break = ['stop', 'stop']
                    Choose_Formation.close()
                else:
                    list_formation.append(ui_cf.listWidget_form_lda.currentItem().text())
                    Choose_Formation.close()

            ui_cf.pushButton_ok_form_lda.clicked.connect(form_mlp_ok)
            Choose_Formation.exec_()

    for f in list_formation:
        cf = session.query(Formation).filter_by(id=int(f.split(' id')[-1])).first()
        data_profile, _ = build_table.build_table_test(analisis='cluster', curr_form=cf)
        working_data_result = pd.concat([working_data_result, data_profile], axis=0, ignore_index=True)
    report = inf_nan_data_report(working_data_result)
    print("variances diagnostic: ")
    report_var, low_var_table = variances_diagnostic(working_data_result)
    report += (f"<br>=== Low Variance Report ===<br>"
               f"n low var: {report_var['n_low_var']}<br>"
               f"fraction removed: {round(report_var['fraction_removed'], 3)}<br>")
    print(low_var_table)
    print("correlation diagnostic: ")
    report_corr, high_corr_table = correlation_diagnostic(working_data_result)

    report += (f"<br>=== High Correlation Report ===<br>"
               f"corr pairs: {report_corr['n_correlation_pairs']}<br>"
               f"to drop features: {report_corr['n_features_to_drop']}<br>"
               f"best features: {report_corr['n_features_after']}<br>"
               f"=== End Report ===")
    print(high_corr_table)

    data = working_data_result.values.tolist()
    new_cluster_obj = ObjectSet(research_id=get_research_id(), analysis_id=get_curr_clust_analys_id(), data=json.dumps(data), report=report)
    session.add(new_cluster_obj)
    session.commit()
    update_list_clust_object()


def remove_clust_object():
    session.query(ObjectSet).filter_by(id=get_curr_clust_object_id()).delete()
    session.commit()
    update_list_clust_object()


def get_curr_clust_analys_id():
    return ui.comboBox_clust_set.currentText().split(' id')[-1]


def get_curr_clust_object_id():
    return ui.comboBox_clust_obj.currentText().split(' id')[-1]


def sync_ui_to_cluster_object_research(clust_object):
    """
    Переключает UI на год/объект/исследование, к которому привязан ObjectSet.
    Возвращает True, если удалось синхронизировать исследование в UI.
    """
    if not clust_object:
        return False

    research = session.query(Research).filter_by(id=clust_object.research_id).first()
    if not research:
        return False

    target_year = research.date_research.strftime('%Y')
    target_object_text = f'{research.object.title} id{research.object_id}'
    target_research_text = f'{research.date_research.strftime("%m.%Y")} id{research.id}'

    year_idx = ui.comboBox_year_research.findText(target_year)
    if year_idx >= 0:
        ui.comboBox_year_research.setCurrentIndex(year_idx)
        update_object()

    object_idx = ui.comboBox_object.findText(target_object_text)
    if object_idx >= 0:
        ui.comboBox_object.setCurrentIndex(object_idx)
        update_research_combobox()

    research_idx = ui.comboBox_research.findText(target_research_text)
    if research_idx >= 0:
        ui.comboBox_research.setCurrentIndex(research_idx)
        update_profile_combobox()

    return True


def update_list_clust_analys():
    ui.comboBox_clust_set.clear()
    for i in session.query(AnalysisCluster).all():
        ui.comboBox_clust_set.addItem(f'{i.title} id{i.id}')

    update_list_clust_param()


def update_list_clust_param():
    ui.listWidget_clust_param.clear()
    try:
        list_param = json.loads(session.query(AnalysisCluster).filter_by(id=get_curr_clust_analys_id()).first().parameter)
        for i in list_param:
            ui.listWidget_clust_param.addItem(i)
    except AttributeError:
        pass


def update_list_clust_object():
    ui.comboBox_clust_obj.clear()
    for clust_obj in session.query(ObjectSet.id, ObjectSet.research_id).filter_by(analysis_id=get_curr_clust_analys_id()).all():
        research = session.query(Research).filter_by(id=clust_obj.research_id).first()
        ui.comboBox_clust_obj.addItem(f'{research.object.title} id{clust_obj.id}')


def show_finite_report():
    obj_set = session.query(ObjectSet.report).filter_by(id=get_curr_clust_object_id()).first()
    set_info(obj_set.report, 'brown')


def inf_nan_data_report(df: pd.DataFrame) -> str:
    """ Расширенная диагностика данных — возвращает строку с отчетом """

    # Выделяем только числовые колонки для проверок на NaN/Inf
    numeric_df = df.select_dtypes(include=[np.number])

    # 1. Базовая статистика
    total_rows = len(df)
    total_cols = len(df.columns)

    # 2. Пропуски (NaN)
    nan = int(df.isna().sum().sum())

    # 3. Бесконечности (Inf) и проблемные строки/колонки
    if not numeric_df.empty:
        inf = int(np.isinf(numeric_df).sum().sum())
        finite_mask = np.isfinite(numeric_df)
        bad_rows = int((~finite_mask).any(axis=1).sum())
        bad_cols = int((~finite_mask).any(axis=0).sum())
    else:
        inf = 0
        bad_rows = 0
        bad_cols = 0

    # 4. Дубликаты строк
    duplicate_rows = int(df.duplicated().sum())

    # 5. Формируем строку отчета
    report = (
            f"<br>=== Data Quality Report ===<br>"
            f"=== {get_object_name()} ===<br>"
            f"Rows: {total_rows}, Cols: {total_cols}<br>"
            f"NaN: {nan}, Inf: {inf}<br>"
            f"Bad Rows: {bad_rows}<br>"
            f"Bad Cols: {bad_cols}<br>"
            f"Duplicate Rows: {duplicate_rows}<br>"
    )

    return report


def variances_diagnostic(df: pd.DataFrame, threshold=1e-6):
    """ Диагностика почти константных признаков """

    variances = df.var(numeric_only=True)

    low_var = variances[variances <= threshold].sort_values()

    report = {
        "n_features_total": len(variances),
        "n_low_var": len(low_var),
        "fraction_removed": len(low_var) / len(variances),
        "threshold": threshold
    }

    low_var_table = pd.DataFrame({
        "feature": low_var.index,
        "variance": low_var.values
    })

    return report, low_var_table


def correlation_diagnostic(df: pd.DataFrame, threshold=0.98):
    """ Диагностика сильно коррелированных признаков """
    set_info("Выполняется диагностика коррелированных признаков", "blue")

    df_sample = df.sample(min(3000, len(df)), random_state=0)
    corr = df_sample.corr().abs()

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    pairs = (
        upper.stack().reset_index().rename(
           columns={
               "level_0": "feature_1",
               "level_1": "feature_2",
               0: "correlation"
           }
        )
    )

    high_corr = pairs[pairs["correlation"] > threshold].sort_values(
        "correlation", ascending=False
    )

    involved_features = set(high_corr["feature_1"]) | set(high_corr["feature_2"])

    to_drop = [
        column for column in upper.columns
        if any(upper[column] > threshold)
    ]

    report = {
        "n_features_total": df.shape[1],
        "n_correlation_pairs": len(high_corr),
        "n_features_involved": len(involved_features),
        "n_features_to_drop": len(to_drop),
        "n_features_after": df.shape[1] - len(to_drop),
        "threshold": threshold
    }
    set_info("Диагностика выполнена", "green")
    return report, high_corr


def clean_features(
        data,

        use_non_finite=False,
        non_finite_mode="impute",  # drop_rows | drop_features | impute

        use_variance_threshold=False,
        variance_threshold=1e-6,

        use_correlation_filter=False,
        correlation_threshold=0.98
):
    # список -> numpy
    X = np.array(data, dtype=float)
    kept_row_indices = list(range(X.shape[0]))

    # --------------------------------------------------
    # 1 удалить первые 3 столбца
    # --------------------------------------------------

    if X.shape[1] > 3:
        X = X[:, 3:]
    else:
        raise ValueError("Table must contain at least 4 columns")

    # --------------------------------------------------
    # 2 обработка NaN / inf
    # --------------------------------------------------

    if use_non_finite:

        X[~np.isfinite(X)] = np.nan

        if non_finite_mode == "drop_rows":

            mask = ~np.isnan(X).any(axis=1)
            X = X[mask]
            kept_row_indices = np.array(kept_row_indices)[mask].tolist()

        elif non_finite_mode == "drop_features":

            mask = ~np.isnan(X).any(axis=0)
            X = X[:, mask]

        elif non_finite_mode == "impute":

            imputer = SimpleImputer(strategy="median")
            X = imputer.fit_transform(X)

        else:
            raise ValueError("non_finite_mode must be drop_rows/drop_features/impute")

    # --------------------------------------------------
    # 3 VarianceThreshold
    # --------------------------------------------------

    if use_variance_threshold and X.shape[1] > 0:
        selector = VarianceThreshold(threshold=variance_threshold)
        try:
            X = selector.fit_transform(X)
        except ValueError:
            set_info('Внимание!!! Недопустимые значения! Включите обработку "Nan / Inf".', 'red')
            return None, []

    # --------------------------------------------------
    # 4 удаление коррелированных признаков
    # --------------------------------------------------

    if use_correlation_filter and X.shape[1] > 1:

        corr = np.corrcoef(X, rowvar=False)
        corr = np.abs(corr)

        keep = np.ones(corr.shape[0], dtype=bool)

        for i in range(corr.shape[0]):
            if not keep[i]:
                continue

            for j in range(i + 1, corr.shape[0]):
                if corr[i, j] > correlation_threshold:
                    keep[j] = False

        X = X[:, keep]

    return X.tolist(), kept_row_indices


def preprocess_features(data, mode="none"):
    """
    Предобработка признаков.

    Параметры
    ----------
    data : list[list[float]]
        Таблица признаков после очистки.

    mode : str
        'none'           — ничего не делать
        'standard'       — StandardScaler
        'robust'         — RobustScaler
        'l2_norm'        — L2 нормировка по строкам
        'row_center'     — центрирование по строкам

    Возвращает
    ----------
    list[list[float]]
    """

    X = np.array(data, dtype=float)

    if mode == "none":
        pass

    elif mode == "standard":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    elif mode == "robust":
        scaler = RobustScaler()
        X = scaler.fit_transform(X)

    elif mode == "l2_norm":
        normalizer = Normalizer(norm="l2")
        X = normalizer.fit_transform(X)

    elif mode == "row_center":
        X = X - X.mean(axis=1, keepdims=True)

    else:
        raise ValueError(
            "mode must be one of: none, standard, robust, l2_norm, row_center"
        )

    return X.tolist()


def apply_pca(
        data,
        mode="fixed_components",  # "fixed_components" | "variance_ratio"
        n_components=20,
        variance_ratio=0.9
):
    """
    Выполняет PCA.

    Parameters
    ----------
    data : list[list] | np.ndarray
        Таблица признаков.

    mode : str
        "fixed_components" - фиксированное число компонент
        "variance_ratio"   - по доле объяснённой дисперсии

    n_components : int
        Число компонент для режима fixed_components

    variance_ratio : float
        Доля дисперсии для режима variance_ratio, например 0.9

    Returns
    -------
    X_pca_list : list[list]
        Преобразованные данные после PCA

    pca_info : dict
        Информация для вывода в GUI:
        {
            "components_after_pca": ...,
            "explained_variance": ...
        }
    """

    X = np.array(data, dtype=float)

    if mode == "fixed_components":
        pca = PCA(n_components=int(n_components))

    elif mode == "variance_ratio":
        pca = PCA(n_components=float(variance_ratio))

    else:
        raise ValueError("mode must be 'fixed_components' or 'variance_ratio'")

    X_pca = pca.fit_transform(X)

    pca_info = {
        "components_after_pca": int(pca.n_components_),
        "explained_variance": float(np.sum(pca.explained_variance_ratio_))
    }

    return X_pca.tolist(), pca_info


def cluster_data(
        data,
        method="kmeans",  # "kmeans" | "hdbscan" | "gmm"

        # KMeans
        kmeans_n_clusters=4,
        kmeans_n_init=10,
        kmeans_random_state=42,

        # HDBSCAN
        hdbscan_min_cluster_size=30,
        hdbscan_min_samples=5,
        hdbscan_metric="euclidean",
        hdbscan_cluster_selection_method="eom",

        # GMM
        gmm_n_components=4,
        gmm_covariance_type="full",
        gmm_random_state=42,
        gmm_reg_covar=1e-6,
        gmm_max_iter=200
):
    """
    Кластеризация данных.

    Parameters
    ----------
    data : list[list] | np.ndarray
        Таблица признаков

    method : str
        "kmeans", "hdbscan", "gmm"

    Returns
    -------
    labels_list : list[int]
        Метка кластера для каждой строки

    cluster_info : dict
        Сводка для GUI
    """

    X = np.array(data, dtype=float)

    if method == "kmeans":
        from sklearn.cluster import KMeans

        model = KMeans(
            n_clusters=int(kmeans_n_clusters),
            n_init=kmeans_n_init,
            random_state=kmeans_random_state
        )

        labels = model.fit_predict(X)

        n_clusters = len(set(labels))
        noise_points = 0
        noise_fraction = 0.0

        cluster_info = {
            "method": "KMeans",
            "n_clusters": int(n_clusters),
            "noise_points": int(noise_points),
            "noise_fraction": float(noise_fraction),
            "inertia": float(model.inertia_)
        }

    elif method == "hdbscan":

        model = hdbscan.HDBSCAN(
            min_cluster_size=int(hdbscan_min_cluster_size),
            min_samples=int(hdbscan_min_samples),
            metric=hdbscan_metric,
            cluster_selection_method=hdbscan_cluster_selection_method
        )

        labels = model.fit_predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_points = int(np.sum(labels == -1))
        noise_fraction = float(noise_points / len(labels)) if len(labels) > 0 else 0.0

        cluster_info = {
            "method": "HDBSCAN",
            "n_clusters": int(n_clusters),
            "noise_points": int(noise_points),
            "noise_fraction": float(noise_fraction)
        }

    elif method == "gmm":
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(
            n_components=int(gmm_n_components),
            covariance_type=gmm_covariance_type,
            random_state=gmm_random_state,
            reg_covar=gmm_reg_covar,
            max_iter=gmm_max_iter
        )

        model.fit(X)
        labels = model.predict(X)

        n_clusters = len(set(labels))
        noise_points = 0
        noise_fraction = 0.0

        cluster_info = {
            "method": "GaussianMixture",
            "n_clusters": int(n_clusters),
            "noise_points": int(noise_points),
            "noise_fraction": float(noise_fraction),
            "bic": float(model.bic(X)),
            "aic": float(model.aic(X))
        }

    else:
        raise ValueError("method must be 'kmeans', 'hdbscan' or 'gmm'")

    return labels.tolist(), cluster_info


def plot_cluster_map(
        label_list,
        data,
        figsize=(12, 8),
        point_size=20,
        title="Cluster map",
        noise_color="gray",
        noise_marker=".",
        noise_label="noise",
        legend=True
):
    """
    Визуализация кластеров на карте (без подписей профилей).

    Parameters
    ----------
    label_list : list[int]
        Метки кластеров.

    data : list[list]
        Исходные данные:
        data[i][1] - x
        data[i][2] - y

    figsize : tuple
        Размер графика.

    point_size : int
        Размер точек.

    noise_color : str
        Цвет шума (label = -1).

    noise_marker : str
        Маркер шума.

    legend : bool
        Показывать легенду.
    """

    labels = np.asarray(label_list)
    arr = np.asarray(data, dtype=float)

    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("data must have at least 3 columns")

    if len(labels) != len(arr):
        raise ValueError("label_list length must match data")

    x = arr[:, 1]
    y = arr[:, 2]

    plt.figure(figsize=figsize)

    unique_labels = np.unique(labels)

    # кластеры
    for label in unique_labels:
        if label == -1:
            continue

        mask = labels == label

        plt.scatter(
            x[mask],
            y[mask],
            s=point_size,
            label=f"cluster {int(label)}",
            alpha=0.85
        )

    # шум
    if -1 in unique_labels:
        mask = labels == -1
        plt.scatter(
            x[mask],
            y[mask],
            s=point_size,
            c=noise_color,
            marker=noise_marker,
            label=noise_label,
            alpha=0.8
        )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)

    if legend:
        plt.legend()

    plt.tight_layout()
    plt.show()


def evaluate_clustering(
        data,
        labels,
        use_silhouette=False,
        use_db=False,
        use_ch=False
):
    """
    Оценка качества кластеризации без эталонной разметки.

    Parameters
    ----------
    data : list[list] | np.ndarray
        Матрица признаков, на которой выполнялась кластеризация.

    labels : list[int] | np.ndarray
        Метки кластеров.

    use_silhouette : bool
        Считать ли silhouette score.

    use_db : bool
        Считать ли Davies-Bouldin index.

    use_ch : bool
        Считать ли Calinski-Harabasz index.

    Returns
    -------
    dict
        {
            "metrics": {...},
            "interpretation": {...},
            "overall_score": int,
            "overall_label": str
        }
    """

    X = np.array(data, dtype=float)
    labels = np.array(labels)

    results = {
        "metrics": {},
        "interpretation": {},
        "overall_score": None,
        "overall_label": None
    }

    # Убираем шум HDBSCAN для внутренних метрик
    mask = labels != -1
    labels_eval = labels[mask]
    X_eval = X[mask]

    unique_clusters = np.unique(labels_eval)

    # Нельзя оценивать, если после удаления шума осталось < 2 кластеров
    if len(unique_clusters) < 2 or len(X_eval) < 2:
        results["overall_score"] = 0
        results["overall_label"] = "Недостаточно кластеров для оценки"
        return results

    scores_for_summary = []

    # --------------------------
    # Silhouette
    # --------------------------
    if use_silhouette:
        val = float(silhouette_score(X_eval, labels_eval))
        results["metrics"]["silhouette"] = val

        if val > 0.5:
            label = "Хорошо"
            text = "Кластеры хорошо отделены друг от друга."
            score = 2
        elif val >= 0.2:
            label = "Нормально"
            text = "Структура кластеров присутствует, но разделение умеренное."
            score = 1
        else:
            label = "Слабо"
            text = "Кластеры плохо отделены или сильно пересекаются."
            score = 0

        results["interpretation"]["silhouette"] = {
            "label": label,
            "text": text
        }
        scores_for_summary.append(score)

    # --------------------------
    # Davies-Bouldin
    # --------------------------
    if use_db:
        val = float(davies_bouldin_score(X_eval, labels_eval))
        results["metrics"]["davies_bouldin"] = val

        if val < 1.0:
            label = "Хорошо"
            text = "Кластеры компактны и хорошо разделены."
            score = 2
        elif val <= 2.0:
            label = "Нормально"
            text = "Кластеры различимы, но разделение неидеально."
            score = 1
        else:
            label = "Слабо"
            text = "Кластеры плохо разделены или слишком растянуты."
            score = 0

        results["interpretation"]["davies_bouldin"] = {
            "label": label,
            "text": text
        }
        scores_for_summary.append(score)

    # --------------------------
    # Calinski-Harabasz
    # --------------------------
    if use_ch:
        val = float(calinski_harabasz_score(X_eval, labels_eval))
        results["metrics"]["calinski_harabasz"] = val

        label = "Справочно"
        text = "Чем больше значение, тем лучше разделение кластеров. Удобно для сравнения разных запусков."
        results["interpretation"]["calinski_harabasz"] = {
            "label": label,
            "text": text
        }

    # --------------------------
    # Общий вердикт
    # --------------------------
    if len(scores_for_summary) == 0:
        results["overall_score"] = None
        results["overall_label"] = "Метрики не выбраны"
        return results

    avg_score = sum(scores_for_summary) / len(scores_for_summary)

    if avg_score >= 1.5:
        overall = "Хорошее качество кластеризации"
    elif avg_score >= 0.75:
        overall = "Приемлемое качество кластеризации"
    else:
        overall = "Слабое качество кластеризации"

    results["overall_score"] = avg_score
    results["overall_label"] = overall

    return results


def build_clustering_report(
        preprocess_mode,
        pca_mode,
        pca_info,
        cluster_info,
        result_info,
        evaluation
):
    """
    Формирует HTML-строку для set_info.
    """

    lines = []

    # --------------------------
    # Настройки
    # --------------------------
    settings = []

    # preprocess
    settings.append(f"Preprocess: {preprocess_mode}")

    # PCA
    if pca_mode == "fixed_components":
        settings.append(f"PCA: {pca_info['components_after_pca']} components")
    elif pca_mode == "variance_ratio":
        settings.append(
            f"PCA: (var={pca_info['explained_variance']:.2f})"
        )
    else:
        settings.append(
            "PCA: off"
        )

    # clustering

    if cluster_info["method"] == "kmeans":
        settings.append(
            f"KMeans (k={cluster_info['kmeans_n']}, init={cluster_info['kmeans_n_init']})"
        )

    elif cluster_info["method"] == "hdbscan":
        settings.append(
            f"HDBSCAN (min size={cluster_info['min_size']}, min sample={cluster_info['min_sample']}, "
            f"type={cluster_info['hdbscan_type']})"
        )

    elif cluster_info["method"] == "gmm":
        settings.append(
            f"GMM (k={cluster_info['n']}, type={cluster_info['gmm_type']})"
        )

    lines.append(" | ".join(settings))
    lines.append("")  # пустая строка
    lines.append(f"Результат: {result_info}")
    lines.append("")  # пустая строка

    # --------------------------
    # Метрики
    # --------------------------
    metrics = evaluation["metrics"]
    interp = evaluation["interpretation"]

    if "silhouette" in metrics and metrics["silhouette"] is not None:
        lines.append(
            f"Silhouette: {metrics['silhouette']:.2f} — {interp['silhouette']['label']}"
        )

    if "davies_bouldin" in metrics and metrics["davies_bouldin"] is not None:
        lines.append(
            f"Davies-Bouldin: {metrics['davies_bouldin']:.2f} — {interp['davies_bouldin']['label']}"
        )

    if "calinski_harabasz" in metrics and metrics["calinski_harabasz"] is not None:
        lines.append(
            f"Calinski-Harabasz: {metrics['calinski_harabasz']:.1f} — {interp['calinski_harabasz']['label']}"
        )

    lines.append("")  # пустая строка

    # --------------------------
    # Итог
    # --------------------------
    lines.append(f"Итог: {evaluation['overall_label']}")

    # --------------------------
    # HTML
    # --------------------------
    return "<br>".join(lines)



def calculate_cluster():
    clust_object_id = get_curr_clust_object_id()
    clust_analys_id = get_curr_clust_analys_id()
    clust_object = session.query(ObjectSet).filter_by(id=clust_object_id).first()
    data = json.loads(clust_object.data)
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

    profile_labels = {}
    invalid_prof_index_count = 0
    duplicate_prof_index_count = 0

    if len(label_list) != len(kept_row_indices):
        set_info(
            f'Внимание: размер labels ({len(label_list)}) не совпадает с числом сохраненных строк '
            f'({len(kept_row_indices)}). Построение профилей выполнено частично.',
            'brown'
        )

    for clean_row_idx, label in enumerate(label_list):
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

        if profile_id not in profile_labels:
            profile_labels[profile_id] = {}

        if trace_index in profile_labels[profile_id]:
            duplicate_prof_index_count += 1

        profile_labels[profile_id][trace_index] = int(label)

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

    plot_cluster_map(label_list, data)

    print(label_list)
    print(clust_info)

    result_eval = evaluate_clustering(
        data_pca,
        label_list,
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
            "gmm_type": gmm_type
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
            "n_points": len(label_list),
            "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
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
        set_info("Не удалось автоматически выбрать исследование/профиль для отрисовки кластеров.", "brown")
