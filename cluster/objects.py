from __future__ import annotations

from .common import *

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
    clust_analys_id = get_curr_clust_analys_id()
    if not str(clust_analys_id).isdigit():
        set_info("Не выбран набор признаков для удаления.", "brown")
        return

    object_count = session.query(ObjectSet).filter_by(analysis_id=int(clust_analys_id)).count()
    cache_count = (
        session.query(ClusterAutoTuningCache)
        .join(ObjectSet, ClusterAutoTuningCache.object_set_id == ObjectSet.id)
        .filter(ObjectSet.analysis_id == int(clust_analys_id))
        .count()
    )
    confirm_text = (
        "Вы уверены, что хотите удалить набор признаков?\n\n"
        f"Будет удалено:\n"
        f"• наборов объектов: {object_count}\n"
        f"• сохраненных AUTO-результатов: {cache_count}\n\n"
        "Это действие нельзя отменить."
    )
    answer = QMessageBox.question(
        MainWindow,
        "Подтверждение удаления",
        confirm_text,
        QMessageBox.Yes | QMessageBox.Cancel,
        QMessageBox.Cancel
    )
    if answer != QMessageBox.Yes:
        return

    object_ids_subquery = session.query(ObjectSet.id).filter_by(analysis_id=int(clust_analys_id)).subquery()
    session.query(ClusterAutoTuningCache).filter(
        ClusterAutoTuningCache.object_set_id.in_(select(object_ids_subquery.c.id))
    ).delete(synchronize_session=False)
    session.query(ObjectSet).filter_by(analysis_id=int(clust_analys_id)).delete(synchronize_session=False)
    session.query(AnalysisCluster).filter_by(id=int(clust_analys_id)).delete(synchronize_session=False)
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
    serialized_data = _serialize_cluster_dataset(data)
    new_cluster_obj = ObjectSet(research_id=get_research_id(), analysis_id=get_curr_clust_analys_id(), data=serialized_data, report=report)
    session.add(new_cluster_obj)
    session.commit()
    update_list_clust_object()


def remove_clust_object():
    clust_object_id = get_curr_clust_object_id()
    if not clust_object_id:
        set_info("Не выбран набор объектов для удаления.", "brown")
        return

    cache_count = session.query(ClusterAutoTuningCache).filter_by(object_set_id=int(clust_object_id)).count()
    confirm_text = (
        "Вы уверены, что хотите удалить набор объектов?\n\n"
        f"Будет удалено сохраненных AUTO-результатов: {cache_count}\n\n"
        "Это действие нельзя отменить."
    )
    answer = QMessageBox.question(
        MainWindow,
        "Подтверждение удаления",
        confirm_text,
        QMessageBox.Yes | QMessageBox.Cancel,
        QMessageBox.Cancel
    )
    if answer != QMessageBox.Yes:
        return

    session.query(ClusterAutoTuningCache).filter_by(object_set_id=int(clust_object_id)).delete(synchronize_session=False)
    session.query(ObjectSet).filter_by(id=int(clust_object_id)).delete(synchronize_session=False)
    session.commit()
    update_list_clust_object()


def get_curr_clust_analys_id():
    return ui.comboBox_clust_set.currentText().split(' id')[-1]


def get_curr_clust_object_id():
    text = str(ui.comboBox_clust_obj.currentText() or "").strip()
    if ' id' not in text:
        return None
    candidate = text.split(' id')[-1].strip()
    if not candidate.isdigit():
        return None
    return int(candidate)


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
    sync_auto_min_cluster_spinbox_with_current_object()
    load_saved_auto_results_for_selected_object()


def show_finite_report():
    obj_set = session.query(ObjectSet.report).filter_by(id=get_curr_clust_object_id()).first()
    if obj_set:
        set_info(obj_set.report, 'brown')


def load_saved_auto_results_for_selected_object() -> None:
    """
    При выборе ObjectSet подгружает последнюю сохраненную таблицу AUTO-кластеризации (если есть).
    """
    run_context = build_cluster_run_context(show_errors=False)
    if run_context is not None:
        load_saved_auto_results_for_context(run_context)
        return

    clust_object_id = get_curr_clust_object_id()
    if not clust_object_id:
        render_auto_results_table([])
        return

    try:
        cache_rows = (
            session.query(ClusterAutoTuningCache)
            .filter_by(object_set_id=int(clust_object_id))
            .order_by(ClusterAutoTuningCache.created_at.desc())
            .all()
        )
    except Exception as exc:
        render_auto_results_table([])
        set_info(f"AUTO: ошибка чтения сохраненного результата: {exc}", "brown")
        return

    cache_row = next((row for row in cache_rows if _cache_row_matches_source_type(row, "gpr")), None)
    if cache_row is None or not cache_row.top_results:
        render_auto_results_table([])
        return

    try:
        cached_results = json.loads(cache_row.top_results)
    except Exception as exc:
        render_auto_results_table([])
        set_info(f"AUTO: ошибка распаковки сохраненного результата: {exc}", "brown")
        return

    if not isinstance(cached_results, list):
        render_auto_results_table([])
        return

    render_auto_results_table(cached_results)
    set_info(
        f"AUTO: загружены сохраненные top-{len(cached_results)} настройки для выбранного набора.",
        "green"
    )


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


