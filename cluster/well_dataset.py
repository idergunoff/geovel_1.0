from __future__ import annotations

from .common import *
from .models import (
    AUTO_SILHOUETTE_ADAPTIVE_ALPHA,
    AUTO_SILHOUETTE_COARSE_MAX_SAMPLES,
    AUTO_SILHOUETTE_MAX_SAMPLES,
    AUTO_SILHOUETTE_MIN_SAMPLES,
    AutoTuningClusterSizeLimits,
)

def _set_cluster_well_collect_button_state(is_actual: bool) -> None:
    button = getattr(ui, 'pushButton_clust_collect_well_log', None)
    if button is None:
        return
    if is_actual:
        button.setStyleSheet('background-color: rgb(143, 240, 164);')
    else:
        button.setStyleSheet('background-color: rgb(255, 166, 166);')


def _invalidate_cluster_well_dataset_data(dataset_id: int) -> int:
    removed_rows = (
        session.query(WellLogClusterDatasetData)
        .filter(WellLogClusterDatasetData.dataset_id == int(dataset_id))
        .delete(synchronize_session=False)
    )
    _set_cluster_well_collect_button_state(False)
    return int(removed_rows or 0)

def update_cluster_well_dataset_combobox(select_dataset_id: int | None = None) -> None:
    """
    Перечитывает список наборов каротажа и заполняет comboBox_cluster_well_set.
    """
    if not hasattr(ui, 'comboBox_cluster_well_set'):
        return

    datasets = session.query(WellLogClusterDataset).order_by(WellLogClusterDataset.created_at, WellLogClusterDataset.id).all()
    combo = ui.comboBox_cluster_well_set
    combo.blockSignals(True)
    combo.clear()
    for dataset in datasets:
        combo.addItem(dataset.name, dataset.id)

    if combo.count() > 0:
        index_to_select = 0
        if select_dataset_id is not None:
            found_index = combo.findData(select_dataset_id)
            if found_index >= 0:
                index_to_select = found_index
        combo.setCurrentIndex(index_to_select)
    combo.blockSignals(False)
    load_cluster_well_dataset_state_to_form(combo.currentData())


def load_cluster_well_dataset_state_to_form(dataset_id: int | None = None) -> None:
    """
    Этап 1.3:
    При выборе dataset загружает связанное состояние вкладки:
    - список скважин набора;
    - список каротажных параметров (canonical + calculator).
    """
    list_well = getattr(ui, 'listWidget_cluster_list_well', None)
    list_log = getattr(ui, 'listWidget_cluster_list_log', None)
    if list_well is None or list_log is None:
        return

    list_well.clear()
    list_log.clear()
    if dataset_id is None:
        _set_cluster_well_collect_button_state(False)
        return

    wells = (
        session.query(WellForCluster)
        .filter(WellForCluster.dataset_id == dataset_id)
        .join(Well, Well.id == WellForCluster.well_id)
        .order_by(Well.name, Well.id)
        .all()
    )
    for row in wells:
        well_name = row.well.name if row.well and row.well.name else f'well_id={row.well_id}'
        marker_text = ''
        full_interval = (
            session.query(func.min(WellLog.begin), func.max(WellLog.end))
            .filter(WellLog.well_id == row.well_id)
            .first()
        )
        if full_interval and full_interval[0] is not None and full_interval[1] is not None:
            full_top = float(full_interval[0])
            full_bottom = float(full_interval[1])
            if full_bottom < full_top:
                full_top, full_bottom = full_bottom, full_top
            if abs(float(row.top_md) - full_top) > 1e-9 or abs(float(row.bottom_md) - full_bottom) > 1e-9:
                marker_text = ' ✓interval'

        text = f'{well_name} [{row.top_md:g} - {row.bottom_md:g}]{marker_text}'
        item = QListWidgetItem(text)
        item.setData(Qt.UserRole, row.well_id)
        list_well.addItem(item)

    canonical_params = (
        session.query(ClusterWellLogParameter)
        .filter(ClusterWellLogParameter.dataset_id == dataset_id)
        .join(CanonicalWellLog, CanonicalWellLog.id == ClusterWellLogParameter.canonical_id)
        .order_by(CanonicalWellLog.canonical_name)
        .all()
    )
    dataset_well_count = (
        session.query(func.count(WellForCluster.id))
        .filter(WellForCluster.dataset_id == dataset_id)
        .scalar()
    ) or 0

    for row in canonical_params:
        canonical_name = row.canonical_name.canonical_name if row.canonical_name else f'canonical_id={row.canonical_id}'
        covered_well_count = (
            session.query(func.count(func.distinct(WellLog.well_id)))
            .join(AliasWellLog, AliasWellLog.alias_name_norm == func.lower(func.trim(WellLog.curve_name)))
            .filter(
                AliasWellLog.canonical_id == row.canonical_id,
                WellLog.well_id.in_(
                    session.query(WellForCluster.well_id).filter(WellForCluster.dataset_id == dataset_id)
                ),
            )
            .scalar()
        ) or 0

        label_text = f'{canonical_name} ({int(covered_well_count)}/{int(dataset_well_count)})'
        item = QListWidgetItem(label_text)
        item.setData(Qt.UserRole, ('canonical', row.canonical_id))
        list_log.addItem(item)

    calculator_params = (
        session.query(ClusterWellLogParameterFromCalculator)
        .filter(ClusterWellLogParameterFromCalculator.dataset_id == dataset_id)
        .join(FeatureCalculator, FeatureCalculator.id == ClusterWellLogParameterFromCalculator.calculator_id)
        .order_by(FeatureCalculator.feature_name)
        .all()
    )
    for row in calculator_params:
        feature_name = row.calculator.feature_name if row.calculator else f'calculator_id={row.calculator_id}'
        item = QListWidgetItem(f'{feature_name} [calc]')
        item.setData(Qt.UserRole, ('calculator', row.calculator_id))
        list_log.addItem(item)

    has_data = (
        session.query(WellLogClusterDatasetData.id)
        .filter(WellLogClusterDatasetData.dataset_id == int(dataset_id))
        .first() is not None
    )
    _set_cluster_well_collect_button_state(has_data)


def open_cluster_well_interval_form(item: QListWidgetItem | None = None) -> None:
    """
    Этап 2.4:
    Двойной клик по скважине набора открывает form_well_log для этой скважины.
    """
    if item is None:
        return

    well_id = item.data(Qt.UserRole)
    try:
        well_id = int(well_id)
    except (TypeError, ValueError):
        QMessageBox.warning(MainWindow, 'Интервал скважины', 'Не удалось определить id выбранной скважины.')
        return

    try:
        selected_item = None
        for idx in range(ui.listWidget_well.count()):
            candidate = ui.listWidget_well.item(idx)
            if candidate is None:
                continue
            candidate_well_id = str(candidate.text()).split(' id')[-1].strip()
            if candidate_well_id == str(well_id):
                selected_item = candidate
                break

        if selected_item is None:
            raise RuntimeError(
                f'Скважина id={well_id} не найдена в основном списке listWidget_well. '
                f'Выберите её на вкладке скважин и повторите.'
            )

        ui.listWidget_well.setCurrentItem(selected_item)
        dataset_well_list = getattr(ui, 'listWidget_cluster_list_well', None)
        selected_dataset_well = dataset_well_list.currentItem() if dataset_well_list is not None else None
        if selected_dataset_well is None:
            QMessageBox.information(MainWindow, 'ADD LOG', 'Выберите скважину из списка набора (listWidget_cluster_list_well).')
            return

        well_id = selected_dataset_well.data(Qt.UserRole)
        try:
            well_id = int(well_id)
        except (TypeError, ValueError):
            QMessageBox.warning(MainWindow, 'ADD LOG', 'Не удалось определить id скважины из списка набора.')
            return

        selected_item = None
        for idx in range(ui.listWidget_well.count()):
            candidate = ui.listWidget_well.item(idx)
            if candidate is None:
                continue
            candidate_well_id = str(candidate.text()).split(' id')[-1].strip()
            if candidate_well_id == str(well_id):
                selected_item = candidate
                break

        if selected_item is None:
            QMessageBox.warning(MainWindow, 'ADD LOG', f'Скважина id={well_id} не найдена в основном списке listWidget_well.')
            return

        ui.listWidget_well.setCurrentItem(selected_item)
        from well_log import show_well_log as open_well_log_form
        open_well_log_form()
    except Exception as exc:
        QMessageBox.warning(MainWindow, 'Интервал скважины', f'Не удалось открыть form_well_log: {exc}')


def add_single_well_log_parameter_to_cluster_dataset() -> None:
    """
    Этап 3.1:
    Кнопка pushButton_cluster_add_well_log:
    - если форма form_well_log уже открыта, выполняет добавление через её кнопку LOG TO CLUST;
    - иначе открывает form_well_log для текущей скважины.
    """
    try:
        app = QtWidgets.QApplication.instance()
        if app is None:
            return

        for widget in app.topLevelWidgets():
            if not isinstance(widget, QtWidgets.QDialog):
                continue
            button = widget.findChild(QtWidgets.QPushButton, 'pushButton_add_well_log_to_cluster')
            if button is None:
                continue
            widget.raise_()
            widget.activateWindow()
            button.click()
            return

        from well_log import show_well_log as open_well_log_form
        open_well_log_form()
    except Exception as exc:
        QMessageBox.warning(MainWindow, 'ADD LOG', f'Не удалось выполнить добавление параметра: {exc}')


def add_all_well_log_parameters_to_cluster_dataset() -> None:
    """
    Этап 3.2 (ADD ALL):
    - добавляет все доступные "чистые" (не calculator) canonical-параметры,
      встречающиеся в скважинах текущего dataset;
    - пропускает неканонизируемые названия и дубли.
    """
    combo = getattr(ui, 'comboBox_cluster_well_set', None)
    if combo is None:
        return

    dataset_id = combo.currentData()
    if dataset_id is None:
        QMessageBox.warning(MainWindow, 'ADD ALL LOG', 'Сначала создайте или выберите набор каротажа.')
        return

    from canonical_well_log_service import resolve_canonical

    well_ids = [
        int(row[0])
        for row in session.query(WellForCluster.well_id).filter(WellForCluster.dataset_id == int(dataset_id)).all()
    ]
    if not well_ids:
        QMessageBox.information(MainWindow, 'ADD ALL LOG', 'В наборе нет скважин. Сначала добавьте скважины.')
        return

    existing_canonical_ids = {
        int(row[0])
        for row in (
            session.query(ClusterWellLogParameter.canonical_id)
            .filter(ClusterWellLogParameter.dataset_id == int(dataset_id))
            .all()
        )
    }

    curves = (
        session.query(WellLog.curve_name)
        .filter(WellLog.well_id.in_(well_ids), WellLog.curve_name.isnot(None), func.trim(WellLog.curve_name) != '')
        .all()
    )

    canonical_ids_to_add: set[int] = set()
    skipped_unmapped = 0
    for (curve_name,) in curves:
        canonical_name = resolve_canonical(curve_name)
        if not canonical_name:
            skipped_unmapped += 1
            continue
        canonical = session.query(CanonicalWellLog.id).filter(CanonicalWellLog.canonical_name == canonical_name).first()
        if canonical is None:
            skipped_unmapped += 1
            continue
        canonical_id = int(canonical[0])
        if canonical_id in existing_canonical_ids:
            continue
        canonical_ids_to_add.add(canonical_id)

    if canonical_ids_to_add:
        session.add_all(
            ClusterWellLogParameter(dataset_id=int(dataset_id), canonical_id=canonical_id)
            for canonical_id in sorted(canonical_ids_to_add)
        )
        removed_data_rows = _invalidate_cluster_well_dataset_data(int(dataset_id))
        session.commit()
    else:
        session.rollback()

    load_cluster_well_dataset_state_to_form(int(dataset_id))
    set_info(
        f'ADD ALL LOG: добавлено {len(canonical_ids_to_add)} параметров, сброшено строк data {removed_data_rows if canonical_ids_to_add else 0}, пропущено без canonical {skipped_unmapped}.',
        'green' if canonical_ids_to_add else 'brown'
    )


def remove_selected_well_log_parameters_from_cluster_dataset() -> None:
    """
    Этап 3.3 (удаление параметров):
    - удаляет выделенные параметры каротажа из списка набора;
    - поддерживает удаление canonical и calculator-параметров;
    - очищает собранные строки data текущего dataset.
    """
    combo = getattr(ui, 'comboBox_cluster_well_set', None)
    list_log = getattr(ui, 'listWidget_cluster_list_log', None)
    if combo is None or list_log is None:
        return

    dataset_id = combo.currentData()
    if dataset_id is None:
        QMessageBox.warning(MainWindow, 'RM LOG', 'Сначала создайте или выберите набор каротажа.')
        return

    selected_items = list_log.selectedItems()
    if not selected_items:
        QMessageBox.information(MainWindow, 'RM LOG', 'Выберите параметры в списке набора для удаления.')
        return

    canonical_ids_to_remove: set[int] = set()
    calculator_ids_to_remove: set[int] = set()
    for item in selected_items:
        payload = item.data(Qt.UserRole)
        if not isinstance(payload, tuple) or len(payload) != 2:
            continue
        entity_type, entity_id = payload
        try:
            entity_id_int = int(entity_id)
        except (TypeError, ValueError):
            continue
        if entity_type == 'canonical':
            canonical_ids_to_remove.add(entity_id_int)
        elif entity_type == 'calculator':
            calculator_ids_to_remove.add(entity_id_int)

    if not canonical_ids_to_remove and not calculator_ids_to_remove:
        QMessageBox.warning(MainWindow, 'RM LOG', 'Не удалось определить id выбранных параметров.')
        return

    removed_canonical_count = 0
    removed_calculator_count = 0
    if canonical_ids_to_remove:
        removed_canonical_count = (
            session.query(ClusterWellLogParameter)
            .filter(
                ClusterWellLogParameter.dataset_id == int(dataset_id),
                ClusterWellLogParameter.canonical_id.in_(canonical_ids_to_remove)
            )
            .delete(synchronize_session=False)
        )
    if calculator_ids_to_remove:
        removed_calculator_count = (
            session.query(ClusterWellLogParameterFromCalculator)
            .filter(
                ClusterWellLogParameterFromCalculator.dataset_id == int(dataset_id),
                ClusterWellLogParameterFromCalculator.calculator_id.in_(calculator_ids_to_remove)
            )
            .delete(synchronize_session=False)
        )

    removed_data_rows = _invalidate_cluster_well_dataset_data(int(dataset_id))
    session.commit()

    load_cluster_well_dataset_state_to_form(int(dataset_id))
    set_info(
        f'RM LOG: удалено canonical {removed_canonical_count}, calculator {removed_calculator_count}, '
        f'удалено строк data {removed_data_rows}.',
        'green'
    )


def clear_all_well_log_parameters_from_cluster_dataset() -> None:
    """
    Полная очистка списка параметров каротажа по кнопке CLEAR:
    - удаляет все canonical и calculator-параметры текущего dataset;
    - очищает собранные строки data текущего dataset.
    """
    combo = getattr(ui, 'comboBox_cluster_well_set', None)
    if combo is None:
        return

    dataset_id = combo.currentData()
    if dataset_id is None:
        QMessageBox.warning(MainWindow, 'CLEAR LOGS', 'Сначала создайте или выберите набор каротажа.')
        return

    answer = QMessageBox.question(
        MainWindow,
        'Подтверждение очистки',
        'Очистить весь список каротажных параметров текущего набора?\n\n'
        'Будут удалены все выбранные параметры и связанные собранные данные.',
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
    )
    if answer != QMessageBox.Yes:
        return

    removed_canonical_count = (
        session.query(ClusterWellLogParameter)
        .filter(ClusterWellLogParameter.dataset_id == int(dataset_id))
        .delete(synchronize_session=False)
    )
    removed_calculator_count = (
        session.query(ClusterWellLogParameterFromCalculator)
        .filter(ClusterWellLogParameterFromCalculator.dataset_id == int(dataset_id))
        .delete(synchronize_session=False)
    )
    removed_data_rows = _invalidate_cluster_well_dataset_data(int(dataset_id))
    session.commit()

    load_cluster_well_dataset_state_to_form(int(dataset_id))
    set_info(
        f'CLEAR LOGS: удалено canonical {removed_canonical_count}, calculator {removed_calculator_count}, '
        f'удалено строк data {removed_data_rows}.',
        'green'
    )


def create_cluster_well_dataset() -> None:
    """
    Этап 1.1:
    - берет имя из lineEdit_string;
    - валидирует;
    - проверяет дубликаты;
    - создает набор и выбирает его в combobox.
    """
    raw_name = ui.lineEdit_string.text()
    dataset_name = raw_name.strip() if raw_name else ''
    if not dataset_name:
        QMessageBox.critical(MainWindow, 'Ошибка', 'Введите имя набора в строке lineEdit_string.')
        return

    existing_dataset = (
        session.query(WellLogClusterDataset)
        .filter(func.lower(WellLogClusterDataset.name) == dataset_name.casefold())
        .first()
    )
    if existing_dataset is not None:
        QMessageBox.critical(MainWindow, 'Ошибка', f'Набор "{dataset_name}" уже существует.')
        update_cluster_well_dataset_combobox(select_dataset_id=existing_dataset.id)
        return

    new_dataset = WellLogClusterDataset(name=dataset_name)
    session.add(new_dataset)
    session.commit()

    update_cluster_well_dataset_combobox(select_dataset_id=new_dataset.id)
    set_info(f'Добавлен набор каротажа "{dataset_name}"', 'green')


def remove_cluster_well_dataset() -> None:
    """
    Этап 1.2:
    - подтверждает удаление набора;
    - удаляет набор и все зависимые сущности (каскад ORM);
    - обновляет combobox и выбирает ближайший доступный набор.
    """
    combo = getattr(ui, 'comboBox_cluster_well_set', None)
    if combo is None:
        return

    current_dataset_id = combo.currentData()
    if current_dataset_id is None:
        QMessageBox.information(MainWindow, 'Удаление набора', 'Нет выбранного набора для удаления.')
        return

    dataset = session.query(WellLogClusterDataset).filter_by(id=current_dataset_id).first()
    if dataset is None:
        update_cluster_well_dataset_combobox()
        QMessageBox.warning(MainWindow, 'Удаление набора', 'Выбранный набор не найден. Список будет обновлен.')
        return

    answer = QMessageBox.question(
        MainWindow,
        'Подтверждение удаления',
        f'Удалить набор "{dataset.name}"?\n\n'
        'Будут удалены связанные скважины, интервалы, каротажи и собранные данные.',
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
    )
    if answer != QMessageBox.Yes:
        return

    current_index = combo.currentIndex()
    session.delete(dataset)
    session.commit()

    next_index = current_index if current_index >= 0 else 0
    update_cluster_well_dataset_combobox()
    if combo.count() > 0:
        combo.setCurrentIndex(min(next_index, combo.count() - 1))
    set_info(f'Набор каротажа "{dataset.name}" удален', 'green')


def add_selected_well_to_cluster_dataset() -> None:
    """
    Этап 2.1 (ADD WELL):
    - берет текущую выбранную скважину из listWidget_well;
    - проверяет, что скважина выбрана;
    - проверяет наличие каротажа (well_log);
    - проверяет отсутствие дубля в текущем наборе;
    - добавляет скважину в набор и обновляет состояние формы.
    """
    combo = getattr(ui, 'comboBox_cluster_well_set', None)
    if combo is None:
        return

    dataset_id = combo.currentData()
    if dataset_id is None:
        QMessageBox.warning(MainWindow, 'ADD WELL', 'Сначала создайте или выберите набор каротажа.')
        return

    current_well_item = getattr(ui, 'listWidget_well', None).currentItem() if hasattr(ui, 'listWidget_well') else None
    if current_well_item is None:
        QMessageBox.warning(MainWindow, 'ADD WELL', 'Выберите скважину в списке listWidget_well.')
        return

    try:
        well_id = int(current_well_item.text().split(' id')[-1])
    except (TypeError, ValueError, AttributeError):
        QMessageBox.warning(MainWindow, 'ADD WELL', 'Не удалось определить id выбранной скважины.')
        return

    well = session.query(Well).filter_by(id=well_id).first()
    if well is None:
        QMessageBox.warning(MainWindow, 'ADD WELL', 'Выбранная скважина не найдена в БД.')
        return

    has_well_log = session.query(WellLog.id).filter(WellLog.well_id == well_id).first() is not None
    if not has_well_log:
        QMessageBox.warning(MainWindow, 'ADD WELL', f'У скважины "{well.name}" отсутствует каротаж (well_log).')
        return

    existing_row = (
        session.query(WellForCluster.id)
        .filter(WellForCluster.dataset_id == dataset_id, WellForCluster.well_id == well_id)
        .first()
    )
    if existing_row is not None:
        QMessageBox.information(MainWindow, 'ADD WELL', f'Скважина "{well.name}" уже добавлена в набор.')
        return

    top_bottom_row = (
        session.query(func.min(WellLog.begin), func.max(WellLog.end))
        .filter(WellLog.well_id == well_id)
        .first()
    )
    top_md = float(top_bottom_row[0]) if top_bottom_row and top_bottom_row[0] is not None else 0.0
    bottom_md = float(top_bottom_row[1]) if top_bottom_row and top_bottom_row[1] is not None else top_md
    if bottom_md < top_md:
        top_md, bottom_md = bottom_md, top_md

    session.add(
        WellForCluster(
            dataset_id=dataset_id,
            well_id=well_id,
            top_md=top_md,
            bottom_md=bottom_md,
        )
    )
    removed_data_rows = _invalidate_cluster_well_dataset_data(int(dataset_id))
    session.commit()

    load_cluster_well_dataset_state_to_form(dataset_id)
    set_info(f'Скважина "{well.name}" добавлена в набор каротажа. Сброшено строк data: {removed_data_rows}.', 'green')


def add_wells_to_cluster_dataset_from_radius() -> None:
    """
    Этап 2.2 (ADD WELLS):
    - берет радиус из spinBox_well_distance;
    - получает скважины вокруг профилей текущего ObjectSet;
    - добавляет только скважины с каротажем;
    - пропускает дубли в текущем наборе.
    """
    combo = getattr(ui, 'comboBox_cluster_well_set', None)
    if combo is None:
        return

    dataset_id = combo.currentData()
    if dataset_id is None:
        QMessageBox.warning(MainWindow, 'ADD WELLS', 'Сначала создайте или выберите набор каротажа.')
        return

    radius_spinbox = getattr(ui, 'spinBox_well_distance', None)
    if radius_spinbox is None:
        QMessageBox.warning(MainWindow, 'ADD WELLS', 'Не найден spinBox_well_distance для чтения радиуса.')
        return
    radius = float(radius_spinbox.value())

    clust_object_id = get_curr_clust_object_id()
    if not clust_object_id:
        QMessageBox.warning(MainWindow, 'ADD WELLS', 'Выберите текущий ObjectSet кластерного анализа.')
        return

    clust_object = session.query(ObjectSet).filter_by(id=int(clust_object_id)).first()
    if clust_object is None or clust_object.research_id is None:
        QMessageBox.warning(MainWindow, 'ADD WELLS', 'Не удалось получить исследование выбранного ObjectSet.')
        return

    # Профили берем по текущему GeoradarObject (а не по ObjectSet):
    # это исключает попадание скважин из профилей других объектов.
    current_object_id = get_object_id()
    if not current_object_id:
        QMessageBox.warning(MainWindow, 'ADD WELLS', 'Не выбран текущий GeoradarObject.')
        return

    profiles = (
        session.query(Profile)
        .join(Research, Profile.research_id == Research.id)
        .filter(Research.object_id == int(current_object_id))
        .order_by(Profile.id)
        .all()
    )

    if not profiles:
        QMessageBox.information(MainWindow, 'ADD WELLS', 'У текущего объекта нет профилей для поиска скважин.')
        return

    existing_well_ids = {
        row[0] for row in session.query(WellForCluster.well_id).filter(WellForCluster.dataset_id == dataset_id).all()
    }

    selected_well_ids: set[int] = set()
    for profile in profiles:
        nearest_wells = get_list_nearest_well(profile.id)
        if not nearest_wells:
            continue
        for near_well in nearest_wells:
            if near_well and near_well[0] is not None and near_well[0].id is not None:
                selected_well_ids.add(int(near_well[0].id))

    if not selected_well_ids:
        QMessageBox.information(MainWindow, 'ADD WELLS', f'В радиусе {radius:g} м от профилей скважины не найдены.')
        return

    wells_to_add = []
    added_count = 0
    skipped_no_log = 0
    skipped_duplicates = 0
    for well_id in sorted(selected_well_ids):
        if well_id in existing_well_ids:
            skipped_duplicates += 1
            continue

        top_bottom_row = (
            session.query(func.min(WellLog.begin), func.max(WellLog.end))
            .filter(WellLog.well_id == well_id)
            .first()
        )
        if top_bottom_row is None or top_bottom_row[0] is None or top_bottom_row[1] is None:
            skipped_no_log += 1
            continue

        top_md = float(top_bottom_row[0])
        bottom_md = float(top_bottom_row[1])
        if bottom_md < top_md:
            top_md, bottom_md = bottom_md, top_md

        wells_to_add.append(
            WellForCluster(
                dataset_id=dataset_id,
                well_id=well_id,
                top_md=top_md,
                bottom_md=bottom_md,
            )
        )
        added_count += 1

    if wells_to_add:
        session.add_all(wells_to_add)
        removed_data_rows = _invalidate_cluster_well_dataset_data(int(dataset_id))
        session.commit()
        load_cluster_well_dataset_state_to_form(dataset_id)
    else:
        session.rollback()

    set_info(
        f'ADD WELLS: добавлено {added_count}, сброшено строк data {removed_data_rows if added_count > 0 else 0}, без каротажа {skipped_no_log}, дублей {skipped_duplicates}.',
        'green' if added_count > 0 else 'brown'
    )


def remove_selected_wells_from_cluster_dataset() -> None:
    """
    Этап 2.3 (ручное удаление скважин):
    - удаляет выделенные скважины из списка скважин набора;
    - очищает связанные интервалы (строки WellForCluster);
    - очищает собранные данные data для текущего dataset
      (на текущей схеме data хранится на уровне dataset).
    """
    combo = getattr(ui, 'comboBox_cluster_well_set', None)
    list_well = getattr(ui, 'listWidget_cluster_list_well', None)
    if combo is None or list_well is None:
        return

    dataset_id = combo.currentData()
    if dataset_id is None:
        QMessageBox.warning(MainWindow, 'RM WELL', 'Сначала создайте или выберите набор каротажа.')
        return

    selected_items = list_well.selectedItems()
    if not selected_items:
        QMessageBox.information(MainWindow, 'RM WELL', 'Выберите скважины в списке набора для удаления.')
        return

    well_ids_to_remove: set[int] = set()
    for item in selected_items:
        well_id = item.data(Qt.UserRole)
        if well_id is None:
            continue
        try:
            well_ids_to_remove.add(int(well_id))
        except (TypeError, ValueError):
            continue

    if not well_ids_to_remove:
        QMessageBox.warning(MainWindow, 'RM WELL', 'Не удалось определить id выбранных скважин.')
        return

    removed_wells_count = (
        session.query(WellForCluster)
        .filter(WellForCluster.dataset_id == dataset_id, WellForCluster.well_id.in_(well_ids_to_remove))
        .delete(synchronize_session=False)
    )
    removed_data_rows = (
        session.query(WellLogClusterDatasetData)
        .filter(WellLogClusterDatasetData.dataset_id == dataset_id)
        .delete(synchronize_session=False)
    )
    session.commit()

    load_cluster_well_dataset_state_to_form(dataset_id)
    set_info(
        f'RM WELL: удалено скважин {removed_wells_count}, удалено строк data {removed_data_rows}.',
        'green'
    )


def clear_all_wells_from_cluster_dataset() -> None:
    """
    Полная очистка списка добавленных скважин по кнопке CLEAR:
    - удаляет все скважины текущего dataset;
    - очищает собранные data текущего dataset.
    """
    combo = getattr(ui, 'comboBox_cluster_well_set', None)
    list_well = getattr(ui, 'listWidget_cluster_list_well', None)
    if combo is None or list_well is None:
        return

    dataset_id = combo.currentData()
    if dataset_id is None:
        QMessageBox.warning(MainWindow, 'CLEAR WELLS', 'Сначала создайте или выберите набор каротажа.')
        return

    answer = QMessageBox.question(
        MainWindow,
        'Подтверждение очистки',
        'Очистить весь список скважин текущего набора?\n\n'
        'Будут удалены все добавленные скважины и связанные собранные данные.',
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
    )
    if answer != QMessageBox.Yes:
        return

    removed_wells_count = (
        session.query(WellForCluster)
        .filter(WellForCluster.dataset_id == dataset_id)
        .delete(synchronize_session=False)
    )
    removed_data_rows = (
        session.query(WellLogClusterDatasetData)
        .filter(WellLogClusterDatasetData.dataset_id == dataset_id)
        .delete(synchronize_session=False)
    )
    session.commit()

    load_cluster_well_dataset_state_to_form(dataset_id)
    set_info(
        f'CLEAR WELLS: удалено скважин {removed_wells_count}, удалено строк data {removed_data_rows}.',
        'green'
    )




def collect_cluster_well_log_dataset_data() -> None:
    combo = getattr(ui, 'comboBox_cluster_well_set', None)
    if combo is None:
        return

    dataset_id = combo.currentData()
    if dataset_id is None:
        QMessageBox.warning(MainWindow, 'COLLECT WELL LOG', 'Сначала создайте или выберите набор каротажа.')
        return

    dataset_id = int(dataset_id)
    wells = session.query(WellForCluster).filter(WellForCluster.dataset_id == dataset_id).order_by(WellForCluster.well_id).all()
    if not wells:
        QMessageBox.critical(MainWindow, 'COLLECT WELL LOG', 'В наборе нет скважин для сборки data.')
        return

    params = (
        session.query(ClusterWellLogParameter)
        .join(CanonicalWellLog, CanonicalWellLog.id == ClusterWellLogParameter.canonical_id)
        .filter(ClusterWellLogParameter.dataset_id == dataset_id)
        .order_by(CanonicalWellLog.canonical_name)
        .all()
    )
    if not params:
        QMessageBox.critical(MainWindow, 'COLLECT WELL LOG', 'В наборе нет выбранных параметров каротажа.')
        return

    wells_without_interval = [row.well.name if row.well else str(row.well_id) for row in wells if row.top_md >= row.bottom_md]
    if wells_without_interval:
        QMessageBox.warning(MainWindow, 'COLLECT WELL LOG', 'У части скважин некорректные интервалы. Исправьте интервалы и повторите.')
        return

    aliases = session.query(AliasWellLog.alias_name_norm, AliasWellLog.canonical_id).all()
    alias_to_canonical = {str(name): int(cid) for name, cid in aliases}

    log_feature_columns = [row.canonical_name.canonical_name for row in params if row.canonical_name]
    columns = ['well_id_depth', WELL_LOG_CLUSTER_SAMPLE_INDEX_FEATURE_NAME] + log_feature_columns
    rows = [columns]

    for wf in wells:
        top_md = float(wf.top_md)
        bottom_md = float(wf.bottom_md)
        # сбор по глубине
        depth_map = {}
        for wl in session.query(WellLog).filter(WellLog.well_id == wf.well_id).all():
            if wl.curve_data is None or wl.step in (None, 0):
                continue
            alias_key = str(wl.curve_name or '').strip().casefold()
            canonical_id = alias_to_canonical.get(alias_key)
            if canonical_id is None:
                continue
            try:
                values = json.loads(wl.curve_data)
            except Exception:
                continue
            if not isinstance(values, list):
                continue
            for idx, value in enumerate(values):
                try:
                    depth = float(wl.begin) + idx * float(wl.step)
                except Exception:
                    continue
                if depth < top_md or depth > bottom_md:
                    continue
                rounded_depth = round(depth, 6)
                key = (rounded_depth)
                if key not in depth_map:
                    depth_map[key] = {}
                depth_map[key][canonical_id] = value

        for sample_index, depth in enumerate(sorted(depth_map.keys())):
            line = [f"{int(wf.well_id)}_{depth:g}", sample_index]
            values_by_canonical = depth_map[depth]
            for param in params:
                line.append(values_by_canonical.get(param.canonical_id, None))
            rows.append(line)

    _invalidate_cluster_well_dataset_data(dataset_id)
    session.add(WellLogClusterDatasetData(dataset_id=dataset_id, data=_serialize_cluster_dataset(rows)))
    session.commit()
    _set_cluster_well_collect_button_state(True)
    set_info(
        f'COLLECT WELL LOG: собрано строк {max(0, len(rows)-1)} и признаков {len(columns)-1} '
        f'(включая {WELL_LOG_CLUSTER_SAMPLE_INDEX_FEATURE_NAME}).',
        'green'
    )

def get_available_well_log_curve_names_with_frequency() -> list[dict[str, int | str]]:
    """
    Возвращает все уникальные названия каротажных кривых из БД с частотностью.

    Используется как источник «всех возможных названий» для формы
    управления canonical/alias (этап 1.3).
    """
    rows = (
        session.query(WellLog.curve_name, func.count(WellLog.id))
        .filter(WellLog.curve_name.isnot(None))
        .group_by(WellLog.curve_name)
        .all()
    )

    aggregated: dict[str, dict[str, int | str]] = {}
    for curve_name, count in rows:
        raw_name = str(curve_name)
        normalized_name = raw_name.strip()
        if not normalized_name:
            continue
        normalized_key = normalized_name.casefold()
        bucket = aggregated.get(normalized_key)
        if bucket is None:
            aggregated[normalized_key] = {
                'curve_name': normalized_name,
                'frequency': int(count),
            }
            continue
        bucket['frequency'] = int(bucket['frequency']) + int(count)

    return sorted(
        aggregated.values(),
        key=lambda item: (-int(item['frequency']), str(item['curve_name']).casefold(), str(item['curve_name']))
    )


def _compute_auto_silhouette_sample_size(
        n_rows: int,
        *,
        stage: str = "coarse",
        estimated_n_clusters: int = 3
) -> int:
    """
    Адаптивно подбирает размер подвыборки для silhouette в AUTO-режиме.

    Формула: alpha * sqrt(n_rows) * estimated_n_clusters с нижним и верхним лимитами.
    """
    n_rows = max(1, int(n_rows))
    estimated_n_clusters = max(2, int(estimated_n_clusters))
    stage_name = str(stage or "coarse").strip().lower()
    stage_cap = int(AUTO_SILHOUETTE_MAX_SAMPLES)
    if stage_name == "coarse":
        stage_cap = min(stage_cap, int(AUTO_SILHOUETTE_COARSE_MAX_SAMPLES))

    adaptive_value = int(AUTO_SILHOUETTE_ADAPTIVE_ALPHA * np.sqrt(n_rows) * estimated_n_clusters)
    return int(min(stage_cap, max(int(AUTO_SILHOUETTE_MIN_SAMPLES), adaptive_value)))


def _build_pilot_sample_for_pca(data, max_rows: int, seed: int = 42) -> Any:
    n_rows = int(len(data)) if data is not None else 0
    if n_rows <= 0 or max_rows <= 0 or n_rows <= int(max_rows):
        return data
    rng = np.random.RandomState(int(seed))
    idx = np.sort(rng.choice(n_rows, size=int(max_rows), replace=False))
    if isinstance(data, np.ndarray):
        return data[idx]
    return np.asarray(data)[idx]


def calculate_auto_min_cluster_sample_limits(
        n_samples: int,
        min_value: int = 1
) -> AutoTuningClusterSizeLimits:
    """
    Вычисляет лимиты для параметра минимального размера кластера в AUTO-режиме.

    - recommended_default_value = ceil(0.05 * N)
    - max_spinbox_value = floor(0.5 * N)
    """
    min_value = max(1, int(min_value))
    n_samples = max(0, int(n_samples))

    recommended_default_value = max(min_value, int(np.ceil(0.05 * n_samples)))
    max_spinbox_value = max(min_value, int(np.floor(0.5 * n_samples)))

    return {
        "min_cluster_samples": int(min_value),
        "recommended_default_value": int(recommended_default_value),
        "max_spinbox_value": int(max_spinbox_value)
    }


def sync_auto_min_cluster_spinbox_with_current_object(*, force_recommended: bool = False) -> None:
    """
    Синхронизирует spinBox минимального размера кластера с текущим ObjectSet.
    """
    spinbox = getattr(ui, "spinBox_cluster_auto_min_n_cluster", None)
    if spinbox is None:
        return

    clust_object_id = get_curr_clust_object_id()
    if not clust_object_id:
        return

    clust_object = session.query(ObjectSet).filter_by(id=clust_object_id).first()
    if clust_object is None:
        return

    n_samples = 0
    try:
        base_data = _deserialize_cluster_dataset(clust_object.data)
        n_samples = len(base_data)
    except Exception:
        pass

    limits = calculate_auto_min_cluster_sample_limits(n_samples, min_value=1)
    old_value = int(spinbox.value())
    spinbox.setMinimum(int(limits["min_cluster_samples"]))

    tooltip_text = (
        "Минимальный размер кластера в образцах для AUTO-подбора. "
        "Кандидаты с кластерами меньше порога помечаются как невалидные."
    )
    spinbox.setToolTip(tooltip_text)
    spinbox.setStatusTip(tooltip_text)
    reset_btn = getattr(ui, "toolButton_cluster_auto_min_n_reset", None)
    if reset_btn is not None:
        reset_btn.setToolTip("Сбросить порог к рекомендованному значению 5% от текущего N.")
    min_label = getattr(ui, "label_46", None)
    if min_label is not None:
        min_label.setToolTip(tooltip_text)
    spinbox.setMaximum(int(limits["max_spinbox_value"]))

    if force_recommended:
        spinbox.setValue(int(limits["recommended_default_value"]))
        return

    if old_value > limits["max_spinbox_value"]:
        spinbox.setValue(int(limits["max_spinbox_value"]))
        set_info(
            f"AUTO: значение 'Минимальный размер кластера' автоматически скорректировано до {limits['max_spinbox_value']} (допустимый максимум для N={n_samples}).",
            "brown"
        )
    elif old_value < limits["min_cluster_samples"]:
        spinbox.setValue(int(limits["min_cluster_samples"]))


def reset_auto_min_cluster_spinbox_to_5_percent() -> None:
    """
    Обработчик кнопки "Сброс к 5%".
    """
    sync_auto_min_cluster_spinbox_with_current_object(force_recommended=True)



def on_cluster_object_selection_changed(*_args) -> None:
    """
    Единый обработчик смены выбранного ObjectSet в combobox.
    """
    show_finite_report()
    load_saved_auto_results_for_selected_object()
    reset_auto_min_cluster_spinbox_to_5_percent()

