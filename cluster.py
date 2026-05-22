import pandas as pd
import base64
import gzip
import hashlib
import json
import random
import gc
import multiprocessing as mp
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from itertools import product
from time import monotonic
from typing import Any, Dict, Literal, Optional, TypedDict

import build_table
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.interpolate import griddata
from sklearn.random_projection import SparseRandomProjection
from draw import draw_radarogram
from func import *
from krige import draw_map

# Runtime-cache результатов кластеризации для быстрой перерисовки по профилям без пересчета.
# Этап 1 (MVP): ключ анализа = id ObjectSet (clust_object_id).
cluster_profile_cache = {}
is_cluster_redraw_in_progress = False


class CandidateConfig(TypedDict):
    """
    Единый контракт кандидата для AUTO-подбора параметров кластеризации.
    """
    scaler_mode: str
    pca_enabled: bool
    pca_mode: Optional[str]
    pca_value: Optional[float]
    method: Literal["kmeans", "hdbscan", "gmm"]
    method_params: Dict[str, Any]


class CandidateMetrics(TypedDict, total=False):
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float


class CandidateStats(TypedDict, total=False):
    n_clusters: int
    noise_fraction: float
    n_samples_eval: int
    pca_components_after: int
    partition_hash: str


class CandidateResult(TypedDict):
    """
    Единый контракт результата прогона кандидата для AUTO-подбора.
    """
    candidate_id: str
    candidate_config: CandidateConfig
    metrics: CandidateMetrics
    stats: CandidateStats
    score: Optional[float]
    status: Literal["ok", "invalid", "error"]
    error_text: str


class AutoTuningClusterSizeLimits(TypedDict):
    min_cluster_samples: int
    recommended_default_value: int
    max_spinbox_value: int


cluster_auto_results_cache: list[CandidateResult] = []


CLUSTER_DATA_GZIP_PREFIX = "gzjson:"
GMM_COVARIANCE_TYPES = ("full", "diag", "tied", "spherical")
AUTO_TRANSFORM_CACHE_MAX_BYTES = 512 * 1024 * 1024
AUTO_TRANSFORM_CACHE_MAX_ITEM_BYTES = 128 * 1024 * 1024
AUTO_SILHOUETTE_MAX_SAMPLES = 5000
AUTO_SILHOUETTE_COARSE_MAX_SAMPLES = 1500
AUTO_SILHOUETTE_ADAPTIVE_ALPHA = 24.0
AUTO_SILHOUETTE_MIN_SAMPLES = 400
AUTO_FINE_SEED_DELTA_FROM_BEST = 0.03
AUTO_PCA_PILOT_ENABLED = True
AUTO_PCA_PILOT_MAX_ROWS = 500
AUTO_TUNING_MAX_ROWS = 20000
AUTO_TRANSFORM_CACHE_MAX_ROWS = 12000
AUTO_TUNING_MAX_WORKING_SET_BYTES = 256 * 1024 * 1024
AUTO_TUNING_MIN_ROWS = 256
AUTO_TUNING_MAX_FEATURES = 512
AUTO_TUNING_REDUCE_FEATURES_THRESHOLD = 10000
AUTO_TUNING_FEATURE_REDUCTION_MODE = "auto"
AUTO_CANDIDATE_HARD_TIMEOUT_SEC: Optional[float] = None
AUTO_CHECKPOINT_SAVE_EVERY = 10


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
        text = f'{well_name} [{row.top_md:g} - {row.bottom_md:g}]'
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
    for row in canonical_params:
        canonical_name = row.canonical_name.canonical_name if row.canonical_name else f'canonical_id={row.canonical_id}'
        item = QListWidgetItem(canonical_name)
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
    session.commit()

    load_cluster_well_dataset_state_to_form(dataset_id)
    set_info(f'Скважина "{well.name}" добавлена в набор каротажа.', 'green')


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

    profiles = (
        session.query(Profile)
        .filter(Profile.research_id == int(clust_object.research_id))
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
        session.commit()
        load_cluster_well_dataset_state_to_form(dataset_id)
    else:
        session.rollback()

    set_info(
        f'ADD WELLS: добавлено {added_count}, без каротажа {skipped_no_log}, дублей {skipped_duplicates}.',
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

def _estimate_array_like_nbytes(value: Any) -> int:
    """
    Грубая оценка объема памяти array-like объекта в байтах.
    """
    if value is None:
        return 0
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if hasattr(value, "nbytes"):
        try:
            return int(value.nbytes)  # type: ignore[attr-defined]
        except Exception:
            pass
    try:
        return int(np.array(value).nbytes)
    except Exception:
        return 0


def _estimate_transform_cache_item_nbytes(value: Any) -> int:
    """
    Оценивает объем одного элемента transform-cache.
    """
    if isinstance(value, tuple):
        return int(sum(_estimate_array_like_nbytes(v) for v in value))
    return _estimate_array_like_nbytes(value)


def _trim_transform_cache(
        transform_cache: "OrderedDict[tuple, Any]",
        cache_sizes: "OrderedDict[tuple, int]",
        cache_total_bytes: int,
        *,
        max_cache_bytes: int
) -> int:
    """
    Поддерживает LRU-ограничение transform-cache по памяти.
    """
    while cache_total_bytes > max_cache_bytes and transform_cache:
        evicted_key, _ = transform_cache.popitem(last=False)
        cache_total_bytes -= int(cache_sizes.pop(evicted_key, 0))
    return max(0, int(cache_total_bytes))


def _sample_rows_for_auto_tuning(data, max_rows: int) -> Any:
    """
    Ограничивает число строк для AUTO-подбора, чтобы снизить риск OOM.
    """
    if data is None:
        return data
    try:
        max_rows = max(2, int(max_rows))
    except (TypeError, ValueError):
        max_rows = AUTO_TUNING_MAX_ROWS
    data_len = len(data)
    if data_len <= max_rows:
        return data
    n_features = None
    if data_len > 0:
        try:
            first_row = data[0]
            n_features = len(first_row)
        except Exception:
            n_features = None
    if n_features and n_features > 0:
        approx_bytes_per_row = int(max(8, n_features * 8 * 6))
        max_rows_by_memory = int(AUTO_TUNING_MAX_WORKING_SET_BYTES // approx_bytes_per_row)
        max_rows = min(max_rows, max(2, max_rows_by_memory))
    max_rows = max(2, min(max_rows, data_len))
    if data_len <= max_rows:
        return data
    rng = np.random.default_rng(42)
    sampled_idx = np.sort(rng.choice(data_len, size=max_rows, replace=False))
    if isinstance(data, np.ndarray):
        return data[sampled_idx]
    return [data[int(i)] for i in sampled_idx]




def _build_sample_indices_for_auto_tuning(data_len: int, max_rows: int, *, seed: int) -> np.ndarray:
    """
    Строит детерминированные индексы подвыборки для AUTO по seed.
    """
    data_len = max(0, int(data_len))
    if data_len == 0:
        return np.array([], dtype=int)
    try:
        max_rows = max(2, int(max_rows))
    except (TypeError, ValueError):
        max_rows = AUTO_TUNING_MAX_ROWS
    if data_len <= max_rows:
        return np.arange(data_len, dtype=int)
    rng = np.random.default_rng(int(seed))
    sampled_idx = np.sort(rng.choice(data_len, size=max_rows, replace=False))
    return np.asarray(sampled_idx, dtype=int)


def _apply_sample_indices(data, sampled_idx: np.ndarray) -> Any:
    if data is None:
        return data
    if sampled_idx is None or len(sampled_idx) == 0:
        return data
    if isinstance(data, np.ndarray):
        return data[sampled_idx]
    return [data[int(i)] for i in sampled_idx]

def _reduce_feature_space_for_auto_tuning(data, max_features: int) -> Any:
    """
    Ограничивает число признаков для AUTO-подбора при очень высокой размерности.
    """
    if data is None:
        return data
    try:
        max_features = max(2, int(max_features))
    except (TypeError, ValueError):
        max_features = AUTO_TUNING_MAX_FEATURES

    data_np = np.asarray(data, dtype=np.float64)
    if data_np.ndim != 2:
        return data
    n_rows = int(data_np.shape[0])
    n_features = int(data_np.shape[1])
    if AUTO_TUNING_FEATURE_REDUCTION_MODE == "off":
        return data_np
    matrix_bytes = int(max(1, n_rows) * max(1, n_features) * 4)
    if AUTO_TUNING_FEATURE_REDUCTION_MODE == "auto":
        if (
                n_features <= int(AUTO_TUNING_REDUCE_FEATURES_THRESHOLD)
                and matrix_bytes <= int(AUTO_TUNING_MAX_WORKING_SET_BYTES)
        ):
            return data_np
        # Оценка безопасной целевой размерности с учетом временных копий.
        approx_copies_factor = 3
        max_features_by_memory = int(
            AUTO_TUNING_MAX_WORKING_SET_BYTES // max(1, n_rows * 4 * approx_copies_factor)
        )
        max_features = max(32, min(int(max_features), int(max_features_by_memory)))
    if n_features <= int(max_features):
        return data_np

    if AUTO_TUNING_FEATURE_REDUCTION_MODE in {"random_projection", "auto"}:
        projector = SparseRandomProjection(
            n_components=max_features,
            dense_output=True,
            random_state=42
        )
        reduced = projector.fit_transform(data_np)
        return np.asarray(reduced, dtype=np.float64)

    feature_idx = np.linspace(0, n_features - 1, num=max_features, dtype=int)
    return np.asarray(data_np[:, feature_idx], dtype=np.float64)


def _serialize_cluster_dataset(data: list[list[Any]]) -> str:
    """
    Сериализует набор данных кластера в компактный JSON,
    а для крупных наборов — в gzip+base64 строку.
    """
    json_payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    encoded_size = len(json_payload.encode("utf-8"))

    # Сжимаем только крупные payload, чтобы снизить размер записи в БД.
    if encoded_size < 10 * 1024 * 1024:
        return json_payload

    compressed = gzip.compress(json_payload.encode("utf-8"), compresslevel=6)
    b64_payload = base64.b64encode(compressed).decode("ascii")
    return f"{CLUSTER_DATA_GZIP_PREFIX}{b64_payload}"


def _deserialize_cluster_dataset(raw_data: str) -> list[list[Any]]:
    """
    Десериализует набор данных кластера из JSON или gzip+base64 формата.
    """
    if not raw_data:
        return []

    if raw_data.startswith(CLUSTER_DATA_GZIP_PREFIX):
        compressed_b64 = raw_data[len(CLUSTER_DATA_GZIP_PREFIX):]
        decompressed = gzip.decompress(base64.b64decode(compressed_b64.encode("ascii")))
        return json.loads(decompressed.decode("utf-8"))

    return json.loads(raw_data)


CLUSTER_MAP_INTERP_RESOLUTION_MIN = 50
CLUSTER_MAP_INTERP_RESOLUTION_MAX = 500
CLUSTER_MAP_INTERP_RESOLUTION_DEFAULT = 200


def _normalize_interpolation_resolution(value: Any) -> int:
    """
    Нормализует разрешение сетки для интерполяции карты кластеров.
    """
    try:
        resolution = int(value)
    except (TypeError, ValueError):
        return CLUSTER_MAP_INTERP_RESOLUTION_DEFAULT
    if resolution < CLUSTER_MAP_INTERP_RESOLUTION_MIN:
        return CLUSTER_MAP_INTERP_RESOLUTION_MIN
    if resolution > CLUSTER_MAP_INTERP_RESOLUTION_MAX:
        return CLUSTER_MAP_INTERP_RESOLUTION_MAX
    return resolution


def _get_ui_control_by_names(*control_names: str) -> Any:
    """
    Возвращает первый найденный UI-контрол по списку имен.
    """
    for name in control_names:
        control = getattr(ui, name, None)
        if control is not None:
            return control
    return None


def _normalize_smoothing_window(window: int) -> int:
    """
    Приводит окно сглаживания к валидному нечетному значению.
    """
    try:
        normalized = int(window)
    except (TypeError, ValueError):
        return 0
    if normalized <= 0:
        return 0
    if normalized < 3:
        normalized = 3
    if normalized % 2 == 0:
        normalized += 1
    return normalized


def _smooth_label_sequence(
        labels: list[int],
        *,
        method: str = "maj",
        window: int = 5,
        preserve_noise: bool = True
) -> list[int]:
    """
    Сглаживает последовательность меток кластера в 1D окне.

    method:
      - maj: мажоритарное голосование (mode)
      - med: медиана по окну
    """
    norm_window = _normalize_smoothing_window(window)
    if norm_window <= 0 or len(labels) <= 1:
        return list(labels)

    radius = norm_window // 2
    method_norm = (method or "maj").strip().lower()
    smoothed = list(labels)

    for idx, center_label in enumerate(labels):
        if preserve_noise and int(center_label) == -1:
            continue

        left = max(0, idx - radius)
        right = min(len(labels), idx + radius + 1)
        neighborhood = [int(v) for v in labels[left:right]]
        if not neighborhood:
            continue

        if preserve_noise:
            neighborhood_wo_noise = [v for v in neighborhood if v != -1]
            if neighborhood_wo_noise:
                neighborhood = neighborhood_wo_noise

        if method_norm == "med":
            target_label = int(np.median(np.array(neighborhood, dtype=float)))
        else:
            target_label = int(Counter(neighborhood).most_common(1)[0][0])
        smoothed[idx] = target_label

    return smoothed


def _smooth_labels_by_profile_trace(
        labels: list[int],
        profile_trace_rows: dict[int, dict[int, int]],
        *,
        method: str = "maj",
        window: int = 5,
        preserve_noise: bool = True
) -> tuple[list[int], int]:
    """
    Сглаживает метки отдельно по каждому профилю и только внутри непрерывных
    сегментов trace_index (без «перетекания» через разрывы).
    """
    smoothed_labels = list(labels)
    changes = 0

    for trace_rows in profile_trace_rows.values():
        if not trace_rows:
            continue
        sorted_trace_row = sorted((int(trace), int(row_idx)) for trace, row_idx in trace_rows.items())
        segment: list[tuple[int, int]] = []

        def flush_segment(items: list[tuple[int, int]]) -> None:
            nonlocal changes
            if not items:
                return
            rows = [row_idx for _, row_idx in items]
            original = [int(labels[row_idx]) for row_idx in rows]
            smoothed_segment = _smooth_label_sequence(
                original,
                method=method,
                window=window,
                preserve_noise=preserve_noise
            )
            for row_idx, old_label, new_label in zip(rows, original, smoothed_segment):
                smoothed_labels[row_idx] = int(new_label)
                if int(old_label) != int(new_label):
                    changes += 1

        for trace_idx, row_idx in sorted_trace_row:
            if not segment:
                segment = [(trace_idx, row_idx)]
                continue
            prev_trace_idx = segment[-1][0]
            if trace_idx == prev_trace_idx + 1:
                segment.append((trace_idx, row_idx))
            else:
                flush_segment(segment)
                segment = [(trace_idx, row_idx)]
        flush_segment(segment)

    return smoothed_labels, changes


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
    ranked_coarse = rank_candidates(coarse_results, weights=weights)
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
        hdbscan_metrics=hdbscan_metrics
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

    combined_ranked = rank_candidates(coarse_results + fine_results, weights=weights)
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
        "fine_results": rank_candidates(fine_results, weights=weights),
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


def render_auto_results_table(results: list[CandidateResult]) -> None:
    """
    Заполняет таблицу результатов AUTO-подбора.
    """
    global cluster_auto_results_cache
    cluster_auto_results_cache = list(results or [])

    table = ui.tableWidget_cluster_auto_result
    headers = [
        "Rank", "Score", "Clusters", "Method", "Scaler", "PCA",
        "PCA comps", "Silhouette", "DB", "CH", "Noise %", "PartHash", "Status"
    ]
    table.clear()
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setRowCount(len(results))

    for row_idx, result in enumerate(results):
        cfg = result.get("candidate_config", {})
        metrics = result.get("metrics", {})
        stats = result.get("stats", {})
        score_val = result.get("score")
        noise = _to_finite_float(stats.get("noise_fraction"))

        status_raw = str(result.get("status", "—"))
        status_view = {
            "ok": "OK",
            "invalid": "INVALID",
            "error": "ERROR"
        }.get(status_raw, status_raw.upper() if status_raw else "—")

        row_values = [
            str(row_idx + 1),
            _safe_num(score_val, precision=4),
            str(stats.get("n_clusters", "—")),
            _candidate_method_short(cfg),
            str(cfg.get("scaler_mode", "—")),
            _candidate_pca_short(cfg),
            str(stats.get("pca_components_after", "—")),
            _safe_num(metrics.get("silhouette"), precision=4),
            _safe_num(metrics.get("davies_bouldin"), precision=4),
            _safe_num(metrics.get("calinski_harabasz"), precision=2),
            (f"{(noise * 100.0):.1f}%" if noise is not None else "—"),
            str(stats.get("partition_hash", "—")),
            status_view
        ]

        for col_idx, value in enumerate(row_values):
            item = QTableWidgetItem(str(value))
            if col_idx in (0, 1, 2, 6, 7, 8, 9, 10, 11, 12):
                item.setTextAlignment(Qt.AlignCenter)
            if col_idx == 12:
                if status_raw == "ok":
                    item.setForeground(QBrush(QColor("darkgreen")))
                elif status_raw == "invalid":
                    item.setForeground(QBrush(QColor("darkorange")))
                else:
                    item.setForeground(QBrush(QColor("darkred")))
            item.setToolTip(
                json.dumps(
                    {
                        "candidate_id": result.get("candidate_id"),
                        "candidate_config": cfg,
                        "error_text": result.get("error_text", "")
                    },
                    ensure_ascii=False
                )
            )
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
    has_cached_results = len(cluster_auto_results_cache) > 0
    has_table_results = table.rowCount() > 0

    if not (has_cached_results or has_table_results):
        set_info("AUTO: таблица результатов уже пустая.", "brown")
        return

    reply = QMessageBox.warning(
        MainWindow,
        "Очистка результатов AUTO",
        "Вы действительно хотите очистить результаты автоподбора?\nЭто действие удалит строки из таблицы и кэш текущей сессии.",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
    )

    if reply != QMessageBox.Yes:
        set_info("AUTO: очистка результатов отменена пользователем.", "brown")
        return

    cluster_auto_results_cache = []
    table.clear()
    table.setRowCount(0)
    table.setColumnCount(0)

    clust_object_id = get_curr_clust_object_id()
    if clust_object_id:
        try:
            (
                session.query(ClusterAutoTuningCache)
                .filter_by(object_set_id=int(clust_object_id))
                .delete(synchronize_session=False)
            )
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
    clust_object_id = get_curr_clust_object_id()
    if clust_object_id is None:
        set_info("AUTO: не выбран объект для кластеризации.", "brown")
        return

    clust_object = session.query(ObjectSet).filter_by(id=clust_object_id).first()
    if clust_object is None:
        set_info(f"AUTO: объект id={clust_object_id} не найден.", "brown")
        return

    try:
        base_data = _deserialize_cluster_dataset(clust_object.data)
    except Exception as exc:
        set_info(f"AUTO: ошибка чтения данных объекта: {exc}", "red")
        return

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
        clust_object_id=int(clust_object_id),
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
        }
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

    if not force_recompute:
        cached_top_results = load_cluster_auto_tuning_cache(
            cache_key=cache_key,
            clust_object_id=int(clust_object_id),
            top_k=top_k
        )
        if cached_top_results:
            render_auto_results_table(cached_top_results)
            best_result = cached_top_results[0]
            if auto_apply_best:
                apply_auto_result_to_ui(best_result)
            set_info(
                f"AUTO {auto_mode}: использованы сохраненные top-{len(cached_top_results)} настройки.",
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
        run_key=cache_key,
        object_set_id=int(clust_object_id),
        clean_kwargs={
            "use_non_finite": ui.checkBox_clust_clean_nan.isChecked(),
            "non_finite_mode": text_method_nan,
            "use_variance_threshold": ui.checkBox_clust_clear_vartresh.isChecked(),
            "use_correlation_filter": ui.checkBox_clust_clear_corr.isChecked()
        }
    )

    top_results = tuning_result.get("top_results", [])
    save_cluster_auto_tuning_cache(
        cache_key=cache_key,
        clust_object_id=int(clust_object_id),
        top_results=top_results,
        top_k=top_k
    )
    render_auto_results_table(top_results)
    best_result = tuning_result.get("best_result")
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
    Запускает AUTO-подбор последовательно для всех ObjectSet в текущем анализе.
    В batch-режиме:
    - min_cluster_samples принудительно = 5% от N для каждого объекта;
    - apply_auto_best отключен;
    - если retune выключен и есть cache, объект пропускается.
    """
    clust_analys_id = get_curr_clust_analys_id()
    if not str(clust_analys_id).isdigit():
        set_info("AUTO BATCH: не выбран набор кластерного анализа.", "brown")
        return

    clust_objects = (
        session.query(ObjectSet.id, ObjectSet.research_id, ObjectSet.data)
        .filter_by(analysis_id=int(clust_analys_id))
        .order_by(ObjectSet.id.asc())
        .all()
    )
    if not clust_objects:
        set_info("AUTO BATCH: нет добавленных наборов объектов для обработки.", "brown")
        return

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

    total_objects = len(clust_objects)
    started_at = monotonic()
    skipped_cached = 0
    calculated = 0
    failed = 0

    set_info(
        f"AUTO BATCH {auto_mode}: старт по {total_objects} объектам | retune={force_recompute} | apply_auto_best=OFF.",
        "blue"
    )

    for idx, clust_obj in enumerate(clust_objects, start=1):
        object_id = int(clust_obj.id)
        object_name = f"object_set_id={object_id}"
        try:
            base_data = _deserialize_cluster_dataset(clust_obj.data)
        except Exception as exc:
            failed += 1
            set_info(f"AUTO BATCH {auto_mode} [{idx}/{total_objects}] {object_name}: FAILED (чтение данных: {exc}).", "red")
            continue

        if not base_data:
            failed += 1
            set_info(f"AUTO BATCH {auto_mode} [{idx}/{total_objects}] {object_name}: FAILED (пустой набор данных).", "brown")
            continue

        auto_random_seed = random.SystemRandom().randrange(1, 2_147_483_647)
        original_rows_count = len(base_data)
        sample_idx = _build_sample_indices_for_auto_tuning(original_rows_count, AUTO_TUNING_MAX_ROWS, seed=auto_random_seed)
        base_data = _apply_sample_indices(base_data, sample_idx)
        sampled_rows_count = len(base_data) if base_data is not None else 0
        if sampled_rows_count < original_rows_count:
            set_info(
                f"AUTO BATCH {auto_mode} [{idx}/{total_objects}] {object_name}: "
                f"использована подвыборка {sampled_rows_count}/{original_rows_count} строк.",
                "brown"
            )

        sample_limits = calculate_auto_min_cluster_sample_limits(len(base_data), min_value=1)
        min_cluster_samples = int(sample_limits["recommended_default_value"])
        cache_key = build_cluster_auto_tuning_cache_key(
            clust_object_id=object_id,
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
            }
        )

        if not force_recompute:
            cached_top_results = load_cluster_auto_tuning_cache(
                cache_key=cache_key,
                clust_object_id=object_id,
                top_k=top_k
            )
            if cached_top_results:
                skipped_cached += 1
                best_cached = cached_top_results[0] if cached_top_results else {}
                set_info(
                    f"AUTO BATCH {auto_mode} [{idx}/{total_objects}] {object_name}: "
                    f"SKIPPED_CACHED top={len(cached_top_results)} score={_safe_num(best_cached.get('score'), 4)} "
                    f"min_cluster_samples={min_cluster_samples}.",
                    "green"
                )
                continue

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
            run_key=cache_key,
            object_set_id=int(object_id),
            clean_kwargs={
                "use_non_finite": ui.checkBox_clust_clean_nan.isChecked(),
                "non_finite_mode": text_method_nan,
                "use_variance_threshold": ui.checkBox_clust_clear_vartresh.isChecked(),
                "use_correlation_filter": ui.checkBox_clust_clear_corr.isChecked()
            }
        )
        top_results = tuning_result.get("top_results", [])
        save_cluster_auto_tuning_cache(
            cache_key=cache_key,
            clust_object_id=object_id,
            top_results=top_results,
            top_k=top_k
        )

        best_result = tuning_result.get("best_result")
        if not best_result or best_result.get("status") != "ok":
            failed += 1
            set_info(
                f"AUTO BATCH {auto_mode} [{idx}/{total_objects}] {object_name}: FAILED "
                f"(нет валидных конфигураций) min_cluster_samples={min_cluster_samples}.",
                "brown"
            )
            continue

        calculated += 1
        best_metrics = best_result.get("metrics", {})
        best_cfg = best_result.get("candidate_config", {})
        set_info(
            f"AUTO BATCH {auto_mode} [{idx}/{total_objects}] {object_name}: CALCULATED "
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
        f"AUTO BATCH {auto_mode}: завершено за {elapsed_sec:.1f} сек | total={total_objects}, "
        f"calculated={calculated}, skipped_cached={skipped_cached}, failed={failed}.",
        "green" if failed == 0 else "brown"
    )


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
    row.updated_at = datetime.datetime.utcnow()
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

        existing_row.created_at = datetime.datetime.utcnow()
        existing_row.top_results = json.dumps(compact_top_results, ensure_ascii=False)
        session.commit()
    except Exception as exc:
        set_info(f"AUTO: ошибка сохранения cache top-5: {exc}", "brown")


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


def switch_cluster_profile(step: int):
    """
    Переключает профиль в comboBox_profile по кругу и перерисовывает:
    - радарограмму (draw_radarogram)
    - кластерную заливку из runtime-cache
    """
    profile_count = ui.comboBox_profile.count()
    if profile_count <= 0:
        set_info("Список профилей пуст. Переключение недоступно.", "brown")
        return

    current_index = ui.comboBox_profile.currentIndex()
    if current_index < 0:
        current_index = 0

    next_index = (current_index + step) % profile_count
    ui.comboBox_profile.setCurrentIndex(next_index)

    draw_radarogram()
    redraw_cluster_for_current_profile_from_cache()


def draw_prev_cluster_profile():
    """
    Отрисовывает предыдущий профиль (по кругу) с результатом кластеризации.
    """
    switch_cluster_profile(-1)


def draw_next_cluster_profile():
    """
    Отрисовывает следующий профиль (по кругу) с результатом кластеризации.
    """
    switch_cluster_profile(1)


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
    clust_object_id = get_curr_clust_object_id()
    if not clust_object_id:
        render_auto_results_table([])
        return

    try:
        cache_row = (
            session.query(ClusterAutoTuningCache)
            .filter_by(object_set_id=int(clust_object_id))
            .order_by(ClusterAutoTuningCache.created_at.desc())
            .first()
        )
    except Exception as exc:
        render_auto_results_table([])
        set_info(f"AUTO: ошибка чтения сохраненного результата: {exc}", "brown")
        return

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
        interpolation_resolution=200,
        title="Cluster map",
        noise_color="gray",
        noise_marker=".",
        noise_label="noise",
        legend=True,
        show_interpolation=True,
        settings_caption: str | None = None
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

    interpolation_resolution : int
        Разрешение регулярной сетки для фоновой интерполяции.
        Чем больше значение, тем более детальная (и более тяжелая по времени)
        отрисовка. Рабочий диапазон: 50..500.

    noise_color : str
        Цвет шума (label = -1).

    noise_marker : str
        Маркер шума.

    legend : bool
        Показывать легенду.

    show_interpolation : bool
        Если True — строится фоновая интерполяция по всей области участка.
        Если False — рисуются только исходные точки.

    settings_caption : str | None
        Дополнительная подпись с параметрами расчета (scaler/PCA/метод и т.д.).
    """

    labels = np.asarray(label_list)
    arr = np.asarray(data, dtype=float)

    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("data must have at least 3 columns")

    if len(labels) != len(arr):
        raise ValueError("label_list length must match data")

    x = arr[:, 1]
    y = arr[:, 2]

    # Явно создаем новую фигуру/ось на каждый вызов, чтобы исключить
    # повторное использование текущей активной оси matplotlib.
    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(labels)
    unique_labels_sorted = sorted([int(v) for v in unique_labels])

    if show_interpolation and len(unique_labels_sorted) > 1 and len(x) >= 3:
        min_x, max_x = float(np.min(x)), float(np.max(x))
        min_y, max_y = float(np.min(y)), float(np.max(y))
        span_x = max(max_x - min_x, 1e-9)
        span_y = max(max_y - min_y, 1e-9)
        pad_x = span_x * 0.03
        pad_y = span_y * 0.03
        min_x -= pad_x
        max_x += pad_x
        min_y -= pad_y
        max_y += pad_y

        grid_n = _normalize_interpolation_resolution(interpolation_resolution)
        grid_x, grid_y = np.meshgrid(
            np.linspace(min_x, max_x, grid_n),
            np.linspace(min_y, max_y, grid_n)
        )

        points = np.column_stack((x, y))
        grid_z = griddata(points, labels.astype(float), (grid_x, grid_y), method="nearest")

        if grid_z is not None and np.any(~np.isnan(grid_z)):
            color_list = [get_cluster_color(label) for label in unique_labels_sorted]
            cmap = ListedColormap(color_list)
            boundaries = [unique_labels_sorted[0] - 0.5]
            boundaries.extend([label + 0.5 for label in unique_labels_sorted])
            norm = BoundaryNorm(boundaries, cmap.N, clip=True)

            ax.pcolormesh(
                grid_x,
                grid_y,
                grid_z,
                shading="auto",
                cmap=cmap,
                norm=norm,
                alpha=0.35,
                zorder=1
            )

    # кластеры
    for label in unique_labels:
        if label == -1:
            continue

        mask = labels == label
        cluster_color = get_cluster_color(label)

        ax.scatter(
            x[mask],
            y[mask],
            s=point_size,
            label=f"cluster {int(label)}",
            c=cluster_color,
            edgecolors=cluster_color,
            alpha=0.95,
            zorder=3
        )

    # шум
    if -1 in unique_labels:
        mask = labels == -1
        noise_cluster_color = get_cluster_color(-1)
        ax.scatter(
            x[mask],
            y[mask],
            s=point_size,
            c=noise_cluster_color if noise_color == "gray" else noise_color,
            marker=noise_marker,
            label=noise_label,
            edgecolors=noise_cluster_color if noise_color == "gray" else noise_color,
            alpha=0.9,
            zorder=4
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    if legend:
        ax.legend()

    if settings_caption:
        fig.text(
            0.01,
            0.01,
            settings_caption,
            ha="left",
            va="bottom",
            fontsize=8,
            color="dimgray"
        )
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    else:
        fig.tight_layout()
    plt.show()


def show_cluster_diagnostics(
        data_for_clustering,
        labels,
        method_name: str,
        model=None
):
    """
    Показывает набор диагностических графиков на одном листе:
    PCA 2D/3D, t-SNE 2D/3D, матрица расстояний, silhouette.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import pairwise_distances, silhouette_samples
    from sklearn.manifold import TSNE

    X = np.asarray(data_for_clustering, dtype=float)
    y = np.asarray(labels, dtype=int)
    if len(X) == 0 or len(X) != len(y):
        return

    if min(X.shape[1], len(X)) < 2:
        return
    uniq_lbl = [v for v in sorted(np.unique(y)) if v != -1]
    centroids = np.array([X[y == lbl].mean(axis=0) for lbl in uniq_lbl]) if uniq_lbl else np.empty((0, X.shape[1]))
    has_centroids = len(centroids) > 0

    pca_2d_model = PCA(n_components=2).fit(X)
    pca_2d = pca_2d_model.transform(X)
    cent_pca_2d = pca_2d_model.transform(centroids) if has_centroids else np.empty((0, 2))

    pca_3d = None
    cent_pca_3d = None
    if min(3, X.shape[1], len(X)) >= 3:
        pca_3d_model = PCA(n_components=3).fit(X)
        pca_3d = pca_3d_model.transform(X)
        cent_pca_3d = pca_3d_model.transform(centroids) if has_centroids else np.empty((0, 3))

    perplexity = max(5, min(30, len(X) - 1))
    tsne_2d_model = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=perplexity)
    if has_centroids:
        comb2 = np.vstack([X, centroids])
        emb2 = tsne_2d_model.fit_transform(comb2)
        tsne_2d, cent_tsne_2d = emb2[:len(X)], emb2[len(X):]
    else:
        tsne_2d = tsne_2d_model.fit_transform(X)
        cent_tsne_2d = np.empty((0, 2))

    tsne_3d = None
    cent_tsne_3d = None
    if min(3, X.shape[1], len(X)) >= 3:
        tsne_3d_model = TSNE(n_components=3, random_state=42, init="pca", learning_rate="auto", perplexity=perplexity)
        if has_centroids:
            comb3 = np.vstack([X, centroids])
            emb3 = tsne_3d_model.fit_transform(comb3)
            tsne_3d, cent_tsne_3d = emb3[:len(X)], emb3[len(X):]
        else:
            tsne_3d = tsne_3d_model.fit_transform(X)
            cent_tsne_3d = np.empty((0, 3))

    fig = plt.figure(figsize=(20, 12))
    # Порядок графиков:
    # [1] PCA 2D, [2] t-SNE 2D, [3] Silhouette
    # [4] PCA 3D, [5] t-SNE 3D, [6] Distance matrix
    ax_pca2 = fig.add_subplot(2, 3, 1)
    ax_tsne2 = fig.add_subplot(2, 3, 2)
    ax_sil = fig.add_subplot(2, 3, 3)
    ax_pca3 = fig.add_subplot(2, 3, 4, projection="3d")
    ax_tsne3 = fig.add_subplot(2, 3, 5, projection="3d")
    ax_dm = fig.add_subplot(2, 3, 6)

    def _plot_2d(ax, pts, title, centroids_2d=None):
        for lbl in sorted(np.unique(y)):
            m = y == lbl
            color = "gray" if lbl == -1 else get_cluster_color(int(lbl))
            ax.scatter(pts[m, 0], pts[m, 1], s=14, c=color, alpha=0.8, label=f"cluster {lbl}")
        if centroids_2d is not None and len(centroids_2d) > 0:
            ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], marker="X", c="black", s=90, label="centroids")
        ax.set_title(title)
        ax.grid(alpha=0.2)

    _plot_2d(ax_pca2, pca_2d, f"{method_name.upper()} • PCA 2D", cent_pca_2d)
    ax_pca2.set_xlabel("PC1")
    ax_pca2.set_ylabel("PC2")
    ax_pca2.legend(loc="best", fontsize=7)

    if pca_3d is not None:
        for lbl in sorted(np.unique(y)):
            m = y == lbl
            color = "gray" if lbl == -1 else get_cluster_color(int(lbl))
            ax_pca3.scatter(pca_3d[m, 0], pca_3d[m, 1], pca_3d[m, 2], s=12, c=color, alpha=0.75)
        if cent_pca_3d is not None and len(cent_pca_3d) > 0:
            ax_pca3.scatter(cent_pca_3d[:, 0], cent_pca_3d[:, 1], cent_pca_3d[:, 2], marker="X", c="black", s=90)
        ax_pca3.set_title("PCA 3D")
    else:
        ax_pca3.set_title("PCA 3D unavailable")

    _plot_2d(ax_tsne2, tsne_2d, "t-SNE 2D", cent_tsne_2d)
    ax_tsne2.legend(loc="best", fontsize=7)

    if tsne_3d is not None:
        for lbl in sorted(np.unique(y)):
            m = y == lbl
            color = "gray" if lbl == -1 else get_cluster_color(int(lbl))
            ax_tsne3.scatter(tsne_3d[m, 0], tsne_3d[m, 1], tsne_3d[m, 2], s=12, c=color, alpha=0.75)
        if cent_tsne_3d is not None and len(cent_tsne_3d) > 0:
            ax_tsne3.scatter(cent_tsne_3d[:, 0], cent_tsne_3d[:, 1], cent_tsne_3d[:, 2], marker="X", c="black", s=90)
        ax_tsne3.set_title("t-SNE 3D")
    else:
        ax_tsne3.set_title("t-SNE 3D unavailable")

    order = np.argsort(y)
    dist_mx = pairwise_distances(X[order], metric="euclidean")
    im = ax_dm.imshow(dist_mx, cmap="turbo", aspect="auto")
    ax_dm.set_title("Distance matrix (ordered)")
    fig.colorbar(im, ax=ax_dm, shrink=0.8)

    mask_valid = y != -1
    y_eval = y[mask_valid]
    X_eval = X[mask_valid]
    if len(X_eval) > 2 and len(np.unique(y_eval)) > 1:
        sil_values = silhouette_samples(X_eval, y_eval)
        y_lower = 10
        for lbl in sorted(np.unique(y_eval)):
            vals = np.sort(sil_values[y_eval == lbl])
            y_upper = y_lower + len(vals)
            ax_sil.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.7, color=get_cluster_color(int(lbl)))
            y_lower = y_upper + 10
        sil_avg = float(np.mean(sil_values))
        ax_sil.axvline(sil_avg, color="red", linestyle="--", linewidth=1.2, label=f"avg={sil_avg:.3f}")
        ax_sil.set_title("Silhouette plot")
        ax_sil.legend(loc="best", fontsize=8)
    else:
        ax_sil.set_title("Silhouette unavailable")

    fig.suptitle(f"Cluster diagnostics • {method_name.upper()}", fontsize=14)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    plt.show()

def evaluate_clustering(
        data,
        labels,
        use_silhouette=False,
        use_db=False,
        use_ch=False,
        max_silhouette_samples: int = AUTO_SILHOUETTE_MAX_SAMPLES
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
        silhouette_sample_size = len(X_eval)
        if max_silhouette_samples is not None:
            try:
                max_silhouette_samples = max(2, int(max_silhouette_samples))
            except (TypeError, ValueError):
                max_silhouette_samples = AUTO_SILHOUETTE_MAX_SAMPLES
            silhouette_sample_size = min(silhouette_sample_size, int(max_silhouette_samples))
        val = float(
            silhouette_score(
                X_eval,
                labels_eval,
                sample_size=silhouette_sample_size if silhouette_sample_size < len(X_eval) else None,
                random_state=42
            )
        )
        results["metrics"]["silhouette"] = val
        results["metrics"]["silhouette_n_samples"] = int(silhouette_sample_size)

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

    smoothing_desc = str(cluster_info.get("smoothing", "off"))
    settings.append(f"Smoothing: {smoothing_desc}")

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
    data = _deserialize_cluster_dataset(clust_object.data)
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
