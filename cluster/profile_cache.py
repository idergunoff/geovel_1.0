from __future__ import annotations

from .common import *

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


