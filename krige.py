from cProfile import label
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy.exc import OperationalError
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from scipy import ndimage
from scipy.spatial import ConvexHull, Delaunay, cKDTree

from func import *
from kriging_utils import (
    fill_kriging_gaps,
    inverse_distance_interpolation,
    prepare_variogram_data,
    savgol_parameters,
)
from qt.choose_formation_map import *
from qt.draw_map_form import *
from report_map import (
    annotate_observations, build_categorical_report_map, build_report_map,
    parse_contour_interval, parse_scale_bounds,
)
from well_map_style import well_marker_color

cmap_list = ['viridis', 'plasma', 'inferno', 'magma', 'cividis','Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
             'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr',
             'seismic', 'twilight', 'twilight_shifted', 'hsv', 'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
             'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c', 'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
             'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
             'turbo', 'nipy_spectral', 'gist_ncar']


dist_func_list = ['euclidean', 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'ice',
                  'hamming', 'jaccard', 'jensenshannon', 'kulczynski1', 'mahalanobis', 'matching', 'minkowski',
                  'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']


def convex_hull_boundary(points):
    hull = ConvexHull(points)
    return hull


def _build_categorical_palette(labels):
    """Build cluster colors consistent with radarogram/diagnostics palette."""
    unique_labels = sorted({int(v) for v in labels})
    base_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173"
    ]
    color_by_label = {}
    for lbl in unique_labels:
        if lbl == -1:
            color_by_label[lbl] = "#808080"
            continue
        color_by_label[lbl] = base_colors[int(lbl) % len(base_colors)]
    return color_by_label


def _inside_hull_mask(xx, yy, points):
    """Build a boolean mask for grid cells inside the data coverage area.

    Stage 2.2 baseline uses convex hull clipping. If hull construction fails
    (e.g. degenerate geometry), keep previous behaviour and do not clip.
    """
    if len(points) < 3:
        return np.ones_like(xx, dtype=bool)

    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        triangulation = Delaunay(hull_points)
        flat_points = np.column_stack([xx.ravel(), yy.ravel()])
        inside_flat = triangulation.find_simplex(flat_points) >= 0
        return inside_flat.reshape(xx.shape)
    except Exception:
        return np.ones_like(xx, dtype=bool)


def _categorical_probability_grid(points, labels, xx, yy, smooth_power=2.0, neighbors_count=12):
    """Probability-style categorical interpolation.

    For each grid point compute class scores via distance weighted voting:
    score_k = sum(1 / d_i^smooth_power) for points of class k.
    Then normalize scores into probabilities and return argmax labels.
    """
    flat_grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    unique_labels = sorted({int(v) for v in labels})
    label_to_index = {lbl: idx for idx, lbl in enumerate(unique_labels)}

    if len(points) == 0 or len(unique_labels) == 0:
        return (
            np.zeros(xx.shape, dtype=int),
            np.zeros(xx.shape + (0,), dtype=float),
            unique_labels
        )

    safe_neighbors = max(1, min(int(neighbors_count), len(points)))
    tree = cKDTree(points)
    distances, neighbor_idx = tree.query(flat_grid_points, k=safe_neighbors)

    if safe_neighbors == 1:
        distances = distances[:, np.newaxis]
        neighbor_idx = neighbor_idx[:, np.newaxis]

    eps = 1e-9
    weights = 1.0 / np.power(np.maximum(distances, eps), max(float(smooth_power), 1e-6))
    neighbor_labels = labels[neighbor_idx]

    scores = np.zeros((flat_grid_points.shape[0], len(unique_labels)), dtype=float)
    for lbl in unique_labels:
        cls_idx = label_to_index[lbl]
        scores[:, cls_idx] = np.sum(weights * (neighbor_labels == lbl), axis=1)

    score_sums = np.sum(scores, axis=1, keepdims=True)
    score_sums[score_sums == 0.0] = 1.0
    probabilities = scores / score_sums
    predicted_class_indices = np.argmax(probabilities, axis=1)
    predicted_labels = np.array(unique_labels, dtype=int)[predicted_class_indices].reshape(xx.shape)
    prob_cube = probabilities.reshape(xx.shape + (len(unique_labels),))
    return predicted_labels, prob_cube, unique_labels


def _calculate_uncertainty_from_probabilities(prob_cube):
    """Margin-based uncertainty for categorical probability output.

    uncertainty = 1 - (p_top1 - p_top2)
    Where top1/top2 are two largest class probabilities for each cell.
    """
    if prob_cube is None or prob_cube.size == 0:
        return None

    if prob_cube.shape[-1] == 1:
        return np.zeros(prob_cube.shape[:2], dtype=float)

    sorted_probabilities = np.sort(prob_cube, axis=-1)
    p_top1 = sorted_probabilities[..., -1]
    p_top2 = sorted_probabilities[..., -2]
    uncertainty = 1.0 - (p_top1 - p_top2)
    return np.clip(uncertainty, 0.0, 1.0)


def _smooth_categorical_labels(class_index_grid, level):
    """Boundary smoothing post-process for categorical class indices."""
    if class_index_grid is None:
        return class_index_grid

    level_norm = str(level).strip().lower()
    iterations_by_level = {
        "off": 0,
        "low": 1,
        "medium": 2,
        "high": 3
    }
    iterations = iterations_by_level.get(level_norm, 0)
    if iterations <= 0:
        return class_index_grid

    is_masked = np.ma.isMaskedArray(class_index_grid)
    if is_masked:
        mask = np.ma.getmaskarray(class_index_grid)
        work = np.asarray(class_index_grid.filled(0), dtype=int)
    else:
        mask = np.zeros(class_index_grid.shape, dtype=bool)
        work = np.asarray(class_index_grid, dtype=int)

    structure = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ],
        dtype=bool
    )
    valid_region = ~mask
    unique_classes = np.unique(work[valid_region]) if np.any(valid_region) else np.array([], dtype=int)
    if unique_classes.size <= 1:
        return class_index_grid

    smoothed = work.copy()
    for _ in range(iterations):
        class_vote_cube = np.zeros(smoothed.shape + (len(unique_classes),), dtype=np.int32)
        for cls_pos, cls_value in enumerate(unique_classes):
            cls_mask = np.logical_and(smoothed == cls_value, valid_region).astype(np.int32)
            class_vote_cube[..., cls_pos] = ndimage.convolve(cls_mask, structure.astype(np.int32), mode="nearest")
        best_class_positions = np.argmax(class_vote_cube, axis=-1)
        smoothed = unique_classes[best_class_positions].astype(int)
        smoothed[mask] = work[mask]

    if is_masked:
        return np.ma.masked_where(mask, smoothed)
    return smoothed


def _enforce_source_point_labels(smoothed_grid, xx, yy, source_points, source_class_indices):
    """Keep local consistency with source points after smoothing."""
    if smoothed_grid is None or len(source_points) == 0:
        return smoothed_grid

    is_masked = np.ma.isMaskedArray(smoothed_grid)
    if is_masked:
        mask = np.ma.getmaskarray(smoothed_grid)
        work = np.asarray(smoothed_grid.filled(0), dtype=int)
    else:
        mask = np.zeros(smoothed_grid.shape, dtype=bool)
        work = np.asarray(smoothed_grid, dtype=int)

    grid_coords = np.column_stack([xx.ravel(), yy.ravel()])
    tree = cKDTree(grid_coords)
    _, nearest_grid_idx = tree.query(source_points, k=1)

    for idx, class_idx in zip(nearest_grid_idx, source_class_indices):
        gx, gy = np.unravel_index(int(idx), work.shape)
        if not mask[gx, gy]:
            work[gx, gy] = int(class_idx)

    if is_masked:
        return np.ma.masked_where(mask, work)
    return work

def show_map():
    r_id = get_research_id()
    r = session.query(Research).filter_by(id=r_id).first()
    if r is None:
        set_info('Для построения карты выберите исследование.', 'red')
        return
    list_x, list_y, list_z = [], [], []
    param = ui.comboBox_param_plast.currentText()
    for profile in r.profiles:
        list_x += (json.loads(profile.x_pulc))
        list_y += (json.loads(profile.y_pulc))
        try:
            if len(profile.formations) == 1:
                if param in list_wavelet_features:
                    form = session.query(literal_column(f'wavelet_feature.{param}')).filter(
                        WaveletFeature.formation_id == profile.formations[0].id).first()
                elif param in list_fractal_features:
                    form = session.query(literal_column(f'fractal_feature.{param}')).filter(
                        FractalFeature.formation_id == profile.formations[0].id).first()
                elif param in list_entropy_features:
                    form = session.query(literal_column(f'entropy_feature.{param}')).filter(
                        EntropyFeature.formation_id == profile.formations[0].id).first()
                elif param in list_nonlinear_features:
                    form = session.query(literal_column(f'nonlinear_feature.{param}')).filter(
                        NonlinearFeature.formation_id == profile.formations[0].id).first()
                elif param in list_morphology_feature:
                    form = session.query(literal_column(f'morphology_feature.{param}')).filter(
                        MorphologyFeature.formation_id == profile.formations[0].id).first()
                elif param in list_frequency_feature:
                    form = session.query(literal_column(f'frequency_feature.{param}')).filter(
                        FrequencyFeature.formation_id == profile.formations[0].id).first()
                elif param in list_envelope_feature:
                    form = session.query(literal_column(f'envelope_feature.{param}')).filter(
                        EnvelopeFeature.formation_id == profile.formations[0].id).first()
                elif param in list_autocorr_feature:
                    form = session.query(literal_column(f'autocorr_feature.{param}')).filter(
                        AutocorrFeature.formation_id == profile.formations[0].id).first()
                elif param in list_emd_feature:
                    form = session.query(literal_column(f'emd_feature.{param}')).filter(
                        EMDFeature.formation_id == profile.formations[0].id).first()
                elif param in list_hht_feature:
                    form = session.query(literal_column(f'hht_feature.{param}')).filter(
                        HHTFeature.formation_id == profile.formations[0].id).first()
                else:
                    form = session.query(literal_column(f'Formation.{param}')).filter(Formation.id == profile.formations[0].id).first()

                list_z += (json.loads(form[0]))
            elif len(profile.formations) > 1:
                Choose_Formation = QtWidgets.QDialog()
                ui_cf = Ui_FormationMAP()
                ui_cf.setupUi(Choose_Formation)
                Choose_Formation.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия
                for f in profile.formations:
                    ui_cf.listWidget_form_map.addItem(f'{f.title} id{f.id}')
                ui_cf.listWidget_form_map.setCurrentRow(0)

                def form_lda_ok():
                    f_id = ui_cf.listWidget_form_map.currentItem().text().split(" id")[-1]
                    if param in list_wavelet_features:
                        form = session.query(literal_column(f'wavelet_feature.{param}')).filter(WaveletFeature.formation_id == f_id).first()
                    elif param in list_fractal_features:
                        form = session.query(literal_column(f'fractal_feature.{param}')).filter(FractalFeature.formation_id == f_id).first()
                    elif param in list_entropy_features:
                        form = session.query(literal_column(f'entropy_feature.{param}')).filter(EntropyFeature.formation_id == f_id).first()
                    elif param in list_nonlinear_features:
                        form = session.query(literal_column(f'nonlinear_feature.{param}')).filter(NonlinearFeature.formation_id == f_id).first()
                    elif param in list_morphology_feature:
                        form = session.query(literal_column(f'morphology_feature.{param}')).filter(MorphologyFeature.formation_id == f_id).first()
                    elif param in list_frequency_feature:
                        form = session.query(literal_column(f'frequency_feature.{param}')).filter(FrequencyFeature.formation_id == f_id).first()
                    elif param in list_envelope_feature:
                        form = session.query(literal_column(f'envelope_feature.{param}')).filter(EnvelopeFeature.formation_id == f_id).first()
                    elif param in list_autocorr_feature:
                        form = session.query(literal_column(f'autocorr_feature.{param}')).filter(AutocorrFeature.formation_id == f_id).first()
                    elif param in list_emd_feature:
                        form = session.query(literal_column(f'emd_feature.{param}')).filter(EMDFeature.formation_id == f_id).first()
                    elif param in list_hht_feature:
                        form = session.query(literal_column(f'hht_feature.{param}')).filter(HHTFeature.formation_id == f_id).first()
                    else:
                        form = session.query(literal_column(f'Formation.{param}')).filter(Formation.id == f_id).first()
                    list_z += (json.loads(form[0]))
                    Choose_Formation.accept()

                ui_cf.pushButton_ok_form_map.clicked.connect(form_lda_ok)
                if ask_in_window(Choose_Formation, MainWindow) != QtWidgets.QDialog.Accepted:
                    return
        except OperationalError:
            set_info('Не выбран пласт', 'red')
            return
    draw_map(list_x, list_y, list_z, param)


class _MapCalculationSignals(QtCore.QObject):
    progress = QtCore.pyqtSignal(str, str)
    completed = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)


class _MapCalculation(QtCore.QRunnable):
    def __init__(self, calculate):
        super().__init__()
        self.calculate = calculate
        self.signals = _MapCalculationSignals()

    def run(self):
        try:
            result = self.calculate(self.signals.progress.emit)
        except Exception as exc:
            self.signals.failed.emit(str(exc) or type(exc).__name__)
        else:
            self.signals.completed.emit(result)


class _DrawMapDialog(QtWidgets.QDialog):
    def start_calculation(self, button, calculate, render):
        if not button.isEnabled():
            return
        self.draw_button = button
        self.render = render
        button.setEnabled(False)
        job = _MapCalculation(calculate)
        job.signals.progress.connect(self.report)
        job.signals.completed.connect(self.show_result)
        job.signals.failed.connect(self.show_error)
        QtCore.QThreadPool.globalInstance().start(job)

    @QtCore.pyqtSlot(str, str)
    def report(self, message, color):
        if self.isVisible():
            set_info(message, color)

    @QtCore.pyqtSlot(object)
    def show_result(self, result):
        self.draw_button.setEnabled(True)
        render, self.render = self.render, None
        if self.isVisible():
            render(result)

    @QtCore.pyqtSlot(str)
    def show_error(self, message):
        self.draw_button.setEnabled(True)
        self.render = None
        self.report(f'Не удалось построить карту: {message}', 'red')


def _map_grid(bounds, step):
    x_min, x_max, y_min, y_max = bounds
    return np.mgrid[x_min - 200:x_max + 200:step, y_min - 200:y_max + 200:step]


def _interpolate_categorical(
    x, y, z, bounds, grid_size, interpolation_method, smoothing_power, clip_to_hull,
    boundary_smoothing_level, enforce_hard_labels, _report,
):
    xx, yy = _map_grid(bounds, grid_size)
    points = np.column_stack([x, y])
    labels = np.asarray(z, dtype=int)
    uncertainty_grid = None
    if interpolation_method == "probability":
        grid_labels, prob_cube, _ = _categorical_probability_grid(
            points=points, labels=labels, xx=xx, yy=yy, smooth_power=smoothing_power
        )
        uncertainty_grid = _calculate_uncertainty_from_probabilities(prob_cube)
        method_title = f"Probability (power={smoothing_power})"
    else:
        tree = cKDTree(points)
        _, nearest_idx = tree.query(np.column_stack([xx.ravel(), yy.ravel()]), k=1)
        grid_labels = labels[nearest_idx].reshape(xx.shape)
        method_title = "Nearest"

    mask = None
    if clip_to_hull:
        mask = ~_inside_hull_mask(xx, yy, points)
        if uncertainty_grid is not None:
            uncertainty_grid = np.ma.masked_where(mask, uncertainty_grid)

    sorted_labels = sorted(set(labels), key=lambda value: (value == -1, value))
    label_to_index = {label: index for index, label in enumerate(sorted_labels)}
    class_index_grid = np.vectorize(label_to_index.__getitem__)(grid_labels)
    if mask is not None:
        class_index_grid = np.ma.masked_where(mask, class_index_grid)

    class_index_grid = _smooth_categorical_labels(class_index_grid, boundary_smoothing_level)
    if enforce_hard_labels:
        source_class_indices = np.array([label_to_index[label] for label in labels])
        class_index_grid = _enforce_source_point_labels(
            smoothed_grid=class_index_grid, xx=xx, yy=yy,
            source_points=points, source_class_indices=source_class_indices,
        )

    return xx, yy, labels, sorted_labels, class_index_grid, uncertainty_grid, method_title


def _interpolate_continuous(
    x, y, z, bounds, grid_size, var_model, estimator, dist_func, bin_func, nlags, apply_filter, filt_power, report,
):
    xx, yy = _map_grid(bounds, grid_size)
    coord = np.column_stack([x, y])
    # Интерполяция значений на сетке scikit-gstat.  A Variogram rejects
    # coincident points (zero-distance pairs); model output can contain
    # several values for the same coordinate, therefore merge them first.
    try:
        coord, z, merged_points = prepare_variogram_data(coord, z, min_unique_points=1)
        x, y = coord[:, 0], coord[:, 1]
        if merged_points:
            report(
                f'Для вариограммы усреднены значения в {merged_points} совпадающих точках.',
                'brown'
            )
    except ValueError as exc:
        raise ValueError(f'Невозможно подготовить данные карты: {exc}') from exc

    if len(coord) == 1 or np.allclose(z, z[0]):
        surface_method = 'Постоянная поверхность'
        z_interp = np.full(xx.shape, z[0], dtype=float)
        report('Все значения карты одинаковы: построена постоянная поверхность.', 'brown')
    else:
        surface_method = f'Кригинг ({var_model})'
        try:
            variogram = Variogram(
                coordinates=coord,
                values=z,
                model=var_model,
                n_lags=min(nlags, len(coord) - 1),
                estimator=estimator,
                dist_func=dist_func,
                bin_func=bin_func,
                fit_sigma='exp'
            )
            kriging = OrdinaryKriging(
                variogram=variogram,
                min_points=min(5, len(coord)),
                max_points=min(20, len(coord)),
                mode='exact'
            )
            z_interp = kriging.transform(xx.flatten(), yy.flatten()).reshape(xx.shape)
            z_interp, filled_cell_count = fill_kriging_gaps(
                z_interp, coord, z, xx, yy
            )
            if filled_cell_count:
                surface_method += f' + IDW ({filled_cell_count} узлов)'
                report(
                    f'Кригинг построен; {filled_cell_count} ячеек без достаточного '
                    'числа соседей локально заполнены методом IDW.',
                    'brown'
                )
        except Exception as exc:
            surface_method = 'IDW'
            report(
                f'Вариограмма не построена ({dist_func}, {bin_func}; '
                f'{type(exc).__name__}: {exc}). '
                'Карта построена резервным методом IDW.',
                'brown'
            )
            z_interp = inverse_distance_interpolation(coord, z, xx, yy)
    if apply_filter:
        if not np.isfinite(z_interp).all():
            # Savitzky-Golay uses a least-squares fit and cannot accept
            # NaN/inf. The map remains useful without this optional step.
            report('Сглаживание пропущено: интерполяция содержит NaN или inf.', 'brown')
        else:
            parameters = savgol_parameters(filt_power, z_interp.shape[-1])
            if parameters is None:
                report('Сглаживание пропущено: размер сетки меньше 3 точек.', 'brown')
            else:
                window_length, polyorder = parameters
                if window_length != filt_power:
                    report(
                        f'Окно сглаживания уменьшено с {filt_power} до {window_length} под размер сетки.',
                        'brown'
                    )
                try:
                    z_interp = savgol_filter(z_interp, window_length, polyorder)
                    surface_method += f', фильтр {window_length}'
                except ValueError as exc:
                    report(f'Сглаживание пропущено: {exc}', 'brown')

    return xx, yy, x, y, z, z_interp, surface_method


def draw_map(list_x, list_y, list_z, param, color_marker=True, profiles=False, list_name=None,
             initial_map_mode=None, report_research_id=None, *, point_labels=None, on_close=None):

    list_x, list_y, list_z = (np.asarray(values).copy() for values in (list_x, list_y, list_z))
    if not (len(list_x) and len(list_x) == len(list_y) == len(list_z)):
        set_info('Для карты требуются одинаковые по длине непустые списки X, Y и значений.', 'red')
        if on_close is not None:
            on_close()
        return

    object_id, research_id = get_object_id(), get_research_id()
    object_title = (
        ui.comboBox_geochem.currentText().split(' id')[0]
        if ui.tabWidget_2.currentWidget().objectName() == 'tab_10'
        else f'{get_object_name()} {get_research_name()}'
    )
    show_profile_wells = ui.checkBox_profile_well.isChecked()
    show_object_wells = ui.checkBox_object_well.isChecked()

    def map_wells(research, selected_object_id):
        if show_object_wells and selected_object_id is not None:
            nearby = get_list_nearest_well_for_object(selected_object_id) or []
        elif show_profile_wells and research is not None:
            nearby = [well for profile in research.profiles for well in (get_list_nearest_well(profile.id) or [])]
        else:
            nearby = []
        return [
            (well.id, (well.x_coord, well.y_coord, well.name, well_marker_color(well)))
            for well, *_ in nearby
        ]

    research = session.get(Research, research_id) if research_id is not None and (profiles or show_profile_wells) else None
    research_profiles = research.profiles if research else []
    nearby_wells = map_wells(research, object_id)
    wells = [well for _, well in nearby_wells]
    report_research_id = research_id if report_research_id is None else report_research_id
    if report_research_id == research_id:
        report_wells = list(dict(nearby_wells).values())
    else:
        report_research = session.get(Research, report_research_id)
        report_wells = list(dict(map_wells(report_research, report_research.object_id)).values()) if report_research else []

    tracks = []
    if report_research_id is not None:
        coordinates = session.query(Profile.x_pulc, Profile.y_pulc).filter(Profile.research_id == report_research_id)
        for raw_x, raw_y in coordinates:
            try:
                x = np.asarray(json.loads(raw_x), dtype=float)
                y = np.asarray(json.loads(raw_y), dtype=float)
            except (TypeError, ValueError):
                continue
            if x.ndim == 1 and x.shape == y.shape and len(x) > 1:
                tracks.append((x, y))

    profile_x, profile_y = [], []
    if profiles:
        for profile in research_profiles:
            try:
                x_coords = json.loads(profile.x_pulc)
                y_coords = json.loads(profile.y_pulc)
            except (TypeError, ValueError):
                set_info(f'Не загружены координаты {profile.title}', 'red')
                continue
            if x_coords and y_coords:
                profile_x.extend(x_coords)
                profile_y.extend(y_coords)

    markers = []
    if param.startswith('Classifier'):
        markers = session.query(MarkerMLP).filter_by(analysis_id=get_MLP_id()).all()
    elif param.startswith('Geochem'):
        markers = session.query(GeochemCategory).filter_by(maket_id=get_maket_id()).all()
    marker_colors = [marker.color for marker in markers]
    legend = '\n'.join(f'{index}-{marker.title}' for index, marker in enumerate(markers, start=1))
    if list_name is not None:
        list_name = list(list_name)

    if point_labels is not None:
        point_labels = np.array(point_labels, dtype=str)

    Draw_Map = _DrawMapDialog()
    ui_dm = Ui_DrawMapForm()
    ui_dm.setupUi(Draw_Map)
    Draw_Map.setWindowTitle(f'{Draw_Map.windowTitle()} — {param}')

    has_labels = point_labels is not None and any(label.strip() for label in point_labels)
    ui_dm.checkBox_show_point_labels.setVisible(has_labels)
    if point_labels is not None:
        ui_dm.checkBox_show_points.setText('Показать точки наблюдений')
        ui_dm.checkBox_report_points.setText('Показать точки наблюдений')
    ui_dm.checkBox_report_profiles.setChecked(profiles)
    ui_dm.comboBox_report_contour_step.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
    ui_dm.comboBox_report_contour_step.addItems([
        'Автоматически', '0,01', '0,05', '0,1', '0,2', '0,5', '1', '2', '5', '10', '20', '50', '100',
    ])

    for i in cmap_list:
        ui_dm.comboBox_cmap.addItem(i)
    ui_dm.comboBox_cmap.setCurrentText('jet')
    for i in dist_func_list:
        ui_dm.comboBox_dist_func.addItem(i)

    def _get_control(*names):
        for name in names:
            ctrl = getattr(ui_dm, name, None)
            if ctrl is None:
                ctrl = Draw_Map.findChild(QtWidgets.QWidget, name)
            if ctrl is not None:
                return ctrl
        return None

    map_mode_combo = _get_control(
        "comboBox_map_mode",
        "comboBox_mode_map",
        "comboBox_render_mode"
    )
    interp_method_combo = _get_control(
        "comboBox_interp_method",
        "comboBox_interpolation_method"
    )

    continuous_controls = [
        _get_control("comboBox_estimator"),
        _get_control("label_27"),
        _get_control("comboBox_var_model"),
        _get_control("label_4"),
        _get_control("comboBox_dist_func"),
        _get_control("label_5"),
        _get_control("comboBox_bin_func"),
        _get_control("label_28"),
        _get_control("spinBox_nlags"),
        _get_control("checkBox_filt"),
        _get_control("spinBox_filt"),
        _get_control("groupBox_kriging")
    ]
    categorical_controls = [
        _get_control("groupBox_categorical"),
        _get_control("checkBox_clip_to_hull"),
        _get_control("checkBox_clip_to_data_area"),
        _get_control("checkBox_show_uncertainty"),
        _get_control("checkBox_uncertainty"),
        _get_control("comboBox_boundary_smooth"),
        _get_control("comboBox_boundary_smoothing"),
        _get_control("spinBox_prob_power"),
        _get_control("doubleSpinBox_prob_power"),
        _get_control("checkBox_hard_labels"),
        _get_control("checkBox_enforce_hard_labels")
    ]
    uncertainty_controls = [
        _get_control("checkBox_show_uncertainty"),
        _get_control("checkBox_uncertainty"),
        _get_control("checkBox_2"),
        _get_control("label_uncertainty"),
        _get_control("doubleSpinBox_uncertainty_alpha"),
        _get_control("spinBox_uncertainty_alpha")
    ]

    def _set_enabled_state(controls, enabled):
        for ctrl in controls:
            if ctrl is not None:
                ctrl.setEnabled(enabled)

    def _is_categorical_mode():
        if map_mode_combo is None:
            return False
        mode_text = map_mode_combo.currentText().strip().lower()
        return "categor" in mode_text or "cluster" in mode_text or mode_text.startswith("cat")

    def _is_report_style():
        return ui_dm.comboBox_map_style.currentIndex() == 1

    def _apply_point_labels_state():
        points_toggle = ui_dm.checkBox_report_points if _is_report_style() else ui_dm.checkBox_show_points
        ui_dm.checkBox_show_point_labels.setEnabled(points_toggle.isChecked())

    def _apply_map_style_state():
        report_style = _is_report_style()
        categorical_mode = _is_categorical_mode()
        numeric_controls = (
            ui_dm.label_report_contour_step, ui_dm.comboBox_report_contour_step,
            ui_dm.label_report_scale_min, ui_dm.lineEdit_report_scale_min,
            ui_dm.label_report_scale_max, ui_dm.lineEdit_report_scale_max, ui_dm.checkBox_report_clip,
        )
        for control in numeric_controls:
            control.setVisible(not categorical_mode)
        ui_dm.groupBox_report_map.setVisible(report_style)
        ui_dm.checkBox_show_points.setVisible(not report_style)
        ui_dm.label_4.setVisible(not (report_style and categorical_mode))
        ui_dm.comboBox_cmap.setVisible(not (report_style and categorical_mode))
        if report_style:
            ui_dm.checkBox_show_contours.setText('Границы классов' if categorical_mode else 'Изолинии')
        else:
            ui_dm.checkBox_show_contours.setText('show contours')
        ui_dm.comboBox_report_contour_step.setEnabled(ui_dm.checkBox_show_contours.isChecked())
        _apply_point_labels_state()

    def _rebuild_interp_method_options():
        if interp_method_combo is None:
            return
        if _is_categorical_mode():
            options = ["Nearest", "Probability"]
            default_value = "Probability"
        else:
            options = ["Kriging"]
            default_value = "Kriging"

        old_text = interp_method_combo.currentText()
        interp_method_combo.blockSignals(True)
        interp_method_combo.clear()
        interp_method_combo.addItems(options)
        if old_text in options:
            interp_method_combo.setCurrentText(old_text)
        else:
            interp_method_combo.setCurrentText(default_value)
        interp_method_combo.blockSignals(False)

    def _apply_interpolation_method_deps():
        if interp_method_combo is None:
            return
        can_use_uncertainty = _is_categorical_mode() and interp_method_combo.currentText().strip().lower() == "probability"
        for ctrl in uncertainty_controls:
            if ctrl is not None:
                ctrl.setEnabled(can_use_uncertainty)

    def _apply_map_mode_state():
        categorical_mode = _is_categorical_mode()
        _set_enabled_state(continuous_controls, not categorical_mode)
        _set_enabled_state(categorical_controls, categorical_mode)
        _rebuild_interp_method_options()
        _apply_interpolation_method_deps()
        _apply_map_style_state()

    if map_mode_combo is not None:
        existing_modes = [map_mode_combo.itemText(i) for i in range(map_mode_combo.count())]
        if not existing_modes:
            map_mode_combo.addItems(["Continuous (Kriging)", "Categorical (Cluster)"])
        map_mode_combo.currentTextChanged.connect(lambda _: _apply_map_mode_state())

    if interp_method_combo is not None:
        interp_method_combo.currentTextChanged.connect(lambda _: _apply_interpolation_method_deps())

    ui_dm.comboBox_map_style.currentIndexChanged.connect(_apply_map_style_state)
    ui_dm.checkBox_show_contours.toggled.connect(_apply_map_style_state)
    ui_dm.checkBox_show_points.toggled.connect(_apply_point_labels_state)
    ui_dm.checkBox_report_points.toggled.connect(_apply_point_labels_state)

    def _checked(*names, default=False):
        control = _get_control(*names)
        return control.isChecked() if control is not None else default

    def show_report(builder, *args, **kwargs):
        try:
            report = builder(*args, **kwargs)
        except ValueError as exc:
            set_info(f'Карта для отчёта: {exc}', 'red')
            return

        preview = QtWidgets.QDialog()
        preview.setWindowTitle(f'Карта для отчёта — {param}')
        canvas = FigureCanvasQTAgg(report)
        toolbar = NavigationToolbar2QT(canvas, preview)
        layout = QtWidgets.QVBoxLayout(preview)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        screen = Draw_Map.screen().availableGeometry()
        width, height = canvas.get_width_height()
        preview.resize(min(width, screen.width()), min(height + toolbar.sizeHint().height(), screen.height() - 40))
        show_work_window(preview)

    def form_lda_ok():
        report_style = _is_report_style()
        categorical_mode = _is_categorical_mode()
        show_contours = _checked("checkBox_show_contours", default=True)
        contour_interval = scale_min = scale_max = None
        if report_style and not categorical_mode:
            try:
                if show_contours:
                    contour_interval = parse_contour_interval(ui_dm.comboBox_report_contour_step.currentText())
                scale_min, scale_max = parse_scale_bounds(
                    ui_dm.lineEdit_report_scale_min.text(), ui_dm.lineEdit_report_scale_max.text(),
                )
            except ValueError as exc:
                set_info(f'Карта для отчёта: {exc}', 'red')
                return

        labels_control = ui_dm.checkBox_show_point_labels
        active_labels = point_labels if has_labels and labels_control.isEnabled() and labels_control.isChecked() else None
        grid_x, grid_y = list_x, list_y
        if report_style:
            samples = np.column_stack([list_x, list_y, list_z]).astype(float)
            if not np.isfinite(samples).all(axis=1).any():
                set_info('Карта для отчёта: нет точек с координатами и значениями.', 'red')
                return
            grid_x, grid_y = samples[np.isfinite(samples[:, :2]).all(axis=1), :2].T
            report_options = dict(
                title=param if categorical_mode else ' — '.join(filter(None, (object_title.strip(), param))),
                show_contours=show_contours,
                show_points=ui_dm.checkBox_report_points.isChecked(),
                profiles=tracks if ui_dm.checkBox_report_profiles.isChecked() else (),
                wells=report_wells,
            )
        bounds = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
        sparse = ui_dm.spinBox_sparse.value()
        x, y, z = list_x[::sparse], list_y[::sparse], list_z[::sparse]
        grid_size = ui_dm.spinBox_grid.value()
        sparse_labels = active_labels[::sparse] if active_labels is not None else None
        if categorical_mode:
            if report_style:
                valid = np.isfinite(samples[::sparse]).all(axis=1)
                x, y, z = samples[::sparse][valid].T
                if not len(z):
                    set_info('Нет данных для построения категориальной карты.', 'red')
                    return
                if sparse_labels is not None:
                    sparse_labels = sparse_labels[valid]
            points = np.column_stack([x, y])
            interpolation_method = interp_method_combo.currentText().strip().lower() if interp_method_combo else "nearest"
            prob_power_ctrl = _get_control("doubleSpinBox_prob_power", "spinBox_prob_power")
            smoothing_power = prob_power_ctrl.value() if prob_power_ctrl is not None else 2.0
            boundary_control = _get_control("comboBox_boundary_smooth", "comboBox_boundary_smoothing")
            boundary_smoothing_level = boundary_control.currentText() if boundary_control else "Off"
            show_uncertainty = _checked("checkBox_show_uncertainty", "checkBox_uncertainty", "checkBox_2")
            show_points = _checked("checkBox_show_points", default=True)
            clip_to_hull = _checked("checkBox_clip_to_hull", "checkBox_clip_to_data_area")
            calculate = partial(
                _interpolate_categorical, x, y, z, bounds, grid_size, interpolation_method, smoothing_power,
                clip_to_hull, boundary_smoothing_level,
                _checked("checkBox_hard_labels", "checkBox_enforce_hard_labels", default=True),
            )

            def render_categorical(result):
                xx, yy, labels, sorted_labels, class_index_grid, uncertainty_grid, method_title = result
                color_by_label = _build_categorical_palette(sorted_labels)
                if report_style:
                    show_report(
                        build_categorical_report_map, xx, yy, class_index_grid, points, labels,
                        {label: color_by_label[label] for label in sorted_labels},
                        grid_step=grid_size, method=f'{method_title}, сглаживание: {boundary_smoothing_level}',
                        clip_to_samples=clip_to_hull, uncertainty=uncertainty_grid if show_uncertainty else None,
                        point_labels=sparse_labels, **report_options,
                    )
                    return

                class_cmap = ListedColormap([color_by_label[lbl] for lbl in sorted_labels])

                fig_interp = plt.figure(figsize=(12, 9))
                plt.pcolormesh(xx, yy, class_index_grid, shading='auto', cmap=class_cmap, alpha=0.55)

                if uncertainty_grid is not None and show_uncertainty:
                    uncertainty_plot = plt.pcolormesh(
                        xx,
                        yy,
                        uncertainty_grid,
                        shading='auto',
                        cmap='binary',
                        alpha=0.35,
                        vmin=0.0,
                        vmax=1.0
                    )
                    cbar = plt.colorbar(uncertainty_plot, fraction=0.046, pad=0.04)
                    cbar.set_label('uncertainty')

                if show_contours:
                    contour_levels = np.arange(-0.5, len(sorted_labels), 1.0)
                    plt.contour(xx, yy, class_index_grid, levels=contour_levels, colors='k', linewidths=0.6, alpha=0.8)

                legend_handles = []
                for lbl in sorted_labels:
                    legend_name = "noise" if lbl == -1 else f"cluster {lbl}"
                    legend_handles.append(Patch(facecolor=color_by_label[lbl], edgecolor='k', label=legend_name))

                if show_points:
                    for lbl in sorted_labels:
                        mask_lbl = labels == lbl
                        marker_style = "x" if lbl == -1 else "o"
                        plt.scatter(
                            x[mask_lbl], y[mask_lbl],
                            c=color_by_label[lbl], s=18 if lbl == -1 else 24,
                            marker=marker_style, edgecolors='k' if lbl != -1 else None,
                            linewidths=0.3
                        )
                    if sparse_labels is not None:
                        annotate_observations(fig_interp.axes[0], points, sparse_labels)

                plt.xlabel('X')
                plt.ylabel('Y')
                plt.title(
                    f'Categorical map: {param}\n'
                    f'Method: {method_title}; smoothing: {boundary_smoothing_level}\n'
                    f'Grid: {grid_size}, sparse: {sparse}'
                )
                plt.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))
                plt.tight_layout()
                fig_interp.show()

            Draw_Map.start_calculation(draw_button, calculate, render_categorical)
            return

        var_model = ui_dm.comboBox_var_model.currentText()
        estimator = ui_dm.comboBox_estimator.currentText()
        dist_func = ui_dm.comboBox_dist_func.currentText()
        bin_func = ui_dm.comboBox_bin_func.currentText()
        filt_power = ui_dm.spinBox_filt.value()
        nlags = ui_dm.spinBox_nlags.value()
        apply_filter = _checked("checkBox_filt")
        color_map = ListedColormap(marker_colors) if color_marker and marker_colors else ui_dm.comboBox_cmap.currentText()
        levels_count = max(1, len(marker_colors) - 1) if marker_colors else 10
        calculate = partial(
            _interpolate_continuous, x, y, z, bounds, grid_size, var_model, estimator, dist_func, bin_func,
            nlags, apply_filter, filt_power,
        )

        clip_to_samples = ui_dm.checkBox_report_clip.isChecked()
        show_points = point_labels is None or ui_dm.checkBox_show_points.isChecked()

        def render_continuous(result):
            xx, yy, x, y, z, z_interp, surface_method = result
            if report_style:
                show_report(
                    build_report_map, xx, yy, z_interp, samples,
                    parameter=param, cmap=color_map, grid_step=grid_size, method=surface_method,
                    contour_interval=contour_interval, clip_to_samples=clip_to_samples,
                    sample_names=list_name, scale_min=scale_min, scale_max=scale_max, point_labels=active_labels,
                    **report_options,
                )
                return

            # Визуализация результатов
            fig_interp = plt.figure(figsize=(12, 9))
            plt.contour(xx, yy, z_interp, levels=levels_count, colors='k', linewidths=0.5)
            plt.pcolormesh(xx, yy, z_interp, shading='auto', cmap=color_map)
            if show_points:
                plt.scatter(x, y, c=z, cmap=color_map)
            plt.colorbar(label=param)
            if show_points:
                plt.scatter(x, y, c=z, marker='.', edgecolors='k', s=0.1)
            if sparse_labels is not None:
                observations = np.column_stack([list_x[::sparse], list_y[::sparse], list_z[::sparse]]).astype(float)
                valid = np.isfinite(observations).all(axis=1)
                annotate_observations(fig_interp.axes[0], observations[valid, :2], sparse_labels[valid])
            if list_name:
                for n_ix in range(len(list_name)):
                    plt.text(list_x[n_ix] + 20, list_y[n_ix] + 20, list_name[n_ix], fontsize=5)

            for well_x, well_y, name, color in wells:
                plt.scatter(well_x, well_y, marker='o', color=color, s=50)
                plt.text(well_x + 20, well_y + 20, name)

            plt.xlabel('X')
            plt.ylabel('Y')

            plt.title(f'{object_title} {param}\n{legend}\nМодель интерполяции: {var_model}'
                      f'\nМетод оценки полувариации: {estimator}\nФункция расстояния: {dist_func}\nФункция разбиения: {bin_func}'
                      # f'\nКоличество ячеек усреднения вариограммы: {nlags}'
                      f'\nФильтр результата: {filt_power if apply_filter else "off"}\n'
                      f'\nШаг сетки: {grid_size} м, узлы: {xx.shape[0]}×{xx.shape[1]}')
            plt.tight_layout()

            if profile_x:
                plt.scatter(profile_x, profile_y, marker='.', edgecolors='k', s=0.1)

            fig_interp.show()

        Draw_Map.start_calculation(draw_button, calculate, render_continuous)

    draw_button = _get_control("pushButton_map", "pushButton_draw_map", "pushButton_draw")
    if draw_button is not None:
        draw_button.clicked.connect(form_lda_ok)
    else:
        set_info("В форме карты не найдена кнопка DRAW (pushButton_map).", "brown")
    from app_settings import bind_form
    bind_form(Draw_Map, "draw_map_form")
    if map_mode_combo is not None and isinstance(initial_map_mode, str):
        for index in range(map_mode_combo.count()):
            if initial_map_mode.strip().lower() in map_mode_combo.itemText(index).lower():
                map_mode_combo.setCurrentIndex(index)
                break
    _apply_map_mode_state()
    if on_close is not None:
        Draw_Map.finished.connect(lambda _: on_close())
    show_work_window(Draw_Map)


def _get_profiles_for_display():
    """Return profiles according to all/year/current profile drawing mode."""
    return get_profiles_for_profile_mode("research", "Построение профилей")


def _draw_profile_label(profile):
    try:
        profile_x = json.loads(profile.x_pulc)
        profile_y = json.loads(profile.y_pulc)

        plt.text(profile_x[0] + 20, profile_y[0] + 20, profile.title, fontsize=10, color='green')
        plt.scatter(profile_x[0], profile_y[0], marker='o', color='green', s=25)
        plt.text(profile_x[-10] + 20, profile_y[-10] + 20, profile.title, fontsize=10, color='orange')
        plt.scatter(profile_x[-1], profile_y[-1], marker='o', color='orange', s=25)
    except (TypeError, IndexError):
        pass


def _draw_object_labels(profiles):
    object_points = {}

    for profile in profiles:
        if not profile.research or not profile.research.object:
            continue

        try:
            profile_x = json.loads(profile.x_pulc)
            profile_y = json.loads(profile.y_pulc)
        except TypeError:
            continue

        object_title = profile.research.object.title
        object_points.setdefault(object_title, {'x': [], 'y': []})
        object_points[object_title]['x'].extend(profile_x)
        object_points[object_title]['y'].extend(profile_y)

    for object_title, points in object_points.items():
        if not points['x'] or not points['y']:
            continue

        label_x = max(points['x']) + 100
        label_y = (min(points['y']) + max(points['y'])) / 2
        plt.text(label_x, label_y, object_title, fontsize=12, color='black', fontweight='bold')


def show_profiles():
    profiles = _get_profiles_for_display()
    if profiles is None:
        return
    list_x, list_y = [], []

    for profile in profiles:
        try:
            list_x += json.loads(profile.x_pulc)
            list_y += json.loads(profile.y_pulc)
        except TypeError:
            set_info(f'Не загружены координаты {profile.title}', 'red')
            pass

    x = np.array(list_x)
    y = np.array(list_y)

    fig_profiles = plt.figure(figsize=(12, 9))

    grid = session.query(Grid).filter(Grid.object_id == get_object_id()).first()
    if grid:
        min_x_r, max_x_r = min([g[0] for g in json.loads(grid.grid_table_r)]), max([g[0] for g in json.loads(grid.grid_table_r)])
        min_y_r, max_y_r = min([g[1] for g in json.loads(grid.grid_table_r)]), max([g[1] for g in json.loads(grid.grid_table_r)])

        min_x_uf, max_x_uf = min([g[0] for g in json.loads(grid.grid_table_uf)]), max([g[0] for g in json.loads(grid.grid_table_uf)])
        min_y_uf, max_y_uf = min([g[1] for g in json.loads(grid.grid_table_uf)]), max([g[1] for g in json.loads(grid.grid_table_uf)])

        min_x_m, max_x_m = min([g[0] for g in json.loads(grid.grid_table_m)]), max([g[0] for g in json.loads(grid.grid_table_m)])
        min_y_m, max_y_m = min([g[1] for g in json.loads(grid.grid_table_m)]), max([g[1] for g in json.loads(grid.grid_table_m)])

        plt.plot([min_x_r, max_x_r, max_x_r, min_x_r, min_x_r], [min_y_r, min_y_r, max_y_r, max_y_r, min_y_r], color='blue', label='сетка рельефа')
        plt.plot([min_x_m, max_x_m, max_x_m, min_x_m, min_x_m], [min_y_m, min_y_m, max_y_m, max_y_m, min_y_m], color='green', label='сетка мощности')
        plt.plot([min_x_uf, max_x_uf, max_x_uf, min_x_uf, min_x_uf], [min_y_uf, min_y_uf, max_y_uf, max_y_uf, min_y_uf], color='red', label='сетка уфы')


    if ui.checkBox_common_grid_for_map.isChecked():
        grid_types = {
            'uf': ('red', 'сетка уфы'),
            'm': ('green', 'сетка мощности'),
            'r': ('blue', 'сетка рельефа')
        }

        for g in session.query(CommonGrid).all():
            if g.type in grid_types:
                color, label = grid_types[g.type]
                points = np.array(json.loads(g.grid_table))[:, :2]
                hull = convex_hull_boundary(points)

                # Рисуем границу
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], color=color, linestyle='-', label=label)

                # Убираем дубликаты в легенде
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))


    plt.scatter(x, y, marker='.', edgecolors='k', s=0.1)
    if ui.checkBox_profile_title.isChecked():
        for profile in profiles:
            _draw_profile_label(profile)

    if ui.checkBox_prof_all.isChecked() or ui.checkBox_profile_year.isChecked():
        _draw_object_labels(profiles)

    if ui.checkBox_profile_well.isChecked() or ui.checkBox_object_well.isChecked():
        if ui.checkBox_object_well.isChecked():
            wells = get_list_nearest_well_for_object(get_object_id())
        else:
            wells = [well for profile in profiles for well in (get_list_nearest_well(profile.id) or [])]
        for well in wells:
            plt.scatter(
                well[0].x_coord,
                well[0].y_coord,
                marker='o',
                color=well_marker_color(well[0]),
                s=50,
            )
            plt.text(well[0].x_coord + 20, well[0].y_coord + 20, well[0].name)

    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.legend()

    plt.title('Профили')
    plt.tight_layout()
    fig_profiles.show()
