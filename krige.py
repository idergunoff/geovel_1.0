from cProfile import label

import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy.exc import OperationalError
from matplotlib.colors import ListedColormap, to_hex
from matplotlib.patches import Patch
from scipy.spatial import ConvexHull, Delaunay, cKDTree

from func import *
from qt.choose_formation_map import *
from qt.draw_map_form import *

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
    unique_labels = sorted({int(v) for v in labels})
    base_colors = [to_hex(plt.get_cmap("tab20")(i)) for i in range(20)]
    color_by_label = {}
    for lbl in unique_labels:
        if lbl == -1:
            color_by_label[lbl] = "#4d4d4d"
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

def show_map():
    global list_z
    r_id = get_research_id()
    r = session.query(Research).filter_by(id=r_id).first()
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
                Choose_Formation.show()
                Choose_Formation.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия
                for f in profile.formations:
                    ui_cf.listWidget_form_map.addItem(f'{f.title} id{f.id}')
                ui_cf.listWidget_form_map.setCurrentRow(0)

                def form_lda_ok():
                    global list_z
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
                    Choose_Formation.close()

                ui_cf.pushButton_ok_form_map.clicked.connect(form_lda_ok)
                Choose_Formation.exec_()
        except OperationalError:
            set_info('Не выбран пласт', 'red')
            return
    draw_map(list_x, list_y, list_z, param)


def draw_map(list_x, list_y, list_z, param, color_marker=True, profiles=False, list_name=None):

    Draw_Map = QtWidgets.QDialog()
    ui_dm = Ui_DrawMapForm()
    ui_dm.setupUi(Draw_Map)
    Draw_Map.show()
    Draw_Map.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

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
        _get_control("label_3"),
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

    if map_mode_combo is not None:
        existing_modes = [map_mode_combo.itemText(i) for i in range(map_mode_combo.count())]
        if not existing_modes:
            map_mode_combo.addItems(["Continuous (Kriging)", "Categorical (Cluster)"])
        map_mode_combo.currentTextChanged.connect(lambda _: _apply_map_mode_state())

    if interp_method_combo is not None:
        interp_method_combo.currentTextChanged.connect(lambda _: _apply_interpolation_method_deps())

    _apply_map_mode_state()


    def form_lda_ok():

        sparse_ctrl = _get_control("spinBox_sparse")
        sparse = sparse_ctrl.value() if sparse_ctrl is not None else 1
        # Создание набора данных
        x = np.array(list_x[::sparse])
        y = np.array(list_y[::sparse])
        coord = np.column_stack([x, y])
        z = np.array(list_z[::sparse])
        grid_ctrl = _get_control("spinBox_grid")
        grid_size = grid_ctrl.value() if grid_ctrl is not None else 75


        # Создание сетки для интерполяции
        # gridx = np.linspace(min(list_x) - 200, max(list_x) + 200, grid_size)
        # gridy = np.linspace(min(list_y) - 200, max(list_y) + 200, grid_size)
        xx, yy = np.mgrid[min(list_x) - 200: max(list_x) + 200: grid_size, min(list_y) - 200: max(list_y) + 200: grid_size]
        # print(xx, yy)



        # Создание объекта OrdinaryKriging
        var_model_ctrl = _get_control("comboBox_var_model")
        estimator_ctrl = _get_control("comboBox_estimator")
        dist_func_ctrl = _get_control("comboBox_dist_func")
        bin_func_ctrl = _get_control("comboBox_bin_func")
        filt_power_ctrl = _get_control("spinBox_filt")
        nlags_ctrl = _get_control("spinBox_nlags")

        var_model = var_model_ctrl.currentText() if var_model_ctrl is not None else "spherical"
        estimator = estimator_ctrl.currentText() if estimator_ctrl is not None else "matheron"
        dist_func = dist_func_ctrl.currentText() if dist_func_ctrl is not None else "euclidean"
        bin_func = bin_func_ctrl.currentText() if bin_func_ctrl is not None else "even"
        filt_power = filt_power_ctrl.value() if filt_power_ctrl is not None else 13
        nlags = nlags_ctrl.value() if nlags_ctrl is not None else 6
        legend = ''
        levels_count = 10
        color_map = ui_dm.comboBox_cmap.currentText()
        if param.startswith('Classifier'):
            markers_mlp = session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).all()
            colors_mlp = [marker.color for marker in markers_mlp]
            color_map = ListedColormap(colors_mlp)
            legend = '\n'.join([f'{n + 1}-{m.title}' for n, m in enumerate(markers_mlp)])
            levels_count = len(markers_mlp) - 1
        elif param.startswith('Geochem'):
            markers_chem = session.query(GeochemCategory).filter_by(maket_id=get_maket_id()).all()
            colors_chem = [marker.color for marker in markers_chem]
            color_map = ListedColormap(colors_chem)
            legend = '\n'.join([f'{n + 1}-{m.title}' for n, m in enumerate(markers_chem)])
            levels_count = len(markers_chem) - 1
        if not color_marker:
            color_map = ui_dm.comboBox_cmap.currentText()

        if _is_categorical_mode():
            # Отдельный пайплайн categorical (stage 2.1 baseline: nearest).
            points = np.column_stack([x, y])
            labels = np.array([int(v) for v in z], dtype=int)
            if len(points) == 0:
                set_info("Нет данных для построения категориальной карты.", "red")
                return

            tree = cKDTree(points)
            _, nearest_idx = tree.query(np.column_stack([xx.ravel(), yy.ravel()]), k=1)
            grid_labels = labels[nearest_idx].reshape(xx.shape)

            clip_ctrl = _get_control("checkBox_clip_to_hull", "checkBox_clip_to_data_area")
            clip_on = bool(clip_ctrl is not None and clip_ctrl.isChecked())
            if clip_on:
                mask = _inside_hull_mask(xx, yy, points)
                grid_labels = np.ma.masked_where(~mask, grid_labels)

            unique_labels = sorted({int(v) for v in labels})
            color_by_label = _build_categorical_palette(unique_labels)
            sorted_labels = sorted(unique_labels, key=lambda v: (v == -1, v))
            label_to_index = {lbl: i for i, lbl in enumerate(sorted_labels)}
            raw_grid = np.asarray(grid_labels.filled(sorted_labels[0]) if np.ma.isMaskedArray(grid_labels) else grid_labels)
            class_index_grid = np.vectorize(lambda val: label_to_index[int(val)])(raw_grid)
            if np.ma.isMaskedArray(grid_labels):
                class_index_grid = np.ma.masked_where(grid_labels.mask, class_index_grid)

            class_cmap = ListedColormap([color_by_label[lbl] for lbl in sorted_labels])

            fig_interp = plt.figure(figsize=(12, 9))
            plt.pcolormesh(xx, yy, class_index_grid, shading='auto', cmap=class_cmap, alpha=0.55)

            show_contours_ctrl = _get_control("checkBox_show_contours")
            if show_contours_ctrl is None or show_contours_ctrl.isChecked():
                contour_levels = np.arange(-0.5, len(sorted_labels), 1.0)
                plt.contour(xx, yy, class_index_grid, levels=contour_levels, colors='k', linewidths=0.6, alpha=0.8)

            legend_handles = []
            for lbl in sorted_labels:
                legend_name = "noise" if lbl == -1 else f"cluster {lbl}"
                legend_handles.append(Patch(facecolor=color_by_label[lbl], edgecolor='k', label=legend_name))

            show_points_ctrl = _get_control("checkBox_show_points")
            if show_points_ctrl is None or show_points_ctrl.isChecked():
                for lbl in sorted_labels:
                    mask_lbl = labels == lbl
                    marker_style = "x" if lbl == -1 else "o"
                    plt.scatter(
                        x[mask_lbl], y[mask_lbl],
                        c=color_by_label[lbl], s=18 if lbl == -1 else 24,
                        marker=marker_style, edgecolors='k' if lbl != -1 else None,
                        linewidths=0.3
                    )

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Categorical map: {param}\nMethod: Nearest\nGrid: {grid_size}, sparse: {sparse}')
            plt.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            fig_interp.show()
            return
        # ok = OrdinaryKriging(x, y, z, variogram_model=var_model, nlags=nlags, weight=weight, verbose=False)
        # print(1)
        # # Интерполяция значений на сетке
        # try:
        #     z_interp, _ = ok.execute("grid", gridx, gridy, backend=vector, n_closest_points=2 if vector == 'C' else None)
        #     print(z_interp)
        # except LinAlgError:
        #     set_info('LinalgError', 'red')
        #     return

        # z_interp = z_interp.reshape(gridx.shape)
        # ok.display_variogram_model()

        # Интерполяция значений на сетке scikit-gstat
        try:
            variogram = Variogram(coordinates=coord, values=z, estimator=estimator, dist_func=dist_func, bin_func=bin_func, fit_sigma='exp')
        except MemoryError:
            set_info('MemoryError', 'red')
            return
        except ValueError:
            set_info('ValueError: variogram', 'red')
            return
        except AttributeError:
            set_info('AttributeError', 'red')
            return
        except:
            set_info('Variogram error', 'red')
            return

        # variogram.fit()
        kriging = OrdinaryKriging(variogram=variogram, min_points=5, max_points=20, mode='exact')
        try:
            z_interp = kriging.transform(xx.flatten(), yy.flatten()).reshape(xx.shape)
        except LinAlgError:
            set_info('LinAlgError: Singular matrix', 'red')
            return
        # print(len(z_interp))
        # print(z_interp)
        filt_checkbox = _get_control("checkBox_filt")
        if filt_checkbox is not None and filt_checkbox.isChecked():
            try:
                z_interp = savgol_filter(z_interp, filt_power, 3)
            except ValueError:
                set_info('ValueError in savgol filter', 'red')
                return

        # интерполяция значений на сетке gstools
        #
        # model = gs.Spherical(dim=2)
        # print(1)
        # ok = gs.krige.Ordinary(model, [x, y], z)
        # print(2)
        # z_interp = ok.__call__((xx, yy))
        # print(z_interp)

        # Визуализация результатов
        fig_interp = plt.figure(figsize=(12, 9))
        plt.contour(xx, yy, z_interp, levels=levels_count, colors='k', linewidths=0.5)
        plt.pcolormesh(xx, yy, z_interp, shading='auto', cmap=color_map)
        plt.scatter(x, y, c=z, cmap=color_map)
        plt.colorbar(label=param)
        plt.scatter(x, y, c=z, marker='.', edgecolors='k', s=0.1)
        if list_name:
            for n_ix in range(len(list_name)):
                plt.text(list_x[n_ix] + 20, list_y[n_ix] + 20, list_name[n_ix], fontsize=5)

        if ui.checkBox_profile_well.isChecked():
            r = session.query(Research).filter_by(id=get_research_id()).first()
            for profile in r.profiles:
                wells = get_list_nearest_well(profile.id)
                if not wells:
                    continue
                for well in wells:
                    plt.scatter(well[0].x_coord, well[0].y_coord, marker='o', color='r', s=50)
                    plt.text(well[0].x_coord + 20, well[0].y_coord + 20, well[0].name)

        plt.xlabel('X')
        plt.ylabel('Y')

        if ui.tabWidget_2.currentWidget().objectName() == 'tab_10':
            object_title = ui.comboBox_geochem.currentText().split(' id')[0]
        else:
            object_title = f'{get_object_name()} {get_research_name()}'

        plt.title(f'{object_title} {param}\n{legend}\nМодель интерполяции: {var_model}'
                  f'\nМетод оценки полувариации: {estimator}\nФункция расстояния: {dist_func}\nФункция разбиения: {bin_func}'
                  # f'\nКоличество ячеек усреднения вариограммы: {nlags}'
                  f'\nФильтр результата: {filt_power if (filt_checkbox is not None and filt_checkbox.isChecked()) else "off"}\n'
                  f'\nРазмер ячеек сетки: {grid_size}x{grid_size}')
        plt.tight_layout()

        if profiles:

            list_prof_x, list_prof_y = [], []

            r = session.query(Research).filter_by(id=get_research_id()).first()
            for profile in r.profiles:
                try:
                    list_prof_x += (json.loads(profile.x_pulc))
                    list_prof_y += (json.loads(profile.y_pulc))
                except TypeError:
                    set_info(f'Не загружены координаты {profile.title}', 'red')
                    pass

            x = np.array(list_prof_x)
            y = np.array(list_prof_y)

            plt.scatter(x, y, marker='.', edgecolors='k', s=0.1)

        fig_interp.show()

    draw_button = _get_control("pushButton_map", "pushButton_draw_map", "pushButton_draw")
    if draw_button is not None:
        draw_button.clicked.connect(form_lda_ok)
    else:
        set_info("В форме карты не найдена кнопка DRAW (pushButton_map).", "brown")
    Draw_Map.exec_()


def show_profiles():
    r_id = get_research_id()
    list_x, list_y = [], []
    if ui.checkBox_prof_all.isChecked():
       for r in session.query(Research).all():
           for profile in r.profiles:
               try:
                   list_x += (json.loads(profile.x_pulc))
                   list_y += (json.loads(profile.y_pulc))
               except TypeError:
                   set_info(f'Не загружены координаты {profile.title}', 'red')
                   pass
    else:

        r = session.query(Research).filter_by(id=r_id).first()
        for profile in r.profiles:
            try:
                list_x += (json.loads(profile.x_pulc))
                list_y += (json.loads(profile.y_pulc))
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
        r = session.query(Research).filter_by(id=r_id).first()
        for profile in r.profiles:
            try:
                plt.text(json.loads(profile.x_pulc)[0] + 20, json.loads(profile.y_pulc)[0] + 20, profile.title, fontsize=10, color='green')
                plt.scatter(json.loads(profile.x_pulc)[0], json.loads(profile.y_pulc)[0], marker='o', color='green', s=25)
                plt.text(json.loads(profile.x_pulc)[-10] + 20, json.loads(profile.y_pulc)[-10] + 20, profile.title, fontsize=10, color='orange')
                plt.scatter(json.loads(profile.x_pulc)[-1], json.loads(profile.y_pulc)[-1], marker='o', color='orange', s=25)
            except TypeError:
                pass

    if ui.checkBox_profile_well.isChecked():
        for profile in r.profiles:
            wells = get_list_nearest_well(profile.id)
            if not wells:
                continue
            for well in wells:
                plt.scatter(well[0].x_coord, well[0].y_coord, marker='o', color='r', s=50)
                plt.text(well[0].x_coord + 20, well[0].y_coord + 20, well[0].name)

    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.legend()

    plt.title('Профили')
    plt.tight_layout()
    fig_profiles.show()
