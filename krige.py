import matplotlib.pyplot as plt
from sqlalchemy.exc import OperationalError

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


def draw_map(list_x, list_y, list_z, param, color_marker=True):

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


    def form_lda_ok():

        sparse = ui_dm.spinBox_sparse.value()
        # Создание набора данных
        x = np.array(list_x[::sparse])
        y = np.array(list_y[::sparse])
        coord = np.column_stack([x, y])
        z = np.array(list_z[::sparse])
        grid_size = ui_dm.spinBox_grid.value()

        # Создание сетки для интерполяции
        # gridx = np.linspace(min(list_x) - 200, max(list_x) + 200, grid_size)
        # gridy = np.linspace(min(list_y) - 200, max(list_y) + 200, grid_size)
        xx, yy = np.mgrid[min(list_x) - 200: max(list_x) + 200: grid_size, min(list_y) - 200: max(list_y) + 200: grid_size]
        # print(xx, yy)



        # Создание объекта OrdinaryKriging
        var_model = ui_dm.comboBox_var_model.currentText()
        estimator = ui_dm.comboBox_estimator.currentText()
        dist_func = ui_dm.comboBox_dist_func.currentText()
        bin_func = ui_dm.comboBox_bin_func.currentText()
        filt_power = ui_dm.spinBox_filt.value()
        nlags = ui_dm.spinBox_nlags.value()
        legend = ''
        levels_count = 10
        color_map = ui_dm.comboBox_cmap.currentText()
        if param == 'lda':
            markers_lda = session.query(MarkerLDA).filter(MarkerLDA.analysis_id == get_LDA_id()).all()
            colors_lda = [marker.color for marker in markers_lda]
            color_map = ListedColormap(colors_lda)
            legend = '\n'.join([f'{n+1}-{m.title}' for n, m in enumerate(markers_lda)])
            levels_count = len(markers_lda) - 1
        elif param.startswith('Classifier'):
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
        if ui_dm.checkBox_filt.isChecked():
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
                  f'\nФильтр результата: {filt_power if ui_dm.checkBox_filt.isChecked() else "off"}\n'
                  f'\nРазмер ячеек сетки: {grid_size}x{grid_size}')
        plt.tight_layout()
        fig_interp.show()


    ui_dm.pushButton_map.clicked.connect(form_lda_ok)
    Draw_Map.exec_()


def show_profiles():
    r_id = get_research_id()
    r = session.query(Research).filter_by(id=r_id).first()
    list_x, list_y = [], []
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


    plt.scatter(x, y, marker='.', edgecolors='k', s=0.1)
    if ui.checkBox_profile_title.isChecked():
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
    plt.legend()

    plt.title('Профили')
    plt.tight_layout()
    fig_profiles.show()
