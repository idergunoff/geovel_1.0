from numpy.linalg import LinAlgError
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
                form = session.query(literal_column(f'Formation.{param}')).filter(Formation.profile_id == profile.id).first()
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
                    form = session.query(literal_column(f'Formation.{param}')).filter(Formation.id == f_id).first()
                    list_z += (json.loads(form[0]))
                    Choose_Formation.close()

                ui_cf.pushButton_ok_form_map.clicked.connect(form_lda_ok)
                Choose_Formation.exec_()
        except OperationalError:
            set_info('Не выбран пласт', 'red')
            return
    draw_map(list_x, list_y, list_z, param)


def draw_map(list_x, list_y, list_z, param):

    Draw_Map = QtWidgets.QDialog()
    ui_dm = Ui_DrawMapForm()
    ui_dm.setupUi(Draw_Map)
    Draw_Map.show()
    Draw_Map.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    for i in cmap_list:
        ui_dm.comboBox_cmap.addItem(i)
    ui_dm.comboBox_cmap.setCurrentText('gist_rainbow')


    def form_lda_ok():

        # Создание набора данных
        x = np.array(list_x)
        y = np.array(list_y)
        z = np.array(list_z)
        grid_size = ui_dm.spinBox_grid.value()

        # Создание сетки для интерполяции
        gridx = np.linspace(min(list_x) - 200, max(list_x) + 200, grid_size)
        gridy = np.linspace(min(list_y) - 200, max(list_y) + 200, grid_size)

        # Создание объекта OrdinaryKriging
        var_model = ui_dm.comboBox_var_model.currentText()
        nlags = ui_dm.spinBox_nlags.value()
        weight = ui_dm.checkBox_weight.isChecked()
        vector = 'vectorized' if ui_dm.checkBox_vector.isChecked() else 'C'
        if param == 'lda':
            markers_lda = session.query(MarkerLDA).filter(MarkerLDA.analysis_id == get_LDA_id()).all()
            colors_lda = [marker.color for marker in markers_lda]
            color_map = ListedColormap(colors_lda)
            legend = '\n'.join([f'{n+1}-{m.title}' for n, m in enumerate(markers_lda)])
            levels_count = len(markers_lda) - 1
        elif param == 'mlp':
            markers_mlp = session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).all()
            colors_mlp = [marker.color for marker in markers_mlp]
            color_map = ListedColormap(colors_mlp)
            legend = '\n'.join([f'{n + 1}-{m.title}' for n, m in enumerate(markers_mlp)])
            levels_count = len(markers_mlp) - 1
        else:
            color_map = ui_dm.comboBox_cmap.currentText()
            legend = ''
            levels_count = 10
        ok = OrdinaryKriging(x, y, z, variogram_model=var_model, nlags=nlags, weight=weight, verbose=True)

        # Интерполяция значений на сетке
        try:
            z_interp, _ = ok.execute("grid", gridx, gridy, backend=vector, n_closest_points=2 if vector == 'C' else None)
        except LinAlgError:
            set_info('LinalgError', 'red')
            return

        # z_interp = z_interp.reshape(gridx.shape)
        # ok.display_variogram_model()

        # Визуализация результатов
        plt.figure(figsize=(12, 9))
        plt.contour(gridx, gridy, z_interp, levels=levels_count, colors='k', linewidths=0.5)
        plt.pcolormesh(gridx, gridy, z_interp, shading='auto', cmap=color_map)
        plt.scatter(x, y, c=z, cmap=color_map)
        plt.colorbar(label=param)
        plt.scatter(x, y, c=z, marker='.', edgecolors='k', s=0.1)
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.title(f'{get_object_name()} {get_research_name()} {param}\n{legend}\nМодель интерполяции: {var_model}\nКоличество ячеек '
                  f'усреднения вариограммы: {nlags}\nЛогистический вес: {weight}\nВекторизованная интерполяция: {vector}'
                  f'\nКол-во ячеек сетки: {grid_size}x{grid_size}')
        plt.tight_layout()
        plt.show()


    ui_dm.pushButton_map.clicked.connect(form_lda_ok)
    Draw_Map.exec_()


