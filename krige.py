from func import *
from qt.choose_formation_map import *

cmap_list = ['viridis', 'plasma', 'inferno', 'magma', 'cividis','Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
             'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr',
             'seismic', 'twilight', 'twilight_shifted', 'hsv', 'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
             'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c', 'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
             'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
             'turbo', 'nipy_spectral', 'gist_ncar']
for i in cmap_list:
    ui.comboBox_cmap.addItem(i)
ui.comboBox_cmap.setCurrentText('gist_rainbow')



def show_map():
    global list_z
    r_id = get_research_id()
    r = session.query(Research).filter_by(id=r_id).first()
    list_x, list_y, list_z = [], [], []
    param = ui.comboBox_param_plast.currentText()
    for profile in r.profiles:
        list_x += (json.loads(profile.x_pulc))
        list_y += (json.loads(profile.y_pulc))
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



    # Создание набора данных
    x = np.array(list_x)
    y = np.array(list_y)
    z = np.array(list_z)
    grid_size = ui.spinBox_grid.value()

    # Создание сетки для интерполяции
    gridx = np.linspace(min(list_x), max(list_x), grid_size)
    gridy = np.linspace(min(list_y), max(list_y), grid_size)

    # Создание объекта OrdinaryKriging
    var_model = ui.comboBox_var_model.currentText()
    nlags = ui.spinBox_nlags.value()
    weight = ui.checkBox_weight.isChecked()
    vector = 'vectorized' if ui.checkBox_vector.isChecked() else 'C'
    clos_win = ui.spinBox_count_distr_lda.value()
    color_map = ui.comboBox_cmap.currentText()
    ok = OrdinaryKriging(x, y, z, variogram_model=var_model, nlags=nlags, weight=weight, verbose=True)

    # Интерполяция значений на сетке
    if ui.checkBox_vector.isChecked():
        z_interp, _ = ok.execute("grid", gridx, gridy)
    else:
        z_interp, _ = ok.execute("grid", gridx, gridy, backend='C', n_closest_points=clos_win)
    # z_interp = z_interp.reshape(gridx.shape)
    # ok.display_variogram_model()



    # Визуализация результатов
    plt.figure(figsize=(12, 9))
    plt.contour(gridx, gridy, z_interp, colors='k', linewidths=0.5)
    plt.pcolormesh(gridx, gridy, z_interp, shading='auto', cmap=color_map)
    plt.scatter(x, y, c=z, cmap=color_map)
    plt.colorbar(label='Z Value')
    plt.scatter(x, y, c=z, marker='.', edgecolors='k', s=0.1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Ordinary Kriging Interpolation')
    plt.tight_layout()
    plt.show()

