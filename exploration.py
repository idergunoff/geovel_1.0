from scipy.interpolate import griddata

from func import *





def add_exploration():
    """ Добавить новое исследование """
    new_expl = Exploration(title=ui.lineEdit_string.text(), object_id=get_object_id())
    session.add(new_expl)
    session.commit()
    update_list_exploration()
    set_info(f'Исследование {new_expl.title} добавлено', 'green')


def remove_exploration():
    """ Удалить исследование """
    current_title = ui.comboBox_expl.currentText()
    result = QtWidgets.QMessageBox.question(
        MainWindow,
        'Удаление исследования',
        f'Вы уверены, что хотите удалить исследование {current_title} со всеми параметрами?',
        QtWidgets.QMessageBox.Yes,
        QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        session.query(ParameterExploration).filter_by(exploration_id=get_exploration_id()).delete()
        session.query(SetPoints).filter_by(exploration_id=get_exploration_id()).delete()
        session.query(Exploration).filter_by(id=get_exploration_id()).delete()
        session.query(PointExploration).filter_by(set_points_id=get_set_point_id()).delete()

        session.commit()
        set_info(f'Исследование "{current_title}" удалено', 'green')
        update_list_exploration()
    else:
        pass
    pass


def add_set_point():
    """ Добавить новый набор точек исследования """
    new_set_point = SetPoints(title=ui.lineEdit_string.text(), exploration_id=get_exploration_id())
    session.add(new_set_point)
    session.commit()
    update_list_set_point()
    set_info(f'Набор точек исследования {new_set_point.title} добавлен', 'green')
    pass


def remove_set_point():
    """ Удалить набор точек исследования """
    set_point_title = ui.comboBox_set_point.currentText()
    result = QtWidgets.QMessageBox.question(
        MainWindow,
        'Удаление набора точек исследования',
        f'Вы уверены, что хотите удалить набор точек исследования {set_point_title}?',
        QtWidgets.QMessageBox.Yes,
        QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        session.query(PointExploration).filter_by(set_points_id=get_set_point_id()).delete()
        session.query(SetPoints).filter_by(id=get_set_point_id()).delete()
        session.commit()
        set_info(f'Набор точек исследования "{set_point_title}" удален', 'green')
        update_list_set_point()
    else:
        pass

def load_point_exploration():
    """ Загрузить набор точек исследования из файла Excel"""
    try:
        file_name = QFileDialog.getOpenFileName(caption='Выберите файл Excel (разделитель ";")', filter='*.xls *.xlsx')[0]
        set_info(file_name, 'blue')
        pd_points = pd.read_excel(file_name, header=0)
        list_cols = list(pd_points.columns)
    except FileNotFoundError:
        return

    pd_points = clean_dataframe(pd_points)

    PointsLoader = QtWidgets.QDialog()
    ui_pt = Ui_Form_load_points()
    ui_pt.setupUi(PointsLoader)
    PointsLoader.show()
    PointsLoader.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    list_combobox = [ui_pt.comboBox_N, ui_pt.comboBox_x, ui_pt.comboBox_y]
    for cmbx in list_combobox:
        for i in list_cols:
            cmbx.addItem(i)

    def load_points():
        ui.progressBar.setMaximum(len(pd_points.index))
        name_cell = ui_pt.comboBox_N.currentText()
        x_cell = ui_pt.comboBox_x.currentText()
        y_cell = ui_pt.comboBox_y.currentText()

        set_id = get_set_point_id()
        exp_id = get_exploration_id()

        for index_i, i in enumerate(pd_points.index):
            ui.progressBar.setValue(index_i + 1)
            p = PointExploration(set_points_id=set_id, x_coord=pd_points.loc[i, x_cell],
                                 y_coord=pd_points.loc[i, y_cell],
                                 title=str(pd_points.loc[i, name_cell]))
            session.add(p)
            session.commit()
            for j, el in enumerate(list_cols):
                if el == name_cell or el == x_cell or el == y_cell:
                    continue
                old_param = session.query(ParameterExploration).filter(
                    ParameterExploration.parameter == el,
                    ParameterExploration.exploration_id == exp_id
                ).first()
                if not old_param:
                    old_param = ParameterExploration(exploration_id=exp_id, parameter=el)
                    session.add(old_param)
                    session.commit()

                par_point = ParameterPoint(point_id=p.id, param_id=old_param.id, value=pd_points.loc[i, list_cols[j]])
                session.add(par_point)
            session.commit()
            update_list_exploration()
            update_list_set_point()
        set_info(f'Добавлены данные исследований из файла', 'green')

    def cancel_points():
        PointsLoader.close()

    ui_pt.buttonBox.accepted.connect(load_points)
    ui_pt.buttonBox.rejected.connect(cancel_points)

    PointsLoader.exec()



def draw_interpolation():
    points = session.query(PointExploration).filter_by(set_points_id=get_set_point_id()).all()
    value_points = []
    for i in points:
        value = session.query(ParameterPoint.value).filter_by(
            param_id=get_parameter_exploration_id(),
            point_id=i.id
        ).first()[0]
        value_points.append(value)
    x_list = [p.x_coord for p in points]
    y_list = [p.y_coord for p in points]

    # print(value_points)
    # print(x_list)
    # print(y_list)

    x_array = np.array(x_list)
    y_array = np.array(y_list)

    npts = 88
    x = np.linspace(np.min(x_array), np.max(x_array), npts)
    y = np.linspace(np.min(y_array), np.max(y_array), npts)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].scatter(x_array, y_array, c=value_points, marker='o', cmap='jet', label='Original Data')
    ax[0, 0].set_title('Sample points on f(X,Y)')


    # Интерполяция тремя способами
    for i, method in enumerate(('nearest', 'linear', 'cubic')):
        Z = griddata((x_array, y_array), value_points, (X, Y), method=method)
        r, c = (i + 1) // 2, (i + 1) % 2
        ax[r, c].contourf(X, Y, Z, cmap='jet', alpha=0.5, levels=20)
        ax[r, c].scatter(x_array, y_array, c=value_points, marker='.', cmap='jet', label='Original Data')
        ax[r, c].set_title("method = '{}'".format(method))

    plt.tight_layout()
    plt.show()



