import numpy as np
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



def add_train_set_point():
    new_set_point = SetPointsTrain(title=ui.lineEdit_string.text(), object_id=get_object_id())
    session.add(new_set_point)
    session.commit()
    update_train_list_set_point()
    update_train_list_point()
    set_info(f'Набор тренировочных точек {new_set_point.title} добавлен', 'green')
    pass


def remove_train_set_point():
    train_set_point_title = ui.comboBox_set_point_2.currentText()
    result = QtWidgets.QMessageBox.question(
        MainWindow,
        'Удаление набора тренировочных точек',
        f'Вы уверены, что хотите удалить набор тренировочных точек {train_set_point_title}?',
        QtWidgets.QMessageBox.Yes,
        QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        session.query(PointTrain).filter_by(set_points_train_id=get_train_set_point_id()).delete()
        session.query(SetPointsTrain).filter_by(id=get_train_set_point_id()).delete()
        session.commit()
        set_info(f'Набор тренировочных точек "{train_set_point_title}" удален', 'green')
        update_train_list_set_point()
        update_train_list_point()
    else:
        pass


# def get_analysis_id():
#     if ui.comboBox_analysis_expl.count() > 0:
#         return ui.comboBox_analysis_expl.currentData()['id']

# def update_list_analysis():
#     ui.listWidget_param_analysis_expl.clear()
#     for i in session.query(ParameterAnalysisExploration).filter_by(analysis_id=get_analysis_id()).all():
#         try:
#             item_text = (f'{i.title} id{i.id}')
#             item = QListWidgetItem(item_text)
#             item.setData(Qt.UserRole, i.id)
#             ui.listWidget_param_analysis_expl.addItem(item)
#         except AttributeError:
#             session.query(ParameterAnalysisExploration).filter_by(id=i.id).delete()
#             session.commit()
#
#
# def update_combobox_analysis():
#     ui.comboBox_analysis_expl.clear()
#     for i in session.query(AnalysisExploration).filter_by(set_points_id=get_set_point_id()).all():
#         ui.comboBox_analysis_expl.addItem(f'{i.title} id{i.id}')
#         ui.comboBox_analysis_expl.setItemData(ui.comboBox_analysis_expl.count() - 1, {'id': i.id})
#     update_list_analysis()


def add_analysis():
    """Добавить новый параметрический анализ"""
    new_analysis = AnalysisExploration(title=ui.lineEdit_string.text(), set_points_id=get_set_point_id())
    session.add(new_analysis)
    session.commit()
    update_analysis_combobox()
    update_analysis_list()
    set_info(f'Параметрический анализ {new_analysis.title} добавлен', 'green')
    pass

def remove_analysis():
    """ Удалить анализ """
    analysis_title = ui.comboBox_analysis_expl.currentText()
    result = QtWidgets.QMessageBox.question(
        MainWindow,
        'Удаление параметрического анализа',
        f'Вы уверены, что хотите удалить параметрический анализ {analysis_title}?',
        QtWidgets.QMessageBox.Yes,
        QtWidgets.QMessageBox.No
    )
    if result == QtWidgets.QMessageBox.Yes:
        session.query(AnalysisExploration).filter_by(id=get_analysis_id()).delete()
        session.commit()
        set_info(f'Параметрический анализ "{analysis_title}" удален', 'green')
        update_analysis_combobox()
        update_analysis_list()
    else:
        pass

def add_all_analysis_parameter_tolist():
    """ Добавляет все параметры из списка в анализ """
    for i in session.query(ParameterExploration).filter_by(exploration_id=get_exploration_id()).all():
        check = get_analysis_id()
        if check is not None:
            param = ParameterAnalysisExploration(analysis_id=get_analysis_id(), parameter_id=i.id, title=i.parameter)
            session.add(param)
        else:
            set_info(f'Для добавления параметра создайте анализ', 'red')
        session.commit()

    update_analysis_list()

def add_analysis_parameter_tolist():
    """ Добавляет выбранный параметр в анализ """
    # ui.listWidget_param_analysis_expl.clear()
    item = session.query(ParameterExploration).filter_by(id=ui.listWidget_param_expl.currentItem().text().split(' id')[-1]).first()
    check = get_analysis_id()
    if check is not None:
        param = ParameterAnalysisExploration(analysis_id=get_analysis_id(), parameter_id=item.id, title=item.parameter)
        session.add(param)
    else:
        set_info(f'Для добавления параметра создайте анализ', 'red')
    session.commit()

    update_analysis_list()

def clear_all_analysis_parameters():
    """ Удаляет все параметры из анализа """
    ch = get_analysis_id()
    if ch is None:
        return
    session.query(ParameterAnalysisExploration).filter_by(analysis_id=get_analysis_id()).delete()
    session.commit()
    update_analysis_list()

def del_analysis_parameter():
    """ Удаляет выбранный параметр из анализа """
    ch = get_analysis_id()
    if ch is None:
        return
    item = session.query(ParameterAnalysisExploration).filter_by(
        id=ui.listWidget_param_analysis_expl.currentItem().text().split(' id')[-1]).first()
    # param = ParameterAnalysisExploration(analysis_id=get_analysis_id(), parameter_id=item.id, title=item.title)
    session.delete(item)
    session.commit()
    update_analysis_list()


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
    def cancel_points():
        PointsLoader.close()

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

        if list_cols is not None:
            for v in [name_cell, x_cell, y_cell]:
                if v not in list_cols:
                    cancel_points()
                else:
                    list_cols.remove(v)

        for index_i, i in enumerate(pd_points.index):
            ui.progressBar.setValue(index_i + 1)
            p = PointExploration(set_points_id=set_id, x_coord=pd_points.loc[i, x_cell],
                                 y_coord=pd_points.loc[i, y_cell],
                                 title=str(pd_points.loc[i, name_cell]))
            session.add(p)
            session.commit()

            for j, el in enumerate(list_cols):
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

    ui_pt.buttonBox.accepted.connect(load_points)
    # ui_pt.buttonBox.rejected.connect(PointsLoader.close())
    PointsLoader.exec()

def update_train_list_exploration():
    """ Обновляем список параметров исследования """
    ui.listWidget_param_analysis_expl.clear()
    for i in session.query(ParameterExploration).filter_by(exploration_id=get_exploration_id()).all():
        try:
            item_text = (f'{i.parameter} id{i.id}')
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, i.id)
            ui.listWidget_param_expl.addItem(item)
        except AttributeError:
            session.query(ParameterExploration).filter_by(id=i.id).delete()
            session.commit()
    update_list_set_point()

def load_train_data():
    """ Загрузить обучающие данные Excel"""
    ch1 = get_exploration_id()
    ch2 = get_set_point_id()
    if ch1 is None or ch2 is None:
        set_info(f'Для добавления данных задайте точки исследования', 'red')
        return

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

    list_combobox = [ui_pt.comboBox_N, ui_pt.comboBox_x, ui_pt.comboBox_y, ui_pt.comboBox_t]
    for cmbx in list_combobox:
        for i in list_cols:
            cmbx.addItem(i)

    def load_train_points():
        ui.progressBar.setMaximum(len(pd_points.index))
        name_cell = ui_pt.comboBox_N.currentText()
        x_cell = ui_pt.comboBox_x.currentText()
        y_cell = ui_pt.comboBox_y.currentText()
        t_cell = ui_pt.comboBox_t.currentText()

        set_id = get_set_point_id()
        exp_id = get_exploration_id()

        for v in [name_cell, x_cell, y_cell, t_cell]:
            if v not in list_cols:
                cancel_points()
            else:
                list_cols.remove(v)

        for index, i_item in enumerate(pd_points.index):
            ui.progressBar.setValue(index + 1)
            p = PointTrain(set_points_train_id=set_id, x_coord=pd_points.loc[i_item, x_cell],
                                 y_coord=pd_points.loc[i_item, y_cell],
                                 target=pd_points.loc[i_item, t_cell],
                                 title=str(pd_points.loc[i_item, name_cell]))
            session.add(p)
            session.commit()

            # for j, el in enumerate(list_cols):
            #     old_param = session.query(ParameterExploration).filter(
            #         ParameterExploration.parameter == el,
            #         ParameterExploration.exploration_id == exp_id).first()
            #     if not old_param:
            #         old_param = ParameterExploration(exploration_id=exp_id, parameter=el, train=True)
            #         session.add(old_param)
            #         session.commit()
            #     par_point = ParameterPoint(point_id=p.id, param_id=old_param.id, value=pd_points.loc[i_item, list_cols[j]])
            #     session.add(par_point)
            # session.commit()
            update_train_list_point()
        set_info(f'Добавлены тренировочные данные из файла', 'green')
    def cancel_points():
        PointsLoader.close()

    ui_pt.buttonBox.accepted.connect(load_train_points)
    # ui_pt.buttonBox.rejected.connect(PointsLoader.close())
    PointsLoader.exec()








def draw_interpolation():
    points = session.query(PointExploration).filter_by(set_points_id=get_set_point_id()).all()
    if not points:
        return
    value_points = []
    for i in points:
        ch = get_parameter_exploration_id()
        if ch is None:
            return
        value = session.query(ParameterPoint.value).filter_by(
            param_id=get_parameter_exploration_id(),
            point_id=i.id
        ).first()[0]
        value_points.append(value)
    x_list = [p.x_coord for p in points]
    y_list = [p.y_coord for p in points]

    x_array = np.array(x_list)
    y_array = np.array(y_list)
    coord = np.column_stack((x_array, y_array))


    npts = 88
    x = np.linspace(np.min(x_array), np.max(x_array), npts)
    y = np.linspace(np.min(y_array), np.max(y_array), npts)
    X, Y = np.meshgrid(x, y)

    xx, yy = np.mgrid[min(x_array) - 200: max(x_array) + 200: 75, min(y_array) - 200: max(y_array) + 200: 75]

    variogram = Variogram(coordinates=coord, values=value_points, model="spherical", fit_method="lm")


    kriging = OrdinaryKriging(variogram=variogram, min_points=5, max_points=20, mode='exact')
    field = kriging.transform(xx.flatten(), yy.flatten()).reshape(xx.shape)
    s2 = kriging.sigma.reshape(xx.shape)

    plt.figure(figsize=(12, 9))
    plt.contour(xx, yy, field, levels=10, colors='k', linewidths=0.5)
    plt.pcolormesh(xx, yy, field, shading='auto', cmap='jet')
    plt.scatter(x_array, y_array, c=value_points, cmap='jet')
    plt.colorbar(label='param')
    plt.scatter(x_array, y_array, c=value_points, marker='o', edgecolors='w', s=0.1)

    plt.tight_layout()
    plt.show()
    #
    # fig, ax = plt.subplots(nrows=2, ncols=2)
    # ax[0, 0].scatter(x_array, y_array, c=value_points, marker='o', cmap='jet', label='Original Data')
    # ax[0, 0].set_title('Sample points on f(X,Y)')
    #
    #
    # # Интерполяция тремя способами
    # for i, method in enumerate(('nearest', 'linear', 'cubic')):
    #     Z = griddata((x_array, y_array), value_points, (X, Y), method=method)
    #     r, c = (i + 1) // 2, (i + 1) % 2
    #     ax[r, c].contourf(X, Y, Z, cmap='jet', alpha=0.5, levels=20)
    #     ax[r, c].scatter(x_array, y_array, c=value_points, marker='.', cmap='jet', label='Original Data')
    #     ax[r, c].set_title("method = '{}'".format(method))
    #
    # plt.tight_layout()
    # plt.show()



