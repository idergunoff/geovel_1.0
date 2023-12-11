import json
from sqlite3 import OperationalError

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from func import *
from krige import draw_map
from random_search import *


def add_exploration():
    """ Добавить новое исследование """
    if ui.lineEdit_string.text() == '':
        return
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
    if ui.lineEdit_string.text() == '':
        return
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
    """ Добавить новый тренировочный набор точек исследования """
    if ui.lineEdit_string.text() == '':
        return
    new_set_point = SetPointsTrain(title=ui.lineEdit_string.text(), object_id=get_object_id())
    session.add(new_set_point)
    session.commit()
    update_train_list()
    update_train_combobox()
    set_info(f'Набор тренировочных точек {new_set_point.title} добавлен', 'green')
    pass


def remove_train_set_point():
    train_set_point_title = ui.comboBox_train_point.currentText()
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
        update_train_list()
        update_train_combobox()
    else:
        pass


def add_analysis():
    """Добавить новый параметрический анализ"""
    if ui.lineEdit_string.text() == '':
        return
    new_analysis = AnalysisExploration(title=ui.lineEdit_string.text(), train_points_id=get_train_set_point_id())
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
        session.query(AnalysisExploration).filter_by(id=get_analysis_expl_id()).delete()
        session.commit()
        set_info(f'Параметрический анализ "{analysis_title}" удален', 'green')
        update_analysis_combobox()
        update_analysis_list()
    else:
        pass

def add_all_analysis_parameter_tolist():
    """ Добавляет все параметры из списка в анализ """
    for i in session.query(ParameterExploration).filter_by(exploration_id=get_exploration_id()).all():
        check = get_analysis_expl_id()
        if check is not None:
            param = ParameterAnalysisExploration(analysis_id=get_analysis_expl_id(), parameter_id=i.id, title=i.parameter)
            session.add(param)
        else:
            set_info(f'Для добавления параметра создайте анализ', 'red')
    session.query(AnalysisExploration).filter_by(id=get_analysis_expl_id()).update({'up_data': False},
                                                                              synchronize_session='fetch')
    session.commit()

    update_analysis_list()

def add_analysis_parameter_tolist():
    """ Добавляет выбранный параметр в анализ """
    try:
        item = session.query(ParameterExploration).filter_by(id=ui.listWidget_param_expl.currentItem().text().split(' id')[-1]).first()
    except:
        return
    check = get_analysis_expl_id()
    if check is not None:
        analysis = session.query(ParameterAnalysisExploration).all()
        for a in analysis:
            if a.parameter_id == item.id:
                return
        param = ParameterAnalysisExploration(analysis_id=get_analysis_expl_id(), parameter_id=item.id, title=item.parameter)
        session.add(param)
    else:
        set_info(f'Для добавления параметра создайте анализ', 'red')
    session.query(AnalysisExploration).filter_by(id=get_analysis_expl_id()).update({'up_data': False},
                                                                              synchronize_session='fetch')
    session.commit()

    update_analysis_list()

def clear_all_analysis_parameters():
    """ Удаляет все параметры из анализа """
    ch = get_analysis_expl_id()
    if ch is None:
        return
    session.query(ParameterAnalysisExploration).filter_by(analysis_id=get_analysis_expl_id()).delete()
    session.query(GeoParameterAnalysisExploration).filter_by(analysis_id=get_analysis_expl_id()).delete()
    session.commit()
    update_analysis_list()

def del_analysis_parameter():
    """ Удаляет выбранный параметр из анализа """
    ch = get_analysis_expl_id()
    if ch is None:
        return

    item = session.query(ParameterAnalysisExploration).filter_by(
        id=ui.listWidget_param_analysis_expl.currentItem().text().split(' id')[-1]).first()
    # param = ParameterAnalysisExploration(analysis_id=get_analysis_id(), parameter_id=item.id, title=item.title)
    if item is not None:
        session.delete(item)
        session.commit()

    item_2 = session.query(GeoParameterAnalysisExploration).filter_by(
        id=ui.listWidget_param_analysis_expl.currentItem().text().split(' id')[-1]).first()
    if item_2 is not None:
        session.delete(item_2)
        session.commit()

    update_analysis_list()


def add_geo_analysis_param():
    """ Добавляет выбранный параметр с георадара в анализ """
    param = ui.comboBox_geovel_param_expl.currentText()
    geo = session.query(GeoParameterAnalysisExploration).filter_by(param=param, analysis_id=get_analysis_expl_id()).first()
    if geo is not None:
        return set_info(f'{param} уже добавлен', 'red')
    geo_param = GeoParameterAnalysisExploration(param=param, analysis_id=get_analysis_expl_id())
    session.add(geo_param)
    session.query(AnalysisExploration).filter_by(id=get_analysis_expl_id()).update({'up_data': False},
                                                                              synchronize_session='fetch')

    session.commit()

    update_analysis_list()


def add_all_geo_analysis_param():
    """ Добавляет все параметры с георадара в анализ """
    for i in range(ui.comboBox_geovel_param_expl.count()):
        param = ui.comboBox_geovel_param_expl.itemText(i)
        geo = session.query(GeoParameterAnalysisExploration).filter_by(param=param,
                                                                       analysis_id=get_analysis_expl_id()).first()
        if geo is None:
            geo_param = GeoParameterAnalysisExploration(param=param, analysis_id=get_analysis_expl_id())
            session.add(geo_param)
            session.commit()
        else:
            set_info(f'{param} уже добавлен', 'red')
    session.query(AnalysisExploration).filter_by(id=get_analysis_expl_id()).update({'up_data': False},
                                                                              synchronize_session='fetch')
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
        PointsLoader.close()

    ui_pt.buttonBox.accepted.connect(load_points)
    ui_pt.buttonBox.rejected.connect(cancel_points)
    PointsLoader.exec_()


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
    ch1 = get_train_set_point_id()
    if ch1 is None:
        set_info(f'Для добавления точек задайте все данные', 'red')
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
        """ Загрузить тренировочный набор точек из файла Excel"""
        ui.progressBar.setMaximum(len(pd_points.index))
        name_cell = ui_pt.comboBox_N.currentText()
        x_cell = ui_pt.comboBox_x.currentText()
        y_cell = ui_pt.comboBox_y.currentText()
        t_cell = ui_pt.comboBox_t.currentText()

        set_id = get_train_set_point_id()
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

            update_train_list_exploration()
            update_train_list()
            update_train_combobox()
        set_info(f'Добавлены тренировочные данные из файла', 'green')
        PointsLoader.close()
    def cancel_points():
        PointsLoader.close()

    ui_pt.buttonBox.accepted.connect(load_train_points)
    ui_pt.buttonBox.rejected.connect(cancel_points)
    PointsLoader.exec()


def draw_interpolation():
    """ Интерполяция по одному параметру на загруженном наборе точек """
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

    param_name = session.query(ParameterExploration.parameter).filter_by(id=get_parameter_exploration_id()).first()[0]
    draw_map(x_list, y_list, value_points, param_name, False)

    # npts = 88
    # x = np.linspace(np.min(x_array), np.max(x_array), npts)
    # y = np.linspace(np.min(y_array), np.max(y_array), npts)
    # X, Y = np.meshgrid(x, y)
    #
    # xx, yy = np.mgrid[min(x_array) - 200: max(x_array) + 200: 75, min(y_array) - 200: max(y_array) + 200: 75]
    #
    # variogram = Variogram(coordinates=coord, values=value_points, model="spherical", fit_method="lm")
    # variogram.plot()
    #
    # kriging = OrdinaryKriging(variogram=variogram, min_points=5, max_points=20, mode='exact')
    # field = kriging.transform(xx.flatten(), yy.flatten()).reshape(xx.shape)
    # s2 = kriging.sigma.reshape(xx.shape)
    #
    # plt.figure(figsize=(12, 9))
    # plt.contour(xx, yy, field, levels=10, colors='k', linewidths=0.5)
    # plt.pcolormesh(xx, yy, field, shading='auto', cmap='jet')
    # plt.scatter(x_array, y_array, c=value_points, cmap='jet')
    # plt.colorbar(label='param')
    # plt.scatter(x_array, y_array, c=value_points, marker='o', edgecolors='w', s=0.1)
    #
    # plt.tight_layout()
    # plt.show()


def train_interpolation():
    """ Интерполяция по одному параметру на тренировочном наборе точек """
    points = session.query(PointExploration).filter_by(set_points_id=get_set_point_id()).all()
    if not points:
        return
    value_points = []
    for i in points:
        ch = get_train_param_id()
        if ch is None:
            return
        p = session.query(ParameterAnalysisExploration).filter_by(id=get_train_param_id()).first()
        value = session.query(ParameterPoint.value).filter_by(
            param_id=p.param.id,
            point_id=i.id
        ).first()[0]
        value_points.append(value)
    x_list = [p.x_coord for p in points]
    y_list = [p.y_coord for p in points]

    x_array = np.array(x_list)
    y_array = np.array(y_list)
    coord = np.column_stack((x_array, y_array))

    t_points = session.query(PointTrain).filter_by(set_points_train_id=get_train_set_point_id()).all()
    if not t_points:
        return
    x_train = [p.x_coord for p in t_points]
    y_train = [p.y_coord for p in t_points]


    xx, yy = np.mgrid[min(x_train) - 200: max(x_train) + 200: 75, min(y_train) - 200: max(y_train) + 200: 75]

    variogram = Variogram(coordinates=coord, values=value_points, model="spherical", fit_method="lm", estimator='matheron', bin_func='even')
    # variogram.plot()

    kriging = OrdinaryKriging(variogram=variogram, min_points=3, max_points=10, mode='exact')
    field = kriging.transform(np.array(x_train), np.array(y_train))
    print(field)
    # plt.figure(figsize=(12, 9))
    # # plt.contour(xx, yy, field, levels=10, colors='k', linewidths=0.5)
    # # plt.pcolormesh(xx, yy, field, shading='auto', cmap='jet')
    # plt.scatter(x_array, y_array, c=value_points, cmap='jet')
    # plt.colorbar(label='param')
    # plt.scatter(x_array, y_array, c=value_points, marker='o', edgecolors='w', s=0.1)
    #
    # plt.tight_layout()
    # plt.show()

def create_grid_points(x, y):
    x_new = []
    y_new = []

    for i in range(len(x)):
        x_point = x[i]
        y_point = y[i]

        # Создаем новые точки сетки вокруг заданных координат
        for x_grid in range(int(x_point) - 25, int(x_point) + 26, 5):
            for y_grid in range(int(y_point) - 25, int(y_point) + 26, 5):
                x_new.append(x_grid)
                y_new.append(y_grid)

    return x_new, y_new


def build_interp_table():
    """ Интерполяция по нескольким параметрам,
                        создание базы данных со значением field для точек исследования """
    start_time = datetime.datetime.now()

    global form_prof
    print("Начало работы: ")
    df = pd.DataFrame(columns=['x_coord', 'y_coord', 'target', 'title'])


    """ Тренировочные точки """
    t_points = session.query(PointTrain).filter_by(set_points_train_id=get_train_set_point_id()).all()
    if not t_points:
        return
    x_train = [p.x_coord for p in t_points]
    y_train = [p.y_coord for p in t_points]

    xx, yy = np.mgrid[min(x_train): max(x_train):,
             min(y_train): max(y_train)]
    x_train, y_train = create_grid_points(x_train, y_train)
    title_train = [p.title for p in t_points]
    target_train = [p.target for p in t_points]

    target_train_new = [[i] * 121 for i in target_train]
    title_train_new = [[i] * 121 for i in title_train]
    target_train, title_train = [], []
    for i in target_train_new:
        target_train.extend(i)

    for i in title_train_new:
        title_train.extend(i)

    df['title'] = title_train
    df['x_coord'] = x_train
    df['y_coord'] = y_train
    df['target'] = target_train

    """ Точки для Вариограмы """

    points = session.query(PointExploration).filter_by(set_points_id=get_set_point_id()).all()
    if not points:
        return
    params = session.query(ParameterAnalysisExploration).filter_by(analysis_id=get_analysis_expl_id()).all()
    ui.progressBar.setMaximum(len(params))

    for index, el in enumerate(params):
        expl = session.query(Exploration).filter_by(id=el.param.exploration_id).first()
        ui.comboBox_expl.setCurrentText(f'{expl.title} id{expl.id}')
        update_list_set_point()
        points = session.query(PointExploration).filter_by(set_points_id=get_set_point_id()).all()

        value_points = []
        for i in points:
            p = session.query(ParameterAnalysisExploration).filter_by(id=el.id).first()
            value = session.query(ParameterPoint.value).filter_by(
                param_id=p.param.id,
                point_id=i.id
            ).first()[0]

            value_points.append(value)

        set_info(f'Обработка параметра {p.title}', 'blue')
        x_list = [p.x_coord for p in points]
        y_list = [p.y_coord for p in points]

        x_array = np.array(x_list)
        y_array = np.array(y_list)
        coord = np.column_stack((x_array, y_array))

        """ Вариограма и Кригинг """
        # variogram = Variogram(coordinates=coord, values=np.array(value_points), model="spherical", fit_method="lm", fit_sigma='exp')
        variogram = Variogram(coordinates=coord, values=np.array(value_points), estimator='matheron', dist_func='euclidean', bin_func='even', fit_sigma='exp')

        try:
            kriging = OrdinaryKriging(variogram=variogram, min_points=2, max_points=20, mode='exact')
            field = kriging.transform(np.array(x_train), np.array(y_train))
            df[el.title] = field
        except LinAlgError:
            set_info(f'LinAlgError - {el.title}', 'red')
        ui.progressBar.setValue(index + 1)

    """ Вариограма и интерполяция для параметров с георадара """
    geo_param = session.query(GeoParameterAnalysisExploration).filter_by(analysis_id=get_analysis_expl_id()).all()
    ui.progressBar.setMaximum(len(geo_param))
    if len(geo_param) > 0:
        profiles = session.query(Profile).filter(Profile.research_id == get_research_id()).all()
        x_prof, y_prof, form_prof = [], [], []

        for profile in profiles:
            x_prof += json.loads(profile.x_pulc)
            y_prof += json.loads(profile.y_pulc)
            if len(profile.formations) == 1:
                form_prof.append(profile.formations[0])
            else:
                Choose_Formation = QtWidgets.QDialog()
                ui_cf = Ui_FormationMAP()
                ui_cf.setupUi(Choose_Formation)
                Choose_Formation.show()
                Choose_Formation.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия
                set_info(f'Выберите пласт для "{profile.title}"', 'blue')
                for f in profile.formations:
                    ui_cf.listWidget_form_map.addItem(f'{f.title} id{f.id}')
                ui_cf.listWidget_form_map.setCurrentRow(0)

                def form_lda_ok():
                    global form_prof
                    f_id = ui_cf.listWidget_form_map.currentItem().text().split(" id")[-1]
                    form = session.query(Formation).filter(Formation.id == f_id).first()
                    form_prof.append(form)
                    Choose_Formation.close()

                ui_cf.pushButton_ok_form_map.clicked.connect(form_lda_ok)
                Choose_Formation.exec_()

        coord_geo = np.column_stack((np.array(x_prof[::5]), np.array(y_prof[::5])))
        for index, g in enumerate(geo_param):
            set_info(f'Обработка параметра {g.param}', 'blue')
            print(f'Обработка параметра {g.param}')
            list_value = []
            for f in form_prof:
                list_value += json.loads(getattr(f, g.param))

            variogram = Variogram(coordinates=coord_geo, values=list_value[::5], estimator='matheron', dist_func='euclidean', bin_func='even', fit_sigma='exp')
            try:
                kriging = OrdinaryKriging(variogram=variogram, min_points=2, max_points=20, mode='exact')
                field = kriging.transform(np.array(x_train), np.array(y_train))
                df[g.param] = field
            except LinAlgError:
                set_info(f"LinAlgError - {g.param}", 'red')
            ui.progressBar.setValue(index + 1)

    train_time = datetime.datetime.now() - start_time
    list_param = params + geo_param
    # print(df['Benzene, (1-methylethyl)-'])
    print(train_time)


    df.to_excel('data_train1.xlsx')

    data_train = json.dumps(df.to_dict())
    session.query(AnalysisExploration).filter_by(id=get_analysis_expl_id()).update({'data': data_train, 'up_data': True},
                                                                              synchronize_session='fetch')
    session.commit()

    return df, list_param

def exploration_MLP():
    """ Тренировка моделей классификаторов """
    data = session.query(AnalysisExploration).filter_by(id=get_analysis_expl_id(), up_data=True).first()
    print(data, get_analysis_expl_id())
    if data is None:
        data_train, list_param = build_interp_table()
    else:
        data_train = pd.DataFrame(json.loads(data.data))

    list_param_mlp = data_train.columns.tolist()[4:]
    # colors = {}
    # for m in session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).all():
    #     colors[m.title] = m.color

    training_sample = imputer.fit_transform(np.array(data_train[list_param_mlp].values.tolist()))
    markup = np.array(sum(data_train[['target']].values.tolist(), []))

    list_marker = [0, 1]

    Classifier = QtWidgets.QDialog()
    ui_cls = Ui_ClassifierForm()
    ui_cls.setupUi(Classifier)
    Classifier.show()
    Classifier.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    ui_cls.spinBox_pca.setMaximum(len(list_param_mlp))
    ui_cls.spinBox_pca.setValue(len(list_param_mlp) // 2)
    def push_checkbutton_extra():
        if ui_cls.checkBox_rfc_ada.isChecked():
            ui_cls.checkBox_rfc_ada.setChecked(False)

    def push_checkbutton_ada():
        if ui_cls.checkBox_rfc_extra.isChecked():
            ui_cls.checkBox_rfc_extra.setChecked(False)

    def push_checkbutton_smote():
        if ui_cls.checkBox_adasyn.isChecked():
            ui_cls.checkBox_adasyn.setChecked(False)

    def push_checkbutton_adasyn():
        if ui_cls.checkBox_smote.isChecked():
            ui_cls.checkBox_smote.setChecked(False)


    def choice_model_classifier(model):
        """ Выбор модели классификатора """
        if model == 'MLPC':
            model_class = MLPClassifier(
                hidden_layer_sizes=tuple(map(int, ui_cls.lineEdit_layer_mlp.text().split())),
                activation=ui_cls.comboBox_activation_mlp.currentText(),
                solver=ui_cls.comboBox_solvar_mlp.currentText(),
                alpha=ui_cls.doubleSpinBox_alpha_mlp.value(),
                max_iter=5000,
                early_stopping=ui_cls.checkBox_e_stop_mlp.isChecked(),
                validation_fraction=ui_cls.doubleSpinBox_valid_mlp.value(),
                random_state=0
            )
            text_model = (f'**MLP**: \nhidden_layer_sizes: '
                          f'({",".join(map(str, tuple(map(int, ui_cls.lineEdit_layer_mlp.text().split()))))}), '
                          f'\nactivation: {ui_cls.comboBox_activation_mlp.currentText()}, '
                          f'\nsolver: {ui_cls.comboBox_solvar_mlp.currentText()}, '
                          f'\nalpha: {round(ui_cls.doubleSpinBox_alpha_mlp.value(), 2)}, '
                          f'\n{"early stopping, " if ui_cls.checkBox_e_stop_mlp.isChecked() else ""}'
                          f'\nvalidation_fraction: {round(ui_cls.doubleSpinBox_valid_mlp.value(), 2)}, ')

        elif model == 'KNNC':
            n_knn = ui_cls.spinBox_neighbors.value()
            weights_knn = 'distance' if ui_cls.checkBox_knn_weights.isChecked() else 'uniform'
            model_class = KNeighborsClassifier(n_neighbors=n_knn, weights=weights_knn, algorithm='auto')
            text_model = f'**KNN**: \nn_neighbors: {n_knn}, \nweights: {weights_knn}, '
        elif model == 'GBC':
            est = ui_cls.spinBox_n_estimators.value()
            l_rate = ui_cls.doubleSpinBox_learning_rate.value()
            model_class = GradientBoostingClassifier(n_estimators=est, learning_rate=l_rate, random_state=0)
            text_model = f'**GBC**: \nn estimators: {round(est, 2)}, \nlearning rate: {round(l_rate, 2)}, '
        elif model == 'G-NB':
            model_class = GaussianNB(var_smoothing=10 ** (-ui_cls.spinBox_gnb_var_smooth.value()))
            text_model = f'**G-NB**: \nvar smoothing: 1E-{str(ui_cls.spinBox_gnb_var_smooth.value())}, '
            # model_class = CategoricalNB()
            # text_model = f'**C-CB**:, '
        elif model == 'DTC':
            spl = 'random' if ui_cls.checkBox_splitter_rnd.isChecked() else 'best'
            model_class = DecisionTreeClassifier(splitter=spl, random_state=0)
            text_model = f'**DTC**: \nsplitter: {spl}, '
        elif model == 'RFC':
            if ui_cls.checkBox_rfc_ada.isChecked():
                model_class = AdaBoostClassifier(n_estimators=ui_cls.spinBox_rfc_n.value(), random_state=0)
                text_model = f'**ABC**: \nn estimators: {ui_cls.spinBox_rfc_n.value()}, '
            elif ui_cls.checkBox_rfc_extra.isChecked():
                model_class = ExtraTreesClassifier(n_estimators=ui_cls.spinBox_rfc_n.value(), class_weight='balanced', bootstrap=True, oob_score=True, random_state=0, n_jobs=-1)
                text_model = f'**ETC**: \nn estimators: {ui_cls.spinBox_rfc_n.value()}, '
            else:
                model_class = RandomForestClassifier(n_estimators=ui_cls.spinBox_rfc_n.value(), class_weight='balanced', bootstrap=True, oob_score=True, random_state=0, n_jobs=-1)
                text_model = f'**RFC**: \nn estimators: {ui_cls.spinBox_rfc_n.value()}, '
        elif model == 'GPC':
            gpc_kernel_width = ui_cls.doubleSpinBox_gpc_wigth.value()
            gpc_kernel_scale = ui_cls.doubleSpinBox_gpc_scale.value()
            n_restart_optimization = ui_cls.spinBox_gpc_n_restart.value()
            multi_class = ui_cls.comboBox_gpc_multi.currentText()
            kernel = gpc_kernel_scale * RBF(gpc_kernel_width)
            model_class = GaussianProcessClassifier(
                kernel=kernel,
                n_restarts_optimizer=n_restart_optimization,
                random_state=0,
                multi_class=multi_class,
                n_jobs=-1
            )
            text_model = (f'**GPC**: \nwidth kernal: {round(gpc_kernel_width, 2)}, \nscale kernal: {round(gpc_kernel_scale, 2)}, '
                          f'\nn restart: {n_restart_optimization}, \nmulti_class: {multi_class} ,')
        elif model == 'QDA':
            model_class = QuadraticDiscriminantAnalysis(reg_param=ui_cls.doubleSpinBox_qda_reg_param.value())
            text_model = f'**QDA**: \nreg_param: {round(ui_cls.doubleSpinBox_qda_reg_param.value(), 2)}, '
        elif model == 'SVC':
            model_class = SVC(kernel=ui_cls.comboBox_svr_kernel.currentText(), probability=True,
                              C=ui_cls.doubleSpinBox_svr_c.value(), random_state=0, class_weight='balanced')
            text_model = (f'**SVC**: \nkernel: {ui_cls.comboBox_svr_kernel.currentText()}, '
                          f'\nC: {round(ui_cls.doubleSpinBox_svr_c.value(), 2)}, ')
        else:
            model_class = QuadraticDiscriminantAnalysis()
            text_model = ''
        return model_class, text_model


    def build_stacking_voting_model():
        """ Построить модель стекинга """
        estimators, list_model = [], []

        if ui_cls.checkBox_stv_mlpc.isChecked():
            mlpc = MLPClassifier(
                hidden_layer_sizes=tuple(map(int, ui_cls.lineEdit_layer_mlp.text().split())),
                activation=ui_cls.comboBox_activation_mlp.currentText(),
                solver=ui_cls.comboBox_solvar_mlp.currentText(),
                alpha=ui_cls.doubleSpinBox_alpha_mlp.value(),
                max_iter=5000,
                early_stopping=ui_cls.checkBox_e_stop_mlp.isChecked(),
                validation_fraction=ui_cls.doubleSpinBox_valid_mlp.value(),
                random_state=0
            )
            estimators.append(('mlpc', mlpc))
            list_model.append('mlpc')

        if ui_cls.checkBox_stv_knnc.isChecked():
            n_knn = ui_cls.spinBox_neighbors.value()
            weights_knn = 'distance' if ui_cls.checkBox_knn_weights.isChecked() else 'uniform'
            knnc = KNeighborsClassifier(n_neighbors=n_knn, weights=weights_knn, algorithm='auto')
            estimators.append(('knnc', knnc))
            list_model.append('knnc')

        if ui_cls.checkBox_stv_gbc.isChecked():
            est = ui_cls.spinBox_n_estimators.value()
            l_rate = ui_cls.doubleSpinBox_learning_rate.value()
            gbc = GradientBoostingClassifier(n_estimators=est, learning_rate=l_rate, random_state=0)
            estimators.append(('gbc', gbc))
            list_model.append('gbc')

        if ui_cls.checkBox_stv_gnb.isChecked():
            gnb = GaussianNB(var_smoothing=10 ** (-ui_cls.spinBox_gnb_var_smooth.value()))
            estimators.append(('gnb', gnb))
            list_model.append('gnb')

        if ui_cls.checkBox_stv_dtc.isChecked():
            spl = 'random' if ui_cls.checkBox_splitter_rnd.isChecked() else 'best'
            dtc = DecisionTreeClassifier(splitter=spl, random_state=0)
            estimators.append(('dtc', dtc))
            list_model.append('dtc')

        if ui_cls.checkBox_stv_rfc.isChecked():
            if ui_cls.checkBox_rfc_ada.isChecked():
                abc = AdaBoostClassifier(n_estimators=ui_cls.spinBox_rfc_n.value(), random_state=0)
                estimators.append(('abc', abc))
                list_model.append('abc')
            elif ui_cls.checkBox_rfc_extra.isChecked():
                etc = ExtraTreesClassifier(n_estimators=ui_cls.spinBox_rfc_n.value(), class_weight='balanced', bootstrap=True, oob_score=True, random_state=0, n_jobs=-1)
                estimators.append(('etc', etc))
                list_model.append('etc')
            else:
                rfc = RandomForestClassifier(n_estimators=ui_cls.spinBox_rfc_n.value(), class_weight='balanced', bootstrap=True, oob_score=True, random_state=0, n_jobs=-1)
                estimators.append(('rfc', rfc))
                list_model.append('rfc')

        if ui_cls.checkBox_stv_gpc.isChecked():
            gpc_kernel_width = ui_cls.doubleSpinBox_gpc_wigth.value()
            gpc_kernel_scale = ui_cls.doubleSpinBox_gpc_scale.value()
            n_restart_optimization = ui_cls.spinBox_gpc_n_restart.value()
            multi_class = ui_cls.comboBox_gpc_multi.currentText()
            kernel = gpc_kernel_scale * RBF(gpc_kernel_width)
            gpc = GaussianProcessClassifier(
                kernel=kernel,
                n_restarts_optimizer=n_restart_optimization,
                random_state=0,
                multi_class=multi_class,
                n_jobs=-1
            )
            estimators.append(('gpc', gpc))
            list_model.append('gpc')

        if ui_cls.checkBox_stv_qda.isChecked():
            qda = QuadraticDiscriminantAnalysis(reg_param=ui_cls.doubleSpinBox_qda_reg_param.value())
            estimators.append(('qda', qda))
            list_model.append('qda')
        if ui_cls.checkBox_stv_svc.isChecked():
            svc = SVC(kernel=ui_cls.comboBox_svr_kernel.currentText(),
                      probability=True, C=ui_cls.doubleSpinBox_svr_c.value(), random_state=0, class_weight='balanced')
            estimators.append(('svc', svc))
            list_model.append('svc')
        final_model, final_text_model = choice_model_classifier(ui_cls.buttonGroup.checkedButton().text())
        list_model_text = ', '.join(list_model)
        if ui_cls.buttonGroup_stack_vote.checkedButton().text() == 'Voting':
            hard_voting = 'hard' if ui_cls.checkBox_voting_hard.isChecked() else 'soft'
            model_class = VotingClassifier(estimators=estimators, voting=hard_voting, n_jobs=-1)
            text_model = f'**Voting**: -{hard_voting}-\n({list_model_text})\n'
            model_name = 'VOT'
        else:
            model_class = StackingClassifier(estimators=estimators, final_estimator=final_model, n_jobs=-1)
            text_model = f'**Stacking**:\nFinal estimator: {final_text_model}\n({list_model_text})\n'
            model_name = 'STACK'
        return model_class, text_model, model_name


    def calc_model_class():
        """ Создание и тренировка модели """
        # global training_sample, markup
        start_time = datetime.datetime.now()
        # Нормализация данных
        scaler = StandardScaler()

        pipe_steps = []
        pipe_steps.append(('scaler', scaler))

        # Разделение данных на обучающую и тестовую выборки
        training_sample_train, training_sample_test, markup_train, markup_test = train_test_split(
            training_sample, markup, test_size=0.20, random_state=1, stratify=markup)

        text_over_sample = ''

        if ui_cls.checkBox_smote.isChecked():
            smote = SMOTE(random_state=0)
            training_sample_train, markup_train = smote.fit_resample(training_sample_train, markup_train)
            text_over_sample = '\nSMOTE'

        if ui_cls.checkBox_adasyn.isChecked():
            adasyn = ADASYN(random_state=0)
            training_sample_train, markup_train = adasyn.fit_resample(training_sample_train, markup_train)
            text_over_sample = '\nADASYN'

        if ui_cls.checkBox_pca.isChecked():
            n_comp = 'mle' if ui_cls.checkBox_pca_mle.isChecked() else ui_cls.spinBox_pca.value()
            pca = PCA(n_components=n_comp, random_state=0)
            pipe_steps.append(('pca', pca))
        text_pca = f'\nPCA: n_components={n_comp}' if ui_cls.checkBox_pca.isChecked() else ''

        if ui_cls.checkBox_stack_vote.isChecked():
            model_class, text_model, model_name = build_stacking_voting_model()
        else:
            model_name = ui_cls.buttonGroup.checkedButton().text()
            model_class, text_model = choice_model_classifier(model_name)

        text_model += text_pca
        text_model += text_over_sample

        pipe_steps.append(('model', model_class))
        pipe = Pipeline(pipe_steps)

        pipe.fit(training_sample_train, markup_train)# Оценка точности на всей обучающей выборке
        train_accuracy = pipe.score(training_sample, markup)
        test_accuracy = pipe.score(training_sample_test, markup_test)

        train_time = datetime.datetime.now() - start_time

        text_model += f'\ntrain_accuracy: {round(train_accuracy, 4)}, test_accuracy: {round(test_accuracy, 4)}, \nвремя обучения: {train_time}'
        set_info(text_model, 'blue')
        preds_train = pipe.predict(training_sample)

        if (ui_cls.checkBox_stack_vote.isChecked() and ui_cls.buttonGroup_stack_vote.checkedButton().text() == 'Voting'
                and ui_cls.checkBox_voting_hard.isChecked()):
            hard_flag = True
        else:
            hard_flag = False
        if not hard_flag:
            preds_proba_train = pipe.predict_proba(training_sample)

            tsne = TSNE(n_components=2, perplexity=5, learning_rate=200, random_state=42)
            train_tsne = tsne.fit_transform(preds_proba_train)
            data_tsne = pd.DataFrame(train_tsne)
            data_tsne['mark'] = preds_train

        if ui_cls.checkBox_cross_val.isChecked():
            kf = StratifiedKFold(n_splits=ui_cls.spinBox_n_cross_val.value(), shuffle=True, random_state=0)

            # if ui_cls.checkBox_smote.isChecked():
            #     smote = SMOTE(random_state=0)
            #     training_sample, markup = smote.fit_resample(training_sample, markup)
            #
            # if ui_cls.checkBox_adasyn.isChecked():
            #     adasyn = ADASYN(random_state=0)
            #     training_sample, markup = adasyn.fit_resample(training_sample, markup)

            scores_cv = cross_val_score(pipe, training_sample, markup, cv=kf, n_jobs=-1)

        if model_name == 'RFC' or model_name == 'GBC' or model_name == 'DTC':
            imp_name_params, imp_params = [], []
            for n, i in enumerate(model_class.feature_importances_):
                if i >= np.mean(model_class.feature_importances_):
                    imp_name_params.append(list_param_mlp[n])
                    imp_params.append(i)


        (fig_train, axes) = plt.subplots(nrows=1, ncols=3)
        fig_train.set_size_inches(25, 10)

        if not hard_flag:
            sns.scatterplot(data=data_tsne, x=0, y=1, hue='mark', s=200, ax=axes[0])
            axes[0].grid()
            axes[0].xaxis.grid(True, "minor", linewidth=.25)
            axes[0].yaxis.grid(True, "minor", linewidth=.25)
            axes[0].set_title(f'Диаграмма рассеяния для канонических значений {model_name}\nдля обучающей выборки и тестовой выборки')
            if len(list_marker) == 2:
                # Вычисляем ROC-кривую и AUC
                preds_test = pipe.predict_proba(training_sample_test)[:, 0]
                fpr, tpr, thresholds = roc_curve(markup_test, preds_test, pos_label=list_marker[0])
                roc_auc = auc(fpr, tpr)

                # Строим ROC-кривую
                axes[1].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
                axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                axes[1].set_xlim([0.0, 1.0])
                axes[1].set_ylim([0.0, 1.05])
                axes[1].set_xlabel('False Positive Rate')
                axes[1].set_ylabel('True Positive Rate')
                axes[1].set_title('ROC-кривая')
                axes[1].legend(loc="lower right")

        title_graph = text_model
        if model_name == 'RFC':
            if not ui_cls.checkBox_rfc_ada.isChecked() or ui_cls.checkBox_rfc_extra.isChecked():
                title_graph += f'\nOOB score: {round(model_class.oob_score_, 7)}'

        if (model_name == 'RFC' or model_name == 'GBC' or model_name == 'DTC') and not ui_cls.checkBox_cross_val.isChecked():
            axes[2].bar(imp_name_params, imp_params)
            axes[2].set_xticklabels(imp_name_params, rotation=90)
            axes[2].set_title('Важность признаков')

        if ui_cls.checkBox_cross_val.isChecked():
            axes[2].bar(range(len(scores_cv)), scores_cv)
            axes[2].set_title('Кросс-валидация')
        fig_train.suptitle(title_graph)
        fig_train.tight_layout()
        fig_train.show()

        if not ui_cls.checkBox_save_model.isChecked():
            return
        result = QtWidgets.QMessageBox.question(
            MainWindow,
            'Сохранение модели',
            f'Сохранить модель {model_name}?',
            QtWidgets.QMessageBox.Yes,
            QtWidgets.QMessageBox.No)
        if result == QtWidgets.QMessageBox.Yes:
            # Сохранение модели в файл с помощью pickle
            path_model = f'models/expl_models/classifier/{model_name}_{round(test_accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y")}.pkl'
            with open(path_model, 'wb') as f:
                pickle.dump(pipe, f)

            new_trained_model = TrainedModelExploration(
                analysis_id=get_analysis_expl_id(),
                title=f'{model_name}_{round(test_accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y")}',
                path_model=path_model,
                list_params=json.dumps(list_param_mlp),
                comment=text_model
            )
            session.add(new_trained_model)
            session.commit()
            update_models_expl_list()
        else:
            pass




    def calc_lof():
        """ Расчет выбросов методом LOF """
        global data_pca, data_tsne, colors, factor_lof

        data_lof = data_train.copy()
        data_lof.drop(['x_coord', 'y_coord', 'title', 'target'], axis=1, inplace=True)

        scaler = StandardScaler()
        training_sample_lof = scaler.fit_transform(data_lof)
        n_LOF = ui_cls.spinBox_lof_neighbor.value()

        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
        data_tsne = tsne.fit_transform(training_sample_lof)

        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(training_sample_lof)

        colors, data_pca, data_tsne, factor_lof, label_lof = calc_lof_model(n_LOF, training_sample_lof)

        Form_LOF = QtWidgets.QDialog()
        ui_lof = Ui_LOF_form()
        ui_lof.setupUi(Form_LOF)
        Form_LOF.show()
        Form_LOF.setAttribute(QtCore.Qt.WA_DeleteOnClose)


        def set_title_lof_form(label_lof):
            ui_lof.label_title_window.setText('Расчет выбросов. Метод LOF (Locally Outlier Factor)\n'
                                              f'Выбросов: {label_lof.tolist().count(-1)} из {len(label_lof)}')

        set_title_lof_form(label_lof)
        ui_lof.spinBox_lof_n.setValue(n_LOF)

        # Визуализация
        draw_lof_tsne(data_tsne, ui_lof)
        draw_lof_pca(data_pca, ui_lof)
        draw_lof_bar(colors, factor_lof, label_lof, ui_lof)
        insert_list_samples(data_train, ui_lof.listWidget_samples, label_lof)
        insert_list_features(data_train, ui_lof.listWidget_features)


        def calc_lof_in_window():
            global data_pca, data_tsne, colors, factor_lof
            colors, data_pca, data_tsne, factor_lof, label_lof = calc_lof_model(ui_lof.spinBox_lof_n.value(), training_sample_lof)
            ui_lof.checkBox_samples.setChecked(False)
            draw_lof_tsne(data_tsne, ui_lof)
            draw_lof_pca(data_pca, ui_lof)
            draw_lof_bar(colors, factor_lof, label_lof, ui_lof)

            set_title_lof_form(label_lof)
            insert_list_samples(data_train, ui_lof.listWidget_samples, label_lof)
            insert_list_features(data_train, ui_lof.listWidget_features)


        def calc_clean_model():
            _, _, _, _, label_lof = calc_lof_model(ui_lof.spinBox_lof_n.value(), training_sample_lof)
            lof_index = [i for i, x in enumerate(label_lof) if x == -1]
            for ix in lof_index:
                prof_well = data_train['prof_well_index'][ix]
                prof_id, well_id, fake_id = int(prof_well.split('_')[0]), int(prof_well.split('_')[1]), int(prof_well.split('_')[2])
                old_list_fake = session.query(MarkupMLP.list_fake).filter(
                    MarkupMLP.analysis_id == get_MLP_id(),
                    MarkupMLP.profile_id == prof_id,
                    MarkupMLP.well_id == well_id
                ).first()[0]
                if old_list_fake:
                    new_list_fake = json.loads(old_list_fake)
                    new_list_fake.append(fake_id)
                else:
                    new_list_fake = [fake_id]
                session.query(MarkupMLP).filter(
                    MarkupMLP.analysis_id == get_MLP_id(),
                    MarkupMLP.profile_id == prof_id,
                    MarkupMLP.well_id == well_id
                ).update({'list_fake': json.dumps(new_list_fake)}, synchronize_session='fetch')
                session.commit()

            Classifier.close()
            Form_LOF.close()
            session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
            session.commit()
            build_table_train(False, 'mlp')
            # update_list_well_markup_mlp()
            # show_regression_form(data_train_clean, list_param)


        def draw_checkbox_samples():
            global data_pca, data_tsne, colors, factor_lof
            if ui_lof.checkBox_samples.isChecked():
                col = ui_lof.listWidget_features.currentItem().text()
                draw_hist_sample_feature(data_train, col, data_train[col][int(ui_lof.listWidget_samples.currentItem().text().split(') ')[0])], ui_lof)
                draw_lof_bar(colors, factor_lof, label_lof, ui_lof)
                draw_lof_pca(data_pca, ui_lof)
            else:
                draw_lof_tsne(data_tsne, ui_lof)
                draw_lof_bar(colors, factor_lof, label_lof, ui_lof)
                draw_lof_pca(data_pca, ui_lof)


        # ui_lof.spinBox_lof_n.valueChanged.connect(calc_lof_in_window)
        ui_lof.pushButton_clean_lof.clicked.connect(calc_clean_model)
        ui_lof.checkBox_samples.clicked.connect(draw_checkbox_samples)
        ui_lof.listWidget_samples.currentItemChanged.connect(draw_checkbox_samples)

        ui_lof.listWidget_features.currentItemChanged.connect(draw_checkbox_samples)
        ui_lof.pushButton_lof.clicked.connect(calc_lof_in_window)

        Form_LOF.exec_()


    def insert_list_samples(data, list_widget, label_lof):
        list_widget.clear()
        for i in data.index:
            list_widget.addItem(f'{i}) {data["title"][i]}')
            if label_lof[int(i)] == -1:
                list_widget.item(int(i)).setBackground(QBrush(QColor('red')))
        list_widget.setCurrentRow(0)


    def insert_list_features(data, list_widget):
        list_widget.clear()
        for col in data.columns:
            if col != 'title' and col != 'target' and col != 'x_coord' and col != 'y_coord':
                list_widget.addItem(col)
        list_widget.setCurrentRow(0)


    def draw_hist_sample_feature(data, feature, value_sample, ui_widget):
        clear_horizontalLayout(ui_widget.horizontalLayout_tsne)
        figure_tsne = plt.figure()
        canvas_tsne = FigureCanvas(figure_tsne)
        figure_tsne.clear()
        ui_widget.horizontalLayout_tsne.addWidget(canvas_tsne)
        sns.histplot(data, x=feature, bins=50)
        plt.axvline(value_sample, color='r', linestyle='dashed', linewidth=2)
        plt.grid()
        figure_tsne.suptitle(f't-SNE')
        figure_tsne.tight_layout()
        canvas_tsne.draw()


    def draw_lof_bar(colors, factor_lof, label_lof, ui_lof):
        clear_horizontalLayout(ui_lof.horizontalLayout_bar)
        figure_bar = plt.figure()
        canvas_bar = FigureCanvas(figure_bar)
        ui_lof.horizontalLayout_bar.addWidget(canvas_bar)
        plt.bar(range(len(label_lof)), factor_lof, color=colors)
        if ui_lof.checkBox_samples.isChecked():
            plt.axvline(int(ui_lof.listWidget_samples.currentItem().text().split(') ')[0]), color='green', linestyle='dashed', linewidth=2)
        figure_bar.suptitle(f'коэффициенты LOF')
        figure_bar.tight_layout()
        canvas_bar.show()


    def draw_lof_pca(data_pca, ui_lof):
        clear_horizontalLayout(ui_lof.horizontalLayout_pca)
        figure_pca = plt.figure()
        canvas_pca = FigureCanvas(figure_pca)
        figure_pca.clear()
        ui_lof.horizontalLayout_pca.addWidget(canvas_pca)
        sns.scatterplot(data=data_pca, x=0, y=1, hue='lof', s=100, palette={-1: 'red', 1: 'blue'})
        if ui_lof.checkBox_samples.isChecked():
            index_sample = int(ui_lof.listWidget_samples.currentItem().text().split(') ')[0])
            plt.axvline(data_pca[0][index_sample], color='green', linestyle='dashed', linewidth=2)
            plt.axhline(data_pca[1][index_sample], color='green', linestyle='dashed', linewidth=2)
        plt.grid()
        figure_pca.suptitle(f'PCA')
        figure_pca.tight_layout()
        canvas_pca.draw()


    def draw_lof_tsne(data_tsne, ui_lof):
        clear_horizontalLayout(ui_lof.horizontalLayout_tsne)
        figure_tsne = plt.figure()
        canvas_tsne = FigureCanvas(figure_tsne)
        figure_tsne.clear()
        ui_lof.horizontalLayout_tsne.addWidget(canvas_tsne)
        sns.scatterplot(data=data_tsne, x=0, y=1, hue='lof', s=100, palette={-1: 'red', 1: 'blue'})
        plt.grid()
        figure_tsne.suptitle(f't-SNE')
        figure_tsne.tight_layout()
        canvas_tsne.draw()


    def calc_lof_model(n_LOF, training_sample):
        global data_pca, data_tsne
        lof = LocalOutlierFactor(n_neighbors=n_LOF)
        label_lof = lof.fit_predict(training_sample)
        factor_lof = -lof.negative_outlier_factor_

        data_tsne_pd = pd.DataFrame(data_tsne)
        data_tsne_pd['lof'] = label_lof

        data_pca_pd = pd.DataFrame(data_pca)
        data_pca_pd['lof'] = label_lof

        colors = ['red' if label == -1 else 'blue' for label in label_lof]

        return colors, data_pca_pd, data_tsne_pd, factor_lof, label_lof


    def class_exit():
        Classifier.close()

    ui_cls.pushButton_random_search.clicked.connect(class_exit)
    ui_cls.pushButton_random_search.clicked.connect(push_random_search)
    ui_cls.pushButton_lof.clicked.connect(calc_lof)
    ui_cls.pushButton_calc.clicked.connect(calc_model_class)
    ui_cls.checkBox_rfc_extra.clicked.connect(push_checkbutton_extra)
    ui_cls.checkBox_rfc_ada.clicked.connect(push_checkbutton_ada)
    ui_cls.checkBox_smote.clicked.connect(push_checkbutton_smote)
    ui_cls.checkBox_adasyn.clicked.connect(push_checkbutton_adasyn)
    Classifier.exec_()


def show_interp_map():
    global form_prof
    start_time = datetime.datetime.now()
    data = pd.DataFrame()

    # список точек по сетке для интерполирования
    p_points = session.query(PointExploration).all()
    x = [p.x_coord for p in p_points]
    y = [p.y_coord for p in p_points]

    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    step_x = 50
    step_y = 50

    x_values = np.arange(x_min, x_max + step_x, step_x)
    y_values = np.arange(y_min, y_max + step_y, step_y)

    x_grid, y_grid = np.meshgrid(x_values, y_values)
    points_grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    data['x_coord'] = x_grid.ravel()
    data['y_coord'] = y_grid.ravel()

    x_grid, y_grid = list(x_grid.ravel()), list(y_grid.ravel())

    # точки для вариограмы
    points = session.query(PointExploration).filter_by(set_points_id=get_set_point_id()).all()
    if not points:
        return
    params = session.query(ParameterAnalysisExploration).filter_by(analysis_id=get_analysis_expl_id()).all()
    ui.progressBar.setMaximum(len(params))

    for index, el in enumerate(params):
        expl = session.query(Exploration).filter_by(id=el.param.exploration_id).first()
        ui.comboBox_expl.setCurrentText(f'{expl.title} id{expl.id}')
        update_list_set_point()
        points = session.query(PointExploration).filter_by(set_points_id=get_set_point_id()).all()

        value_points = []
        for i in points:
            p = session.query(ParameterAnalysisExploration).filter_by(id=el.id).first()
            value = session.query(ParameterPoint.value).filter_by(
                param_id=p.param.id,
                point_id=i.id
            ).first()[0]

            value_points.append(value)
        set_info(f'Обработка параметра {p.title}', 'blue')
        print(f'({index + 1}/{len(params)}) Обработка параметра {p.title}')
        x_list = [p.x_coord for p in points]
        y_list = [p.y_coord for p in points]

        x_array = np.array(x_list)
        y_array = np.array(y_list)
        coord = np.column_stack((x_array, y_array))

        # вариограма и кригинг
        variogram = Variogram(coordinates=coord, values=np.array(value_points), estimator='matheron', dist_func='euclidean',
                              bin_func='even', fit_sigma='linear', model='spherical', fit_method='trf')

        try:
            kriging = OrdinaryKriging(variogram=variogram, min_points=2, max_points=30, mode='exact')
            field = kriging.transform(np.array(x_grid), np.array(y_grid))
            x_grid = [x_i for inx, x_i in enumerate(x_grid) if not np.isnan(field[inx])]
            y_grid = [y_i for iny, y_i in enumerate(y_grid) if not np.isnan(field[iny])]
            list_nan = [i for i, val in enumerate(field) if np.isnan(val)]
            field = [i for i in field if not np.isnan(i)]
            for i in list_nan:
                data.drop(i, inplace=True)
            data[el.param.parameter] = field
            data.reset_index(drop=True, inplace=True)
        except LinAlgError:
            set_info(f"LinAlgError - {el.param}", 'red')
        ui.progressBar.setValue(index + 1)

    # Георадар
    geo_param = session.query(GeoParameterAnalysisExploration).filter_by(analysis_id=get_analysis_expl_id()).all()
    ui.progressBar.setMaximum(len(geo_param))
    if len(geo_param) > 0:
        profiles = session.query(Profile).filter(Profile.research_id == get_research_id()).all()
        x_prof, y_prof, form_prof = [], [], []

        for profile in profiles:
            x_prof += json.loads(profile.x_pulc)
            y_prof += json.loads(profile.y_pulc)
            if len(profile.formations) == 1:
                form_prof.append(profile.formations[0])
            else:
                Choose_Formation = QtWidgets.QDialog()
                ui_cf = Ui_FormationMAP()
                ui_cf.setupUi(Choose_Formation)
                Choose_Formation.show()
                Choose_Formation.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия
                set_info(f'Выберите пласт для "{profile.title}"', 'blue')
                for f in profile.formations:
                    ui_cf.listWidget_form_map.addItem(f'{f.title} id{f.id}')
                ui_cf.listWidget_form_map.setCurrentRow(0)

                def form_lda_ok():
                    global form_prof
                    f_id = ui_cf.listWidget_form_map.currentItem().text().split(" id")[-1]
                    form = session.query(Formation).filter(Formation.id == f_id).first()
                    form_prof.append(form)
                    Choose_Formation.close()

                ui_cf.pushButton_ok_form_map.clicked.connect(form_lda_ok)
                Choose_Formation.exec_()

        coord_geo = np.column_stack((np.array(x_prof[::5]), np.array(y_prof[::5])))
        for index, g in enumerate(geo_param):
            set_info(f'Обработка параметра {g.param}', 'blue')
            print(f'({index + 1}/{len(geo_param)}) Обработка параметра георадара {g.param}')
            list_value = []
            for f in form_prof:
                list_value += json.loads(getattr(f, g.param))

            variogram = Variogram(coordinates=coord_geo, values=list_value[::5], estimator='matheron',
                                  dist_func='euclidean', bin_func='even', fit_sigma='exp')
            try:
                kriging = OrdinaryKriging(variogram=variogram, min_points=2, max_points=20, mode='exact')
                field = kriging.transform(np.array(x_grid), np.array(y_grid))
                x_grid = [x_i for inx, x_i in enumerate(x_grid) if not np.isnan(field[inx])]
                y_grid = [y_i for iny, y_i in enumerate(y_grid) if not np.isnan(field[iny])]
                list_nan = [i for i, val in enumerate(field) if np.isnan(val)]
                field = [i for i in field if not np.isnan(i)]
                for i in list_nan:
                    data.drop(i, inplace=True)
                data[g.param] = field
                data.reset_index(drop=True, inplace=True)
            except LinAlgError:
                set_info(f"LinAlgError - {g.param}", 'red')
            ui.progressBar.setValue(index + 1)
    # data = data.dropna()
    data.to_excel('new_data.xlsx')
    train_time = datetime.datetime.now() - start_time
    print(train_time)

def draw_interp_map():

    pass




