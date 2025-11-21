import json

import matplotlib.pyplot as plt
import numpy as np
from mlxtend.cluster import Kmeans
from pandas.core.common import random_state
from scipy.cluster.vq import kmeans

from draw import draw_radarogram, draw_formation, draw_fill, draw_fake, plot_groups_with_smoothed_hull, \
    plot_graphs_by_group
from func import *
from build_table import *
from krige import draw_map
from random_param_reg import push_random_param_reg
from feature_selection import *


""" Структура модели PyTorch """
class RegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, dropout_rate, activation_fn):
        super(RegressionModel, self).__init__()

        layers = []
        current_input_dim = input_dim

        activation_dict = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU()
        }

        if activation_fn not in activation_dict:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

        activation_layer = activation_dict[activation_fn]

        for units in hidden_units:
            layers.append(nn.Linear(current_input_dim, units))
            layers.append(activation_layer)
            layers.append(nn.Dropout(dropout_rate))
            current_input_dim = units

        layers.append(nn.Linear(current_input_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()
        return self.model(x).squeeze(1).float()


def rename_model_reg():
    """Переименовать модель"""
    model = session.query(TrainedModelReg).filter_by(id=ui.listWidget_trained_model_reg.currentItem().data(
        Qt.UserRole)).first()
    RenameModel = QtWidgets.QDialog()
    ui_rnm = Ui_FormRenameModel()
    ui_rnm.setupUi(RenameModel)
    RenameModel.show()
    RenameModel.setAttribute(Qt.WA_DeleteOnClose)
    ui_rnm.lineEdit.setText(model.title)

    def rename_model():
        model.title = ui_rnm.lineEdit.text()
        session.commit()
        update_list_trained_models_regmod()
        RenameModel.close()

    ui_rnm.buttonBox.accepted.connect(rename_model)
    ui_rnm.buttonBox.rejected.connect(RenameModel.close)

    RenameModel.exec_()


def add_regression_model():
    """Добавить новый набор для обучения регрессионной модели"""
    if ui.lineEdit_string.text() == '':
        set_info('Введите название модели в поле ввода текста в верхней части главного окна', 'red')
        QMessageBox.critical(MainWindow, 'Ошибка', 'Введите название модели!')
        return
    new_reg_mod = AnalysisReg(title=ui.lineEdit_string.text())
    session.add(new_reg_mod)
    session.commit()
    update_list_reg()
    set_info(f'Добавлен новый анализ MLP - "{ui.lineEdit_string.text()}"', 'green')


def remove_reg():
    """Удалить анализ MLP"""
    mlp_title = get_regmod_title()
    result = QtWidgets.QMessageBox.question(ui.listWidget_well_mlp, 'Remove Regression Model',
                                            f'Вы уверены, что хотите удалить модель "{mlp_title}"?',
                                            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        session.query(ParameterReg).filter_by(analysis_id=get_regmod_id()).delete()
        session.query(MarkupReg).filter_by(analysis_id=get_regmod_id()).delete()
        session.query(AnalysisReg).filter_by(id=get_regmod_id()).delete()
        session.query(ExceptionReg).filter_by(analysis_id=get_regmod_id()).delete()
        for model in session.query(TrainedModelReg).filter_by(analysis_id=get_regmod_id()).all():
            os.remove(model.path_model)
            session.delete(model)
        session.commit()
        set_info(f'Удалена модель - "{mlp_title}"', 'green')
        update_list_reg()
    elif result == QtWidgets.QMessageBox.No:
        pass


def update_list_reg():
    """Обновить список наборов для обучения регрессионной модели"""
    ui.comboBox_regmod.clear()
    for i in session.query(AnalysisReg.id, AnalysisReg.title).order_by(AnalysisReg.title).all():
        ui.comboBox_regmod.addItem(f'{i.title} id{i.id}')
    update_list_well_markup_reg()
    update_list_param_regmod(db=True)
    update_list_trained_models_regmod()


def add_well_markup_reg():
    """Добавить новую обучающую скважину для обучения регрессионной модели"""
    analysis_id = get_regmod_id()
    well_id = get_well_id()
    profile_id = get_profile_id()
    formation_id = get_formation_id()

    if analysis_id and well_id and profile_id and formation_id:
        remove_all_param_geovel_reg()

        if ui.checkBox_profile_intersec.isChecked():
            x_prof = json.loads(session.query(Profile.x_pulc).filter(Profile.id == profile_id).first()[0])
            inter = session.query(Intersection).filter(Intersection.id == well_id).first()
            well_dist = ui.spinBox_well_dist_reg.value()
            start = inter.i_profile - well_dist if inter.i_profile - well_dist > 0 else 0
            stop = inter.i_profile + well_dist if inter.i_profile + well_dist < len(x_prof) else len(x_prof)
            list_measure = list(range(start, stop))
            new_markup_reg = MarkupReg(analysis_id=analysis_id, well_id=well_id, profile_id=profile_id,
                                       formation_id=formation_id, target_value=round(inter.temperature, 2),
                                       list_measure=json.dumps(list_measure), type_markup='intersection')
        else:
            target_value = ui.doubleSpinBox_target_val.value()
            well = session.query(Well).filter(Well.id == well_id).first()
            x_prof = json.loads(session.query(Profile.x_pulc).filter(Profile.id == profile_id).first()[0])
            y_prof = json.loads(session.query(Profile.y_pulc).filter(Profile.id == profile_id).first()[0])
            index, _ = closest_point(well.x_coord, well.y_coord, x_prof, y_prof)
            well_dist = ui.spinBox_well_dist_reg.value()
            start = index - well_dist if index - well_dist > 0 else 0
            stop = index + well_dist if index + well_dist < len(x_prof) else len(x_prof)
            list_measure = list(range(start, stop))
            new_markup_reg = MarkupReg(analysis_id=analysis_id, well_id=well_id, profile_id=profile_id,
                                       formation_id=formation_id, target_value=target_value,
                                       list_measure=json.dumps(list_measure))
        session.add(new_markup_reg)
        session.commit()
        set_info(f'Добавлена новая обучающая скважина для регрессионной модели - "{get_well_name()} со значенем '
                 f'{new_markup_reg.target_value}"', 'green')
        update_list_well_markup_reg()
    else:
        set_info('выбраны не все параметры - скважина/пересечение и пласт', 'red')
        QMessageBox.critical(MainWindow, 'Ошибка', 'выберите все параметры для добавления обучающей скважины!')


def add_all_well_markup_reg():
    """Добавить все скважины объекта для обучения регрессионной модели"""
    analysis_id = get_regmod_id()

    list_formation = get_list_formation()
    if not list_formation:
        return
    profiles = session.query(Profile).filter(Profile.research_id == get_research_id()).all()
    ui.progressBar.setMaximum(len(profiles))
    for np, p in enumerate(profiles):
        ui.progressBar.setValue(np + 1)
        wells = get_list_nearest_well(p.id)
        if wells:
            x_prof = json.loads(session.query(Profile.x_pulc).filter(Profile.id == p.id).first()[0])
            y_prof = json.loads(session.query(Profile.y_pulc).filter(Profile.id == p.id).first()[0])
            for w in wells:
                if session.query(MarkupReg).filter(MarkupReg.analysis_id == analysis_id,
                                                   MarkupReg.well_id == w[0].id,
                                                   MarkupReg.profile_id == p.id).count() > 0:
                    set_info(f'Скважина {w[0].name} на профиле {p.title} уже добавлена', 'red')
                    continue
                index, _ = closest_point(w[0].x_coord, w[0].y_coord, x_prof, y_prof)
                well_dist = ui.spinBox_well_dist_reg.value()
                start = index - well_dist if index - well_dist > 0 else 0
                stop = index + well_dist if index + well_dist < len(x_prof) else len(x_prof)
                list_measure = list(range(start, stop))
                new_markup_reg = MarkupReg(analysis_id=analysis_id, well_id=w[0].id, profile_id=p.id,
                                           formation_id=list_formation[np].split(' id')[-1], target_value=0,
                                           list_measure=json.dumps(list_measure))
                session.add(new_markup_reg)
                set_info(f'Добавлена новая обучающая скважина для регрессионной модели - "{w[0].name} - {p.title}"', 'green')
    session.commit()
    update_list_well_markup_reg()


def update_list_well_markup_reg():
    """Обновить список обучающих скважин"""
    ui.listWidget_well_regmod.clear()
    count_markup, count_measure, count_fake = 0, 0, 0
    for i in session.query(MarkupReg).filter(MarkupReg.analysis_id == get_regmod_id()).all():
        try:
            fake = len(json.loads(i.list_fake)) if i.list_fake else 0
            measure = len(json.loads(i.list_measure))
            if i.type_markup == 'intersection':
                try:
                    inter_name = session.query(Intersection.name).filter(Intersection.id == i.well_id).first()[0]
                except TypeError:
                    session.query(MarkupReg).filter(MarkupReg.id == i.id).delete()
                    session.commit()
                    set_info(f'Обучающая скважина удалена из-за отсутствия пересечения', 'red')
                    continue
                item = (f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | {inter_name.split("_")[0]} | '
                        f'{measure - fake} | {i.target_value} | id{i.id}')
            else:
                item = (f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | {i.well.name} | '
                        f'{measure - fake} | {i.target_value} | id{i.id}')
            ui.listWidget_well_regmod.addItem(item)
            count_markup += 1
            count_measure += measure - fake
            count_fake += fake
        except AttributeError:
            set_info(f'Параметр для профиля {i.profile.title} удален из-за отсутствия одного из параметров', 'red')
            session.delete(i)
            session.commit()
    if ui.checkBox_sort_reg_markup.isChecked():
        ui.listWidget_well_regmod.sortItems(Qt.AscendingOrder)
    ui.label_count_markup_reg.setText(f'<i><u>{count_markup}</u></i> обучающих скважин; <i><u>{count_measure}</u></i> '
                                      f'измерений; <i><u>{count_fake}</u></i> выбросов')
    update_list_param_regmod(db=True)
    update_list_param_reg_no_update()


def sorting_markup_regmod():
    if ui.checkBox_sort_reg_markup.isChecked():
        ui.listWidget_well_regmod.sortItems(Qt.AscendingOrder)
    else:
        update_list_well_markup_reg()


def remove_well_markup_reg():
    """ Удалить скважину """

    markup = session.query(MarkupReg).filter(MarkupReg.id == get_markup_regmod_id()).first()
    if not markup:
        return
    if markup.type_markup == 'intersection':
        skv_name = session.query(Intersection.name).filter(Intersection.id == markup.well_id).first()[0]
    else:
        skv_name = session.query(Well.name).filter(Well.id == markup.well_id).first()[0]
    prof_name = session.query(Profile.title).filter(Profile.id == markup.profile_id).first()[0]
    regmod_name = session.query(AnalysisReg.title).filter(AnalysisReg.id == markup.analysis_id).first()[0]
    result = QtWidgets.QMessageBox.question(ui.listWidget_well_regmod, 'Remove markup for regression model',
                                            f'Вы уверены, что хотите удалить скважину "{skv_name}" на '
                                            f'профиле "{prof_name}" из обучающей модели регрессионного анализа "{regmod_name}"?',
                                            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        session.delete(markup)
        session.commit()
        set_info(f'Удалена обучающая скважина для регрессионного анализа - "{ui.listWidget_well_regmod.currentItem().text()}"', 'green')
        update_list_well_markup_reg()
    elif result == QtWidgets.QMessageBox.No:
        pass


def choose_markup_regmod():
    """
        Функция выбора маркера регрессионной модели
        Выбирает маркер, на основе сохраненных данных из базы данных, и затем обновляет все соответствующие виджеты
        пользовательского интерфейса
    """

    # Получение информации о маркере из БД по его ID
    markup = session.query(MarkupReg).filter(MarkupReg.id == get_markup_regmod_id()).first()
    # Если ID маркера не найден в БД, то функция завершается
    if not markup:
        return

    # Установка соответствующих значений виджетов пользовательского интерфейса
    ui.comboBox_year_research.setCurrentText(markup.profile.research.date_research.strftime('%Y'))
    update_object()
    ui.comboBox_object.setCurrentText(f'{markup.profile.research.object.title} id{markup.profile.research.object_id}')
    update_research_combobox()
    ui.comboBox_research.setCurrentText(
        f'{markup.profile.research.date_research.strftime("%m.%Y")} id{markup.profile.research_id}')
    update_profile_combobox()
    count_measure = len(json.loads(session.query(Profile.signal).filter(Profile.id == markup.profile_id).first()[0]))
    ui.comboBox_profile.setCurrentText(f'{markup.profile.title} ({count_measure} измерений) id{markup.profile_id}')
    draw_radarogram()
    ui.comboBox_plast.setCurrentText(f'{markup.formation.title} id{markup.formation_id}')
    draw_formation()
    draw_intersection(markup.well_id) if markup.type_markup == 'intersection' else draw_well(markup.well_id)
    ui.doubleSpinBox_target_val.setValue(markup.target_value)
    list_measure = json.loads(markup.list_measure)  # Получение списка измерений
    list_fake = json.loads(markup.list_fake) if markup.list_fake else []  # Получение списка пропущенных измерений
    list_up = json.loads(markup.formation.layer_up.layer_line)  # Получение списка с верхними границами формации
    list_down = json.loads(markup.formation.layer_down.layer_line)  # Получение списка со снижными границами формации
    y_up = [list_up[i] for i in list_measure]  # Создание списка верхних границ для отображения
    y_down = [list_down[i] for i in list_measure]  # Создание списка нижних границ для отображения
    # Обновление маркера с конкретными данными о верхней и нижней границах и цветом
    draw_fill(list_measure, y_up, y_down, 'blue')
    draw_fake(list_fake, list_up, list_down)


def update_all_well_markup_reg():
    """ Обновляет количество измерений для всех скважин """

    for mrp in session.query(MarkupReg).filter(MarkupReg.analysis_id == get_regmod_id()).all():
        if mrp.type_markup == 'intersection':
            return

        x_prof = json.loads(session.query(Profile.x_pulc).filter(Profile.id == mrp.profile_id).first()[0])
        well_dist = ui.spinBox_well_dist_reg.value()
        form_id = get_formation_id()
        well = session.query(Well).filter(Well.id == mrp.well_id).first()
        y_prof = json.loads(session.query(Profile.y_pulc).filter(Profile.id == mrp.profile_id).first()[0])
        index, _ = closest_point(well.x_coord, well.y_coord, x_prof, y_prof)
        start = index - well_dist if index - well_dist > 0 else 0
        stop = index + well_dist if index + well_dist < len(x_prof) else len(x_prof)
        list_measure = list(range(start, stop))
        session.query(MarkupReg).filter(MarkupReg.id == mrp.id).update(
            {'list_measure': json.dumps(list_measure)},
            synchronize_session='fetch')
    session.commit()
    update_list_well_markup_reg()


def update_well_markup_reg():
    """ Обновить измерения для скважины объекта """

    markup = session.query(MarkupReg).filter(MarkupReg.id == get_markup_regmod_id()).first()
    if not markup:
        return
    x_prof = json.loads(session.query(Profile.x_pulc).filter(Profile.id == markup.profile_id).first()[0])
    target_value = ui.doubleSpinBox_target_val.value()
    well_dist = ui.spinBox_well_dist_reg.value()
    form_id = get_formation_id()
    if markup.type_markup == 'intersection':
        well = session.query(Intersection).filter(Intersection.id == markup.well_id).first()
        start = well.i_profile - well_dist if well.i_profile - well_dist > 0 else 0
        stop = well.i_profile + well_dist if well.i_profile + well_dist < len(x_prof) else len(x_prof)
    else:
        well = session.query(Well).filter(Well.id == markup.well_id).first()
        y_prof = json.loads(session.query(Profile.y_pulc).filter(Profile.id == markup.profile_id).first()[0])
        index, _ = closest_point(well.x_coord, well.y_coord, x_prof, y_prof)
        start = index - well_dist if index - well_dist > 0 else 0
        stop = index + well_dist if index + well_dist < len(x_prof) else len(x_prof)
    list_measure = list(range(start, stop))
    session.query(MarkupReg).filter(MarkupReg.id == get_markup_regmod_id()).update(
        {'target_value': target_value, 'list_measure': json.dumps(list_measure), 'formation_id': form_id},
        synchronize_session='fetch')
    session.commit()
    set_info(f'Изменена обучающая скважина для регрессионной модели - "{well.name}"', 'green')
    update_list_well_markup_reg()


def split_well_train_test():
    """ Разделение выборки на train и test методами clusters/greedy"""

    markups = session.query(MarkupReg).filter_by(analysis_id=get_regmod_id()).all()
    list_mkp_id = [mkp.id for mkp in markups]
    list_data = [[mkp.well.x_coord, mkp.well.y_coord, mkp.target_value] for mkp in markups]

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(list_data)
    if ui.radioButton_clusters.isChecked():
        kmeans = KMeans(n_clusters=5, random_state=42).fit(data_scaled)
        labels = kmeans.labels_

        test_ids, train_ids = [], []
        for label in np.unique(labels):
            cluster_indices = np.where(labels == label)[0]
            np.random.shuffle(cluster_indices)
            test_size = int(len(cluster_indices) * 0.2)
            test_cluster_indices = cluster_indices[:test_size]
            train_cluster_indices = cluster_indices[test_size:]

            test_ids.extend([list_mkp_id[i] for i in test_cluster_indices])
            train_ids.extend([list_mkp_id[i] for i in train_cluster_indices])
    elif ui.radioButton_object.isChecked():

        obj_id = get_object_id()
        train_ids = [mkp.id for mkp in markups if mkp.profile.research.object_id != obj_id]
        test_ids = [mkp.id for mkp in markups if mkp.profile.research.object_id == obj_id]


    else:
        def min_distance_to_selected(point, selected_points):
            if not selected_points:
                return np.inf
            distance = np.linalg.norm(point - data_scaled[selected_points], axis=1)
            return np.min(distance)

        selected_indices = [np.random.choice(range(len(data_scaled)))]

        while len(selected_indices) < len(data_scaled) * 0.2:
            distances = np.array([min_distance_to_selected(point, selected_indices) for point in data_scaled])
            next_point_index = np.argmax(distances)
            selected_indices.append(next_point_index)

        test_ids = [list_mkp_id[i] for i in selected_indices]
        train_ids = [list_mkp_id[i] for i in range(len(list_mkp_id)) if i not in selected_indices]

    x_coords = [mkp.well.x_coord for mkp in markups]
    y_coords = [mkp.well.y_coord for mkp in markups]
    values = [mkp.target_value for mkp in markups]


    test_x = [mkp.well.x_coord for mkp in markups if mkp.id in test_ids]
    test_y = [mkp.well.y_coord for mkp in markups if mkp.id in test_ids]
    test_values = [mkp.target_value for mkp in markups if mkp.id in test_ids]

    plt.figure(figsize=(15, 12))

    sc = plt.scatter(x_coords, y_coords, c=values, cmap='viridis', s=200, alpha=0.7, label='TRAIN')
    plt.scatter(test_x, test_y, c=test_values, cmap='viridis', s=200, edgecolors='red', linewidths=2.5, alpha=0.7, label='TEST')

    plt.colorbar(sc, label='value')

    plt.legend()
    plt.show()

    result = QtWidgets.QMessageBox.question(
        MainWindow,
        'Train/Test',
        f'Разделить выборку?',
        QtWidgets.QMessageBox.Yes,
        QtWidgets.QMessageBox.No)

    if result == QtWidgets.QMessageBox.Yes:
        if ui.lineEdit_string.text() == '':
            set_info('Введите название для разделении выборки', 'red')
            return
        old_regmod = session.query(AnalysisReg).filter_by(id=get_regmod_id()).first()
        new_regmod_train = AnalysisReg(title=f'{ui.lineEdit_string.text()}_train')
        new_regmod_test = AnalysisReg(title=f'{ui.lineEdit_string.text()}_test')
        session.add(new_regmod_train)
        session.add(new_regmod_test)
        session.commit()
        for old_markup in session.query(MarkupReg).filter_by(analysis_id=get_regmod_id()):
            new_markup = MarkupReg(
                analysis_id=new_regmod_test.id if old_markup.id in test_ids else new_regmod_train.id,
                well_id=old_markup.well_id,
                profile_id=old_markup.profile_id,
                formation_id=old_markup.formation_id,
                target_value=old_markup.target_value,
                list_measure=old_markup.list_measure,
                type_markup=old_markup.type_markup
            )
            session.add(new_markup)
        session.commit()
        update_list_reg()
        set_info(f'Выборка разделена на {ui.lineEdit_string.text()}_train и {ui.lineEdit_string.text()}_test', 'green')


def set_updata_false_reg():
    analysis = session.query(AnalysisReg).filter_by(id=get_regmod_id()).first()
    try:
        filepath = Path(analysis.data)
        if filepath.exists():
            filepath.unlink()
    except (OSError, TypeError):
        pass
    analysis.up_data = False
    session.commit()


def add_param_signal_reg():
    """ Добавление одного параметра Signal """
    set_updata_false_reg()
    param = ui.comboBox_signal_reg.currentText()
    if session.query(ParameterReg).filter_by(
            analysis_id=get_regmod_id(),
            parameter=param
    ).count() == 0:
        add_param_regmod(param)
        # update_list_param_regmod()
        set_color_button_updata_regmod()
        update_list_param_reg_no_update()
    else:
        set_info(f'Параметр {param} уже добавлен', 'red')


def add_param_crl_reg():
    """ Добавление одного параметра CRL """
    set_updata_false_reg()
    if session.query(ParameterReg).filter_by(
            analysis_id=get_regmod_id(),
            parameter='CRL'
    ).count() == 0:
        add_param_regmod('CRL')
        # update_list_param_mlp()
        set_color_button_updata_regmod()
        update_list_param_reg_no_update()
        set_info(f'Параметр CRL добавлен', 'green')
    else:
        set_info(f'Параметр CRL уже добавлен', 'red')


def add_param_crl_nf_reg():
    """ Добавление параметра CRL NF """
    set_updata_false_reg()
    if session.query(ParameterReg).filter_by(
            analysis_id=get_regmod_id(),
            parameter='CRL_NF'
    ).count() == 0:
        add_param_regmod('CRL_NF')
        # update_list_param_mlp()
        set_color_button_updata_regmod()
        set_info(f'Параметр CRL_NF добавлен', 'green')
        update_list_param_reg_no_update()
    else:
        set_info(f'Параметр CRL_NF уже добавлен', 'red')


def add_all_param_signal_reg():
    """ Добавление всех параметров Signal """
    set_updata_false_reg()
    list_param_signal = ['Signal_Abase', 'Signal_diff', 'Signal_At', 'Signal_Vt', 'Signal_Pht', 'Signal_Wt']
    for param in list_param_signal:
        if session.query(ParameterReg).filter_by(
                analysis_id=get_regmod_id(),
                parameter=param
        ).count() == 0:
            add_param_regmod(param)
        else:
            set_info(f'Параметр {param} уже добавлен', 'red')
    # update_list_param_regmod()
    set_color_button_updata_regmod()
    update_list_param_reg_no_update()


def add_param_geovel_reg():
    """ Добавление основных атрибутов """
    set_updata_false_reg()
    param = ui.comboBox_geovel_param_reg.currentText()
    if not param in list_all_additional_features + ['X', 'Y']:
        for m in session.query(MarkupReg).filter(MarkupReg.analysis_id == get_regmod_id()).all():
            if not session.query(literal_column(f'Formation.{param}')).filter(Formation.id == m.formation_id).first()[0]:
                set_info(f'Параметр {param} отсутствует для профиля {m.profile.title}', 'red')
                return
    if session.query(ParameterReg).filter_by(
            analysis_id=get_regmod_id(),
            parameter= param
    ).count() == 0:
        add_param_regmod(param)
        # update_list_param_regmod()
        set_color_button_updata_regmod()
        update_list_param_reg_no_update()
    else:
        set_info(f'Параметр {param} уже добавлен', 'red')


def add_all_param_geovel_reg():
    """ Добавление всех основных атрибутов """

    new_list_param = ['X', 'Y'] +list_param_geovel + list_all_additional_features
    for param in list_param_geovel:
        for m in session.query(MarkupReg).filter(MarkupReg.analysis_id == get_regmod_id()).all():
            if not session.query(literal_column(f'Formation.{param}')).filter(Formation.id == m.formation_id).first()[0]:
                if param in new_list_param:
                    set_info(f'Параметр {param} отсутствует для профиля {m.profile.title}', 'red')
                    new_list_param.remove(param)
    for param in new_list_param:
        if session.query(ParameterReg).filter(ParameterReg.analysis_id == get_regmod_id()).filter(
                ParameterReg.parameter == param).count() > 0:
            set_info(f'Параметр {param} уже добавлен', 'red')
            continue
        add_param_regmod(param)
    session.commit()
    set_updata_false_reg()
    # update_list_param_regmod()
    update_list_param_reg_no_update()
    set_color_button_updata_regmod()


def add_param_profile_reg():
    """ Добавление параметров PROF """
    set_updata_false_reg()
    param = ui.comboBox_prof_ftr_reg.currentText()
    if session.query(ParameterReg).filter_by(
            analysis_id=get_regmod_id(),
            parameter= param
    ).count() == 0:
        add_param_regmod(param)
        set_color_button_updata_regmod()
        update_list_param_reg_no_update()
    else:
        set_info(f'Параметр {param} уже добавлен', 'red')


def add_all_param_profile_reg():
    """ Добавление всех параметров PROF """
    set_updata_false_reg()
    for param in list_all_additional_features:
        if param in ['fractal_dim', 'hht_marg_spec_min']:
            continue
        if session.query(ParameterReg).filter(ParameterReg.analysis_id == get_regmod_id()).filter(
                ParameterReg.parameter == f'prof_{param}').count() > 0:
            set_info(f'Параметр "prof_{param}" уже добавлен', 'red')
            continue
        add_param_regmod(f'prof_{param}')
    set_color_button_updata_regmod()
    update_list_param_reg_no_update()


def add_param_distr_reg():
    """ Добавление параметров distr """

    for param in session.query(ParameterReg).filter(ParameterReg.analysis_id == get_regmod_id()).all():
        if param.parameter.startswith(f'distr_{ui.comboBox_atrib_distr_reg.currentText()}'):
            session.query(ParameterReg).filter_by(id=param.id).update({
                'parameter': f'distr_{ui.comboBox_atrib_distr_reg.currentText()}_{ui.spinBox_count_distr_reg.value()}'
            }, synchronize_session='fetch')
            session.commit()
            update_list_param_regmod()
            set_info(f'В параметры добавлены {ui.spinBox_count_distr_reg.value()} интервалов распределения по '
                     f'{ui.comboBox_atrib_distr_reg.currentText()}', 'green')
            return
    add_param_regmod('distr')
    set_updata_false_reg()
    update_list_param_reg_no_update()
    set_color_button_updata_regmod()
    set_info(f'В параметры добавлены {ui.spinBox_count_distr_reg.value()} интервалов распределения по '
             f'{ui.comboBox_atrib_distr_reg.currentText()}', 'green')


def add_param_sep_reg():
    """ Добавление параметров SEP """

    for param in session.query(ParameterReg).filter(ParameterReg.analysis_id == get_regmod_id()).all():
        if param.parameter.startswith(f'sep_{ui.comboBox_atrib_distr_reg.currentText()}'):
            session.query(ParameterReg).filter_by(id=param.id).update({
                'parameter': f'sep_{ui.comboBox_atrib_distr_reg.currentText()}_{ui.spinBox_count_distr_reg.value()}'
            }, synchronize_session='fetch')
            session.commit()
            update_list_param_regmod()
            set_info(f'В параметры добавлены средние значения разделения на {ui.spinBox_count_distr_reg.value()} интервалов по '
                     f'{ui.comboBox_atrib_distr_reg.currentText()}', 'green')
            return
    add_param_regmod('sep')
    set_updata_false_reg()
    # update_list_param_regmod()
    set_color_button_updata_regmod()
    update_list_param_reg_no_update()
    set_info(f'В параметры добавлены средние значения разделения на {ui.spinBox_count_distr_reg.value()} интервалов по '
             f'{ui.comboBox_atrib_distr_reg.currentText()}', 'green')


def add_all_param_distr_reg():
    """ Добавление всех параметров distr """

    list_distr = ['distr_Abase', 'distr_diff', 'distr_At', 'distr_Vt', 'distr_Pht', 'distr_Wt', 'distr_SigCRL',
                  'sep_Abase', 'sep_diff', 'sep_At', 'sep_Vt', 'sep_Pht', 'sep_Wt', 'sep_SigCRL']
    count = ui.spinBox_count_distr_reg.value()
    for param in session.query(ParameterReg).filter(ParameterReg.analysis_id == get_regmod_id()).all():
        if param.parameter.startswith('distr') or param.parameter.startswith('sep'):
            session.query(ParameterReg).filter_by(id=param.id).delete()
            session.commit()
    for distr_param in list_distr:
        new_param = f'{distr_param}_{count}'
        new_param_reg = ParameterReg(analysis_id=get_regmod_id(), parameter=new_param)
        session.add(new_param_reg)
    set_updata_false_reg()
    session.commit()
    # update_list_param_regmod()
    set_color_button_updata_regmod()
    update_list_param_reg_no_update()
    set_info(f'Добавлены все параметры распределения по {count} интервалам', 'green')


def add_param_mfcc_reg():
    """ Добавление параметров mfcc """

    for param in session.query(ParameterReg).filter(ParameterReg.analysis_id == get_regmod_id()).all():
        if param.parameter.startswith(f'mfcc_{ui.comboBox_atrib_mfcc_reg.currentText()}'):
            session.query(ParameterReg).filter_by(id=param.id).update({
                'parameter': f'mfcc_{ui.comboBox_atrib_mfcc_reg.currentText()}_{ui.spinBox_count_mfcc_reg.value()}'
            }, synchronize_session='fetch')
            session.commit()
            update_list_param_regmod()
            set_info(f'В параметры добавлены {ui.spinBox_count_mfcc_reg.value()} кепстральных коэффициентов '
                     f'{ui.comboBox_atrib_mfcc_reg.currentText()}', 'green')
            return
    add_param_regmod('mfcc')
    set_updata_false_reg()
    session.commit()
    # update_list_param_regmod()
    set_color_button_updata_regmod()
    update_list_param_reg_no_update()
    set_info(f'В параметры добавлены {ui.spinBox_count_mfcc_reg.value()} кепстральных коэффициентов '
             f'{ui.comboBox_atrib_mfcc_reg.currentText()}', 'green')


def add_all_param_mfcc_reg():
    """ Добавление всех параметров mfcc """

    list_mfcc = ['mfcc_Abase', 'mfcc_diff', 'mfcc_At', 'mfcc_Vt', 'mfcc_Pht', 'mfcc_Wt', 'mfcc_SigCRL']
    count = ui.spinBox_count_mfcc_reg.value()
    for param in session.query(ParameterReg).filter(ParameterReg.analysis_id == get_regmod_id()).all():
        if param.parameter.startswith('mfcc'):
            session.query(ParameterReg).filter_by(id=param.id).delete()
            session.commit()
    for mfcc_param in list_mfcc:
        new_param = f'{mfcc_param}_{count}'
        new_param_mlp = ParameterReg(analysis_id=get_regmod_id(), parameter=new_param)
        session.add(new_param_mlp)
    set_updata_false_reg()
    session.commit()
    # update_list_param_regmod()
    set_color_button_updata_regmod()
    update_list_param_reg_no_update()
    set_info(f'Добавлены коэффициенты mfcc по всем параметрам по {count} интервалам', 'green')


def add_predict_reg():
    """ Добавление параметра с предсказанями обученной модели"""

    try:
        predict = session.query(ProfileModelPrediction).filter_by(
            id=ui.listWidget_model_pred.currentItem().text().split(' id')[-1]
        ).first()
    except AttributeError:
        set_info('Выберите модель в Model Prediction', 'red')
        return
    param = f'model_{predict.type_model}_id{predict.model_id}'

    if session.query(ParameterReg).filter_by(analysis_id=get_regmod_id(), parameter=param).count() > 0:
        set_info(f'Параметр {param} уже добавлен', 'red')
        return
    else:
        new_param_reg = ParameterReg(analysis_id=get_regmod_id(), parameter=param)
        session.add(new_param_reg)
        set_updata_false_reg()
        session.commit()
        set_color_button_updata_regmod()
        update_list_param_reg_no_update()
        set_info(f'Добавлен параметр {param}', 'green')


def remove_param_geovel_reg():
    """ Удаление одного параметра """

    try:
        param = ui.listWidget_param_reg.currentItem().text().split(' ')[0]
    except AttributeError:
        set_info('Выберите параметр', 'red')
        return
    if param:
        if (param.startswith('distr') or param.startswith('sep') or param.startswith('mfcc') or
                param.startswith('Signal') or param.startswith('CRL')):
            for p in session.query(ParameterReg).filter(ParameterReg.analysis_id == get_regmod_id()).all():
                if p.parameter.startswith('_'.join(param.split('_')[:-1])):
                    session.query(ParameterReg).filter_by(id=p.id).delete()
                    session.commit()
        else:
            session.query(ParameterReg).filter_by(analysis_id=get_regmod_id(), parameter=param ).delete()
        session.commit()
        ui.listWidget_param_reg.takeItem(ui.listWidget_param_reg.currentRow())
        set_updata_false_reg()
        set_color_button_updata_regmod()
    update_list_param_reg_no_update()


def remove_all_param_geovel_reg():
    """ Удаление всех параметров """

    session.query(ParameterReg).filter_by(analysis_id=get_regmod_id()).delete()
    session.commit()
    set_updata_false_reg()
    update_list_param_regmod()
    update_list_param_reg_no_update()


def update_list_param_reg_no_update():
    """ Обновление списка параметров БЕЗ сбора таблицы """

    data = session.query(AnalysisReg.up_data).filter_by(id=get_regmod_id()).first()
    try:
        if data[0]:
            return
    except TypeError:
        return 
    list_param_reg = session.query(ParameterReg).filter_by(analysis_id=get_regmod_id()).all()
    list_param_reg.sort(key=lambda x: x.parameter)
    ui.listWidget_param_reg.clear()
    for param in list_param_reg:
        i_item = QListWidgetItem(f'{param.parameter}')
        if param.parameter.startswith('model'):
            model_id = int(param.parameter.split('_id')[-1])
            if param.parameter.startswith('model_reg'):
                model = session.query(TrainedModelReg).filter_by(id=model_id).first()
            else:
                model = session.query(TrainedModelClass).filter_by(id=model_id).first()
            i_item.setToolTip(model.title)
        ui.listWidget_param_reg.addItem(i_item)
        i_item.setBackground(QBrush(QColor('#FFFAD5')))


def update_list_param_regmod(db=False):
    """ Обновление списка параметров со сбором таблицы """

    try:
        data_train, list_param = build_table_train(db, 'regmod')
    except TypeError:
        return
    if data_train is None:
        return
    ui.listWidget_param_reg.clear()
    list_param_reg = data_train.columns.tolist()[2:]
    for param in list_param_reg:
        i_item = QListWidgetItem(param)

        if param.startswith('model'):
            model_id = int(param.split('_id')[-1])
            if param.startswith('model_reg'):
                model = session.query(TrainedModelReg).filter_by(id=model_id).first()
            else:
                model = session.query(TrainedModelClass).filter_by(id=model_id).first()
            i_item.setToolTip(model.title)

        ui.listWidget_param_reg.addItem(i_item)

    ui.label_count_param_regmod.setText(f'<i><u>{ui.listWidget_param_reg.count()}</u></i> параметров')
    update_list_trained_models_regmod()
    set_color_button_updata_regmod()
    update_line_edit_exception_reg()


def update_line_edit_exception_reg():
    """ Обновление строки исключений Exception для параметров """

    ui.lineEdit_signal_except_reg.clear()
    ui.lineEdit_crl_except_reg.clear()
    except_reg = session.query(ExceptionReg).filter_by(analysis_id=get_regmod_id()).first()
    if except_reg:
        ui.lineEdit_signal_except_reg.setText(except_reg.except_signal)
        ui.lineEdit_crl_except_reg.setText(except_reg.except_crl)


def set_color_button_updata_regmod():
    """ Изменение цвета кнопки UP DATA"""

    reg = session.query(AnalysisReg).filter(AnalysisReg.id == get_regmod_id()).first()
    btn_color = 'background-color: rgb(191, 255, 191);' if reg.up_data else 'background-color: rgb(255, 185, 185);'
    ui.pushButton_updata_regmod.setStyleSheet(btn_color)


def add_param_list_reg():
    """ Получить параметры из строки """

    analysis_id = get_regmod_id()
    session.query(ParameterReg).filter_by(analysis_id=get_regmod_id()).delete()
    set_updata_false_reg()
    session.commit()
    check_except = False
    for i in ui.lineEdit_string.text().split('//'):
        param = i.split('_')
        if param[0] == 'sig':
            if param[1] == 'CRL':
                if session.query(ParameterReg).filter_by(
                        analysis_id=analysis_id,
                        parameter='CRL'
                ).count() == 0:

                    new_param_reg = ParameterReg(analysis_id=get_regmod_id(), parameter='CRL')
                    session.add(new_param_reg)
                    session.commit()

                else:
                    set_info(f'Параметр CRL уже добавлен', 'red')
            elif param[1] == 'CRLNF':
                if session.query(ParameterReg).filter_by(
                        analysis_id=analysis_id,
                        parameter='CRL_NF'
                ).count() == 0:
                    new_param_reg = ParameterReg(analysis_id=get_regmod_id(), parameter='CRL_NF')
                    session.add(new_param_reg)
                    session.commit()
                else:
                    set_info(f'Параметр CRL_NF уже добавлен', 'red')
            else:
                if session.query(ParameterReg).filter_by(
                        analysis_id=analysis_id,
                        parameter=f'Signal_{param[1]}'
                ).count() == 0:
                    new_param_reg = ParameterReg(analysis_id=get_regmod_id(), parameter=f'Signal_{param[1]}')
                    session.add(new_param_reg)
                    session.commit()
                else:
                    set_info(f'Параметр Signal_{param[1]} уже добавлен', 'red')
            if not check_except:
                str_exeption = f'1-{param[2]},{f"{str(512 - int(param[3]))}-512" if int(param[3]) > 0  else ""}'
                session.query(ExceptionReg).filter_by(analysis_id=analysis_id).update({'except_signal': str_exeption,
                                                                              'except_crl': str_exeption},
                                                                             synchronize_session='fetch')
                session.commit()
                check_except = True
        elif param[0] in ['distr', 'sep', 'mfcc']:
            if param[1] == 'CRL':
                new_param_mlp = ParameterReg(analysis_id=get_regmod_id(), parameter=f'{param[0]}_SigCRL_{param[2]}')
            else:
                new_param_mlp = ParameterReg(analysis_id=get_regmod_id(), parameter=i)
            session.add(new_param_mlp)
            session.commit()
        else:
            if session.query(ParameterReg).filter_by(
                    analysis_id=analysis_id,
                    parameter=i
            ).count() == 0:
                new_param_reg = ParameterReg(analysis_id=get_regmod_id(), parameter=i)
                session.add(new_param_reg)
                session.commit()
            else:
                set_info(f'Параметр {i} уже добавлен', 'red')
    set_color_button_updata_regmod()
    update_line_edit_exception_reg()
    update_list_param_reg_no_update()

#
# def str_to_interval(string):
#     if string == '':
#         return ''
#     parts = string.split("-")
#     result = [float(part.replace(",", ".")) for part in parts]
#     return result

def train_regression_model():
    """ Расчет регрессионной модели """
    data_train, list_param_name = build_table_train(True, 'regmod')
    list_param_reg = get_list_param_numerical_for_train(list_param_name)
    list_nan_param, count_nan = set(), 0
    # for i in data_train.index:
    #     for param in list_param_reg:
    #         if pd.isna(data_train[param][i]):
    #             count_nan += 1
    #             list_nan_param.add(param)
    #         if data_train[param][i] == np.inf or data_train[param][i] == -np.inf:
    #             data_train[param][i] = 0
    #             count_nan += 1
    #             list_nan_param.add(param)
    # Используем vectorized операции вместо циклов
    nan_mask = data_train[list_param_reg].isna()
    inf_mask = np.isinf(data_train[list_param_reg])

    # Подсчет NaN и inf значений
    count_nan = nan_mask.sum().sum() + inf_mask.sum().sum()

    # Добавление параметров с NaN или inf в set
    list_nan_param = set(nan_mask.columns[nan_mask.any() | inf_mask.any()])

    # Замена inf на 0
    data_train[list_param_reg] = data_train[list_param_reg].replace([np.inf, -np.inf], 0)

    if count_nan > 0:
        list_col = data_train.columns.tolist()
        data_train = pd.DataFrame(imputer.fit_transform(data_train), columns=list_col)

        analysis_data = session.query(AnalysisReg).filter_by(id=get_regmod_id()).first()
        try:
            data_train.to_parquet(analysis_data.data)
        except OSError:
            p_sep = os.path.sep
            name = f'{analysis_data.title}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
            filepath = f'data_tables{p_sep}reg{p_sep}{name}.parquet'
            data_train.to_parquet(filepath)
            session.query(AnalysisReg).filter_by(id=analysis_data.id).update({'data': str(filepath)}, synchronize_session='fetch')
            session.commit()

        set_info(f'Заполнены пропуски в {count_nan} параметрах {", ".join(list_nan_param)}', 'red')

    training_sample = data_train[list_param_reg].values.tolist()
    target = sum(data_train[['target_value']].values.tolist(), [])

    Regressor = QtWidgets.QDialog()
    ui_r = Ui_RegressorForm()
    ui_r.setupUi(Regressor)
    Regressor.show()
    Regressor.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    ui_r.spinBox_pca.setMaximum(len(list_param_reg))
    ui_r.spinBox_pca.setValue(len(list_param_reg) // 2)

    def update_list_saved_mask():
        ui_r.listWidget_mask_param.clear()
        for i in session.query(ParameterMask).all():
            item = QListWidgetItem(f'{i.count_param} id{i.id}')
            item.setToolTip(i.mask_info)
            ui_r.listWidget_mask_param.addItem(item)

    update_list_saved_mask()

    def build_torch_model(training_sample_train):
        """ Сбор модели PyTorch """

        output_dim = 1

        epochs = ui_r.spinBox_epochs_torch.value()
        learning_rate = ui_r.doubleSpinBox_lr_torch.value()
        hidden_units = list(map(int, ui_r.lineEdit_layers_torch.text().split()))
        dropout_rate = ui_r.doubleSpinBox_dropout_torch.value()
        weight_decay = ui_r.doubleSpinBox_decay_torch.value()

        if ui_r.comboBox_activation_torch.currentText() == 'ReLU':
            activation_function = 'relu'
        elif ui_r.comboBox_activation_torch.currentText() == 'Sigmoid':
            activation_function = 'sigmoid'
        elif ui_r.comboBox_activation_torch.currentText() == 'Tanh':
            activation_function = 'tanh'

        if ui_r.comboBox_optimizer_torch.currentText() == 'Adam':
            optimizer = torch.optim.Adam
        elif ui_r.comboBox_optimizer_torch.currentText() == 'SGD':
            optimizer = torch.optim.SGD

        if ui_r.comboBox_loss_torch.currentText() == 'MSE':
            loss_function = nn.MSELoss
        elif ui_r.comboBox_loss_torch.currentText() == 'MAE':
            loss_function = nn.L1Loss
        elif ui_r.comboBox_loss_torch.currentText() == 'HuberLoss':
            loss_function = nn.HuberLoss
        elif ui_r.comboBox_loss_torch.currentText() == 'SmoothL1Loss':
            loss_function = nn.SmoothL1Loss

        patience = 0
        early_stopping_flag = False
        if ui_r.checkBox_estop_torch.isChecked():
            early_stopping_flag = True
            patience = ui_r.spinBox_stop_patience.value()

        early_stopping = EarlyStopping(
            monitor='valid_loss',
            patience=patience,
            threshold=1e-4,
            threshold_mode='rel',
            lower_is_better=True,
        )
        if ui_r.checkBox_pca.isChecked():
            pca = PCA(n_components=ui_r.spinBox_pca.value())
            training_sample_train = pca.fit_transform(training_sample_train)

        model = RegressionModel(np.array(training_sample_train).shape[1], output_dim, hidden_units, dropout_rate,
                                activation_function)

        model_class = NeuralNetRegressor(
            model,
            max_epochs=epochs,
            lr=learning_rate,
            optimizer=optimizer,
            criterion=loss_function,
            optimizer__weight_decay=weight_decay,
            iterator_train__batch_size=32,
            callbacks=[early_stopping] if early_stopping_flag else None,
            train_split=ValidSplit(cv=5),
            verbose=0
        )

        text_model = '*** TORCH NN *** \n' + 'learning_rate: ' + str(learning_rate) + '\nhidden units: ' + str(
            hidden_units) + '\nweight decay: ' + str(weight_decay) + '\ndropout rate: ' + str(dropout_rate) + \
                     '\nactivation_func: ' + activation_function + '\noptimizer: ' + \
                     ui_r.comboBox_optimizer_torch.currentText() + '\ncriterion: ' + \
                     ui_r.comboBox_loss_torch.currentText() + '\nepochs: ' + str(epochs)

        return model_class, text_model


    def choice_model_regressor(model, x_train):
        """ Выбор модели регрессии """
        if model == 'MLPR':
            model_reg = MLPRegressor(
                hidden_layer_sizes=tuple(map(int, ui_r.lineEdit_layer_mlp.text().split())),
                activation=ui_r.comboBox_activation_mlp.currentText(),
                solver=ui_r.comboBox_solvar_mlp.currentText(),
                alpha=ui_r.doubleSpinBox_alpha_mlp.value(),
                learning_rate_init=ui_r.doubleSpinBox_lr_mlp.value(),
                max_iter=5000,
                early_stopping=ui_r.checkBox_e_stop_mlp.isChecked(),
                validation_fraction=ui_r.doubleSpinBox_valid_mlp.value(),
                random_state=0
            )
            text_model = (
                f'**MLPR**: \n'
                f'hidden_layer_sizes: ({",".join(map(str, tuple(map(int, ui_r.lineEdit_layer_mlp.text().split()))))}), '
                f'\nactivation: {ui_r.comboBox_activation_mlp.currentText()}, '
                f'\nsolver: {ui_r.comboBox_solvar_mlp.currentText()}, '
                f'\nalpha: {round(ui_r.doubleSpinBox_alpha_mlp.value(), 2)}, '
                f'\nlearning_rate: {round(ui_r.doubleSpinBox_lr_mlp.value(), 2)}'
                f'\n{"early stopping, " if ui_r.checkBox_e_stop_mlp.isChecked() else ""}'
                f'\nvalidation_fraction: {round(ui_r.doubleSpinBox_valid_mlp.value(), 2)}'
            )

        elif model == 'KNNR':
            n_knn = ui_r.spinBox_neighbors.value()
            weights_knn = 'distance' if ui_r.checkBox_knn_weights.isChecked() else 'uniform'
            model_reg = KNeighborsRegressor(n_neighbors=n_knn, weights=weights_knn, algorithm='auto')
            text_model = f'**KNNR**: \nn_neighbors: {n_knn}, \nweights: {weights_knn}, '

        elif model == 'GBR':
            est = ui_r.spinBox_n_estimators_gbr.value()
            l_rate = ui_r.doubleSpinBox_learning_rate_gbr.value()
            model_reg = GradientBoostingRegressor(n_estimators=est,
                                                  learning_rate=l_rate,
                                                  max_depth=ui_r.spinBox_depth_gbr.value(),
                                                  min_samples_split=ui_r.spinBox_min_sample_split_gbr.value(),
                                                  min_samples_leaf=ui_r.spinBox_min_sample_leaf_gbr.value(),
                                                  subsample=ui_r.doubleSpinBox_subsample_gbr.value(),
                                                  random_state=0)
            text_model = f'**GBR**: \nn estimators: {round(est, 2)}, \nlearning rate: {round(l_rate, 2)}, ' \
                         f'max_depth: {ui_r.spinBox_depth_gbr.value()}, \nmin_samples_split: ' \
                         f'{ui_r.spinBox_min_sample_split_gbr.value()}, \nmin_samples_leaf: ' \
                         f'{ui_r.spinBox_min_sample_leaf_gbr.value()} \nsubsample: ' \
                         f'{round(ui_r.doubleSpinBox_subsample_gbr.value(), 2)}, '

        elif model == 'LR':
            model_reg = LinearRegression(fit_intercept=ui_r.checkBox_fit_intercept.isChecked())
            text_model = f'**LR**: \nfit_intercept: {"on" if ui_r.checkBox_fit_intercept.isChecked() else "off"}, '

        elif model == 'DTR':
            spl = 'random' if ui_r.checkBox_splitter_rnd.isChecked() else 'best'
            model_reg = DecisionTreeRegressor(splitter=spl, random_state=0)
            text_model = f'**DTR**: \nsplitter: {spl}, '

        elif model == 'RFR':
            model_reg = RandomForestRegressor(n_estimators=ui_r.spinBox_rfr_n.value(),
                                              max_depth=ui_r.spinBox_depth_rfr.value(),
                                              min_samples_split=ui_r.spinBox_min_sample_split.value(),
                                              min_samples_leaf=ui_r.spinBox_min_sample_leaf.value(),
                                              max_features=ui_r.comboBox_max_features_rfr.currentText(),
                                              oob_score=True, random_state=0, n_jobs=-1)
            text_model = f'**RFR**: \nn estimators: {ui_r.spinBox_rfr_n.value()}, ' \
                         f'\nmax_depth: {ui_r.spinBox_depth_rfr.value()},' \
                         f'\nmin_samples_split: {ui_r.spinBox_min_sample_split.value()}, ' \
                         f'\nmin_samples_leaf: {ui_r.spinBox_min_sample_leaf.value()}, ' \
                         f'\nmax_features: {ui_r.comboBox_max_features_rfr.currentText()}, '

        elif model == 'ABR':
            model_reg = AdaBoostRegressor(n_estimators=ui_r.spinBox_rfr_n.value(), random_state=0)
            text_model = f'**ABR**: \nn estimators: {ui_r.spinBox_rfr_n.value()}, '

        elif model == 'ETR':
            model_reg = ExtraTreesRegressor(n_estimators=ui_r.spinBox_rfr_n.value(), bootstrap=True, oob_score=True, random_state=0, n_jobs=-1)
            text_model = f'**ETR**: \nn estimators: {ui_r.spinBox_rfr_n.value()}, '

        elif model == 'GPR':
            constant = ui_r.doubleSpinBox_gpr_const.value()
            scale = ui_r.doubleSpinBox_gpr_scale.value()
            n_restart_optimization = ui_r.spinBox_gpr_n_restart.value()
            kernel = ConstantKernel(constant) * RBF(scale)
            model_reg = GaussianProcessRegressor(
                kernel=kernel,
                alpha=ui_r.doubleSpinBox_gpr_alpha.value(),
                n_restarts_optimizer=n_restart_optimization,
                random_state=0
                )
            text_model = (f'**GPR**: '
                          f'\nkernal: {kernel}, '
                          f'\nscale: {round(scale, 3)}, '
                          f'\nconstant: {round(constant, 3)}, '
                          f'\nn restart: {n_restart_optimization} ,')

        elif model == 'SVR':
            model_reg = SVR(kernel=ui_r.comboBox_svr_kernel.currentText(), C=ui_r.doubleSpinBox_svr_c.value(),
                            epsilon=ui_r.doubleSpinBox_epsilon_svr.value())
            text_model = (f'**SVR**: \nkernel: {ui_r.comboBox_svr_kernel.currentText()}, '
                          f'\nC: {round(ui_r.doubleSpinBox_svr_c.value(), 2)}, '
                          f'\nepsilon: {round(ui_r.doubleSpinBox_epsilon_svr.value(), 2)}')

        elif model == 'EN':
            model_reg = ElasticNet(
                alpha=ui_r.doubleSpinBox_alpha.value(),
                l1_ratio=ui_r.doubleSpinBox_l1_ratio.value(),
                random_state=0
            )
            text_model = (f'**EN**: \nalpha: {round(ui_r.doubleSpinBox_alpha.value(), 2)}, '
                          f'\nl1_ratio: {round(ui_r.doubleSpinBox_l1_ratio.value(), 2)}, ')

        elif model == 'LSS':
            model_reg = Lasso(alpha=ui_r.doubleSpinBox_alpha.value(), random_state=0)
            text_model = f'**LSS**: \nalpha: {round(ui_r.doubleSpinBox_alpha.value(), 2)}, '

        elif model == 'XGB':
            model_reg = XGBRegressor(n_estimators=ui_r.spinBox_n_estimators_xgb.value(),
                                     learning_rate=ui_r.doubleSpinBox_learning_rate_xgb.value(),
                                     max_depth=ui_r.spinBox_depth_xgb.value(),
                                     alpha=ui_r.doubleSpinBox_alpha_xgb.value(), booster='gbtree', random_state=0)
            text_model = f'**XGB**: \nn estimators: {ui_r.spinBox_n_estimators_xgb.value()}, ' \
                         f'\nlearning_rate: {ui_r.doubleSpinBox_learning_rate_xgb.value()}, ' \
                         f'\nmax_depth: {ui_r.spinBox_depth_xgb.value()} \nalpha: {ui_r.doubleSpinBox_alpha_xgb.value()}'

        elif model == 'LGBM':
            model_reg = lgb.LGBMRegressor(
                objective='regression',
                verbosity=-1,
                boosting_type='gbdt',
                reg_alpha=ui_r.doubleSpinBox_l1_lgbm.value(),
                reg_lambda=ui_r.doubleSpinBox_l2_lgbm.value(),
                num_leaves=ui_r.spinBox_lgbm_num_leaves.value(),
                colsample_bytree=ui_r.doubleSpinBox_lgbm_feature.value(),
                subsample=ui_r.doubleSpinBox_lgbm_subsample.value(),
                subsample_freq=ui_r.spinBox_lgbm_sub_freq.value(),
                min_child_samples=ui_r.spinBox_lgbm_child.value(),
                learning_rate=ui_r.doubleSpinBox_lr_lgbm.value(),
                n_estimators=ui_r.spinBox_estim_lgbm.value(),
            )

            text_model = f'**LGBM**: \nlambda_1: {ui_r.doubleSpinBox_l1_lgbm.value()}, ' \
                         f'\nlambda_2: {ui_r.doubleSpinBox_l2_lgbm.value()}, ' \
                         f'\nnum_leaves: {ui_r.spinBox_lgbm_num_leaves.value()}, ' \
                         f'\nfeature_fraction: {ui_r.doubleSpinBox_lgbm_feature.value()}, ' \
                         f'\nsubsample: {ui_r.doubleSpinBox_lgbm_subsample.value()}, ' \
                         f'\nsubsample_freq: {ui_r.spinBox_lgbm_sub_freq.value()}, ' \
                         f'\nmin_child_samples: {ui_r.spinBox_lgbm_child.value()}, ' \
                         f'\nlearning_rate: {ui_r.doubleSpinBox_lr_lgbm.value()}, ' \
                         f'\nn_estimators: {ui_r.spinBox_estim_lgbm.value()}'

        elif model == 'TORCH':
            model_reg, text_model = build_torch_model(x_train)

        else:
            model_reg = QuadraticDiscriminantAnalysis()
            text_model = ''

        return model_reg, text_model


    def build_stacking_voting_model():
        """ Построить модель стекинга """
        estimators, list_model = [], []

        if ui_r.checkBox_stv_mlpr.isChecked():
            mlpr = MLPRegressor(
                hidden_layer_sizes=tuple(map(int, ui_r.lineEdit_layer_mlp.text().split())),
                activation=ui_r.comboBox_activation_mlp.currentText(),
                solver=ui_r.comboBox_solvar_mlp.currentText(),
                alpha=ui_r.doubleSpinBox_alpha_mlp.value(),
                max_iter=5000,
                early_stopping=ui_r.checkBox_e_stop_mlp.isChecked(),
                validation_fraction=ui_r.doubleSpinBox_valid_mlp.value(),
                random_state=0
            )
            estimators.append(('mlpr', mlpr))
            list_model.append('mlpr')

        if ui_r.checkBox_stv_knnr.isChecked():
            n_knn = ui_r.spinBox_neighbors.value()
            weights_knn = 'distance' if ui_r.checkBox_knn_weights.isChecked() else 'uniform'
            knnr = KNeighborsRegressor(n_neighbors=n_knn, weights=weights_knn, algorithm='auto')
            estimators.append(('knnr', knnr))
            list_model.append('knnr')

        if ui_r.checkBox_stv_gbr.isChecked():
            est = ui_r.spinBox_n_estimators.value()
            l_rate = ui_r.doubleSpinBox_learning_rate.value()
            gbr = GradientBoostingRegressor(n_estimators=est, learning_rate=l_rate, random_state=0)
            estimators.append(('gbr', gbr))
            list_model.append('gbr')

        if ui_r.checkBox_stv_dtr.isChecked():
            spl = 'random' if ui_r.checkBox_splitter_rnd.isChecked() else 'best'
            dtr = DecisionTreeRegressor(splitter=spl, random_state=0)
            estimators.append(('dtr', dtr))
            list_model.append('dtr')

        if ui_r.checkBox_stv_rfr.isChecked():
            rfr = RandomForestRegressor(n_estimators=ui_r.spinBox_rfr_n.value(), oob_score=True, random_state=0, n_jobs=-1)
            estimators.append(('rfr', rfr))
            list_model.append('rfr')

        if ui_r.checkBox_stv_abr.isChecked():
            abr = AdaBoostRegressor(n_estimators=ui_r.spinBox_rfr_n.value(), random_state=0)
            estimators.append(('abr', abr))

        if ui_r.checkBox_stv_etr.isChecked():
            etr = ExtraTreesRegressor(n_estimators=ui_r.spinBox_rfr_n.value(), bootstrap=True, oob_score=True, random_state=0, n_jobs=-1)
            estimators.append(('etr', etr))
            list_model.append('etr')

        if ui_r.checkBox_stv_gpr.isChecked():
            gpc_kernel_width = ui_r.doubleSpinBox_gpc_wigth.value()
            gpc_kernel_scale = ui_r.doubleSpinBox_gpc_scale.value()
            n_restart_optimization = ui_r.spinBox_gpc_n_restart.value()
            kernel = gpc_kernel_scale * RBF(gpc_kernel_width)
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=n_restart_optimization,
                random_state=0
            )
            estimators.append(('gpr', gpr))
            list_model.append('gpr')

        if ui_r.checkBox_stv_svr.isChecked():
            svr = SVR(kernel=ui_r.comboBox_svr_kernel.currentText(),
                      C=ui_r.doubleSpinBox_svr_c.value())
            estimators.append(('svr', svr))
            list_model.append('svr')

        if ui_r.checkBox_stv_lr.isChecked():
            lr = LinearRegression(fit_intercept=ui_r.checkBox_fit_intercept.isChecked())
            estimators.append(('lr', lr))
            list_model.append('lr')

        if ui_r.checkBox_stv_en.isChecked():
            en = ElasticNet(
                alpha=ui_r.doubleSpinBox_alpha.value(),
                l1_ratio=ui_r.doubleSpinBox_l1_ratio.value(),
                random_state=0
            )
            estimators.append(('en', en))
            list_model.append('en')

        if ui_r.checkBox_stv_lasso.isChecked():
            lss = Lasso(alpha=ui_r.doubleSpinBox_alpha.value(), random_state=0)
            estimators.append(('lss', lss))
            list_model.append('lss')

        final_model, final_text_model = choice_model_regressor(ui_r.buttonGroup.checkedButton().text())
        list_model_text = ', '.join(list_model)

        if ui_r.buttonGroup_stack_vote.checkedButton().text() == 'Voting':
            model_class = VotingRegressor(estimators=estimators, n_jobs=-1)
            text_model = f'**Voting**: \n({list_model_text})\n'
            model_name = 'VOT'
        else:
            model_class = StackingRegressor(estimators=estimators, final_estimator=final_model, n_jobs=-1)
            text_model = f'**Stacking**:\nFinal estimator: {final_text_model}\n({list_model_text})\n'
            model_name = 'STACK'
        return model_class, text_model, model_name

    def add_model_reg_to_lineup():
        """ Добавить модель в LineUp """

        scaler = StandardScaler()

        pipe_steps = []
        pipe_steps.append(('scaler', scaler))

        if ui_r.checkBox_pca.isChecked():
            n_comp = 'mle' if ui_r.checkBox_pca_mle.isChecked() else ui_r.spinBox_pca.value()
            pca = PCA(n_components=n_comp, random_state=0)
            pipe_steps.append(('pca', pca))
        text_pca = f'\nPCA: n_components={n_comp}' if ui_r.checkBox_pca.isChecked() else ''

        if ui_r.checkBox_stack_vote.isChecked():
            model_class, text_model, model_name = build_stacking_voting_model()
        else:
            model_name = ui_r.buttonGroup.checkedButton().text()
            model_class, text_model = choice_model_regressor(model_name)

        if ui_r.checkBox_baggig.isChecked():
            model_class = BaggingRegressor(base_estimator=model_class, n_estimators=ui_r.spinBox_bagging.value(),
                                           random_state=0, n_jobs=-1)
        bagging_text = f'\nBagging: n_estimators={ui_r.spinBox_bagging.value()}' if ui_r.checkBox_baggig.isChecked() else ''

        text_model += text_pca
        text_model += bagging_text

        pipe_steps.append(('model', model_class))
        pipe = Pipeline(pipe_steps)

        except_reg = session.query(ExceptionReg).filter_by(analysis_id=get_regmod_id()).first()

        new_lineup = LineupTrain(
            type_ml = 'reg',
            analysis_id = get_regmod_id(),
            list_param = json.dumps(list_param_reg),
            list_param_short = json.dumps(list_param_name),
            except_signal = except_reg.except_signal,
            except_crl = except_reg.except_crl,
            text_model=text_model,
            model_name=model_name,
            over_sampling = 'none',
            pipe = pickle.dumps(pipe)
        )
        session.add(new_lineup)
        session.commit()

        set_info(f'Модель {model_name} добавлена в очередь\n{text_model}', 'green')


    def calc_searched_param_model(trial, ui_rs, x_train, y_train, x_test, y_test):
        pipe_steps = []
        text_scaler = ''
        text_model = ''

        if ui_rs.checkBox_stdscaler.isChecked():
            std_scaler = StandardScaler()
            pipe_steps.append(('scaler', std_scaler))
            text_scaler += '\nStandardScaler'
        if ui_rs.checkBox_robscaler.isChecked():
            robust_scaler = RobustScaler()
            pipe_steps.append(('scaler', robust_scaler))
            text_scaler += '\nRobustScaler'
        if ui_rs.checkBox_mnmxscaler.isChecked():
            minmax_scaler = MinMaxScaler()
            pipe_steps.append(('scaler', minmax_scaler))
            text_scaler += '\nMinMaxScaler'
        if ui_rs.checkBox_mxabsscaler.isChecked():
            maxabs_scaler = MaxAbsScaler()
            pipe_steps.append(('scaler', maxabs_scaler))
            text_scaler += '\nMaxAbsScaler'

        if ui_rs.checkBox_pca.isChecked():
            comp = trial.suggest_int('n_components', ui_rs.spinBox_pca.value(), ui_rs.spinBox_pca_lim.value())
            n_comp = 'mle' if ui_rs.checkBox_pca_mle.isChecked() else comp
            pca = PCA(n_components=n_comp, random_state=0)
            pipe_steps.append(('pca', pca))
        text_pca = f'\nPCA: n_components={n_comp}' if ui_rs.checkBox_pca.isChecked() else ''

        if ui_rs.radioButton_xgb.isChecked():
            model = XGBRegressor(
                objective='reg:squarederror',
                n_estimators=trial.params['n_estimators'],
                learning_rate=trial.params['learning_rate'],
                max_depth=trial.params['max_depth'],
                # alpha=trial.params['alpha']
            )
            text_model += f'**XGB**: \nn estimators: {trial.params["n_estimators"]}, ' \
                         f'\nlearning_rate: {round(trial.params["learning_rate"], 5),} ' \
                         f'\nmax_depth: {trial.params["max_depth"]}'

        if ui_rs.radioButton_rfr.isChecked():
            model = RandomForestRegressor(
                n_estimators=trial.params['n_estimators'],
                max_depth=trial.params['max_depth'],
                min_samples_split=trial.params['min_samples_split'],
                min_samples_leaf=trial.params['min_samples_leaf'],
                max_features=trial.params['max_features']
            )
            text_model = f'**RFR**: \nn estimators: {trial.params["n_estimators"]}, ' \
                         f'\nmax_depth: {trial.params["max_depth"]},' \
                         f'\nmin_samples_split: {trial.params["min_samples_split"]}, ' \
                         f'\nmin_samples_leaf: {trial.params["min_samples_leaf"]}, ' \
                         f'\nmax_features: {trial.params["max_features"]}, '

        if ui_rs.radioButton_mlpr.isChecked():
            hidden_layer_sizes = tuple(trial.params[f'hidden_layer_size_{i}'] for i in range(trial.params['n_layers']))
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=trial.params['activation'],
                solver=trial.params['solver'],
                alpha=trial.params['alpha'],
                learning_rate_init=trial.params['learning_rate_init'],
                max_iter=ui_rs.spinBox_iter_mlp.value(),
                random_state=42
            )
            text_model = (
                f'**MLPR**: \n'
                f'hidden_layer_sizes: ({hidden_layer_sizes}), '
                f'\nactivation: {trial.params["activation"]}, '
                f'\nsolver: {trial.params["solver"]}, '
                f'\nalpha: {round(trial.params["alpha"], 5)}, '
                f'\nlearning_rate: {round(trial.params["learning_rate_init"], 5)}'
                f'\nmax_iter: {ui_rs.spinBox_iter_mlp.value()}'
            )
        if ui_rs.radioButton_svr.isChecked():
            model = SVR(
                C=trial.params['C'],
                kernel=trial.params['kernel'],
                epsilon=trial.params['epsilon']
            )
            text_model = (f'**SVR**: \nkernel: {trial.params["kernel"]}, '
                          f'\nC: {round(trial.params["C"], 5)}, '
                          f'\nepsilon: {round(trial.params["epsilon"], 5)}')
        if ui_rs.radioButton_gbr.isChecked():
            model = GradientBoostingRegressor(
                n_estimators=trial.params['n_estimators'],
                learning_rate=trial.params['learning_rate'],
                max_depth=trial.params['max_depth'],
                min_samples_split=trial.params['min_samples_split'],
                min_samples_leaf=trial.params['min_samples_leaf'],
                subsample=trial.params['subsample']
            )
            text_model = f'**GBR**: \nn_estimators: {trial.params["n_estimators"]}, ' \
                         f'\nlearning rate: {round(trial.params["learning_rate"], 4)}, ' \
                         f'\nmax_depth: {trial.params["max_depth"]}, ' \
                         f'\nmin_samples_split: {trial.params["min_samples_split"]}' \
                         f'\nmin_samples_leaf: {trial.params["min_samples_leaf"]}, ' \
                         f'\nsubsample: {trial.params["subsample"]}'
        if ui_rs.radioButton_gpr.isChecked():
            kernel = ConstantKernel(trial.params["constant_value"]) * RBF(length_scale=trial.params["length_scale"])
            model = GaussianProcessRegressor(kernel=kernel,
                                             alpha=round(trial.params["alpha"], 4),
                                             n_restarts_optimizer=trial.params["n_restarts_optimizer"]
                                             )
            text_model = f'**GPR**: \nkernel: {kernel}' \
                         f'\nconstant_value: {round(trial.params["constant_value"], 4)}, ' \
                         f'\nlength_scale: {round(trial.params["length_scale"], 4)}, ' \
                         f'\nalpha: {round(trial.params["alpha"], 4)}, ' \
                         f'\nn_restarts_optimizer: {trial.params["n_restarts_optimizer"]}'

        if ui_rs.radioButton_lgbm.isChecked():
            model = lgb.LGBMRegressor(
                objective='regression',
                verbosity=-1,
                boosting_type='gbdt',
                reg_alpha=trial.params["reg_alpha"],
                reg_lambda=trial.params["reg_lambda"],
                num_leaves=trial.params["num_leaves"],
                colsample_bytree=trial.params["colsample_bytree"],
                subsample=trial.params["subsample"],
                subsample_freq=trial.params["subsample_freq"],
                min_child_samples=trial.params["min_child_samples"],
                learning_rate=trial.params["learning_rate"],
                n_estimators=trial.params["n_estimators"]
            )
            text_model = f'**LGBMReg**: \nl1: {round(trial.params["reg_alpha"], 4)}' \
                         f'\nl2: {round(trial.params["reg_lambda"], 4)}, ' \
                         f'\nnum_leaves: {trial.params["num_leaves"]}, ' \
                         f'\nfeature_fraction: {round(trial.params["colsample_bytree"], 4)}, ' \
                         f'\nsubsample: {round(trial.params["subsample"], 4)}' \
                         f'\nsubsample_freq: {round(trial.params["subsample_freq"], 4)}, ' \
                         f'\nlearning_rate: {round(trial.params["learning_rate"], 4)}'  \
                         f'\nmin_child_samples: {trial.params["min_child_samples"]}, ' \
                         f'\nn_estimators: {trial.params["n_estimators"]}'

        model_name = ui_rs.buttonGroup.checkedButton().text()
        text_model += text_scaler
        text_model += text_pca

        pipe_steps.append(('model', model))
        pipe = Pipeline(pipe_steps)
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        errors = y_pred - y_test

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # График фактических vs. предсказанных значений
        axs[0, 0].scatter(y_test, y_pred, alpha=0.7)
        axs[0, 0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
        axs[0, 0].set_xlabel('Истинные значения')
        axs[0, 0].set_ylabel('Предсказанные значения')
        axs[0, 0].set_title('Истинные vs. Предсказанные значения')
        axs[0, 0].grid(True)
        # График ошибки предсказания
        axs[0, 1].scatter(y_test, errors, alpha=0.7)
        axs[0, 1].axhline(y=0, color='r', linestyle='--')
        axs[0, 1].set_xlabel('Истинные значения')
        axs[0, 1].set_ylabel('Ошибка предсказания')
        axs[0, 1].set_title('Ошибка предсказания vs. Истинные значения')
        axs[0, 1].grid(True)
        # График распределения ошибок
        axs[1, 0].hist(errors, bins=30, edgecolor='k', alpha=0.7)
        axs[1, 0].set_xlabel('Ошибка предсказания')
        axs[1, 0].set_ylabel('Частота')
        axs[1, 0].set_title('Распределение ошибок предсказания')
        axs[1, 0].grid(True)

        fig.delaxes(axs[1, 1])
        plt.tight_layout()
        plt.show()

        if ui_rs.checkBox_save_model.isChecked():
            result = QtWidgets.QMessageBox.question(
                MainWindow,
                'Сохранение модели',
                f'Сохранить модель {model_name}?',
                QtWidgets.QMessageBox.Yes,
                QtWidgets.QMessageBox.No)
            if result == QtWidgets.QMessageBox.Yes:
                # Сохранение модели в файл с помощью pickle
                path_model = f'models/regression/{model_name}_{round(r2, 3)}_{datetime.datetime.now().strftime("%d%m%y")}.pkl'
                if os.path.exists(path_model):
                    path_model = f'models/regression/{model_name}_{round(r2, 3)}_{datetime.datetime.now().strftime("%d%m%y_%H%M%S")}.pkl'
                with open(path_model, 'wb') as f:
                    pickle.dump(pipe, f)

                new_trained_model = TrainedModelReg(
                    analysis_id=get_regmod_id(),
                    title=f'{model_name}_{round(r2, 3)}_{datetime.datetime.now().strftime("%d%m%y")}',
                    path_model=path_model,
                    list_params=json.dumps(list_param_name),
                    except_signal=ui.lineEdit_signal_except_reg.text(),
                    except_crl=ui.lineEdit_crl_except_reg.text(),
                    comment=text_model
                )
                session.add(new_trained_model)
                session.commit()
                update_list_trained_models_regmod()
            else:
                pass

    def random_search_reg():
        RandomSearchReg = QtWidgets.QDialog()
        ui_rs = Ui_RandomSearchReg()
        ui_rs.setupUi(RandomSearchReg)
        RandomSearchReg.show()
        RandomSearchReg.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        x_train, x_test, y_train, y_test = train_test_split(
            training_sample, target, test_size=0.2, random_state=42
        )

        def create_data():
            if ui_rs.radioButton_xgb.isChecked():
                data = pd.DataFrame(columns=['n_estimators', 'learning_rate', 'max_depth', 'r2', 'mse'])
            if ui_rs.radioButton_rfr.isChecked():
                data = pd.DataFrame(columns=['n_estimators', 'max_depth', 'min_samples_split',
                                               'min_samples_leaf', 'max_features', 'r2', 'mse'])
            if ui_rs.radioButton_mlpr.isChecked():
                data = pd.DataFrame(columns=['layers', 'hidden_layer_sizes', 'activation', 'solver', 'alpha',
                                'learning_rate_init', 'max_iter', 'r2', 'mse'])
            if ui_rs.radioButton_svr.isChecked():
                data = pd.DataFrame(columns=['C', 'epsilon', 'kernel', 'r2', 'mse'])
            if ui_rs.radioButton_gbr.isChecked():
                data = pd.DataFrame(columns=['n_estimators', 'learning_rate', 'max_depth', 'min_samples_split',
                                             'min_samples_leaf', 'subsample', 'r2', 'mse'])
            if ui_rs.radioButton_gpr.isChecked():
                data = pd.DataFrame(columns=['length_scale', 'constant_value', 'alpha', 'n_restarts_optimizer',
                                'r2', 'mse'])

            if ui_rs.radioButton_lgbm.isChecked():
                data = pd.DataFrame(columns=['lambda_l1', 'lambda_l2', 'num_leaves', 'colsample_bytree',
                                             'subsample', 'subsample_freq', 'min_child_samples',
                                             'learning_rate', 'n_estimators', 'r2', 'mse'])
            return data


        def search_param():
            """ Подбор оптимальных параметров с помощью Optuna """

            global data
            filename, _ = QFileDialog.getSaveFileName(caption="Сохранить результаты подбора параметров?",
                                                      filter="Excel Files (*.xlsx)")
            ui_rs.plainTextEdit.clear()
            start_time = datetime.datetime.now()
            data = create_data()

            def create_objective(data):
                def objective(trial):
                    global data
                    print("Trial number:", trial.number)
                    ui_rs.plainTextEdit.appendPlainText("Trial number: " + str(trial.number))
                    pipe_steps = []

                    if ui_rs.checkBox_stdscaler.isChecked():
                        std_scaler = StandardScaler()
                        pipe_steps.append(('scaler', std_scaler))
                    if ui_rs.checkBox_robscaler.isChecked():
                        robust_scaler = RobustScaler()
                        pipe_steps.append(('scaler', robust_scaler))
                    if ui_rs.checkBox_mnmxscaler.isChecked():
                        minmax_scaler = MinMaxScaler()
                        pipe_steps.append(('scaler', minmax_scaler))
                    if ui_rs.checkBox_mxabsscaler.isChecked():
                        maxabs_scaler = MaxAbsScaler()
                        pipe_steps.append(('scaler', maxabs_scaler))

                    if ui_rs.checkBox_pca.isChecked():
                        comp = trial.suggest_int('n_components', ui_rs.spinBox_pca.value(), ui_rs.spinBox_pca_lim.value())
                        n_comp = 'mle' if ui_rs.checkBox_pca_mle.isChecked() else comp
                        pca = PCA(n_components=n_comp, random_state=0)
                        pipe_steps.append(('pca', pca))

                    try:
                        if ui_rs.radioButton_xgb.isChecked():
                            estim = [ui_rs.spinBox_estim_xgb_min.value(), ui_rs.spinBox_estim_xgb_max.value()]
                            lr = [ui_rs.doubleSpinBox_lr_xgb_min.value(), ui_rs.doubleSpinBox_lr_xgb_max.value()]
                            depth = [ui_rs.spinBox_depth_xgb_min.value(), ui_rs.spinBox_depth_xgb_max.value()]
                            alpha = [ui_rs.doubleSpinBox_alpha_xgb_min.value(), ui_rs.doubleSpinBox_alpha_xgb_max.value()]
                            model = XGBRegressor(
                                objective='reg:squarederror',
                                n_estimators=trial.suggest_int('n_estimators', estim[0], estim[1]),
                                learning_rate=trial.suggest_float('learning_rate', lr[0], lr[1], log=True),
                                max_depth=trial.suggest_int('max_depth', depth[0], depth[1]),
                                # alpha=trial.suggest_float('alpha', alpha[0], alpha[1], log=True)
                            )
                            new_row = pd.Series({
                                'n_estimators': model.n_estimators,
                                'learning_rate': model.learning_rate,
                                'max_depth': model.max_depth,
                                # 'alpha': model.alpha
                            })

                        if ui_rs.radioButton_rfr.isChecked():
                            estim = [ui_rs.spinBox_estim_rfr_min.value(), ui_rs.spinBox_estim_rfr_max.value()]
                            depth = [ui_rs.spinBox_depth_rfr_min.value(), ui_rs.spinBox_depth_rfr_max.value()]
                            min_split = [ui_rs.spinBox_min_split_rfr_min.value(), ui_rs.spinBox_min_split_rfr_max.value()]
                            min_leaf = [ui_rs.spinBox_min_leaf_rfr_min.value(), ui_rs.spinBox_min_leaf_rfr_max.value()]
                            max_features = []
                            if ui_rs.checkBox_none_rfr.isChecked():
                                max_features.append(None)
                            if ui_rs.checkBox_sqrt_rfr.isChecked():
                                max_features.append('sqrt')
                            if ui_rs.checkBox_log2_rfr.isChecked():
                                max_features.append('log2')
                            model = RandomForestRegressor(
                                n_estimators=trial.suggest_int('n_estimators', estim[0], estim[1]),
                                max_depth=trial.suggest_int('max_depth', depth[0], depth[1]),
                                min_samples_split=trial.suggest_int('min_samples_split', min_split[0], min_split[1]),
                                min_samples_leaf=trial.suggest_int('min_samples_leaf', min_leaf[0], min_leaf[1]),
                                max_features=trial.suggest_categorical('max_features', max_features)
                            )
                            new_row = pd.Series({
                                'n_estimators': model.n_estimators,
                                'max_depth': model.max_depth,
                                'min_samples_split': model.min_samples_split,
                                'min_samples_leaf': model.min_samples_leaf,
                                'max_features': model.max_features
                            })

                        if ui_rs.radioButton_mlpr.isChecked():
                            neurons = [ui_rs.spinBox_neurons_min.value(), ui_rs.spinBox_neurons_max.value()]
                            layers = [ui_rs.spinBox_layers_min.value(), ui_rs.spinBox_layers_max.value()]

                            n_layers = trial.suggest_int('n_layers', layers[0], layers[1])
                            hidden_layer_sizes = tuple(
                                trial.suggest_int(f'hidden_layer_size_{i}', neurons[0], neurons[1])
                                for i in range(n_layers)
                            )
                            activation = []
                            if ui_rs.checkBox_relu.isChecked():
                                activation.append('relu')
                            if ui_rs.checkBox_tanh.isChecked():
                                activation.append('tanh')
                            if ui_rs.checkBox_logistic.isChecked():
                                activation.append('logistic')

                            solver = []
                            if ui_rs.checkBox_adam.isChecked():
                                solver.append('adam')
                            if ui_rs.checkBox_sgd.isChecked():
                                solver.append('sgd')

                            alpha = [ui_rs.doubleSpinBox_alpha_mlp_min.value(), ui_rs.doubleSpinBox_alpha_mlp_max.value()]
                            lr = [ui_rs.doubleSpinBox_lr_mlp_min.value(), ui_rs.doubleSpinBox_lr_mlp_max.value()]
                            iter = ui_rs.spinBox_iter_mlp.value()
                            model = MLPRegressor(
                                hidden_layer_sizes=hidden_layer_sizes,
                                activation=trial.suggest_categorical('activation', activation),
                                solver=trial.suggest_categorical('solver', solver),
                                alpha=trial.suggest_float('alpha', alpha[0], alpha[1], log=True),
                                learning_rate_init=trial.suggest_float('learning_rate_init', lr[0], lr[1], log=True),
                                max_iter=iter,
                                random_state=42
                            )
                            new_row = pd.Series({
                                'layers': n_layers,
                                'hidden_layer_sizes': model.hidden_layer_sizes,
                                'activation': model.activation,
                                'solver': model.solver,
                                'alpha': model.alpha,
                                'learning_rate_init': model.learning_rate_init,
                                'max_iter': model.max_iter
                            })

                        if ui_rs.radioButton_svr.isChecked():
                            C = [ui_rs.doubleSpinBox_C_min.value(), ui_rs.doubleSpinBox_C_max.value()]
                            epsilon = [ui_rs.doubleSpinBox_epsilon_min.value(), ui_rs.doubleSpinBox_epsilon_max.value()]
                            kernel = []
                            if ui_rs.checkBox_rbf.isChecked():
                                kernel.append('rbf')
                            if ui_rs.checkBox_linear.isChecked():
                                kernel.append('linear')
                            if ui_rs.checkBox_poly.isChecked():
                                kernel.append('poly')
                            if ui_rs.checkBox_sigmoid.isChecked():
                                kernel.append('sigmoid')
                            model = SVR(C=trial.suggest_float('C', C[0], C[1], log=True),
                                        epsilon=trial.suggest_float('epsilon', epsilon[0], epsilon[1]),
                                        kernel=trial.suggest_categorical('kernel', kernel)
                                        )
                            new_row = pd.Series({
                                'C': model.C,
                                'epsilon': model.epsilon,
                                'kernel': model.kernel
                            })

                        if ui_rs.radioButton_gbr.isChecked():
                            estim = [ui_rs.spinBox_estim_gbr_min.value(), ui_rs.spinBox_estim_gbr_max.value()]
                            lr = [ui_rs.doubleSpinBox_lr_gbr_min.value(), ui_rs.doubleSpinBox_lr_gbr_max.value()]
                            depth = [ui_rs.spinBox_depth_gbr_min.value(), ui_rs.spinBox_depth_gbr_max.value()]
                            min_split = [ui_rs.spinBox_min_split_gbr_min.value(), ui_rs.spinBox_min_split_gbr_max.value()]
                            min_leaf = [ui_rs.spinBox_min_leaf_gbr_min.value(), ui_rs.spinBox_min_leaf_gbr_max.value()]
                            subsample = [ui_rs.doubleSpinBox_subsample_gbr_min.value(), ui_rs.doubleSpinBox_subsample_gbr_max.value()]

                            model = GradientBoostingRegressor(
                                n_estimators=trial.suggest_int('n_estimators', estim[0], estim[1]),
                                learning_rate=trial.suggest_float('learning_rate', lr[0], lr[1], log=True),
                                max_depth=trial.suggest_int('max_depth', depth[0], depth[1]),
                                min_samples_split=trial.suggest_int('min_samples_split', min_split[0], min_split[1]),
                                min_samples_leaf=trial.suggest_int('min_samples_leaf', min_leaf[0], min_leaf[1]),
                                subsample=trial.suggest_float('subsample', subsample[0], subsample[1])
                            )
                            new_row = pd.Series({
                                'n_estimators': model.n_estimators,
                                'learning_rate': model.learning_rate,
                                'max_depth': model.max_depth,
                                'min_samples_split': model.min_samples_split,
                                'min_samples_leaf': model.min_samples_leaf,
                                'subsample': model.subsample
                            })

                        if ui_rs.radioButton_gpr.isChecked():
                            scale = [ui_rs.doubleSpinBox_gpr_scale_min.value(), ui_rs.doubleSpinBox_gpr_scale_max.value()]
                            constant = [ui_rs.doubleSpinBox_gpr_const_min.value(), ui_rs.doubleSpinBox_gpr_const_max.value()]
                            alpha = [ui_rs.doubleSpinBox_gpr_alpha_min.value(), ui_rs.doubleSpinBox_gpr_alpha_max.value()]
                            n = [ui_rs.spinBox_gpr_n_restart_min.value(), ui_rs.spinBox_gpr_n_restart_max.value()]
                            length_scale = trial.suggest_float('length_scale', scale[0], scale[1])
                            constant_value = trial.suggest_float('constant_value', constant[0], constant[1])
                            kernel = ConstantKernel(constant_value) * RBF(length_scale=length_scale)
                            model = GaussianProcessRegressor(kernel=kernel,
                                                             alpha=trial.suggest_float('alpha', alpha[0], alpha[1]),
                                                             n_restarts_optimizer=trial.suggest_int('n_restarts_optimizer',
                                                                                                    n[0], n[1]))
                            new_row = pd.Series({
                                'length_scale': length_scale,
                                'constant_value': constant_value,
                                'alpha': model.alpha,
                                'n_restarts_optimizer': model.n_restarts_optimizer
                            })

                        if ui_rs.radioButton_lgbm.isChecked():
                            l1 = [ui_rs.doubleSpinBox_l1_lgbm_min.value(), ui_rs.doubleSpinBox_l1_lgbm_max.value()]
                            l2 = [ui_rs.doubleSpinBox_l2_lgbm_min.value(), ui_rs.doubleSpinBox_l2_lgbm_max.value()]
                            leaves = [ui_rs.spinBox_lgbm_num_leaves_min.value(), ui_rs.spinBox_lgbm_num_leaves_max.value()]
                            feature = [ui_rs.doubleSpinBox_lgbm_feature_min.value(), ui_rs.doubleSpinBox_lgbm_feature_max.value()]
                            subsample = [ui_rs.doubleSpinBox_lgbm_subsample_min.value(), ui_rs.doubleSpinBox_lgbm_subsample_max.value()]
                            subsample_freq = [ui_rs.spinBox_lgbm_sub_freq_min.value(), ui_rs.spinBox_lgbm_sub_freq_max.value()]
                            min_child = [ui_rs.spinBox_lgbm_child_min.value(), ui_rs.spinBox_lgbm_child_max.value()]
                            lr = [ui_rs.doubleSpinBox_lr_lgbm_min.value(), ui_rs.doubleSpinBox_lr_lgbm_max.value()]
                            estim = [ui_rs.spinBox_estim_lgbm_min.value(), ui_rs.spinBox_estim_lgbm_max.value()]

                            model = lgb.LGBMRegressor(
                                objective='regression',
                                verbosity=-1,
                                boosting_type='gbdt',
                                reg_alpha=trial.suggest_float('reg_alpha', l1[0], l1[1]),
                                reg_lambda=trial.suggest_float('reg_lambda', l2[0], l2[1]),
                                num_leaves=trial.suggest_int('num_leaves', leaves[0], leaves[1]),
                                colsample_bytree=trial.suggest_float('colsample_bytree', feature[0], feature[1]),
                                subsample=trial.suggest_float('subsample', subsample[0], subsample[1]),
                                subsample_freq=trial.suggest_int('subsample_freq', subsample_freq[0], subsample_freq[1]),
                                min_child_samples=trial.suggest_int('min_child_samples', min_child[0], min_child[1]),
                                learning_rate=trial.suggest_float('learning_rate', lr[0], lr[1]),
                                n_estimators=trial.suggest_int('n_estimators', estim[0], estim[1])
                            )

                            new_row = pd.Series({
                                'lambda_l1': model.reg_alpha,
                                'lambda_l2': model.reg_lambda,
                                'num_leaves': model.num_leaves,
                                'colsample_bytree': model.colsample_bytree,
                                'subsample': model.subsample,
                                'subsample_freq': model.subsample_freq,
                                'min_child_samples': model.min_child_samples,
                                'learning_rate': model.learning_rate,
                                'n_estimators': model.n_estimators
                            })


                        pipe_steps.append(('model', model))
                        pipe = Pipeline(pipe_steps)
                        pipe.fit(x_train, y_train)
                        y_pred = pipe.predict(x_test)
                        r2 = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        new_row['r2'] = r2
                        new_row['mse'] = mse
                        ui_rs.plainTextEdit.appendPlainText("R2: " + str(r2))
                        ui_rs.plainTextEdit.appendPlainText("MSE: " + str(mse))
                        print('r2: ', r2)
                        print('mse: ', mse)
                        new_row_df = pd.DataFrame([new_row])
                        data = pd.concat([data, new_row_df], ignore_index=True)
                        pd.set_option('display.max_columns', None)
                        print(data)
                        return r2

                    except ValueError as e:
                        print(f"Trial failed due to: {e}")
                        ui_rs.plainTextEdit.appendPlainText(f"Trial failed due to: {e}")
                        return float('-inf')

                return objective


            # Запуск оптимизации
            # optuna.logging.set_verbosity(optuna.logging.CRITICAL)
            try:
                objective_func = create_objective(data)
                study = optuna.create_study(direction='maximize')
                study.optimize(objective_func, n_trials=ui_rs.spinBox_trials.value())

                # Вывод лучших параметров
                print("\nNumber of finished trials:", len(study.trials))
                print("Best trial:")
                ui_rs.plainTextEdit.appendPlainText("\nNumber of finished trials: " + str(len(study.trials)) +
                                                    "\nBest trial: ")
                trial = study.best_trial

                print("Value: ", trial.value)
                ui_rs.plainTextEdit.appendPlainText("Value: " + str(trial.value))
                print("Params: ")
                ui_rs.plainTextEdit.appendPlainText("Params: ")
                for key, value in trial.params.items():
                    ui_rs.plainTextEdit.appendPlainText(f"    {key}: {value}")
                    print(f"    {key}: {value}")

                if filename:
                    data.to_excel(filename, index=False)
            except optuna.exceptions.OptunaError as e:
                print(f"Optimization stopped: {e}")

            end_time = datetime.datetime.now() - start_time
            print(end_time)
            print('\n\n')

            calc_searched_param_model(trial, ui_rs, x_train, y_train, x_test, y_test)

        ui_rs.pushButton_search_param.clicked.connect(search_param)
        RandomSearchReg.exec_()

    def calc_model_reg():
        """ Создание и тренировка модели """
        nonlocal training_sample, list_param_reg

        start_time = datetime.datetime.now()
        # Нормализация данных
        text_scaler = ''

        pipe_steps = []
        if ui_r.checkBox_stdscaler_reg.isChecked():
            std_scaler = StandardScaler()
            pipe_steps.append(('scaler', std_scaler))
            text_scaler += '\nStandardScaler'
        if ui_r.checkBox_robscaler_reg.isChecked():
            robust_scaler = RobustScaler()
            pipe_steps.append(('scaler', robust_scaler))
            text_scaler += '\nRobustScaler'
        if ui_r.checkBox_mnmxscaler_reg.isChecked():
            minmax_scaler = MinMaxScaler()
            pipe_steps.append(('scaler', minmax_scaler))
            text_scaler += '\nMinMaxScaler'
        if ui_r.checkBox_mxabsscaler_reg.isChecked():
            maxabs_scaler = MaxAbsScaler()
            pipe_steps.append(('scaler', maxabs_scaler))
            text_scaler += '\nMaxAbsScaler'

        if ui_r.checkBox_mask_param.isChecked():
            list_param = get_list_param_by_mask(ui_r.listWidget_mask_param.currentItem().text().split(" id")[-1])
            training_sample = np.array(data_train[list_param].values.tolist())

        # Разделение данных на обучающую и тестовую выборки
        x_train, x_test, y_train, y_test = train_test_split(
            training_sample, target, test_size=0.2, random_state=42
        )

        if ui_r.checkBox_pca.isChecked():
            n_comp = 'mle' if ui_r.checkBox_pca_mle.isChecked() else ui_r.spinBox_pca.value()
            pca = PCA(n_components=n_comp, random_state=0)
            pipe_steps.append(('pca', pca))
        text_pca = f'\nPCA: n_components={n_comp}' if ui_r.checkBox_pca.isChecked() else ''

        if ui_r.checkBox_stack_vote.isChecked():
            model_class, text_model, model_name = build_stacking_voting_model()
        else:
            model_name = ui_r.buttonGroup.checkedButton().text()
            model_class, text_model = choice_model_regressor(model_name, x_train)

        if ui_r.checkBox_baggig.isChecked():
            model_class = BaggingRegressor(base_estimator=model_class, n_estimators=ui_r.spinBox_bagging.value(),
                                            random_state=0, n_jobs=-1)
        bagging_text = f'\nBagging: n_estimators={ui_r.spinBox_bagging.value()}' if ui_r.checkBox_baggig.isChecked() else ''

        text_model += text_scaler
        text_model += text_pca
        text_model += bagging_text

        pipe_steps.append(('model', model_class))
        pipe = Pipeline(pipe_steps)

        kf = KFold(n_splits=ui_r.spinBox_n_cross_val.value(), shuffle=True, random_state=0)
        if ui_r.checkBox_cross_val.isChecked():
            # kf = KFold(n_splits=ui_r.spinBox_n_cross_val.value(), shuffle=True, random_state=0)
            list_train, list_test, n_cross = [], [], 1
            for train_index, test_index in kf.split(training_sample):
                list_train.append(train_index.tolist())
                list_test.append(test_index.tolist())
                n_cross += 1
            scores_cv = cross_val_score(pipe, training_sample, np.array(target, dtype=np.float32), cv=kf, n_jobs=-1)
            n_max = np.argmax(scores_cv)
            train_index, test_index = list_train[n_max], list_test[n_max]

            x_train = [training_sample[i] for i in train_index]
            x_test = [training_sample[i] for i in test_index]

            y_train = [target[i] for i in train_index]
            y_test = [target[i] for i in test_index]

        pipe.fit(x_train, np.array(y_train, dtype=np.float32))
        y_pred = pipe.predict(x_test)

        r2 = r2_score(y_test, y_pred)
        accuracy = round(pipe.score(x_test, y_test), 5)
        # mse = round(mean_squared_error(y_test, y_pred), 5)
        mse = np.mean((y_test - y_pred) ** 2)
        print('accuracy ', accuracy, 'r2 ', r2, 'mse ', mse)


        cv_text = (
            f'\nКРОСС-ВАЛИДАЦИЯ\nОценки на каждом разбиении:\n {" / ".join(str(round(val, 2)) for val in scores_cv)}'
            f'\nСредн.: {round(scores_cv.mean(), 2)} '
            f'\nR2: {round(r2, 2)} '
            f'Станд. откл.: {round(scores_cv.std(), 2)}') if ui_r.checkBox_cross_val.isChecked() else ''

        train_time = datetime.datetime.now() - start_time

        set_info(f'Модель {model_name}:\n точность: {accuracy} '
                 f' Mean Squared Error:\n {mse}, \n R2: {r2}\n время обучения: {train_time}', 'blue')
        y_remain = [round(y_test[i] - y_pred[i], 5) for i in range(len(y_pred))]


        data_graph = pd.DataFrame({
            'y_test': y_test,
            'y_pred': y_pred,
            'y_remain': y_remain
        })

        try:
            ipm_name_params, imp_params = [], []
            if ui_r.checkBox_mask_param.isChecked():
                list_param_reg = get_list_param_by_mask(ui_r.listWidget_mask_param.currentItem().text().split(" id")[-1])
            for n, i in enumerate(pipe['model'].feature_importances_):
                if i >= np.mean(pipe['model'].feature_importances_):
                    ipm_name_params.append(list_param_reg[n])
                    imp_params.append(i)

            fig, axes = plt.subplots(nrows=2, ncols=2)
            fig.set_size_inches(15, 10)
            fig.suptitle(f'Модель {model_name}:\n точность: {accuracy} '
                 f' Mean Squared Error:\n {mse}, \n R2: {round(r2, 2)} \n время обучения: {train_time}' + cv_text)
            sns.scatterplot(data=data_graph, x='y_test', y='y_pred', ax=axes[0, 0])
            sns.regplot(data=data_graph, x='y_test', y='y_pred', ax=axes[0, 0])
            sns.scatterplot(data=data_graph, x='y_test', y='y_remain', ax=axes[1, 0])
            sns.regplot(data=data_graph, x='y_test', y='y_remain', ax=axes[1, 0])
            if ui_r.checkBox_cross_val.isChecked():
                axes[0, 1].bar(range(len(scores_cv)), scores_cv)
                axes[0, 1].set_title('Кросс-валидация')
            else:
                axes[0, 1].bar(ipm_name_params, imp_params)
                axes[0, 1].set_xticklabels(ipm_name_params, rotation=90)
            try:
                sns.histplot(data=data_graph, x='y_remain', kde=True, ax=axes[1, 1])
            except MemoryError:
                pass
            fig.tight_layout()
            fig.show()
        except AttributeError:
            fig, axes = plt.subplots(nrows=2, ncols=2)
            fig.set_size_inches(15, 10)
            fig.suptitle(f'Модель {model_name}:\n точность: {accuracy} '
                          f' Mean Squared Error:\n {mse}, R2: {round(r2, 2)} \n время обучения: {train_time}' + cv_text)
            sns.scatterplot(data=data_graph, x='y_test', y='y_pred', ax=axes[0, 0])
            sns.regplot(data=data_graph, x='y_test', y='y_pred', ax=axes[0, 0])
            sns.scatterplot(data=data_graph, x='y_test', y='y_remain', ax=axes[1, 0])
            sns.regplot(data=data_graph, x='y_test', y='y_remain', ax=axes[1, 0])
            if ui_r.checkBox_cross_val.isChecked():
                axes[0, 1].bar(range(len(scores_cv)), scores_cv)
                axes[0, 1].set_title('Кросс-валидация')
            try:
                sns.histplot(data=data_graph, x='y_remain', kde=True, ax=axes[1, 1])
            except MemoryError:
                pass
            fig.tight_layout()
            fig.show()


        # if 'GBR' in model_name or 'LGBM' in model_name or 'XGB' in model_name or 'RFR' in model_name:
            print('find dependence')
        Form_PDP = QtWidgets.QDialog()
        ui_pdp = Ui_Dialog_PDP()
        ui_pdp.setupUi(Form_PDP)
        Form_PDP.show()
        Form_PDP.setAttribute(QtCore.Qt.WA_DeleteOnClose)


        for p in list_param_reg:
            ui_pdp.comboBox_pdp.addItem(f'{p}')

        for p in list_param_reg:
            ui_pdp.comboBox_pdp_2.addItem(f'{p}')

        def draw_dependence():
            param = ui_pdp.comboBox_pdp.currentText()
            param_2 = ui_pdp.comboBox_pdp_2.currentText()
            if ui_pdp.checkBox_2_params.isChecked():
                PartialDependenceDisplay.from_estimator(pipe, x_test, [(param, param_2)], feature_names=list_param_reg,
                                                        kind='average')
            else:
                PartialDependenceDisplay.from_estimator(pipe, x_test, [param], feature_names=list_param_reg, kind='both')
            plt.show()



        ui_pdp.pushButton_pdp.clicked.connect(draw_dependence)
        Form_PDP.exec_()



        if not ui_r.checkBox_save_model.isChecked():
            return
        result = QtWidgets.QMessageBox.question(
            MainWindow,
            'Сохранение модели',
            f'Сохранить модель {model_name}?',
            QtWidgets.QMessageBox.Yes,
            QtWidgets.QMessageBox.No)
        if result == QtWidgets.QMessageBox.Yes:
            # Сохранение модели в файл с помощью pickle
            path_model = f'models/regression/{model_name}_{round(accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y")}.pkl'
            if os.path.exists(path_model):
                path_model = f'models/regression/{model_name}_{round(accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y_%H%M%S")}.pkl'
            with open(path_model, 'wb') as f:
                pickle.dump(pipe, f)

            new_trained_model = TrainedModelReg(
                analysis_id=get_regmod_id(),
                title=f'{model_name}_{round(accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y")}',
                path_model=path_model,
                list_params=json.dumps(list_param_name),
                except_signal=ui.lineEdit_signal_except_reg.text(),
                except_crl=ui.lineEdit_crl_except_reg.text(),
                comment = text_model
            )
            session.add(new_trained_model)
            session.commit()

            if ui_r.checkBox_mask_param.isChecked():
                mask_id = int(ui_r.listWidget_mask_param.currentItem().text().split(' id')[-1])
                new_trained_model_mask = TrainedModelRegMask(
                    model_id=new_trained_model.id,
                    mask_id=mask_id
                )
                session.add(new_trained_model_mask)
                session.commit()

            update_list_trained_models_regmod()
        else:
            pass


    def calc_cov():
        """ Кросс-объектная валидация """
        nonlocal list_param_reg

        def get_obj_title(prof_well_index):
            prof_id = prof_well_index.split('_')[0]
            obj = session.query(GeoradarObject).join(Research).join(Profile).filter(Profile.id == prof_id).first()
            return obj.title

        def get_coordinates_x(prof_well_index):
            well_id = int(prof_well_index.split('_')[1])
            try:
                return session.query(Well).filter(Well.id == well_id).first().x_coord
            except AttributeError:
                return None

        def get_coordinates_y(prof_well_index):
            well_id = int(prof_well_index.split('_')[1])
            try:
                return session.query(Well).filter(Well.id == well_id).first().y_coord
            except AttributeError:
                return None

        data_train_cov = data_train.copy()
        data_train_cov['obj_title'] = data_train_cov['prof_well_index'].apply(get_obj_title)
        data_train_cov['x_coord'] = data_train_cov['prof_well_index'].apply(get_coordinates_x)
        data_train_cov['y_coord'] = data_train_cov['prof_well_index'].apply(get_coordinates_y)


        if ui_r.checkBox_mask_param.isChecked():
            list_param_reg = get_list_param_by_mask(ui_r.listWidget_mask_param.currentItem().text().split(" id")[-1])

        training_sample = np.array(data_train_cov[list_param_reg].values.tolist())

        markup = np.array(sum(data_train[['target_value']].values.tolist(), []))
        groups = np.array(sum(data_train_cov[['obj_title']].values.tolist(), []))


        # Нормализация данных
        text_scaler = ''

        pipe_steps = []
        if ui_r.checkBox_stdscaler_reg.isChecked():
            std_scaler = StandardScaler()
            pipe_steps.append(('scaler', std_scaler))
            text_scaler += '\nStandardScaler'
        if ui_r.checkBox_robscaler_reg.isChecked():
            robust_scaler = RobustScaler()
            pipe_steps.append(('scaler', robust_scaler))
            text_scaler += '\nRobustScaler'
        if ui_r.checkBox_mnmxscaler_reg.isChecked():
            minmax_scaler = MinMaxScaler()
            pipe_steps.append(('scaler', minmax_scaler))
            text_scaler += '\nMinMaxScaler'
        if ui_r.checkBox_mxabsscaler_reg.isChecked():
            maxabs_scaler = MaxAbsScaler()
            pipe_steps.append(('scaler', maxabs_scaler))
            text_scaler += '\nMaxAbsScaler'

        if ui_r.checkBox_pca.isChecked():
            n_comp = 'mle' if ui_r.checkBox_pca_mle.isChecked() else ui_r.spinBox_pca.value()
            pca = PCA(n_components=n_comp, random_state=0)
            pipe_steps.append(('pca', pca))
        text_pca = f'\nPCA: n_components={n_comp}' if ui_r.checkBox_pca.isChecked() else ''

        model_name = ui_r.buttonGroup.checkedButton().text()
        model_class, text_model = choice_model_regressor(model_name, training_sample)

        text_model += text_scaler
        text_model += text_pca

        pipe_steps.append(('model', model_class))
        pipe = Pipeline(pipe_steps)

        # logo = LeaveOneGroupOut()

        # Передаём groups и logo как cv
        # scores = cross_val_score(pipe, training_sample, markup, cv=logo, groups=groups)

        scores = []
        group_order = []
        group_sizes = []
        group_r2 = []
        graphs = []
        all_list = []

        ui.progressBar.setMaximum(len(set(list(groups))))
        n_progress = 1
        for train_idx, test_idx in LeaveOneGroupOut().split(training_sample, markup, groups):
            ui.progressBar.setValue(n_progress)
            start_time = datetime.datetime.now()

            if ui_r.checkBox_cov_percent.isChecked():
                if len(test_idx) / len(markup) < ui_r.spinBox_cov_percent.value() / 100:
                    n_progress += 1
                    continue

            pipe.fit(training_sample[train_idx], markup[train_idx])
            y_pred = pipe.predict(training_sample[test_idx])
            score = pipe.score(training_sample[test_idx], markup[test_idx])
            r2 = r2_score(markup[test_idx], y_pred)

            for_graph = pd.DataFrame({'x': markup[test_idx], 'y': y_pred})

            test_group = np.unique(groups[test_idx])[0]

            group_size = int(len(test_idx)/10) if len(test_idx)%10 == 0 else int(len(test_idx)/10) + 1


            all_list.append([score, test_group, group_size, r2, for_graph])

            finish_time = datetime.datetime.now()
            inter_time = finish_time - start_time
            set_info(f'Качество "{test_group}": {score}. Время выполнения: {inter_time}, осталось: {(len(set(list(groups))) - n_progress) * inter_time}', 'blue')
            n_progress += 1


        # Сортируем по group_sizes

        all_list = sorted(all_list, key=lambda x: x[2], reverse=True)
        for i in all_list:
            scores.append(i[0])
            group_order.append(i[1])
            group_sizes.append(i[2])
            group_r2.append(i[3])
            graphs.append(i[4])


        # Создаём словарь: group -> score
        group_to_score = dict(zip(group_order, scores))

        # Добавляем столбец score для каждой строки по obj_title
        data_train_cov['group_score'] = data_train_cov['obj_title'].map(group_to_score)

        # Собираем подписи
        labels = [f"{g}\n(n={s}\nr2={r2})" for g, s, r2 in zip(group_order, group_sizes, group_r2)]

        # Рисуем график
        plt.figure(figsize=(15, 12))
        bars = plt.bar(range(len(scores)), scores, color='skyblue', edgecolor='black')

        # Подписываем столбики снизу
        plt.xticks(ticks=range(len(scores)), labels=labels, rotation=90, fontsize=12)

        # Значения сверху столбиков
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

        plt.title(f'{model_name}\nMean acc: {np.mean(scores):.2f} Std: {np.std(scores):.2f}\nMean r2: {np.mean(group_r2):.2f} Std: {np.std(group_r2):.2f}')
        plt.ylabel('Score')
        # plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        plot_groups_with_smoothed_hull(data_train_cov)
        plot_graphs_by_group(graphs, group_order)

    def calc_model_regression_by_cvw():
        """ Кросс-валидация по скважинам """
        nonlocal list_param_reg

        if ui_r.checkBox_mask_param.isChecked():
            list_param_reg = get_list_param_by_mask(ui_r.listWidget_mask_param.currentItem().text().split(" id")[-1])

        training_sample = np.array(data_train[list_param_reg].values.tolist())

        # Нормализация данных
        text_scaler = ''

        pipe_steps = []
        if ui_r.checkBox_stdscaler_reg.isChecked():
            std_scaler = StandardScaler()
            pipe_steps.append(('scaler', std_scaler))
            text_scaler += '\nStandardScaler'
        if ui_r.checkBox_robscaler_reg.isChecked():
            robust_scaler = RobustScaler()
            pipe_steps.append(('scaler', robust_scaler))
            text_scaler += '\nRobustScaler'
        if ui_r.checkBox_mnmxscaler_reg.isChecked():
            minmax_scaler = MinMaxScaler()
            pipe_steps.append(('scaler', minmax_scaler))
            text_scaler += '\nMinMaxScaler'
        if ui_r.checkBox_mxabsscaler_reg.isChecked():
            maxabs_scaler = MaxAbsScaler()
            pipe_steps.append(('scaler', maxabs_scaler))
            text_scaler += '\nMaxAbsScaler'

        if ui_r.checkBox_pca.isChecked():
            n_comp = 'mle' if ui_r.checkBox_pca_mle.isChecked() else ui_r.spinBox_pca.value()
            pca = PCA(n_components=n_comp, random_state=0)
            pipe_steps.append(('pca', pca))
        text_pca = f'\nPCA: n_components={n_comp}' if ui_r.checkBox_pca.isChecked() else ''

        model_name = ui_r.buttonGroup.checkedButton().text()
        model_class, text_model = choice_model_regressor(model_name, training_sample)

        text_model += text_scaler
        text_model += text_pca

        pipe_steps.append(('model', model_class))
        pipe = Pipeline(pipe_steps)


        list_well = [i.split('_')[1] for i in data_train['prof_well_index'].values.tolist()]
        data_train_by_well = data_train.copy()
        data_train_by_well['well_id'] = list_well

        list_cvw = split_list_cvw(list_well, 5)

        list_accuracy = []

        ui.progressBar.setMaximum(len(list_cvw))
        for n_cv, lcvw in enumerate(list_cvw):
            ui.progressBar.setValue(n_cv + 1)

            cvw_row, cvw_col = n_cv // 3, n_cv % 3

            data_test_well = data_train_by_well[data_train_by_well['well_id'].isin(lcvw)]
            data_train_well = data_train_by_well[~data_train_by_well['well_id'].isin(lcvw)]

            training_sample_train = np.array(data_train_well[list_param_reg].values.tolist())
            training_sample_test = np.array(data_test_well[list_param_reg].values.tolist())
            markup_train = np.array(sum(data_train_well[['target_value']].values.tolist(), []))
            markup_test = np.array(sum(data_test_well[['target_value']].values.tolist(), []))

            pipe.fit(training_sample_train, markup_train)

            # Оценка точности на всей обучающей выборке
            test_accuracy = pipe.score(training_sample_test, markup_test)

            list_accuracy.append(test_accuracy)

        plt.bar(range(5), list_accuracy)
        plt.show()


    def calc_lof():
        """ Расчет выбросов методом LOF """
        global data_pca, data_tsne, colors, factor_lof

        data_lof = data_train.copy()
        data_lof.drop(['prof_well_index', 'target_value'], axis=1, inplace=True)

        scaler = StandardScaler()
        training_sample_lof = scaler.fit_transform(data_lof)
        n_LOF = ui_r.spinBox_lof_neighbor.value()

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
                old_list_fake = session.query(MarkupReg.list_fake).filter(
                    MarkupReg.analysis_id == get_regmod_id(),
                    MarkupReg.profile_id == prof_id,
                    MarkupReg.well_id == well_id
                ).first()[0]
                if old_list_fake:
                    new_list_fake = json.loads(old_list_fake)
                    new_list_fake.append(fake_id)
                else:
                    new_list_fake = [fake_id]
                session.query(MarkupReg).filter(
                    MarkupReg.analysis_id == get_regmod_id(),
                    MarkupReg.profile_id == prof_id,
                    MarkupReg.well_id == well_id
                ).update({'list_fake': json.dumps(new_list_fake)}, synchronize_session='fetch')
                session.commit()

            Regressor.close()
            Form_LOF.close()

            new_data_train = data_train.drop(data_train.index[lof_index]).reset_index(drop=True)

            analysis_reg = session.query(AnalysisReg).filter_by(id=get_regmod_id()).first()
            filepath = analysis_reg.data
            new_data_train.to_parquet(filepath)

            # session.query(AnalysisReg).filter_by(id=get_regmod_id()).update(
            #     {'data': json.dumps(new_data_train.to_dict())}, synchronize_session='fetch')
            # session.commit()
            # build_table_train(False, 'regmod')
            update_list_well_markup_reg()
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
            list_widget.addItem(f'{i}) {data["prof_well_index"][i]}')
            if label_lof[int(i)] == -1:
                list_widget.item(int(i)).setBackground(QBrush(QColor('red')))
        list_widget.setCurrentRow(0)


    def insert_list_features(data, list_widget):
        list_widget.clear()
        for col in data.columns:
            if col != 'prof_well_index' and col != 'mark':
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

    def call_feature_selection():
        feature_selection_calc(data_train[list_param_reg], data_train['target_value'], mode='reg')





    ###########################################################
    ################## Генетический алгоритм ##################
    ###########################################################

    def genetic_algorithm():

        GenAlg = QtWidgets.QDialog()
        ui_ga = Ui_GeneticForm()
        ui_ga.setupUi(GenAlg)
        GenAlg.show()
        GenAlg.setAttribute(Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

        m_width, m_height = get_width_height_monitor()
        GenAlg.resize(m_width - 500, m_height - 400)

        def get_gen_an():
            return session.query(GeneticAlgorithmReg).filter_by(
                id=ui_ga.comboBox_gen_analysis.currentText().split(' id')[-1]).first()

        def update_combobox_gen_an():
            ui_ga.comboBox_gen_analysis.clear()
            gen_analysis = session.query(GeneticAlgorithmReg).filter_by(analysis_id=get_regmod_id()).order_by(
                desc(GeneticAlgorithmReg.id)).all()
            for ga in gen_analysis:
                ui_ga.comboBox_gen_analysis.addItem(f'{ga.type_problem} {ga.title} id{ga.id}')

        def save_mask():
            ga = get_gen_an()
            if ga:
                try:
                    list_p = json.loads(ga.list_params)
                    with open(ga.checkfile_path, "rb") as f:
                        data = pickle.load(f)
                except FileNotFoundError:
                    return

                selected_mask = ui_ga.listWidget_population.currentItem().text().split(' N')

                for x, fobj in zip(data["X"], data["F"]):
                    try:
                        if fobj[0] == float(selected_mask[0]) and fobj[1] == int(selected_mask[1]):
                            print(f'{fobj[0]} N{fobj[1]}')
                            print(x)
                            flattened_mask = [m[0] for m in x]
                            mask_param = [p for p, m in zip(list_p, flattened_mask) if m]
                            print(mask_param)

                            info = (f'regression analysis: {ui.comboBox_regmod.currentText()}\n'
                                    f'pareto analysis: {ui_ga.comboBox_gen_analysis.currentText()}\n')
                            new_mask = ParameterMask(
                                count_param=len(mask_param),
                                mask=json.dumps(mask_param),
                                mask_info=info
                            )
                            session.add(new_mask)
                            session.commit()
                            QMessageBox.information(MainWindow, 'Info', f'Маска сохранена\n{info}')
                            update_list_saved_mask()
                            update_list_mask()
                            break
                    except IndexError:
                        if fobj == float(selected_mask[0]) and np.sum(x) == int(selected_mask[1]):
                            print(f'{fobj} N{np.sum(x)}')
                            print(x)
                            flattened_mask = [m[0] for m in x]
                            mask_param = [p for p, m in zip(list_p, flattened_mask) if m]
                            print(mask_param)

                            info = (f'regression analysis: {ui.comboBox_regmod.currentText()}\n'
                                    f'pareto analysis: {ui_ga.comboBox_gen_analysis.currentText()}\n')
                            new_mask = ParameterMask(
                                count_param=len(mask_param),
                                mask=json.dumps(mask_param),
                                mask_info=info
                            )
                            session.add(new_mask)
                            session.commit()
                            QMessageBox.information(MainWindow, 'Info', f'Маска сохранена\n{info}')
                            update_list_saved_mask()
                            update_list_mask()
                            break
            else:
                return

        def update_list_mask():
            ui_ga.listWidget_save_mask.clear()
            for i in session.query(ParameterMask).all():
                item = QListWidgetItem(f'{i.count_param} id{i.id}')
                item.setToolTip(i.mask_info)
                ui_ga.listWidget_save_mask.addItem(item)


        def remove_mask():
            session.query(ParameterMask).filter_by(
                id=ui_ga.listWidget_save_mask.currentItem().text().split(' id')[-1]).delete()
            session.commit()
            update_list_mask()
            update_list_saved_mask()


        def update_list_population():
            ui_ga.listWidget_population.clear()
            ga = get_gen_an()
            if ga:
                try:
                    with open(ga.checkfile_path, "rb") as f:
                        data = pickle.load(f)
                except FileNotFoundError:
                    return

                for x, fobj in zip(data["X"], data["F"]):
                    try:
                        ui_ga.listWidget_population.addItem(f'{fobj[0]} N{fobj[1]}')
                    except IndexError:
                        ui_ga.listWidget_population.addItem(f'{fobj} N{np.sum(x)}')

                ui_ga.lcdNumber_generation.display(data["ngen"])
                if ga.type_problem == 'min':
                    ui_ga.radioButton_pareto_min.setChecked(True)
                if ga.type_problem == 'max':
                    ui_ga.radioButton_pareto_max.setChecked(True)
                if ga.type_problem == 'no':
                    ui_ga.radioButton_pareto_no.setChecked(True)

                # update_list_params()
                draw_pareto_front(data)

        def show_population():
            # Очищаем существующий layout перед добавлением новой таблицы
            while ui_ga.verticalLayout_table_pop.count():
                item = ui_ga.verticalLayout_table_pop.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()

            ga = get_gen_an()

            if ga:
                n_features = len(list_param_reg)
                list_p = json.loads(ga.list_params)

                problem = Problem(n_features, 1 if ui_ga.radioButton_pareto_no.isChecked() else 2)

                # создаем отдельный объект Binary для каждой переменной
                for i in range(n_features):
                    problem.types[i] = Binary(1)  # Указываем размерность 1 для каждой переменной

                problem.directions[0] = Problem.MAXIMIZE  # Максимизация средней accuracy
                if ui_ga.radioButton_pareto_min.isChecked():
                    problem.directions[1] = Problem.MINIMIZE  # Минимизация числа признаков
                elif ui_ga.radioButton_pareto_max.isChecked():
                    problem.directions[1] = Problem.MAXIMIZE
                else:
                    pass

                try:
                    with open(ga.checkfile_path, "rb") as f:
                        saved = pickle.load(f)
                except FileNotFoundError:
                    QMessageBox.critical(GenAlg, "Error", "File not found")
                    return

                pop = []
                for x, fobj in zip(saved["X"], saved["F"]):
                    s = Solution(problem)
                    flat_x = [bool(v[0]) if isinstance(v, (list, tuple)) else bool(v)
                              for v in x]
                    s.variables[:] = flat_x
                    s.objectives[:] = fobj
                    s.evaluated = True
                    pop.append(s)

                data_dict = {}
                for sol in pop:
                    acc = sol.objectives[0]
                    n_feat = sum(sol.variables)
                    col_name = f"{acc:.4f}-{n_feat}"

                    # Добавляем каждое значение как список значений для каждого признака
                    for i, val in enumerate(sol.variables):
                        if i not in data_dict:
                            data_dict[i] = {}
                        data_dict[i][col_name] = val

                df = pd.DataFrame.from_dict(data_dict, orient='index')
                df.index = list_p

                rows, cols = df.shape

                table = QTableWidget(rows, cols)

                table.setHorizontalHeaderLabels(df.columns.tolist())
                table.setVerticalHeaderLabels(df.index.tolist())

                # заполняем ячейки
                for row in range(df.shape[0]):
                    for col in range(df.shape[1]):
                        val = df.iat[row, col]
                        item = QTableWidgetItem()
                        # можно не показывать текст, оставить пусто
                        # item.setText("1" if val else "0")
                        color = QColor('#ABF37F') if val else QColor('#FF8080')
                        item.setBackground(color)
                        table.setItem(row, col, item)

                ui_ga.verticalLayout_table_pop.addWidget(table)

        # def update_list_params():
        #     ui_ga.listWidget_features.clear()
        #     ga = get_gen_an()
        #     if ga:
        #         list_p = json.loads(ga.list_params)
        #         for i_param in tqdm(list_p):
        #             check_box_widget = QCheckBox(i_param)
        #             # check_box_widget.setChecked(True)
        #             list_item = QListWidgetItem()
        #             ui_ga.listWidget_features.addItem(list_item)
        #             ui_ga.listWidget_features.setItemWidget(list_item, check_box_widget)
        #

        #
        # def update_list_features():
        #     ga = get_gen_an()
        #     if ga:
        #         with open(ga.checkfile_path, "rb") as f:
        #             data = pickle.load(f)
        #
        #         list_x = []
        #         try:
        #             point = ui_ga.listWidget_population.currentItem().text().split(' N')
        #         except AttributeError:
        #             return
        #         for x, fobj in zip(data["X"], data["F"]):
        #             if str(fobj[0]) == point[0] and fobj[1] == int(point[1]):
        #                 list_x = list(x)
        #                 break
        #
        #         if list_x:
        #             for i in range(ui_ga.listWidget_features.count()):
        #                 checkbox = ui_ga.listWidget_features.itemWidget(ui_ga.listWidget_features.item(i))
        #                 checkbox.setChecked(list_x[i][0])

        def show_gen_an_info():
            ui_ga.textEdit_info.clear()
            ga = get_gen_an()
            if ga:
                ui_ga.textEdit_info.append(ga.type_problem)
                ui_ga.textEdit_info.append(ga.title)
                ui_ga.textEdit_info.append(ga.pipeline)
                ui_ga.textEdit_info.append(ga.comment)
                ui_ga.spinBox_pop_size.setValue(ga.population_size)
                # with open(ga.checkfile_path, "rb") as f:
                #     data = pickle.load(f)
                # ui_ga.lcdNumber_generation.display(data["ngen"])

                update_list_population()

        def draw_pareto_front(data):

            clear_layout(ui_ga.verticalLayout_pareto)
            figure_pareto = plt.figure()
            canvas_pareto = FigureCanvas(figure_pareto)
            mpl_toolbar = NavigationToolbar(canvas_pareto, GenAlg)
            ui_ga.verticalLayout_pareto.addWidget(mpl_toolbar)
            ui_ga.verticalLayout_pareto.addWidget(canvas_pareto)

            # Создание осей внутри фигуры
            ax = figure_pareto.add_subplot(111)

            # Построение точек на графике
            if isinstance(data["F"][0], (float, int)):
                counts = [np.sum(x) for x in data["X"]]
                accuracy = [data["F"]]
            else:

                counts = [f[1] for f in data["F"]]
                accuracy = [f[0] for f in data["F"]]

            ax.scatter(counts, accuracy, alpha=0.5)
            ax.set_xlabel("Количество признаков")
            ax.set_ylabel("Точность модели")
            ax.set_title("Парето-фронт")
            ax.grid(True)

            # Обновление канвы
            canvas_pareto.draw()

        def start_gen_algorithm():

            data_train_cov = data_train.copy()
            data_train_cov['obj_title'] = data_train_cov['prof_well_index'].apply(get_obj_title)

            training_sample = data_train_cov[list_param_reg]

            markup = data_train_cov[['target_value']]
            groups = data_train_cov[['obj_title']]

            # Нормализация данных
            text_scaler = ''

            pipe_steps = []
            if ui_r.checkBox_stdscaler_reg.isChecked():
                std_scaler = StandardScaler()
                pipe_steps.append(('scaler', std_scaler))
                text_scaler += '\nStandardScaler'
            if ui_r.checkBox_robscaler_reg.isChecked():
                robust_scaler = RobustScaler()
                pipe_steps.append(('scaler', robust_scaler))
                text_scaler += '\nRobustScaler'
            if ui_r.checkBox_mnmxscaler_reg.isChecked():
                minmax_scaler = MinMaxScaler()
                pipe_steps.append(('scaler', minmax_scaler))
                text_scaler += '\nMinMaxScaler'
            if ui_r.checkBox_mxabsscaler_reg.isChecked():
                maxabs_scaler = MaxAbsScaler()
                pipe_steps.append(('scaler', maxabs_scaler))
                text_scaler += '\nMaxAbsScaler'

            if ui_r.checkBox_pca.isChecked():
                n_comp = 'mle' if ui_r.checkBox_pca_mle.isChecked() else ui_r.spinBox_pca.value()
                pca = PCA(n_components=n_comp, random_state=0)
                pipe_steps.append(('pca', pca))
            text_pca = f'\nPCA: n_components={n_comp}' if ui_r.checkBox_pca.isChecked() else ''

            model_name = ui_r.buttonGroup.checkedButton().text()
            model_class, text_model = choice_model_regressor(model_name, training_sample)

            text_model += text_scaler
            text_model += text_pca

            pipe_steps.append(('model', model_class))
            pipe = Pipeline(pipe_steps)

            title = f'{model_name}_{len(list_param_reg)}_{str(ui_ga.spinBox_pop_size.value())}'
            if ui_ga.radioButton_pareto_min.isChecked():
                p_type = 'min'
            elif ui_ga.radioButton_pareto_max.isChecked():
                p_type = 'max'
            else:
                p_type = 'no'
            ga = session.query(GeneticAlgorithmReg).filter_by(
                analysis_id=get_regmod_id(),
                title=title,
                pipeline=text_model,
                list_params=json.dumps(list_param_reg),
                population_size=ui_ga.spinBox_pop_size.value(),
                type_problem=p_type
            ).first()
            if not ga:
                ga = new_gen_an(model_name, text_model, list_param_reg, p_type)

            # Определение задачи
            n_features = training_sample.shape[1]

            problem = Problem(n_features, 1 if ui_ga.radioButton_pareto_no.isChecked() else 2)

            # создаем отдельный объект Binary для каждой переменной
            for i in range(n_features):
                problem.types[i] = Binary(1)  # Указываем размерность 1 для каждой переменной

            problem.directions[0] = Problem.MAXIMIZE  # Максимизация средней accuracy
            if ui_ga.radioButton_pareto_min.isChecked():
                problem.directions[1] = Problem.MINIMIZE  # Минимизация числа признаков
            elif ui_ga.radioButton_pareto_max.isChecked():
                problem.directions[1] = Problem.MAXIMIZE
            else:
                pass

            # Целевая функция
            def objectives(features):

                selected_features = np.array(features, dtype=int)
                if np.sum(selected_features) == 0:
                    return [0, n_features]

                # Выбор активных признаков
                training_sample_subset = np.array(training_sample.loc[:, selected_features == 1].values.tolist())

                markup_subset = np.array(sum(markup.values.tolist(), []))
                groups_subset = np.array(sum(groups.values.tolist(), []))

                scores = []

                # Нормализация данных
                text_scaler = ''

                pipe_steps = []
                if ui_r.checkBox_stdscaler_reg.isChecked():
                    std_scaler = StandardScaler()
                    pipe_steps.append(('scaler', std_scaler))
                    text_scaler += '\nStandardScaler'
                if ui_r.checkBox_robscaler_reg.isChecked():
                    robust_scaler = RobustScaler()
                    pipe_steps.append(('scaler', robust_scaler))
                    text_scaler += '\nRobustScaler'
                if ui_r.checkBox_mnmxscaler_reg.isChecked():
                    minmax_scaler = MinMaxScaler()
                    pipe_steps.append(('scaler', minmax_scaler))
                    text_scaler += '\nMinMaxScaler'
                if ui_r.checkBox_mxabsscaler_reg.isChecked():
                    maxabs_scaler = MaxAbsScaler()
                    pipe_steps.append(('scaler', maxabs_scaler))
                    text_scaler += '\nMaxAbsScaler'

                if ui_r.checkBox_pca.isChecked():
                    n_comp = 'mle' if ui_r.checkBox_pca_mle.isChecked() else ui_r.spinBox_pca.value()
                    pca = PCA(n_components=n_comp, random_state=0)
                    pipe_steps.append(('pca', pca))
                text_pca = f'\nPCA: n_components={n_comp}' if ui_r.checkBox_pca.isChecked() else ''

                model_name = ui_r.buttonGroup.checkedButton().text()
                model_class, text_model = choice_model_regressor(model_name, training_sample)

                text_model += text_scaler
                text_model += text_pca

                pipe_steps.append(('model', model_class))
                pipe = Pipeline(pipe_steps)

                ui.progressBar.setMaximum(len(set(list(groups_subset))))
                n_progress = 1

                for train_idx, test_idx in LeaveOneGroupOut().split(training_sample_subset, markup_subset,
                                                                    groups_subset):
                    ui.progressBar.setValue(n_progress)

                    if ui_r.checkBox_cov_percent.isChecked():
                        if len(test_idx) / len(markup_subset) < ui_r.spinBox_cov_percent.value() / 100:
                            n_progress += 1
                            continue

                    pipe.fit(training_sample_subset[train_idx], markup_subset[train_idx])
                    score = pipe.score(training_sample_subset[test_idx], markup_subset[test_idx])
                    scores.append(score)

                count = np.sum(selected_features)
                print(np.mean(scores), count)
                ui_ga.progressBar_pop.setValue(ui_ga.progressBar_pop.value() + 1)

                if ui_ga.radioButton_pareto_no.isChecked():
                    return [np.mean(scores)]
                else:
                    return [np.mean(scores), count]

            problem.function = objectives

            # --- Параметры сохранения и выполнения ---
            population_size = ui_ga.spinBox_pop_size.value()  # Размер популяции
            total_generations = ui_ga.spinBox_n_gen.value()  # Общее количество поколений для выполнения
            save_interval = ui_ga.spinBox_save_int.value()  # Сохранять каждые N поколений
            checkpoint_file = ga.checkfile_path  # Файл для сохранения состояния

            ui_ga.progressBar_pop.setMaximum(population_size)
            ui_ga.progressBar_gen.setMaximum(total_generations)

            # --- Логика загрузки или инициализации ---
            start_gen = 0
            if os.path.exists(checkpoint_file):
                try:
                    print(f"Загрузка состояния из файла: {checkpoint_file}")

                    algorithm, start_gen = load_checkpoint(problem, checkpoint_file)
                    print(f"Возобновление с поколения {start_gen + 1}")

                except Exception as e:

                    print(f"Ошибка при загрузке файла {checkpoint_file}: {e}")
                    print("Начинаем новый запуск.")

                    if ui_ga.radioButton_pareto_no.isChecked():
                        algorithm = GeneticAlgorithm(problem, population_size=population_size)
                    else:
                        algorithm = NSGAII(problem, population_size=population_size)


            else:

                print("Файл состояния не найден. Начинаем новый запуск.")

                if ui_ga.radioButton_pareto_no.isChecked():
                    algorithm = GeneticAlgorithm(problem, population_size=population_size)
                else:
                    algorithm = NSGAII(problem, population_size=population_size)

            # --- Основной цикл выполнения с сохранением ---
            print(f"Запуск оптимизации с поколения {start_gen + 1} до {start_gen + total_generations}")

            n_gen = 0
            for gen in range(start_gen, total_generations + start_gen):
                ui_ga.lcdNumber_generation.display(gen)
                ui_ga.progressBar_pop.setValue(0)
                print(f"Поколение {gen + 1}/{total_generations + start_gen}...")
                algorithm.step()  # Выполняем одно поколение

                # Проверяем, нужно ли сохраняться
                if (gen + 1) % save_interval == 0:
                    print(f"Сохранение состояния в {checkpoint_file} после поколения {gen + 1}...")
                    try:
                        save_population(algorithm, checkpoint_file)
                        print("Состояние успешно сохранено.")
                    except Exception as e:
                        print(f"Ошибка при сохранении состояния: {e}")

                n_gen += 1
                ui_ga.progressBar_gen.setValue(n_gen)

            print("Оптимизация завершена.")

            # Получение результатов после завершения цикла
            results = algorithm.result
            # Дальнейшая обработка результатов...
            for solution in results:
                print(solution.objectives)
                print(solution.variables)

            update_combobox_gen_an()

        def new_gen_an(model_name, text_model, list_param, p_type):
            title = f'{model_name}_{len(list_param)}_{str(ui_ga.spinBox_pop_size.value())}'
            p_sep = os.path.sep
            filepath = f'genetic{p_sep}reg{p_sep}{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}{title}.pkl'
            new_gen = GeneticAlgorithmReg(
                analysis_id=get_regmod_id(),
                title=title,
                pipeline=text_model,
                checkfile_path=filepath,
                list_params=json.dumps(list_param),
                population_size=ui_ga.spinBox_pop_size.value(),
                type_problem=p_type
            )
            session.add(new_gen)
            session.commit()
            return new_gen

        def get_obj_title(prof_well_index):
            prof_id = prof_well_index.split('_')[0]
            obj = session.query(GeoradarObject).join(Research).join(Profile).filter(Profile.id == prof_id).first()
            return obj.title

        def save_population(alg, fname):
            data = dict(
                X=[s.variables[:] for s in alg.population],
                F=[
                    s.objectives[0] if len(s.objectives) == 1 else s.objectives[:]
                    for s in alg.population
                ],
                nfe=alg.nfe,
                rng=random.getstate(),
                ngen=alg.nfe // alg.population_size
            )
            with open(fname, "wb") as f:
                pickle.dump(data, f)

        def load_checkpoint(problem, fname, is_master_node=False):
            with open(fname, "rb") as f:
                data = pickle.load(f)

            pop = []
            for x, fobj in zip(data["X"], data["F"]):
                s = Solution(problem)
                s.variables[:] = x
                if problem.nobjs == 1:
                    # Одноцелевой: fobj — это скаляр
                    s.objectives[0] = fobj
                else:
                    # Многоцелевой: fobj — это список
                    s.objectives[:] = fobj
                s.evaluated = True
                pop.append(s)
                print(x)
                print(fobj)

            n_features = problem.nvars
            crossover = HUX()
            mutation = BitFlip(probability=1 / n_features)
            variator = CompoundOperator(crossover, mutation)
            # Выбор алгоритма по количеству целей
            if problem.nobjs == 1:
                alg = GeneticAlgorithm(problem,
                                       generator=InjectedPopulation(pop),
                                       variator=variator,
                                       population_size=len(pop))
            else:
                alg = NSGAII(problem,
                             generator=InjectedPopulation(pop),
                             variator=variator,
                             population_size=len(pop))

            alg.nfe = data["nfe"]
            if is_master_node:
                random.setstate(data["rng"])  # воспроизводимость
            else:
                random.seed()  # новое зерно из /dev/urandom

            alg.initialize()

            return alg, data["ngen"]

        def remove_gen_an():
            gen_an = get_gen_an()
            if os.path.exists(gen_an.checkfile_path):
                os.unlink(gen_an.checkfile_path)
            session.delete(gen_an)
            session.commit()
            update_combobox_gen_an()

        def _read_pop(fname, problem):
            with open(fname, "rb") as f:
                d = pickle.load(f)

            pop = []
            for x, fobj in zip(d["X"], d["F"]):
                s = Solution(problem)
                s.variables[:] = x
                if problem.nobjs == 1:
                    # Одноцелевой: fobj — это скаляр
                    s.objectives[0] = fobj
                else:
                    # Многоцелевой: fobj — это список
                    s.objectives[:] = fobj
                s.evaluated = True
                pop.append(s)
            return pop, d["nfe"]

        def _write_pop(pop, nfe, fname):
            data = dict(
                X=[s.variables[:] for s in pop],

                F=[
                    s.objectives[0] if len(s.objectives) == 1 else s.objectives[:]
                    for s in pop
                ],
                nfe=nfe,
                rng=random.getstate(),  # актуальное состояние ГСЧ
                ngen=nfe // len(pop) if len(pop) else 0
            )
            with open(fname, "wb") as f:
                pickle.dump(data, f)

        def select_best(population, k):
            """Возвратить k решений по рангу+crowding (NSGA‑II style)."""
            # Применяем nondominated_sort к популяции (функция модифицирует объекты)
            nondominated_sort(population)

            # Группируем решения по рангам
            ranks = {}
            for solution in population:
                if not hasattr(solution, 'rank'):
                    print("Warning: solution does not have 'rank' attribute after nondominated_sort")
                    continue

                rank = solution.rank
                if rank not in ranks:
                    ranks[rank] = []
                ranks[rank].append(solution)

            # Теперь мы имеем словарь, где ключи - ранги, значения - списки решений
            selected = []

            # Обрабатываем ранги в порядке возрастания (сначала лучшие)
            for rank in sorted(ranks.keys()):
                front = ranks[rank]
                crowding_distance(front)  # нужно для сортировки
                front.sort(key=lambda s: -s.crowding_distance)

                space_left = k - len(selected)
                selected.extend(front[:space_left])  # добираем столько, сколько нужно
                if len(selected) >= k:
                    break  # набрали k, выходим

            return selected

        def merge_checkpoints_to_file(problem,
                                      filenames: list[str],
                                      out_fname: str,
                                      target_size: int | None = None,
                                      nfe_mode: str = "max"):
            """Склеить pkl-файлы и записать новый.

            Parameters
            ----------
            problem      : ваш объект Problem (нужен для Solution)
            filenames    : список путей к pkl-файлам
            out_fname    : куда сохранить объединённый файл
            target_size  : None -> не ограничивать;
                           k    -> оставить k лучших по crowding
            nfe_mode     : 'max'  -> взять max(nfe)  из файлов;
                           'sum'  -> сумму; любое др. -> 0
            """
            all_pop, nfe_list = [], []

            for fn in filenames:
                pop, nfe = _read_pop(fn, problem)
                all_pop.extend(pop)
                nfe_list.append(nfe)

            if target_size:
                front = select_best(all_pop, target_size)
            else:
                front = select_best(all_pop, len(all_pop))

            # 3. выбираем счётчик nfe
            if nfe_mode == "max":
                new_nfe = max(nfe_list)
            elif nfe_mode == "sum":
                new_nfe = sum(nfe_list)
            else:
                new_nfe = 0

            # 4. сохраняем

            _write_pop(front, new_nfe, out_fname)
            print(f"Записан объединённый чек‑пойнт «{out_fname}» "
                  f"({len(front)} решений, nfe={new_nfe})")

        def add_file_gen_an():
            file_name_new = QFileDialog.getOpenFileName(filter='Pickle files (*.pkl)')[0]
            if file_name_new:
                gen_an = get_gen_an()
                file_name = gen_an.checkfile_path

                n_features = training_sample.shape[1]

                problem = Problem(n_features, 1 if ui_ga.radioButton_pareto_no.isChecked() else 2)

                # создаем отдельный объект Binary для каждой переменной
                for i in range(n_features):
                    problem.types[i] = Binary(1)  # Указываем размерность 1 для каждой переменной

                problem.directions[0] = Problem.MAXIMIZE  # Максимизация средней accuracy
                if ui_ga.radioButton_pareto_min.isChecked():
                    problem.directions[1] = Problem.MINIMIZE  # Минимизация числа признаков
                elif ui_ga.radioButton_pareto_max.isChecked():
                    problem.directions[1] = Problem.MAXIMIZE
                else:
                    pass

                merge_checkpoints_to_file(problem, [file_name, file_name_new], file_name,
                                          target_size=ui_ga.spinBox_pop_size.value(), nfe_mode="max")

                update_list_population()

        ui_ga.pushButton_start_gen.clicked.connect(start_gen_algorithm)
        ui_ga.comboBox_gen_analysis.currentIndexChanged.connect(show_gen_an_info)
        ui_ga.comboBox_gen_analysis.currentIndexChanged.connect(show_population)
        ui_ga.toolButton_remove_gen_an.clicked.connect(remove_gen_an)
        # ui_ga.listWidget_population.currentItemChanged.connect(update_list_features)
        ui_ga.pushButton_add_file.clicked.connect(add_file_gen_an)
        ui_ga.pushButton_save_mask.clicked.connect(save_mask)
        ui_ga.pushButton_remove_mask.clicked.connect(remove_mask)

        update_list_mask()
        update_combobox_gen_an()

        GenAlg.exec_()

    ui_r.pushButton_add_to_lineup.clicked.connect(add_model_reg_to_lineup)
    ui_r.pushButton_lof.clicked.connect(calc_lof)
    ui_r.pushButton_calc.clicked.connect(calc_model_reg)
    ui_r.pushButton_search_param.clicked.connect(random_search_reg)
    ui_r.pushButton_random_param.clicked.connect(Regressor.close)
    ui_r.pushButton_random_param.clicked.connect(push_random_param_reg)
    ui_r.pushButton_feature_selection.clicked.connect(call_feature_selection)
    ui_r.pushButton_cov.clicked.connect(calc_cov)
    ui_r.pushButton_gen_alg.clicked.connect(genetic_algorithm)
    ui_r.pushButton_cvw.clicked.connect(calc_model_regression_by_cvw)
    Regressor.exec_()




# def train_regression_model_old():
#     """ Расчет модели """
#     data_train, list_param = build_table_train(True, 'regmod')
#     list_param = get_list_param_numerical(list_param)
#     training_sample = data_train[list_param].values.tolist()
#     target = sum(data_train[['target_value']].values.tolist(), [])
#
#     Form_Regmod = QtWidgets.QDialog()
#     ui_frm = Ui_Form_formation_ai()
#     ui_frm.setupUi(Form_Regmod)
#     Form_Regmod.show()
#     Form_Regmod.setAttribute(QtCore.Qt.WA_DeleteOnClose)
#
#     def calc_regression_model():
#         start_time = datetime.datetime.now()
#         model = ui_frm.comboBox_model_ai.currentText()
#
#         pipe_steps = []
#         scaler = StandardScaler()
#         pipe_steps.append(('scaler', scaler))
#
#         x_train, x_test, y_train, y_test = train_test_split(
#             training_sample, target, test_size=0.2, random_state=42
#         )
#
#         model_name, model_regression = choose_regression_model(model)
#
#         pipe_steps.append(('model', model_regression))
#         pipe = Pipeline(pipe_steps)
#
#         if ui_frm.checkBox_cross_val.isChecked():
#             data_train_cross = data_train.copy()
#             kf = KFold(n_splits=ui_frm.spinBox_n_cross_val.value(), shuffle=True, random_state=0)
#             list_train, list_test, n_cross = [], [], 1
#             for train_index, test_index in kf.split(training_sample):
#                 list_train.append(train_index.tolist())
#                 list_test.append(test_index.tolist())
#                 list_test_to_table = ['x' if i in test_index.tolist() else 'o' for i in range(len(data_train.index))]
#                 data_train_cross[f'sample {n_cross}'] = list_test_to_table
#                 n_cross += 1
#             scores_cv = cross_val_score(pipe, training_sample, target, cv=kf)
#             n_max = np.argmax(scores_cv)
#             train_index, test_index = list_train[n_max], list_test[n_max]
#
#             x_train = [training_sample[i] for i in train_index]
#             x_test = [training_sample[i] for i in test_index]
#
#             y_train = [target[i] for i in train_index]
#             y_test = [target[i] for i in test_index]
#             if ui_frm.checkBox_cross_val_save.isChecked():
#                 fn = QFileDialog.getSaveFileName(caption="Сохранить выборку в таблицу",
#                                                  directory='table_cross_val.xlsx',
#                                                  filter="Excel Files (*.xlsx)")
#                 data_train_cross.to_excel(fn[0])
#
#             # print("Оценки на каждом разбиении:", scores_cv)
#             # print("Средняя оценка:", scores_cv.mean())
#             # print("Стандартное отклонение оценок:", scores_cv.std())
#
#         cv_text = (
#             f'\nКРОСС-ВАЛИДАЦИЯ\nОценки на каждом разбиении:\n {" / ".join(str(round(val, 2)) for val in scores_cv)}'
#             f'\nСредн.: {round(scores_cv.mean(), 2)} '
#             f'Станд. откл.: {round(scores_cv.std(), 2)}') if ui_frm.checkBox_cross_val.isChecked() else ''
#
#         pipe.fit(x_train, y_train)
#         y_pred = pipe.predict(x_test)
#
#         accuracy = round(pipe.score(x_test, y_test), 5)
#         mse = round(mean_squared_error(y_test, y_pred), 5)
#
#         train_time = datetime.datetime.now() - start_time
#         set_info(f'Модель {model}:\n точность: {accuracy} '
#                  f' Mean Squared Error:\n {mse}, \n время обучения: {train_time}', 'blue')
#         y_remain = [round(y_test[i] - y_pred[i], 5) for i in range(len(y_pred))]
#
#
#         data_graph = pd.DataFrame({
#             'y_test': y_test,
#             'y_pred': y_pred,
#             'y_remain': y_remain
#         })
#         try:
#             ipm_name_params, imp_params = [], []
#             for n, i in enumerate(pipe.feature_importances_):
#                 if i >= np.mean(pipe.feature_importances_):
#                     ipm_name_params.append(list_param[n])
#                     imp_params.append(i)
#
#             fig, axes = plt.subplots(nrows=2, ncols=2)
#             fig.set_size_inches(15, 10)
#             fig.suptitle(f'Модель {model}:\n точность: {accuracy} '
#                  f' Mean Squared Error:\n {mse}, \n время обучения: {train_time}' + cv_text)
#             sns.scatterplot(data=data_graph, x='y_test', y='y_pred', ax=axes[0, 0])
#             sns.regplot(data=data_graph, x='y_test', y='y_pred', ax=axes[0, 0])
#             sns.scatterplot(data=data_graph, x='y_test', y='y_remain', ax=axes[1, 0])
#             sns.regplot(data=data_graph, x='y_test', y='y_remain', ax=axes[1, 0])
#             if ui_frm.checkBox_cross_val.isChecked():
#                 axes[0, 1].bar(range(len(scores_cv)), scores_cv)
#                 axes[0, 1].set_title('Кросс-валидация')
#             else:
#                 axes[0, 1].bar(ipm_name_params, imp_params)
#                 axes[0, 1].set_xticklabels(ipm_name_params, rotation=90)
#             sns.histplot(data=data_graph, x='y_remain', kde=True, ax=axes[1, 1])
#             fig.tight_layout()
#             fig.show()
#         except AttributeError:
#             fig, axes = plt.subplots(nrows=2, ncols=2)
#             fig.set_size_inches(15, 10)
#             fig.suptitle(f'Модель {model}:\n точность: {accuracy} '
#                           f' Mean Squared Error:\n {mse}, \n время обучения: {train_time}' + cv_text)
#             sns.scatterplot(data=data_graph, x='y_test', y='y_pred', ax=axes[0, 0])
#             sns.regplot(data=data_graph, x='y_test', y='y_pred', ax=axes[0, 0])
#             sns.scatterplot(data=data_graph, x='y_test', y='y_remain', ax=axes[1, 0])
#             sns.regplot(data=data_graph, x='y_test', y='y_remain', ax=axes[1, 0])
#             if ui_frm.checkBox_cross_val.isChecked():
#                 axes[0, 1].bar(range(len(scores_cv)), scores_cv)
#                 axes[0, 1].set_title('Кросс-валидация')
#             sns.histplot(data=data_graph, x='y_remain', kde=True, ax=axes[1, 1])
#             fig.tight_layout()
#             fig.show()
#         if not ui_frm.checkBox_save.isChecked():
#             return
#         result = QtWidgets.QMessageBox.question(
#             MainWindow,
#             'Сохранение модели',
#             f'Сохранить модель {model}?',
#             QtWidgets.QMessageBox.Yes,
#             QtWidgets.QMessageBox.No)
#         if result == QtWidgets.QMessageBox.Yes:
#             # Сохранение модели в файл с помощью pickle
#             path_model = f'models/regression/{model_name}_{round(accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y")}.pkl'
#             with open(path_model, 'wb') as f:
#                 pickle.dump(pipe, f)
#
#             new_trained_model = TrainedModelReg(
#                 analysis_id=get_regmod_id(),
#                 title=f'{model_name}_{round(accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y")}',
#                 path_model=path_model,
#                 list_params=json.dumps(list_param),
#             )
#             session.add(new_trained_model)
#             session.commit()
#             update_list_trained_models_regmod()
#         else:
#             pass
#
#     def choose_regression_model(model):
#         if model == 'LinearRegression':
#             model_regression = LinearRegression(fit_intercept=ui_frm.checkBox_fit_intercept.isChecked())
#             model_name = 'LR'
#
#         elif model == 'DecisionTreeRegressor':
#             spl = 'random' if ui_frm.checkBox_splitter_rnd.isChecked() else 'best'
#             model_regression = DecisionTreeRegressor(splitter=spl, random_state=0)
#             model_name = 'DTR'
#
#         elif model == 'KNeighborsRegressor':
#             model_regression = KNeighborsRegressor(
#                 n_neighbors=ui_frm.spinBox_neighbors.value(),
#                 weights='distance' if ui_frm.checkBox_knn_weights.isChecked() else 'uniform',
#                 algorithm=ui_frm.comboBox_knn_algorithm.currentText()
#             )
#             model_name = 'KNNR'
#
#         elif model == 'SVR':
#             model_regression = SVR(kernel=ui_frm.comboBox_svr_kernel.currentText(),
#                                    C=ui_frm.doubleSpinBox_svr_c.value())
#             model_name = 'SVR'
#
#         elif model == 'MLPRegressor':
#             layers = tuple(map(int, ui_frm.lineEdit_layer_mlp.text().split()))
#             model_regression = MLPRegressor(
#                 hidden_layer_sizes=layers,
#                 activation=ui_frm.comboBox_activation_mlp.currentText(),
#                 solver=ui_frm.comboBox_solvar_mlp.currentText(),
#                 alpha=ui_frm.doubleSpinBox_alpha_mlp.value(),
#                 max_iter=5000,
#                 early_stopping=ui_frm.checkBox_e_stop_mlp.isChecked(),
#                 validation_fraction=ui_frm.doubleSpinBox_valid_mlp.value(),
#                 random_state=0
#             )
#             model_name = 'MLPR'
#
#         elif model == 'GradientBoostingRegressor':
#             model_regression = GradientBoostingRegressor(
#                 n_estimators=ui_frm.spinBox_n_estimators.value(),
#                 learning_rate=ui_frm.doubleSpinBox_learning_rate.value(),
#                 random_state=0
#             )
#             model_name = 'GBR'
#
#         elif model == 'ElasticNet':
#             model_regression = ElasticNet(
#                 alpha=ui_frm.doubleSpinBox_alpha.value(),
#                 l1_ratio=ui_frm.doubleSpinBox_l1_ratio.value(),
#                 random_state=0
#             )
#             model_name = 'EN'
#
#         elif model == 'Lasso':
#             model_regression = Lasso(alpha=ui_frm.doubleSpinBox_alpha.value(), random_state=0)
#             model_name = 'Lss'
#
#         else:
#             model_regression = LinearRegression(fit_intercept=ui_frm.checkBox_fit_intercept.isChecked())
#             model_name = 'LR'
#
#         return model_name, model_regression
#
#     ui_frm.pushButton_calc_model.clicked.connect(calc_regression_model)
#     Form_Regmod.exec_()


def update_list_trained_models_regmod():
    """  Обновление списка моделей """
    models = session.query(TrainedModelReg).filter(TrainedModelReg.analysis_id == get_regmod_id()).all()
    ui.listWidget_trained_model_reg.clear()
    for model in models:
        item_text = model.title
        item = QListWidgetItem(item_text)
        item.setData(Qt.UserRole, model.id)
        tool_tip_text = (f'{round(os.path.getsize(model.path_model) / 1048576, 4)} МБ\n'
                        f'ID {model.id}\n'
                        f'{model.comment}\n'
                        f'Количество параметров: '
                        f'{len(get_list_param_numerical(json.loads(model.list_params), model))}')
        if model.model_mask:
            tool_tip_text += f'\nMASK {str(model.model_mask[0].mask.count_param)}'
        item.setToolTip(tool_tip_text)
        ui.listWidget_trained_model_reg.addItem(item)
    ui.listWidget_trained_model_reg.setCurrentRow(0)


def remove_trained_model_regmod():
    """ Удаление модели """
    model = session.query(TrainedModelReg).filter_by(id=ui.listWidget_trained_model_reg.currentItem().data(Qt.UserRole)).first()
    model_mask = session.query(TrainedModelRegMask).filter_by(model_id=model.id).first()
    if model_mask:
        session.delete(model_mask)
    os.remove(model.path_model)
    session.delete(model)
    session.commit()
    update_list_trained_models_regmod()
    set_info(f'Модель {model.title} удалена', 'blue')


def calc_profile_model_regmod():
    """ Расчет профиля обученной моделью """

    try:
        working_data, curr_form = build_table_test('regmod')
    except TypeError:
        QMessageBox.critical(MainWindow, 'Ошибка', 'Недостаточно  параметров для рассчета по выбранной модели',
                             QMessageBox.Ok)
        return
    Choose_RegModel = QtWidgets.QDialog()
    ui_rm = Ui_FormRegMod()
    ui_rm.setupUi(Choose_RegModel)
    Choose_RegModel.show()
    Choose_RegModel.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    def calc_class_model():
        model = session.query(TrainedModelReg).filter_by(
        id=ui.listWidget_trained_model_reg.currentItem().data(Qt.UserRole)).first()
        list_param_num = get_list_param_numerical(json.loads(model.list_params), model)

        model_mask = session.query(TrainedModelRegMask).filter_by(model_id=model.id).first()
        if model_mask:
            list_param_num = sorted(json.loads(session.query(ParameterMask).filter_by(id=model_mask.mask_id).first().mask))

        try:
            working_sample = working_data[list_param_num].values.tolist()
        except KeyError:
            QMessageBox.critical(MainWindow, 'Ошибка', 'Для расчёта выбранной модели не хватает параметров',
                                 QMessageBox.Ok)
            return

        with open(model.path_model, 'rb') as f:
            reg_model = pickle.load(f)

        try:
            y_pred = reg_model.predict(working_sample)

        except ValueError:
            working_sample = [[np.nan if np.isinf(x) else x for x in y] for y in working_sample]
            data = imputer.fit_transform(working_sample)
            y_pred = reg_model.predict(data)

            for i in working_data.index:
                p_nan = [working_data.columns[ic + 3] for ic, v in enumerate(working_data.iloc[i, 3:].tolist()) if
                         np.isnan(v)]
                if len(p_nan) > 0:
                    set_info(f'Внимание для измерения "{i}" отсутствуют параметры "{", ".join(p_nan)}", поэтому расчёт для'
                             f' этого измерения может быть не корректен', 'red')
        ui.graph.clear()
        number = list(range(1, len(y_pred) + 1))
        # Создаем кривую и кривую, отфильтрованную с помощью savgol_filter
        curve = pg.PlotCurveItem(x=number, y=y_pred)
        curve_filter = pg.PlotCurveItem(x=number, y=savgol_filter(y_pred, 31, 3),
                                        pen=pg.mkPen(color='red', width=2.4))
        # Добавляем кривую и отфильтрованную кривую на график для всех пластов
        ui.graph.addItem(curve)
        ui.graph.addItem(curve_filter)

        result = QtWidgets.QMessageBox.question(ui.listWidget_well_regmod, 'Сохранение результата',
                                                'Вы хотите сохранить результат модели?',
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if result == QtWidgets.QMessageBox.Yes:
            new_prof_model_pred = ProfileModelPrediction(
                profile_id=get_profile_id(),
                type_model='reg',
                model_id=model.id,
                prediction=json.dumps(y_pred.tolist())
            )
            session.add(new_prof_model_pred)
            session.commit()
            set_info(f'Результат расчета модели "{model.title}" для профиля {get_profile_name()} сохранен', 'green')
            update_list_model_prediction()

    ui_rm.pushButton_calc_model.clicked.connect(calc_class_model)
    Choose_RegModel.exec_()


def calc_object_model_regmod():
    """ Расчет объекта обученной моделью """

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
    # working_data_result = pd.DataFrame()
    # list_formation = []
    # for n, prof in enumerate(session.query(Profile).filter(Profile.research_id == get_research_id()).all()):
    #     count_measure = len(json.loads(session.query(Profile.signal).filter(Profile.id == prof.id).first()[0]))
    #     ui.comboBox_profile.setCurrentText(f'{prof.title} ({count_measure} измерений) id{prof.id}')
    #     set_info(f'Профиль {prof.title} ({count_measure} измерений)', 'blue')
    #     update_formation_combobox()
    #     if len(prof.formations) == 1:
    #         list_formation.append(f'{prof.formations[0].title} id{prof.formations[0].id}')
    #         # ui.comboBox_plast.setCurrentText(f'{prof.formations[0].title} id{prof.formations[0].id}')
    #     elif len(prof.formations) > 1:
    #         Choose_Formation = QtWidgets.QDialog()
    #         ui_cf = Ui_FormationLDA()
    #         ui_cf.setupUi(Choose_Formation)
    #         Choose_Formation.show()
    #         Choose_Formation.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия
    #         for f in prof.formations:
    #             ui_cf.listWidget_form_lda.addItem(f'{f.title} id{f.id}')
    #         ui_cf.listWidget_form_lda.setCurrentRow(0)
    #
    #         def form_mlp_ok():
    #             list_formation.append(ui_cf.listWidget_form_lda.currentItem().text())
    #             # ui.comboBox_plast.setCurrentText(ui_cf.listWidget_form_lda.currentItem().text())
    #             Choose_Formation.close()
    #
    #         ui_cf.pushButton_ok_form_lda.clicked.connect(form_mlp_ok)
    #         Choose_Formation.exec_()
    if ui.checkBox_save_prof_reg.isChecked():

        model = session.query(TrainedModelReg).filter_by(id=ui.listWidget_trained_model_reg.currentItem().data(Qt.UserRole)).first()
        with open(model.path_model, 'rb') as f:
            reg_model = pickle.load(f)

        list_param_num = get_list_param_numerical(json.loads(model.list_params), model)

        model_mask = session.query(TrainedModelRegMask).filter_by(model_id=model.id).first()
        if model_mask:
            list_param_num = sorted(json.loads(session.query(ParameterMask).filter_by(id=model_mask.mask_id).first().mask))

    for n, prof in enumerate(session.query(Profile).filter(Profile.research_id == get_research_id()).all()):
        count_measure = len(json.loads(session.query(Profile.signal).filter(Profile.id == prof.id).first()[0]))
        ui.comboBox_profile.setCurrentText(f'{prof.title} ({count_measure} измерений) id{prof.id}')
        update_formation_combobox()
        ui.comboBox_plast.setCurrentText(list_formation[n])
        working_data, curr_form = build_table_test('regmod')

        if ui.checkBox_save_prof_reg.isChecked():

            if session.query(ProfileModelPrediction).filter_by(
                    profile_id=get_profile_id(), model_id=model.id, type_model='reg').count() == 0:
                working_data_profile = working_data.copy()
                working_sample_profile = working_data_profile[list_param_num].values.tolist()

                try:
                    y_pred = reg_model.predict(working_sample_profile)
                except ValueError:
                    working_sample = [[np.nan if np.isinf(x) else x for x in y] for y in working_sample_profile]
                    data = imputer.fit_transform(working_sample)
                    y_pred = reg_model.predict(data)

                new_prof_model_pred = ProfileModelPrediction(
                    profile_id=prof.id,
                    type_model='reg',
                    model_id=model.id,
                    prediction=json.dumps(y_pred.tolist())
                )
                session.add(new_prof_model_pred)
                session.commit()
                set_info(f'Результат расчета модели "{model.title}" для профиля {prof.title} сохранен', 'green')

        working_data_result = pd.concat([working_data_result, working_data], axis=0, ignore_index=True)

    update_list_model_prediction()
    working_data_result_copy = working_data_result.copy()

    Choose_RegModel = QtWidgets.QDialog()
    ui_rm = Ui_FormRegMod()
    ui_rm.setupUi(Choose_RegModel)
    Choose_RegModel.show()
    Choose_RegModel.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    def calc_regmod():
        model = session.query(TrainedModelReg).filter_by(id=ui.listWidget_trained_model_reg.currentItem().data(Qt.UserRole)).first()
        with open(model.path_model, 'rb') as f:
            reg_model = pickle.load(f)

        list_param_num = get_list_param_numerical(json.loads(model.list_params), model)

        model_mask = session.query(TrainedModelRegMask).filter_by(model_id=model.id).first()
        if model_mask:
            list_param_num = sorted(json.loads(session.query(ParameterMask).filter_by(id=model_mask.mask_id).first().mask))

        working_sample = working_data_result_copy[list_param_num].values.tolist()

        try:
            y_pred = reg_model.predict(working_sample)
        except ValueError:
            working_sample = [[np.nan if np.isinf(x) else x for x in y] for y in working_sample]
            data = imputer.fit_transform(working_sample)
            y_pred = reg_model.predict(data)

            for i in working_data_result_copy.index:
                p_nan = [working_data_result_copy.columns[ic + 3] for ic, v in enumerate(working_data_result_copy.iloc[i, 3:].tolist()) if
                         np.isnan(v)]
                if len(p_nan) > 0:
                    set_info(f'Внимание для измерения "{i}" отсутствуют параметры "{", ".join(p_nan)}", поэтому расчёт для'
                             f' этого измерения может быть не корректен', 'red')


        working_data_result_copy['value'] = y_pred
        x = list(working_data_result_copy['x_pulc'])
        y = list(working_data_result_copy['y_pulc'])
        z = list(working_data_result_copy['value'])

        draw_map(x, y, z, ui.listWidget_trained_model_reg.currentItem().text(), False)
        result1 = QMessageBox.question(MainWindow, 'Сохранение', 'Сохранить результаты расчёта регрессионной модели?', QMessageBox.Yes,
                                       QMessageBox.No)
        if result1 == QMessageBox.Yes:
            result2 = QMessageBox.question(MainWindow, 'Сохранение', 'Сохранить только результаты расчёта?', QMessageBox.Yes,
                                           QMessageBox.No)
            if result2 == QMessageBox.Yes:
                list_col = ['x_pulc', 'y_pulc', 'value']
                working_data_result_excel = working_data_result_copy[list_col]
            else:
                working_data_result_excel = working_data_result.copy()
            try:
                file_name = f'{get_object_name()}_{get_research_name()}__модель_{ui.listWidget_trained_model_reg.currentItem().text()}.xlsx'
                fn = QFileDialog.getSaveFileName(
                    caption=f'Сохранить результат регрессионной модели "{get_object_name()}_{get_research_name()}" в таблицу',
                    directory=file_name,
                    filter="Excel Files (*.xlsx)")
                working_data_result_excel.to_excel(fn[0])
                set_info(f'Таблица сохранена в файл: {fn[0]}', 'green')
            except ValueError:
                pass
        else:
            pass

    ui_rm.pushButton_calc_model.clicked.connect(calc_regmod)
    Choose_RegModel.exec_()


def copy_regmod():
    if ui.lineEdit_string.text() == '':
        set_info('Введите название для копии анализа', 'red')
        return
    old_regmod = session.query(AnalysisReg).filter_by(id=get_regmod_id()).first()
    new_regmod = AnalysisReg(title=ui.lineEdit_string.text())
    session.add(new_regmod)
    session.commit()
    for old_markup in session.query(MarkupReg).filter_by(analysis_id=get_regmod_id()):
        new_markup = MarkupReg(
            analysis_id=new_regmod.id,
            well_id=old_markup.well_id,
            profile_id=old_markup.profile_id,
            formation_id=old_markup.formation_id,
            target_value=old_markup.target_value,
            list_measure=old_markup.list_measure,
            type_markup=old_markup.type_markup
        )
        session.add(new_markup)
    session.commit()
    update_list_reg()
    set_info(f'Скопирован набор для регрессионного анализа - "{old_regmod.title}"', 'green')


def copy_regmod_by_except():
    if ui.lineEdit_string.text() == '':
        set_info('Введите ID анализа для проверки наличия в нем скважин', 'red')
        return
    old_regmod = session.query(AnalysisReg).filter_by(id=get_regmod_id()).first()
    check_regmod = session.query(AnalysisReg).filter_by(id=int(ui.lineEdit_string.text())).first()
    if not check_regmod:
        set_info('Нет анализа с таким ID для проверки', 'red')
        return
    new_regmod = AnalysisReg(title=f'{old_regmod.title}_check_by_ID{ui.lineEdit_string.text()}')
    session.add(new_regmod)
    session.commit()
    for old_markup in session.query(MarkupReg).filter_by(analysis_id=get_regmod_id()):
        if not session.query(MarkupReg).filter_by(
                analysis_id=check_regmod.id,
                well_id=old_markup.well_id,
                profile_id=old_markup.profile_id
        ).first():
            continue
        new_markup = MarkupReg(
            analysis_id=new_regmod.id,
            well_id=old_markup.well_id,
            profile_id=old_markup.profile_id,
            formation_id=old_markup.formation_id,
            target_value=old_markup.target_value,
            list_measure=old_markup.list_measure,
            type_markup=old_markup.type_markup
        )
        session.add(new_markup)
    session.commit()
    update_list_reg()
    set_info(f'Скопирован набор для регрессионного анализа - "{old_regmod.title}"', 'green')


def rename_regmod():
    session.query(AnalysisReg).filter(AnalysisReg.id == get_regmod_id()).update(
        {"title": ui.lineEdit_string.text()}, synchronize_session='fetch')

    update_list_reg()


def calc_corr_regmod():
    """ Расчет корреляции """

    if not session.query(AnalysisReg).filter(AnalysisReg.id == get_regmod_id()).first().up_data:
        update_list_param_regmod()
    data_train, list_param = build_table_train(True, 'regmod')
    data_corr = data_train.iloc[:, 2:]

    list_param = list(data_corr.columns)
    corr_gist = []
    for i in data_corr.corr():
        corr_gist.append(np.abs(data_corr.corr()[i]).mean())

    fig = plt.figure(figsize=(20, 12))
    colors = ['#57e389', '#f66151'] * (len(list_param) // 2)
    if len(list_param) > 60:
        plt.bar(list_param, corr_gist, align='edge', width=1, color=colors)
        plt.tick_params(rotation=90)
        plt.grid()
    else:
        ax = plt.subplot2grid((1, 3), (0, 0), colspan=2)
        sns.heatmap(data_corr.corr(), xticklabels=list_param, yticklabels=list_param, cmap='RdYlGn', annot=True)
        ax = plt.subplot2grid((1, 3), (0, 2))
        ax.barh(range(1, len(list_param) + 1), corr_gist, align='edge', tick_label=list_param, color=colors)
        plt.xlim(np.min(corr_gist) - 0.05, np.max(corr_gist) + 0.05)
        plt.ylim(1, len(list_param) + 1)
        ax.invert_yaxis()
        ax.grid()
    fig.tight_layout()
    fig.show()


def anova_regmod():
    """ Отображение ANOVA """

    Anova = QtWidgets.QDialog()
    ui_anova = Ui_Anova()
    ui_anova.setupUi(Anova)
    Anova.show()
    Anova.setAttribute(QtCore.Qt.WA_DeleteOnClose) # атрибут удаления виджета после закрытия

    # ui_anova.graphicsView.setBackground('w')

    data_plot, list_param = build_table_train(True, 'regmod')

    figure = plt.figure()
    canvas = FigureCanvas(figure)
    ui_anova.horizontalLayout.addWidget(canvas)


    for i in sorted(data_plot.columns.tolist()[2:]):
        ui_anova.listWidget.addItem(i)

    def draw_graph_anova():
        figure.clear()
        param = ui_anova.listWidget.currentItem().text()
        sns.kdeplot(data=data_plot, x=param, y='target_value', fill=True)
        sns.scatterplot(data=data_plot, x=param, y='target_value')
        sns.regplot(data=data_plot, x=param, y='target_value', line_kws={'color': 'red'})

        # x = data_plot[param]
        # y = data_plot['target_value']
        #
        # a, b = np.polyfit(x, y, deg=1)
        # y_est = a * x + b
        # y_err = x.std() * np.sqrt(1 / len(x) +
        #                           (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
        #
        # plt.plot(x, y_est, '-', 'k', linewidth=2)
        # plt.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.75)
        plt.grid()
        figure.suptitle(f'Коэффициент корреляции: {np.corrcoef(data_plot[param], data_plot["target_value"])[0, 1]}')
        figure.tight_layout()
        canvas.draw()

    ui_anova.listWidget.currentItemChanged.connect(draw_graph_anova)

    Anova.exec_()


def clear_fake_reg():
    """ Удаление выбросов """

    session.query(MarkupReg).filter(MarkupReg.analysis_id == get_regmod_id()).update({'list_fake': None},
                                                                                  synchronize_session='fetch')
    session.commit()
    set_info(f'Выбросы для анализа "{ui.comboBox_regmod.currentText()}" очищены.', 'green')
    set_updata_false_reg()
    build_table_train(False, 'regmod')
    update_list_well_markup_reg()


def get_model_param_list_reg():
    """ Функция для вывода параметров модели и сравнения с параметрами других моделей """

    try:
        model = session.query(TrainedModelReg).filter_by(id=ui.listWidget_trained_model_reg.currentItem().data(Qt.UserRole)).first()
    except AttributeError:
        QMessageBox.critical(MainWindow, 'Не выбрана модель', 'Не выбрана модель.')
        set_info('Не выбрана модель', 'red')
        return

    FormParams = QtWidgets.QDialog()
    ui_p = Ui_Form_ModelParams()
    ui_p.setupUi(FormParams)
    FormParams.show()
    FormParams.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    ui_p.label.setText(model.title + ' parameters: ')
    ui_p.label_crl.setText(ui_p.label_crl.text() + model.except_crl)
    ui_p.label_signal.setText(ui_p.label_signal.text() + model.except_signal)
    params = json.loads(model.list_params)
    ui_p.listWidget_parameters.addItems(sorted(params))
    ui_p.label_count.setText(ui_p.label_count.text() + str(len(params)))

    def highlight_common_items(list_widget, common_params):
        for index in range(list_widget.count()):
            item = list_widget.item(index)
            if item.text() in common_params:
                item.setBackground(QtGui.QColor("green"))

    def add_and_compare_models(type_model):
        ui_p.listWidget_parameters_2.clear()
        ui_p.label_2.setText('Model: ')
        ui_p.label_crl_2.setText('Except CRL: ')
        ui_p.label_signal_2.setText('Except Signal: ')
        ui_p.label_count_2.setText('Params count: ')
        [ui_p.listWidget_parameters.item(i).setBackground(QtGui.QColor("white"))
            for i in range(ui_p.listWidget_parameters.count())]

        try:
            if type_model == 'cls':
                model_compare = session.query(TrainedModelClass).filter_by(
                    id=ui.listWidget_trained_model_class.currentItem().data(Qt.UserRole)).first()
            elif type_model == 'reg':
                model_compare = session.query(TrainedModelReg).filter_by(
                    id=ui.listWidget_trained_model_reg.currentItem().data(Qt.UserRole)).first()
        except AttributeError:
            QMessageBox.critical(MainWindow, 'Не выбрана модель', 'Не выбрана модель.')
            set_info('Не выбрана модель', 'red')
            return

        if not model_compare:
            set_info('Не выбрана модель', 'red')
            return

        ui_p.label_2.setText(ui_p.label_2.text() + model_compare.title)
        ui_p.label_crl_2.setText(ui_p.label_crl_2.text() + model_compare.except_crl)
        ui_p.label_signal_2.setText(ui_p.label_signal_2.text() + model_compare.except_signal)
        params_compare = json.loads(model_compare.list_params)
        ui_p.listWidget_parameters_2.addItems(sorted(params_compare))
        ui_p.label_count_2.setText(ui_p.label_count_2.text() + str(len(params_compare)))

        common_params = list(filter(lambda x: x in params_compare, params))
        highlight_common_items(ui_p.listWidget_parameters_2, common_params)
        highlight_common_items(ui_p.listWidget_parameters, common_params)

        ui_p.label_common_param.setText('Совпало параметров: ' + str(len(common_params)))

    ui_p.pushButton_add_class.clicked.connect(lambda: add_and_compare_models('cls'))
    ui_p.pushButton_add_reg.clicked.connect(lambda: add_and_compare_models('reg'))

    FormParams.exec_()



def update_trained_model_reg_comment():
    """ Изменить комментарий обученной модели """
    try:
        an = session.query(TrainedModelReg).filter_by(id=ui.listWidget_trained_model_reg.currentItem().data(Qt.UserRole)).first()
    except AttributeError:
        QMessageBox.critical(MainWindow, 'Не выбрана модель', 'Не выбрана модель.')
        set_info('Не выбрана модель', 'red')
        return

    FormComment = QtWidgets.QDialog()
    ui_cmt = Ui_Form_comment()
    ui_cmt.setupUi(FormComment)
    FormComment.show()
    FormComment.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    ui_cmt.textEdit.setText(an.comment)

    def update_comment():
        session.query(TrainedModelReg).filter_by(id=an.id).update({'comment': ui_cmt.textEdit.toPlainText()}, synchronize_session='fetch')
        session.commit()
        update_list_trained_models_regmod()
        FormComment.close()

    ui_cmt.buttonBox.accepted.connect(update_comment)

    FormComment.exec_()


def export_model_reg():
    """ Экспорт модели """
    try:
        model = session.query(TrainedModelReg).filter_by(
        id=ui.listWidget_trained_model_reg.currentItem().data(Qt.UserRole)).first()
    except AttributeError:
        return set_info('Не выбрана модель', 'red')

    analysis = session.query(AnalysisReg).filter_by(id=model.analysis_id).first()

    model_parameters = {
        'analysis_title': analysis.title,
        'title': model.title,
        'list_params': model.list_params,
        'comment': model.comment,
        'except_crl': model.except_crl,
        'except_signal': model.except_signal
    }

    # Сохранение словаря с параметрами в файл *.pkl
    with open('model_parameters.pkl', 'wb') as parameters_file:
        pickle.dump(model_parameters, parameters_file)
    try:
        filename = QFileDialog.getSaveFileName(caption='Экспорт регрессионной модели', directory=f'{model.title}.zip', filter="*.zip")[0]
        with zipfile.ZipFile(filename, 'w') as zip:
            zip.write('model_parameters.pkl', 'model_parameters.pkl')
            zip.write(model.path_model, 'model.pkl')

        set_info(f'Модель {model.title} экспортирована в файл {filename}', 'blue')
    except FileNotFoundError:
        pass


def import_model_reg():
    """ Импорт модели """
    try:
        filename = QFileDialog.getOpenFileName(caption='Импорт регрессионной модели', filter="*.zip")[0]

        with zipfile.ZipFile(filename, 'r') as zip:
            zip.extractall('extracted_data')

            with open('extracted_data/model_parameters.pkl', 'rb') as parameters_file:
                loaded_parameters = pickle.load(parameters_file)

            # Загрузка модели из файла *.pkl
            with open('extracted_data/model.pkl', 'rb') as model_file:
                loaded_model = pickle.load(model_file)
    except FileNotFoundError:
        return

    analysis_title = loaded_parameters['analysis_title']
    model_name = loaded_parameters['title']
    list_params = loaded_parameters['list_params']
    comment = loaded_parameters['comment']
    except_crl = loaded_parameters['except_crl']
    except_signal = loaded_parameters['except_signal']

    path_model = f'models/regression/{model_name}.pkl'

    with open(path_model, 'wb') as f:
        pickle.dump(loaded_model, f)

    analysis = session.query(AnalysisReg).filter_by(title=analysis_title).first()
    if not analysis:
        new_analisys = AnalysisReg(title=analysis_title)
        session.add(new_analisys)
        session.commit()
        analysis = new_analisys
        update_list_reg()

    new_trained_model = TrainedModelReg(
        analysis_id=analysis.id,
        title=model_name,
        path_model=path_model,
        list_params=list_params,
        except_crl=except_crl,
        except_signal=except_signal,
        comment=comment
    )
    session.add(new_trained_model)
    session.commit()
    try:
        shutil.rmtree('extracted_data')
        os.remove('model_parameters.pkl')
    except FileNotFoundError:
        pass

    update_list_trained_models_regmod()
    set_info(f'Модель {model_name} добавлена', 'blue')


def export_well_markup_reg():
    """ Сохранение тренировочных точек """

    dir = QFileDialog.getExistingDirectory()
    if not dir:
        return

    markups = session.query(MarkupReg).filter_by(analysis_id=get_regmod_id()).all()
    ui.progressBar.setMaximum(len(markups))
    for n,i in enumerate(markups):
        if i.type_markup != 'intersection':
            return set_info('Функция только для данных мониторинга', 'red')
        ui.progressBar.setValue(n+1)
        save_markup(i.id, dir, n)
    set_info('Тренирочные точки сохранены в папку ' + dir, 'blue')


def save_markup(markup_id, dir, num):
    fig = plt.figure(figsize=(14, 14))
    ax1 = fig.add_subplot(211)
    all_x, all_y = [], []
    reg_markup = session.query(MarkupReg).filter_by(id=markup_id).first()
    profiles = session.query(Profile).filter(Profile.research_id == reg_markup.formation.profile.research_id).all()
    data_incl = session.query(ParameterHWell).join(HorizontalWell).filter(
        HorizontalWell.object_id == reg_markup.formation.profile.research.object_id,
        ParameterHWell.parameter == 'Инклинометрия').all()
    for incl in data_incl:
        if incl.h_well.thermograms:
            coord_inc = json.loads(incl.data)
            all_x.extend(coord[0] for coord in coord_inc)
            all_y.extend(coord[1] for coord in coord_inc)
            continue
    ax1.scatter(all_x, all_y, marker='.', s=1, color='blue')
    for p in profiles:
        xs = json.loads(p.x_pulc)
        ys = json.loads(p.y_pulc)
        ax1.scatter(xs, ys, marker='.', s=1, color='green')
    for p in profiles:
        for intersection in p.intersections:
            if intersection.id == reg_markup.well_id:
                size, fonts, color_mark, color_text = 130, 10, 'red', 'red'
            else:
                size, fonts, color_mark, color_text = 50, 7, 'orange', 'gray'
            ax1.scatter(intersection.x_coord, intersection.y_coord, marker='o', s=size, color=color_mark)
            ax1.text(intersection.x_coord + 25, intersection.y_coord + 25, round(intersection.temperature, 1),
                     fontsize=fonts, color=color_text)
    ax1.set_aspect('equal')
    map_name = f'{reg_markup.formation.profile.research.object.title} - {reg_markup.formation.profile.research.date_research.strftime("%d.%m.%Y")}'
    ax1.set_title(map_name)
    ax2 = fig.add_subplot(212)
    ints = session.query(Intersection).filter_by(id=reg_markup.well_id).first()
    therm = json.loads(ints.thermogram.therm_data)
    data_therm = [i for i in therm if len(i) > 2]
    x_therm = [i[0] for i in therm]
    y_therm = [i[1] for i in therm]
    ax2.plot(x_therm, y_therm)
    ax2.vlines(data_therm[ints.i_therm][0], 0, ints.temperature, color='green', linestyles='dashed', lw=2)
    ax2.hlines(xmin=min(x_therm), xmax=data_therm[ints.i_therm][0], y=ints.temperature, color='red',
               linestyles='dashed')
    ax2.text(min(x_therm) + 25, ints.temperature - 5, 't=' + str(round(ints.temperature, 1)) + '°C', fontsize=16,
             color='red')
    ax2.text(data_therm[ints.i_therm][0] + 25, ints.temperature - 10, 'пр. ' + ints.profile.title, fontsize=16,
             color='green')
    ax2.set_xlim(min(x_therm), max(x_therm))
    ax2.grid()
    graph_name = f'скважина {ints.thermogram.h_well.title}, термограмма {ints.thermogram.date_time.strftime("%d.%m.%Y")}'
    ax2.set_title(graph_name)
    fig.tight_layout()
    fig.savefig(f'{dir}/{num+1}_{map_name}_{graph_name}.png')
    plt.close(fig)

def add_signal_except_reg():
    """ Список исключений для параметра Signal """
    check_str = parse_range_exception(ui.lineEdit_signal_except_reg.text())
    if not check_str:
        set_info('Неверный формат диапазона исключений', 'red')
        return
    except_line = '' if check_str == -1 else ui.lineEdit_signal_except_reg.text()
    excetp_signal = session.query(ExceptionReg).filter_by(analysis_id=get_regmod_id()).first()
    if excetp_signal:
        excetp_signal.except_signal = except_line
    else:
        new_except = ExceptionReg(
            analysis_id=get_regmod_id(),
            except_signal=except_line
        )
        session.add(new_except)
    set_updata_false_reg()
    session.commit()
    set_color_button_updata_regmod()
    set_info('Исключения добавлены', 'green')


def add_crl_except_reg():
    """ Список исключений для параметра Crl """
    check_str = parse_range_exception(ui.lineEdit_crl_except_reg.text())
    if not check_str:
        set_info('Неверный формат диапазона исключений', 'red')
        return
    except_line = '' if check_str == -1 else ui.lineEdit_crl_except_reg.text()
    except_crl = session.query(ExceptionReg).filter_by(analysis_id=get_regmod_id()).first()
    if except_crl:
        except_crl.except_crl = except_line
    else:
        new_except = ExceptionReg(
            analysis_id=get_regmod_id(),
            except_crl=except_line
        )
        session.add(new_except)
    set_updata_false_reg()
    session.commit()
    set_color_button_updata_regmod()
    set_info('Исключения добавлены', 'green')


def list_param_reg_to_lineEdit():
    """ Выгрузка параметров в строку """

    model = session.query(TrainedModelReg).filter_by(id=ui.listWidget_trained_model_reg.currentItem().data(
        Qt.UserRole)).first()

    if not model:
        return
    ex_sig, ex_crl = model.except_signal.split(','), model.except_crl.split(',')
    sig_up = ex_sig[0].split('-')[1] if model.except_signal else '0'
    crl_up = ex_crl[0].split('-')[1] if model.except_crl else '0'
    sig_down = str(int(ex_sig[-1].split('-')[1]) - int(ex_sig[-1].split('-')[0])) if model.except_signal else '512'
    crl_down = str(int(ex_sig[-1].split('-')[1]) - int(ex_sig[-1].split('-')[0])) if model.except_crl else '512'


    list_param_model = []
    for param in json.loads(model.list_params):
        parts = param.split('_')
        if param.startswith('Signal'):
            list_param_model.append(f'sig_{parts[1]}_{sig_up}_{sig_down}')
        elif param == 'CRL':
            list_param_model.append(f'sig_CRL_{crl_up}_{crl_down}')
        elif param == 'CRL_NF':
            list_param_model.append(f'sig_CRLNF_{crl_up}_{crl_down}')
        else:
            list_param_model.append(param)
    print(list_param_model)
    line_param = '//'.join(list_param_model)
    print(line_param)
    ui.lineEdit_string.setText(line_param)


def get_feature_importance_reg():
    """ Вывод наиболее значимых параметров для моделей, поддерживающих feature_importances """

    model = session.query(TrainedModelReg).filter_by(id=ui.listWidget_trained_model_reg.currentItem().data(
        Qt.UserRole)).first()

    if not model:
        return

    if 'GBC' in model.title or 'LGBM' in model.title or 'RFC' in model.title or 'XGB' in model.title:
        with open(model.path_model, 'rb') as f:
            reg_model = pickle.load(f)

        params = json.loads(model.list_params)

        full_params = get_list_param_numerical(params, model)
        feature_importances = reg_model.named_steps['model'].feature_importances_

        feature_importance_df = pd.DataFrame({
            'Feature': full_params,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        print(feature_importance_df.head(30))


    pass


def markup_to_excel_reg():
    """ Сохранение результатов анализа в файл Excel """

    list_col = ['Целевое значение', 'Объект', 'Профиль', 'Интервал', 'Измерения', 'Выбросы', 'Скважина',
                'Альтитуда', 'Удаленность', 'X', 'Y', 'Примечание']
    analisis = session.query(AnalysisReg).filter_by(id=get_regmod_id()).first()
    pd_markup = pd.DataFrame(columns=list_col)
    ui.progressBar.setMaximum(len(analisis.markups))
    n = 1
    for mrp in analisis.markups:
        ui.progressBar.setValue(n)
        mrp_dict = dict()
        mrp_dict['Целевое значение'] = mrp.target_value
        mrp_dict['Объект'] = f'{mrp.profile.research.object.title}_{mrp.profile.research.date_research.year}'
        mrp_dict['Профиль'] = mrp.profile.title
        mrp_dict['Интервал'] = mrp.formation.title
        mrp_dict['Измерения'] = mrp.list_measure
        mrp_dict['Выбросы'] = mrp.list_fake
        mrp_dict['Скважина'] = mrp.well.name
        mrp_dict['Альтитуда'] = mrp.well.alt
        mrp_dict['X'] = mrp.well.x_coord
        mrp_dict['Y'] = mrp.well.y_coord
        mrp_dict['Удаленность'] = closest_point(mrp.well.x_coord, mrp.well.y_coord, json.loads(mrp.profile.x_pulc),
                                                json.loads(mrp.profile.y_pulc))[1]
        list_opt = []
        for opt in mrp.well.well_optionally:
            list_opt.append(f'{opt.option} - {opt.value}')
        mrp_dict['Примечание'] = "\n".join(list_opt)

        pd_markup = pd.concat([pd_markup, pd.DataFrame(data=mrp_dict, columns=list_col, index=[0])], axis = 0, ignore_index=True)
        n += 1

    file_name = QFileDialog.getSaveFileName(
        None,
        "Сохранить результаты анализа",
        "",
        "Excel Files (*.xlsx)"
    )[0]
    if not file_name:
        return
    # Добавляем .xlsx, если его нет в конце
    if not file_name.lower().endswith('.xlsx'):
        file_name += '.xlsx'
    pd_markup.to_excel(file_name, index=False)

    # Сохранение с фиксированной шириной столбцов
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        pd_markup.to_excel(writer, index=False, sheet_name='Результаты анализа')

        workbook = writer.book
        worksheet = writer.sheets['Результаты анализа']

        # Создаем форматы
        header_format = workbook.add_format({
            'bold': True,
            'align': 'center',
            'valign': 'vcenter',
            'border': 1
        })

        cell_format = workbook.add_format({
            'text_wrap': True,
            'align': 'left',
            'valign': 'vcenter'
        })

        # Устанавливаем фиксированную ширину для всех столбцов (в символах)
        column_widths = {
            'Целевое значение': 18,
            'Объект': 18,
            'Профиль': 15,
            'Интревал': 20,
            'Измерения': 30,
            'Выбросы': 10,
            'Скважина': 10,
            'Альтитуда': 12,
            'Удаленность': 12,
            'X': 10,
            'Y': 10,
            'Примечание': 40
        }

        # Применяем к заголовкам
        worksheet.set_row(0, None, header_format)

        # Применяем ширину и формат
        for idx, col_name in enumerate(pd_markup.columns):
            worksheet.set_column(
                idx, idx,
                width=column_widths.get(col_name, 15),  # 15 - ширина по умолчанию
                cell_format=cell_format
            )


