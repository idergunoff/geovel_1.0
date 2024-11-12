import pandas as pd
from PyQt5.QtWidgets import QDialog

from draw import draw_radarogram, draw_formation, draw_fill, draw_fake, draw_fill_result, remove_poly_item
from formation_ai import get_model_id
from func import *
from build_table import *
from nn_torch_classifier import *
from krige import draw_map
from qt.choose_formation_lda import *
from random_search import push_random_search
from classification_func import train_classifier
from regression import update_list_reg


def add_mlp():
    """Добавить новый анализ MLP"""
    if ui.lineEdit_string.text() == '':
        set_info('Введите название анализа', 'red')
        return
    new_mlp = AnalysisMLP(title=ui.lineEdit_string.text())
    session.add(new_mlp)
    session.commit()
    update_list_mlp()
    set_info(f'Добавлен новый анализ MLP - "{ui.lineEdit_string.text()}"', 'green')


def copy_mlp():
    """Скопировать анализ MLP"""
    if ui.lineEdit_string.text() == '':
        set_info('Введите название для копии анализа', 'red')
        return
    old_mlp = session.query(AnalysisMLP).filter_by(id=get_MLP_id()).first()
    new_mlp = AnalysisMLP(title=ui.lineEdit_string.text())
    session.add(new_mlp)
    session.commit()
    for old_marker in old_mlp.markers:
        new_marker = MarkerMLP(analysis_id=new_mlp.id, title=old_marker.title, color=old_marker.color)
        session.add(new_marker)
        for old_markup in session.query(MarkupMLP).filter_by(analysis_id=get_MLP_id(), marker_id=old_marker.id):
            new_markup = MarkupMLP(
                analysis_id=new_mlp.id,
                well_id=old_markup.well_id,
                profile_id=old_markup.profile_id,
                formation_id=old_markup.formation_id,
                marker_id=new_marker.id,
                list_measure=old_markup.list_measure,
                type_markup=old_markup.type_markup
            )
            session.add(new_markup)
    session.commit()
    update_list_mlp()
    set_info(f'Скопирован анализ MLP - "{old_mlp.title}"', 'green')


def copy_mlp_to_lda():
    """Скопировать анализ MLP в LDA"""
    from lda import update_list_lda

    if ui.lineEdit_string.text() == '':
        set_info('Введите название для копии анализа', 'red')
        QMessageBox.critical(MainWindow, 'Ошибка', 'Введите название для копии анализа в поле в верхней части главного окна')
        return
    old_mlp = session.query(AnalysisMLP).filter_by(id=get_MLP_id()).first()
    new_lda = AnalysisLDA(title=ui.lineEdit_string.text())
    session.add(new_lda)
    session.commit()
    for old_marker in old_mlp.markers:
        new_marker = MarkerLDA(analysis_id=new_lda.id, title=old_marker.title, color=old_marker.color)
        session.add(new_marker)
        for old_markup in session.query(MarkupMLP).filter_by(analysis_id=get_MLP_id(), marker_id=old_marker.id):
            new_markup = MarkupLDA(
                analysis_id=new_lda.id,
                well_id=old_markup.well_id,
                profile_id=old_markup.profile_id,
                formation_id=old_markup.formation_id,
                marker_id=new_marker.id,
                list_measure=old_markup.list_measure,
                type_markup=old_markup.type_markup
            )
            session.add(new_markup)
    session.commit()
    build_table_train_no_db('lda', new_lda.id, [])
    update_list_lda()
    set_info(f'Скопирован анализ MLP - "{old_mlp.title}"', 'green')


def copy_mlp_to_regmod():
    """Скопировать анализ MLP в регрессионную модель"""
    if ui.lineEdit_string.text() == '':
        set_info('Введите название для копии анализа', 'red')
        return
    old_mlp = session.query(AnalysisMLP).filter_by(id=get_MLP_id()).first()
    new_regmod = AnalysisReg(title=ui.lineEdit_string.text())
    session.add(new_regmod)
    session.commit()

    for old_markup in session.query(MarkupMLP).filter_by(analysis_id=get_MLP_id()):
        if old_markup.type_markup == 'intersection':
            intersection = session.query(Intersection).filter(Intersection.id == old_markup.well_id).first()
            well_dist = ui.spinBox_well_dist_reg.value()
            start = intersection.i_profile - well_dist if intersection.i_profile - well_dist > 0 else 0
            len_prof = len(json.loads(intersection.profile.x_pulc))
            stop = intersection.i_profile + well_dist if intersection.i_profile + well_dist < len_prof else len_prof
            list_measure = list(range(start, stop))
            new_markup = MarkupReg(
                analysis_id=new_regmod.id,
                well_id=old_markup.well_id,
                profile_id=old_markup.profile_id,
                formation_id=old_markup.formation_id,
                target_value=round(intersection.temperature, 2),
                list_measure=json.dumps(list_measure),
                type_markup=old_markup.type_markup
            )
        else:
            well = session.query(Well).filter(Well.id == old_markup.well_id).first()
            x_prof = json.loads(session.query(Profile.x_pulc).filter(Profile.id == old_markup.profile_id).first()[0])
            y_prof = json.loads(session.query(Profile.y_pulc).filter(Profile.id == old_markup.profile_id).first()[0])
            index, _ = closest_point(well.x_coord, well.y_coord, x_prof, y_prof)
            well_dist = ui.spinBox_well_dist_reg.value()
            start = index - well_dist if index - well_dist > 0 else 0
            stop = index + well_dist if index + well_dist < len(x_prof) else len(x_prof)
            list_measure = list(range(start, stop))
            new_markup = MarkupReg(
                analysis_id=new_regmod.id,
                well_id=old_markup.well_id,
                profile_id=old_markup.profile_id,
                formation_id=old_markup.formation_id,
                target_value=0,
                list_measure=json.dumps(list_measure),
                type_markup=old_markup.type_markup
            )
        session.add(new_markup)
    session.commit()
    update_list_reg()
    set_info(f'Скопирован анализ MLP - "{old_mlp.title}"', 'green')

def remove_mlp():
    """Удалить анализ MLP"""
    mlp_title = get_mlp_title()
    result = QtWidgets.QMessageBox.question(ui.listWidget_well_mlp, 'Remove markup MLP',
                                            f'Вы уверены, что хотите удалить модель MLP "{mlp_title}"?',
                                            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        session.query(ParameterMLP).filter_by(analysis_id=get_MLP_id()).delete()
        session.query(MarkerMLP).filter_by(analysis_id=get_MLP_id()).delete()
        session.query(MarkupMLP).filter_by(analysis_id=get_MLP_id()).delete()
        session.query(AnalysisMLP).filter_by(id=get_MLP_id()).delete()
        session.query(ExceptionMLP).filter_by(analysis_id=get_MLP_id()).delete()
        for model in session.query(TrainedModelClass).filter_by(analysis_id=get_MLP_id()).all():
            os.remove(model.path_model)
            session.delete(model)
        session.commit()
        set_info(f'Удалена модель MLP - "{mlp_title}"', 'green')
        update_list_mlp()
    elif result == QtWidgets.QMessageBox.No:
        pass


def update_list_mlp(db=False):
    """Обновить список анализов MLP"""
    ui.comboBox_mlp_analysis.clear()
    for i in session.query(AnalysisMLP.id, AnalysisMLP.title).order_by(AnalysisMLP.title).all():
        ui.comboBox_mlp_analysis.addItem(f'{i.title} id{i.id}')
    if db:
        update_list_marker_mlp_db()
    else:
        update_list_marker_mlp()
    update_list_trained_models_class()


def add_marker_mlp():
    """Добавить новый маркер MLP"""
    if ui.lineEdit_string.text() == '':
        set_info('Введите название маркера', 'red')
        return
    if session.query(MarkerMLP).filter_by(title=ui.lineEdit_string.text(), analysis_id=get_MLP_id()).count () > 0:
        session.query(MarkerMLP).filter_by(title=ui.lineEdit_string.text(), analysis_id=get_MLP_id()).update(
            {'color': ui.pushButton_color.text()}, synchronize_session='fetch')
        set_info(f'Изменен цвет маркера MLP - "{ui.lineEdit_string.text()}"', 'green')
    else:
        new_marker = MarkerMLP(title=ui.lineEdit_string.text(), analysis_id=get_MLP_id(), color=ui.pushButton_color.text())
        session.add(new_marker)
        set_info(f'Добавлен новый маркер MLP - "{ui.lineEdit_string.text()}"', 'green')
    session.commit()
    update_list_marker_mlp()


def edit_marker_mlp():
    """Редактировать маркер MLP"""
    if ui.lineEdit_string.text() == '':
        set_info('Введите новое название маркера', 'red')
        return
    session.query(MarkerMLP).filter_by(id=get_marker_mlp_id()).update(
            {'title': ui.lineEdit_string.text()}, synchronize_session='fetch')
    session.commit()
    set_info(f'Изменено название маркера - "{ui.lineEdit_string.text()}"', 'green')
    update_list_marker_mlp()


def remove_marker_mlp():
    """Удалить маркер MLP"""
    marker_title = get_marker_mlp_title()
    result = QtWidgets.QMessageBox.question(ui.listWidget_well_mlp, 'Remove marker MLP',
                                            f'В модели {session.query(MarkupMLP).filter_by(marker_id=get_marker_mlp_id()).count()} скважин отмеченных '
                                            f'этим маркером. Вы уверены, что хотите удалить маркер MLP "{marker_title}" вместе с обучающими скважинами'
                                            f' из модели "{get_mlp_title()}"?',
                                            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        session.query(MarkupMLP).filter_by(marker_id=get_marker_mlp_id()).delete()
        session.query(MarkerMLP).filter_by(id=get_marker_mlp_id()).delete()
        session.commit()
        set_info(f'Удалена маркер MLP - "{marker_title}"', 'green')
        update_list_mlp()
    elif result == QtWidgets.QMessageBox.No:
        pass


def update_list_marker_mlp():
    """Обновить список маркеров MLP"""
    ui.comboBox_mark_mlp.clear()
    for i in session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).order_by(MarkerMLP.id).all():
        item = f'{i.title} id{i.id}'
        ui.comboBox_mark_mlp.addItem(f'{i.title} id{i.id}')
        ui.comboBox_mark_mlp.setItemData(ui.comboBox_mark_mlp.findText(item), QBrush(QColor(i.color)),
                                         Qt.BackgroundRole)
    update_list_well_markup_mlp()
    update_list_param_mlp(False)


def update_list_marker_mlp_db():
    """Обновить список маркеров MLP"""
    ui.comboBox_mark_mlp.clear()
    time = datetime.datetime.now()
    for i in session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).order_by(MarkerMLP.id).all():
        item = f'{i.title} id{i.id}'
        ui.comboBox_mark_mlp.addItem(f'{i.title} id{i.id}')
        ui.comboBox_mark_mlp.setItemData(ui.comboBox_mark_mlp.findText(item), QBrush(QColor(i.color)),
                                         Qt.BackgroundRole)
    update_list_well_markup_mlp()
    # update_list_param_mlp(True)


def add_well_markup_mlp():
    """Добавить новую обучающую скважину для MLP"""
    analysis_id = get_MLP_id()
    well_id = get_well_id()
    profile_id = get_profile_id()
    formation_id = get_formation_id()
    marker_id = get_marker_mlp_id()

    if analysis_id and well_id and profile_id and marker_id and formation_id:
        remove_all_param_geovel_mlp()
        # for param in get_list_param_mlp():
        #     if not session.query(literal_column(f'Formation.{param}')).filter(Formation.id == formation_id).first()[0]:
        #         set_info(f'Параметр {param} отсутствует для профиля {get_profile_name()}', 'red')
        #         return
        if ui.checkBox_profile_intersec.isChecked():
            x_prof = json.loads(session.query(Profile.x_pulc).filter(Profile.id == profile_id).first()[0])
            inter = session.query(Intersection).filter(Intersection.id == well_id).first()
            well_dist = ui.spinBox_well_dist_mlp.value()
            start = inter.i_profile - well_dist if inter.i_profile - well_dist > 0 else 0
            stop = inter.i_profile + well_dist if inter.i_profile + well_dist < len(x_prof) else len(x_prof)
            list_measure = list(range(start, stop))
            new_markup_mlp = MarkupMLP(analysis_id=analysis_id, well_id=well_id, profile_id=profile_id,
                                       marker_id=marker_id, formation_id=formation_id,
                                       list_measure=json.dumps(list_measure), type_markup='intersection')
        else:
            well = session.query(Well).filter(Well.id == well_id).first()
            x_prof = json.loads(session.query(Profile.x_pulc).filter(Profile.id == profile_id).first()[0])
            y_prof = json.loads(session.query(Profile.y_pulc).filter(Profile.id == profile_id).first()[0])
            index, _ = closest_point(well.x_coord, well.y_coord, x_prof, y_prof)
            well_dist = ui.spinBox_well_dist_mlp.value()
            start = index - well_dist if index - well_dist > 0 else 0
            stop = index + well_dist if index + well_dist < len(x_prof) else len(x_prof)
            list_measure = list(range(start, stop))
            new_markup_mlp = MarkupMLP(analysis_id=analysis_id, well_id=well_id, profile_id=profile_id,
                                       marker_id=marker_id, formation_id=formation_id,
                                       list_measure=json.dumps(list_measure))
        session.add(new_markup_mlp)
        session.commit()
        set_info(f'Добавлена новая обучающая скважина для MLP - "{get_well_name()} {get_marker_mlp_title()}"', 'green')
        update_list_well_markup_mlp()
    else:
        set_info('выбраны не все параметры', 'red')


def add_profile_mlp():
    """Добавить часть профиля в обучающую выборку MLP"""
    AddProfClass = QtWidgets.QDialog()
    ui_apc = Ui_AddProfileClass()
    ui_apc.setupUi(AddProfClass)
    AddProfClass.show()
    AddProfClass.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    form = session.query(Formation).filter(Formation.id == get_formation_id()).first()
    ui_apc.spinBox_from.setMaximum(len(json.loads(form.T_top)))
    ui_apc.spinBox_to.setMaximum(len(json.loads(form.T_top)))
    ui_apc.spinBox_from.setValue(0)
    ui_apc.spinBox_to.setValue(len(json.loads(form.T_top)))
    ui_apc.label_question.setText(f'Добавить часть профиля {form.profile.title}')


    def draw_border_addition():
        global l_from, l_to
        if globals().get('l_from') and globals().get('l_to'):
            radarogramma.removeItem(l_from)
            radarogramma.removeItem(l_to)
        l_from = pg.InfiniteLine(pos=ui_apc.spinBox_from.value(), angle=90, pen=pg.mkPen(color='white', width=2, dash=[8, 2]))
        l_to = pg.InfiniteLine(pos=ui_apc.spinBox_to.value(), angle=90, pen=pg.mkPen(color='white', width=2, dash=[8, 2]))
        radarogramma.addItem(l_from)
        radarogramma.addItem(l_to)


    def add_profile_mlp_to_db():
        new_markup_mlp = MarkupMLP(
            analysis_id = get_MLP_id(),
            well_id = 0,
            profile_id = get_profile_id(),
            formation_id = get_formation_id(),
            marker_id = get_marker_mlp_id(),
            list_measure = json.dumps(list(range(ui_apc.spinBox_from.value(), ui_apc.spinBox_to.value()))),
            type_markup = 'profile'
        )
        session.add(new_markup_mlp)
        session.commit()
        set_info(f'Добавлена новая обучающий профиль для MLP - "{get_profile_name()}"', 'green')
        update_list_well_markup_mlp()
        add_profile_mlp_close()


    def add_profile_mlp_close():
        radarogramma.removeItem(l_from)
        radarogramma.removeItem(l_to)
        AddProfClass.close()


    draw_border_addition()
    ui_apc.spinBox_from.valueChanged.connect(draw_border_addition)
    ui_apc.spinBox_to.valueChanged.connect(draw_border_addition)
    ui_apc.buttonBox.accepted.connect(add_profile_mlp_to_db)
    ui_apc.buttonBox.rejected.connect(add_profile_mlp_close)

    AddProfClass.exec_()



def update_well_markup_mlp():
    markup = session.query(MarkupMLP).filter(MarkupMLP.id == get_markup_mlp_id()).first()
    if not markup:
        return
    if markup.type_markup == 'intersection':
        x_prof = json.loads(session.query(Profile.x_pulc).filter(Profile.id == markup.profile_id).first()[0])
        well = session.query(Intersection).filter(Intersection.id == markup.well_id).first()
        well_dist = ui.spinBox_well_dist_mlp.value()
        start = well.i_profile - well_dist if well.i_profile - well_dist > 0 else 0
        stop = well.i_profile + well_dist if well.i_profile + well_dist < len(x_prof) else len(x_prof)
        list_measure = list(range(start, stop))
    elif markup.type_markup == 'profile':
        list_measure = json.loads(markup.list_measure)
    else:
        well = session.query(Well).filter(Well.id == markup.well_id).first()
        x_prof = json.loads(session.query(Profile.x_pulc).filter(Profile.id == markup.profile_id).first()[0])
        y_prof = json.loads(session.query(Profile.y_pulc).filter(Profile.id == markup.profile_id).first()[0])
        index, _ = closest_point(well.x_coord, well.y_coord, x_prof, y_prof)
        well_dist = ui.spinBox_well_dist_mlp.value()
        start = index - well_dist if index - well_dist > 0 else 0
        stop = index + well_dist if index + well_dist < len(x_prof) else len(x_prof)
        list_measure = list(range(start, stop))
    form_id = get_formation_id()
    session.query(MarkupMLP).filter(MarkupMLP.id == get_markup_mlp_id()).update(
        {'marker_id': get_marker_mlp_id(), 'list_measure': json.dumps(list_measure), 'formation_id': form_id})
    session.commit()
    if markup.type_markup == 'profile':
        set_info(f'Изменен обучающий профиль для MLP - "{get_profile_name()}"', 'green')
    else:
        set_info(f'Изменена обучающая скважина для MLP - "{well.name} {get_marker_mlp_title()}"', 'green')
    update_list_well_markup_mlp()


def remove_well_markup_mlp():
    markup = session.query(MarkupMLP).filter(MarkupMLP.id == get_markup_mlp_id()).first()
    if not markup:
        return
    n_widget = ui.listWidget_well_mlp.currentRow()
    skv_name = 'profile markup' if markup.type_markup == 'profile' else session.query(Well.name).filter(Well.id == markup.well_id).first()[0]
    prof_name = session.query(Profile.title).filter(Profile.id == markup.profile_id).first()[0]
    mlp_name = session.query(AnalysisMLP.title).filter(AnalysisMLP.id == markup.analysis_id).first()[0]
    result = QtWidgets.QMessageBox.question(ui.listWidget_well_mlp, 'Remove markup MLP',
                                            f'Вы уверены, что хотите удалить скважину "{skv_name}" на '
                                            f'профиле "{prof_name}" из обучающей модели MLP-анализа "{mlp_name}"?',
                                            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        session.delete(markup)
        session.commit()
        set_info(f'Удалена обучающая скважина для MLP - "{ui.listWidget_well_mlp.currentItem().text()}"', 'green')
        update_list_well_markup_mlp()
        ui.listWidget_well_mlp.setCurrentRow(n_widget)
    elif result == QtWidgets.QMessageBox.No:
        pass





def choose_marker_mlp():
    # Функция выбора маркера MLP
    # Выбирает маркер, на основе сохраненных данных из базы данных, и затем обновляет все соответствующие виджеты
    # пользовательского интерфейса

    # Получение информации о маркере из БД по его ID
    markup = session.query(MarkupMLP).filter(MarkupMLP.id == get_markup_mlp_id()).first()
    # Если ID маркера не найден в БД, то функция завершается
    if not markup:
        return

    # Установка соответствующих значений виджетов пользовательского интерфейса
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
    if markup.type_markup == 'profile':
        pass
    else:
        draw_intersection(markup.well_id) if markup.type_markup == 'intersection' else draw_well(markup.well_id)
    list_measure = json.loads(markup.list_measure)  # Получение списка измерений
    list_fake = json.loads(markup.list_fake) if markup.list_fake else []  # Получение списка пропущенных измерений
    list_up = json.loads(markup.formation.layer_up.layer_line)  # Получение списка с верхними границами формации
    list_down = json.loads(markup.formation.layer_down.layer_line)  # Получение списка со снижными границами формации
    y_up = [list_up[i] for i in list_measure]  # Создание списка верхних границ для отображения
    y_down = [list_down[i] for i in list_measure]  # Создание списка нижних границ для отображения
    # Обновление маркера с конкретными данными о верхней и нижней границах и цветом
    draw_fill(list_measure, y_up, y_down, markup.marker.color)
    draw_fake(list_fake, list_up, list_down)


def split_well_train_test_mlp():
    markups = session.query(MarkupMLP).filter_by(analysis_id=get_MLP_id()).all()
    list_markers = [mrk.id for mrk in session.query(MarkerMLP).filter_by(analysis_id=get_MLP_id()).all()]
    list_mkp_id = [mkp.id for mkp in markups if not mkp.type_markup]
    list_data = [[mkp.well.x_coord, mkp.well.y_coord, list_markers.index(mkp.marker.id)] for mkp in markups if not mkp.type_markup]

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(list_data)
    if ui.radioButton_clusters_mlp.isChecked():
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

    x_coords = [mkp.well.x_coord for mkp in markups if not mkp.type_markup]
    y_coords = [mkp.well.y_coord for mkp in markups if not mkp.type_markup]
    values = [list_markers.index(mkp.marker.id) for mkp in markups if not mkp.type_markup]


    test_x = [mkp.well.x_coord for mkp in markups if mkp.id in test_ids]
    test_y = [mkp.well.y_coord for mkp in markups if mkp.id in test_ids]
    test_values = [list_markers.index(mkp.marker.id) for mkp in markups if mkp.id in test_ids]

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
        old_mlp = session.query(AnalysisMLP).filter_by(id=get_MLP_id()).first()
        new_mlp_train = AnalysisMLP(title=f'{ui.lineEdit_string.text()}_train')
        new_mlp_test = AnalysisMLP(title=f'{ui.lineEdit_string.text()}_test')
        session.add(new_mlp_train)
        session.add(new_mlp_test)
        session.commit()
        for old_marker in old_mlp.markers:
            new_marker_train = MarkerMLP(analysis_id=new_mlp_train.id, title=old_marker.title, color=old_marker.color)
            new_marker_test = MarkerMLP(analysis_id=new_mlp_test.id, title=old_marker.title, color=old_marker.color)
            session.add(new_marker_train)
            session.add(new_marker_test)
            session.commit()

            for old_markup in session.query(MarkupMLP).filter_by(analysis_id=get_MLP_id(), marker_id=old_marker.id):
                if old_markup.type_markup:
                    continue
                new_markup = MarkupMLP(
                    analysis_id=new_mlp_test.id if old_markup.id in test_ids else new_mlp_train.id,
                    well_id=old_markup.well_id,
                    profile_id=old_markup.profile_id,
                    formation_id=old_markup.formation_id,
                    marker_id=new_marker_test.id if old_markup.id in test_ids else new_marker_train.id,
                    list_measure=old_markup.list_measure
                )
                session.add(new_markup)
        session.commit()
        update_list_mlp()
        set_info(f'Выборка разделена на {ui.lineEdit_string.text()}_train и {ui.lineEdit_string.text()}_test', 'green')


def add_param_signal_mlp():
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    param = ui.comboBox_signal_mlp.currentText()
    if session.query(ParameterMLP).filter_by(
            analysis_id=get_MLP_id(),
            parameter=param
    ).count() == 0:
        add_param_mlp(param)
        update_list_param_mlp_no_update()
        set_color_button_updata()
    else:
        set_info(f'Параметр {param} уже добавлен', 'red')


def add_all_param_signal_mlp():
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    list_param_signal = ['Signal_Abase', 'Signal_diff', 'Signal_At', 'Signal_Vt', 'Signal_Pht', 'Signal_Wt']
    for param in list_param_signal:
        if session.query(ParameterMLP).filter_by(
                analysis_id=get_MLP_id(),
                parameter=param
        ).count() == 0:
            add_param_mlp(param)
        else:
            set_info(f'Параметр {param} уже добавлен', 'red')
    update_list_param_mlp_no_update()
    set_color_button_updata()


def add_param_crl_mlp():
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    if session.query(ParameterMLP).filter_by(
            analysis_id=get_MLP_id(),
            parameter='CRL'
    ).count() == 0:
        add_param_mlp('CRL')
        update_list_param_mlp_no_update()
        set_color_button_updata()
        set_info(f'Параметр CRL добавлен', 'green')
    else:
        set_info(f'Параметр CRL уже добавлен', 'red')


def add_param_crl_nf_mlp():
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    if session.query(ParameterMLP).filter_by(
            analysis_id=get_MLP_id(),
            parameter='CRL_NF'
    ).count() == 0:
        add_param_mlp('CRL_NF')
        update_list_param_mlp_no_update()
        set_color_button_updata()
        set_info(f'Параметр CRL_NF добавлен', 'green')
    else:
        set_info(f'Параметр CRL_NF уже добавлен', 'red')


def add_param_geovel_mlp():
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    param = ui.comboBox_geovel_param_mlp.currentText()
    if not param in list_all_additional_features + ['X', 'Y']:
        for m in session.query(MarkupMLP).filter(MarkupMLP.analysis_id == get_MLP_id()).all():
            if not session.query(literal_column(f'Formation.{param}')).filter(Formation.id == m.formation_id).first()[0]:
                set_info(f'Параметр {param} отсутствует для профиля {m.profile.title}', 'red')
                return
    if session.query(ParameterMLP).filter_by(
            analysis_id=get_MLP_id(),
            parameter= param
    ).count() == 0:
        add_param_mlp(param)
        set_color_button_updata()
        update_list_param_mlp_no_update()
    else:
        set_info(f'Параметр {param} уже добавлен', 'red')


def add_param_list_mlp():
    analysis_id = get_MLP_id()
    session.query(ParameterMLP).filter_by(analysis_id=get_MLP_id()).delete()
    session.query(AnalysisMLP).filter_by(id=analysis_id).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    check_except = False
    for i in ui.lineEdit_string.text().split('//'):
        param = i.split('_')
        if param[0] == 'sig':
            if param[1] == 'CRL':
                if session.query(ParameterMLP).filter_by(
                        analysis_id=analysis_id,
                        parameter='CRL'
                ).count() == 0:

                    new_param_mlp = ParameterMLP(analysis_id=get_MLP_id(), parameter='CRL')
                    session.add(new_param_mlp)
                    session.commit()

                else:
                    set_info(f'Параметр CRL уже добавлен', 'red')
            elif param[1] == 'CRLNF':
                if session.query(ParameterMLP).filter_by(
                        analysis_id=analysis_id,
                        parameter='CRL_NF'
                ).count() == 0:
                    new_param_mlp = ParameterMLP(analysis_id=get_MLP_id(), parameter='CRL_NF')
                    session.add(new_param_mlp)
                    session.commit()
                else:
                    set_info(f'Параметр CRL_NF уже добавлен', 'red')
            else:
                if session.query(ParameterMLP).filter_by(
                        analysis_id=analysis_id,
                        parameter=f'Signal_{param[1]}'
                ).count() == 0:
                    new_param_mlp = ParameterMLP(analysis_id=get_MLP_id(), parameter=f'Signal_{param[1]}')
                    session.add(new_param_mlp)
                    session.commit()
                else:
                    set_info(f'Параметр Signal_{param[1]} уже добавлен', 'red')
            if not check_except:
                str_exeption = f'1-{param[2]},{f"{str(512 - int(param[3]))}-512" if int(param[3]) > 0  else ""}'
                session.query(ExceptionMLP).filter_by(analysis_id=analysis_id).update({'except_signal': str_exeption,
                                                                              'except_crl': str_exeption},
                                                                             synchronize_session='fetch')
                session.commit()
                check_except = True
        elif param[0] in ['distr', 'sep', 'mfcc']:
            if param[1] == 'CRL':
                new_param_mlp = ParameterMLP(analysis_id=get_MLP_id(), parameter=f'{param[0]}_SigCRL_{param[2]}')
                session.add(new_param_mlp)
            else:
                new_param_mlp = ParameterMLP(analysis_id=get_MLP_id(), parameter=i)
                session.add(new_param_mlp)
            session.commit()
        else:
            if session.query(ParameterMLP).filter_by(
                    analysis_id=analysis_id,
                    parameter=i
            ).count() == 0:
                new_param_mlp = ParameterMLP(analysis_id=get_MLP_id(), parameter=i)
                session.add(new_param_mlp)
                session.commit()
            else:
                set_info(f'Параметр {i} уже добавлен', 'red')
    set_color_button_updata()
    update_list_param_mlp_no_update()
    update_line_edit_exception_mlp()


def add_all_param_geovel_mlp():
    new_list_param = ['X', 'Y'] + list_param_geovel + list_all_additional_features
    for param in list_param_geovel:
        for m in session.query(MarkupMLP).filter(MarkupMLP.analysis_id == get_MLP_id()).all():
            if not session.query(literal_column(f'Formation.{param}')).filter(Formation.id == m.formation_id).first()[0]:
                if param in new_list_param:
                    set_info(f'Параметр {param} отсутствует для профиля {m.profile.title}', 'red')
                    new_list_param.remove(param)
    for param in new_list_param:
        if session.query(ParameterMLP).filter(ParameterMLP.analysis_id == get_MLP_id()).filter(
                ParameterMLP.parameter == param).count() > 0:
            set_info(f'Параметр {param} уже добавлен', 'red')
            continue
        add_param_mlp(param)
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    set_color_button_updata()
    update_list_param_mlp_no_update()


def add_param_profile_mlp():
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    param = ui.comboBox_prof_ftr_mlp.currentText()
    if session.query(ParameterMLP).filter_by(
            analysis_id=get_MLP_id(),
            parameter= param
    ).count() == 0:
        add_param_mlp(param)
        set_color_button_updata()
        update_list_param_mlp_no_update()
    else:
        set_info(f'Параметр {param} уже добавлен', 'red')


def add_all_param_profile_mlp():
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    for param in list_all_additional_features:
        if param in ['fractal_dim', 'hht_marg_spec_min']:
            continue
        if session.query(ParameterMLP).filter(ParameterMLP.analysis_id == get_MLP_id()).filter(
                ParameterMLP.parameter == f'prof_{param}').count() > 0:
            set_info(f'Параметр "prof_{param}" уже добавлен', 'red')
            continue
        add_param_mlp(f'prof_{param}')
    set_color_button_updata()
    update_list_param_mlp_no_update()


def add_param_distr_mlp():
    for param in session.query(ParameterMLP).filter(ParameterMLP.analysis_id == get_MLP_id()).all():
        if param.parameter.startswith(f'distr_{ui.comboBox_atrib_distr_mlp.currentText()}'):
            session.query(ParameterMLP).filter_by(id=param.id).update({
                'parameter': f'distr_{ui.comboBox_atrib_distr_mlp.currentText()}_{ui.spinBox_count_distr_mlp.value()}'
            }, synchronize_session='fetch')
            session.commit()
            update_list_param_mlp()
            set_info(f'В параметры добавлены {ui.spinBox_count_distr_mlp.value()} интервалов распределения по '
                     f'{ui.comboBox_atrib_distr_mlp.currentText()}', 'green')
            return
    add_param_mlp('distr')
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    set_color_button_updata()
    update_list_param_mlp_no_update()
    set_info(f'В параметры добавлены {ui.spinBox_count_distr_mlp.value()} интервалов распределения по '
             f'{ui.comboBox_atrib_distr_mlp.currentText()}', 'green')


def add_param_sep_mlp():
    for param in session.query(ParameterMLP).filter(ParameterMLP.analysis_id == get_MLP_id()).all():
        if param.parameter.startswith(f'sep_{ui.comboBox_atrib_distr_mlp.currentText()}'):
            session.query(ParameterMLP).filter_by(id=param.id).update({
                'parameter': f'sep_{ui.comboBox_atrib_distr_mlp.currentText()}_{ui.spinBox_count_distr_mlp.value()}'
            }, synchronize_session='fetch')
            session.commit()
            update_list_param_mlp()
            set_info(f'В параметры добавлены средние значения разделения на {ui.spinBox_count_distr_mlp.value()} интервалов по '
                     f'{ui.comboBox_atrib_distr_mlp.currentText()}', 'green')
            return
    add_param_mlp('sep')
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    set_color_button_updata()
    update_list_param_mlp_no_update()
    set_info(f'В параметры добавлены средние значения разделения на {ui.spinBox_count_distr_mlp.value()} интервалов по '
             f'{ui.comboBox_atrib_distr_mlp.currentText()}', 'green')


def add_all_param_distr_mlp():
    list_distr = ['distr_Abase', 'distr_diff', 'distr_At', 'distr_Vt', 'distr_Pht', 'distr_Wt', 'distr_SigCRL', 'sep_Abase', 'sep_diff', 'sep_At', 'sep_Vt', 'sep_Pht', 'sep_Wt', 'sep_SigCRL']
    count = ui.spinBox_count_distr_mlp.value()
    for param in session.query(ParameterMLP).filter(ParameterMLP.analysis_id == get_MLP_id()).all():
        if param.parameter.startswith('distr') or param.parameter.startswith('sep'):
            session.query(ParameterMLP).filter_by(id=param.id).delete()
            session.commit()
    for distr_param in list_distr:
        new_param = f'{distr_param}_{count}'
        new_param_mlp = ParameterMLP(analysis_id=get_MLP_id(), parameter=new_param)
        session.add(new_param_mlp)
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    set_color_button_updata()
    update_list_param_mlp_no_update()
    set_info(f'Добавлены все параметры распределения по {count} интервалам', 'green')


def add_param_mfcc_mlp():
    for param in session.query(ParameterMLP).filter(ParameterMLP.analysis_id == get_MLP_id()).all():
        if param.parameter.startswith(f'mfcc_{ui.comboBox_atrib_mfcc_mlp.currentText()}'):
            session.query(ParameterMLP).filter_by(id=param.id).update({
                'parameter': f'mfcc_{ui.comboBox_atrib_mfcc_mlp.currentText()}_{ui.spinBox_count_mfcc_mlp.value()}'
            }, synchronize_session='fetch')
            session.commit()
            update_list_param_mlp()
            set_info(f'В параметры добавлены {ui.spinBox_count_mfcc_mlp.value()} кепстральных коэффициентов '
                     f'{ui.comboBox_atrib_mfcc_mlp.currentText()}', 'green')
            return
    add_param_mlp('mfcc')
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    set_color_button_updata()
    update_list_param_mlp_no_update()
    set_info(f'В параметры добавлены {ui.spinBox_count_mfcc_mlp.value()} кепстральных коэффициентов '
             f'{ui.comboBox_atrib_mfcc_mlp.currentText()}', 'green')


def add_all_param_mfcc_mlp():
    list_mfcc = ['mfcc_Abase', 'mfcc_diff', 'mfcc_At', 'mfcc_Vt', 'mfcc_Pht', 'mfcc_Wt', 'mfcc_SigCRL']
    count = ui.spinBox_count_mfcc_mlp.value()
    for param in session.query(ParameterMLP).filter(ParameterMLP.analysis_id == get_MLP_id()).all():
        if param.parameter.startswith('mfcc'):
            session.query(ParameterMLP).filter_by(id=param.id).delete()
            session.commit()
    for mfcc_param in list_mfcc:
        new_param = f'{mfcc_param}_{count}'
        new_param_mlp = ParameterMLP(analysis_id=get_MLP_id(), parameter=new_param)
        session.add(new_param_mlp)
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    set_color_button_updata()
    update_list_param_mlp_no_update()
    set_info(f'Добавлены коэффициенты mfcc по всем параметрам по {count} интервалам', 'green')


def add_predict_mlp():
    try:
        predict = session.query(ProfileModelPrediction).filter_by(
            id=ui.listWidget_model_pred.currentItem().text().split(' id')[-1]
        ).first()
    except AttributeError:
        set_info('Выберите модель в Model Prediction', 'red')
        return
    param = f'model_{predict.type_model}_id{predict.model_id}'
    if session.query(ParameterMLP).filter_by(analysis_id=get_MLP_id(), parameter=param).count() > 0:
        set_info(f'Параметр {param} уже добавлен', 'red')
        return
    else:
        new_param_mlp = ParameterMLP(analysis_id=get_MLP_id(), parameter=param)
        session.add(new_param_mlp)
        session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
        session.commit()
        update_list_param_mlp_no_update()
        set_color_button_updata()
        set_info(f'Добавлен параметр {param}', 'green')


def remove_param_geovel_mlp():
    param = ui.listWidget_param_mlp.currentItem().text().split(' ')[0]
    if param:
        if (param.startswith('distr') or param.startswith('sep') or param.startswith('mfcc') or
                param.startswith('Signal') or param.startswith('CRL')):
            for p in session.query(ParameterMLP).filter(ParameterMLP.analysis_id == get_MLP_id()).all():
                if p.parameter.startswith('_'.join(param.split('_')[:-1])):
                    session.query(ParameterMLP).filter_by(id=p.id).delete()
                    session.commit()
        else:
            session.query(ParameterMLP).filter_by(analysis_id=get_MLP_id(), parameter=param ).delete()
        session.commit()
        ui.listWidget_param_mlp.takeItem(ui.listWidget_param_mlp.currentRow())
        session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
        session.commit()
        set_color_button_updata()
        update_list_param_mlp_no_update()


def remove_all_param_geovel_mlp():
    session.query(ParameterMLP).filter_by(analysis_id=get_MLP_id()).delete()
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    update_list_param_mlp()
    update_list_param_mlp_no_update()


def update_list_well_markup_mlp():
    """Обновление списка обучающих скважин MLP"""
    print('update_list_well_markup_mlp')
    ui.listWidget_well_mlp.clear()
    count_markup, count_measure, count_fake = 0, 0, 0
    for i in session.query(MarkupMLP).filter(MarkupMLP.analysis_id == get_MLP_id()).all():
        try:
            fake = len(json.loads(i.list_fake)) if i.list_fake else 0
            measure = len(json.loads(i.list_measure))
            if i.type_markup == 'intersection':
                try:
                    inter_name = session.query(Intersection.name).filter(Intersection.id == i.well_id).first()[0]
                except TypeError:
                    session.query(MarkupMLP).filter(MarkupMLP.id == i.id).delete()
                    set_info(f'Обучающая скважина удалена из-за отсутствия пересечения', 'red')
                    session.commit()
                    continue
                item = f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | {inter_name} | {measure - fake} из {measure} | id{i.id}'
            elif i.type_markup == 'profile':
                item = f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | | {measure - fake} из {measure} | id{i.id}'
            else:
                item = f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | {i.well.name} | {measure - fake} из {measure} | id{i.id}'
            ui.listWidget_well_mlp.addItem(item)
            i_item = ui.listWidget_well_mlp.findItems(item, Qt.MatchContains)[0]
            i_item.setBackground(QBrush(QColor(i.marker.color)))
            count_markup += 1
            count_measure += measure - fake
            count_fake += fake
            # ui.listWidget_well_mlp.setItemData(ui.listWidget_well_mlp.findText(item), QBrush(QColor(i.marker.color)), Qt.BackgroundRole)
        except AttributeError:
            set_info(f'Параметр для профиля {i.profile.title} удален из-за отсутствия одного из параметров', 'red')
            session.delete(i)
            session.commit()
    ui.label_count_markup_mlp.setText(f'<i><u>{count_markup}</u></i> обучающих скважин; '
                                      f'<i><u>{count_measure}</u></i> измерений; '
                                      f'<i><u>{count_fake}</u></i> выбросов')
    update_list_param_mlp(db=True)
    update_list_param_mlp_no_update()


def update_list_param_mlp_no_update():
    data = session.query(AnalysisMLP.up_data).filter_by(id=get_MLP_id()).first()
    if data[0]:
        return
    print('update_list_param_mlp_no_update')
    list_param_mlp = session.query(ParameterMLP).filter_by(analysis_id=get_MLP_id()).all()
    list_param_mlp.sort(key=lambda x: x.parameter)
    ui.listWidget_param_mlp.clear()
    for param in list_param_mlp:
        i_item = QListWidgetItem(f'{param.parameter}')
        ui.listWidget_param_mlp.addItem(i_item)
        i_item.setBackground(QBrush(QColor('#FFFAD5')))


def update_list_param_mlp(db=False):
    start_time = datetime.datetime.now()
    data_train, list_param = build_table_train(db, 'mlp')
    list_marker = get_list_marker_mlp('georadar')
    ui.listWidget_param_mlp.clear()
    list_param_mlp = data_train.columns.tolist()[2:]
    for param in list_param_mlp:
        if ui.checkBox_kf.isChecked():
            groups = []
            for mark in list_marker:
                groups.append(data_train[data_train['mark'] == mark][param].values.tolist())
            F, p = f_oneway(*groups)
            if np.isnan(F) or np.isnan(p):
                # if (not param.startswith('distr') and not param.startswith('sep') and not param.startswith('mfcc') and
                #         not param.startswith('Signal')):
                #     session.query(ParameterMLP).filter_by(analysis_id=get_MLP_id(), parameter=param).delete()
                #     data_train.drop(param, axis=1, inplace=True)
                #     session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'data': json.dumps(data_train.to_dict())}, synchronize_session='fetch')
                #     session.commit()
                #     set_info(f'Параметр {param} удален', 'red')
                ui.listWidget_param_mlp.addItem(param)
                continue
            ui.listWidget_param_mlp.addItem(f'{param} \t\tF={round(F, 2)} p={round(p, 3)}')
            if F < 1 or p > 0.05:
                i_item = ui.listWidget_param_mlp.findItems(f'{param} \t\tF={round(F, 2)} p={round(p, 3)}', Qt.MatchContains)[0]
                i_item.setBackground(QBrush(QColor('red')))
        else:
            ui.listWidget_param_mlp.addItem(param)
    ui.label_count_param_mlp.setText(f'<i><u>{ui.listWidget_param_mlp.count()}</u></i> параметров')
    set_color_button_updata()
    update_list_trained_models_class()
    update_line_edit_exception_mlp()


def update_line_edit_exception_mlp():
    ui.lineEdit_signal_except.clear()
    ui.lineEdit_crl_except.clear()
    except_mlp = session.query(ExceptionMLP).filter_by(analysis_id=get_MLP_id()).first()
    if except_mlp:
        ui.lineEdit_signal_except.setText(except_mlp.except_signal)
        ui.lineEdit_crl_except.setText(except_mlp.except_crl)


def set_color_button_updata():
    mlp = session.query(AnalysisMLP).filter(AnalysisMLP.id == get_MLP_id()).first()
    btn_color = 'background-color: rgb(191, 255, 191);' if mlp.up_data else 'background-color: rgb(255, 185, 185);'
    ui.pushButton_updata_mlp.setStyleSheet(btn_color)


def prep_data_train(data):
    list_param_mlp = data.columns.tolist()[2:]

    training_sample = data[list_param_mlp].values.tolist()
    markup = sum(data[['mark']].values.tolist(), [])

    # Нормализация данных
    scaler = StandardScaler()
    training_sample_norm = scaler.fit_transform(training_sample)

    # Разделение данных на обучающую и тестовую выборки
    training_sample_train, training_sample_test, markup_train, markup_test = train_test_split(
        training_sample_norm, markup, test_size=0.20, random_state=1)
    return list_param_mlp, training_sample_train, training_sample_test, markup_train, markup_test


def draw_MLP():
    """ Тренировка моделей классификаторов """
    data_train, list_param = build_table_train(True, 'mlp')
    list_param_mlp = data_train.columns.tolist()[2:]

    colors = {}
    for m in session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).all():
        colors[m.title] = m.color

    train_classifier(data_train, list_param_mlp, list_param, colors, 'mark', 'prof_well_index', 'georadar')


def remove_trained_model_class():
    """ Удаление модели """
    model = session.query(TrainedModelClass).filter_by(id=ui.listWidget_trained_model_class.currentItem().data(Qt.UserRole)).first()
    os.remove(model.path_model)
    session.delete(model)
    session.commit()
    update_list_trained_models_class()
    set_info(f'Модель {model.title} удалена', 'blue')


# def torch_predict_proba(model, data):
#     scaler = StandardScaler()
#     working_data = scaler.fit_transform(data)
#     data_tensor = torch.tensor(working_data).float()
#     print('data_tensor ', data_tensor)
#     dataset = TensorDataset(data_tensor)
#     data = DataLoader(dataset, batch_size=20, shuffle=False)
#     model.eval()
#     predictions = []
#     mark_pred = []
#     with torch.no_grad():
#         for batch in data:
#             xb = batch[0].float()
#             pred_batch = model(xb)
#             predictions.extend([np.hstack((pred.numpy(), 1 - pred.numpy())) for pred in pred_batch])
#             mark_pred.extend([pred.numpy() for pred in pred_batch])
#     mark = [item for m in mark_pred for item in m]
#     print('proba predictions ', predictions)
#     print('proba mark ', mark)
#     return predictions, mark


def calc_class_profile():
    """  Расчет профиля по выбранной модели классификатора """
    try:
        working_data, curr_form = build_table_test('mlp')
    except TypeError:
        return
    working_data_class = working_data.copy()

    Choose_RegModel = QtWidgets.QDialog()
    ui_rm = Ui_FormRegMod()
    ui_rm.setupUi(Choose_RegModel)
    Choose_RegModel.show()
    Choose_RegModel.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    def calc_class_model():
        labels = set_marks()
        labels_dict = {value: key for key, value in labels.items()}
        model = session.query(TrainedModelClass).filter_by(
            id=ui.listWidget_trained_model_class.currentItem().data(Qt.UserRole)).first()

        if 'torch' in model.title:
            with open(model.path_model, 'rb') as f:
                class_model = pickle.load(f)
            # list_cat = [i.title for i in session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).all()]
            list_cat = list(class_model.classes_)
            print('torch list_cat ', list_cat)
        else:
            with open(model.path_model, 'rb') as f:
                class_model = pickle.load(f)
            list_cat = list(class_model.classes_)
            print('sklearn list_cat ', list_cat)
        list_param_num = get_list_param_numerical(json.loads(model.list_params), model)
        try:
            working_sample = working_data_class[list_param_num].values.tolist()
        except KeyError:
            set_info('Не совпадает количество признаков для данной модели. Выберите нужную модель и '
                     'рассчитайте заново', 'red')
            QMessageBox.critical(MainWindow, 'Ошибка', 'Не совпадает количество признаков для данной модели.')
            return

        try:
            mark = class_model.predict(working_sample)
            probability = class_model.predict_proba(working_sample)
        except ValueError:
            working_sample = [[np.nan if np.isinf(x) else x for x in y] for y in working_sample]
            data = imputer.fit_transform(working_sample)
            try:
                mark = class_model.predict(data)
            except ValueError:
                set_info('Не совпадает количество признаков для данной модели. Выберите нужную модель и '
                         'рассчитайте заново', 'red')
                QMessageBox.critical(MainWindow, 'Ошибка', 'Не совпадает количество признаков для данной модели.')
                return
            probability = class_model.predict_proba(data)

            for i in working_data_class.index:
                p_nan = [working_data_class.columns[ic + 3] for ic, v in enumerate(working_data_class.iloc[i, 3:].tolist()) if
                         np.isnan(v)]
                if len(p_nan) > 0:
                    set_info(f'Внимание для измерения "{i}" отсутствуют параметры "{", ".join(p_nan)}", поэтому расчёт для'
                             f' этого измерения может быть не корректен', 'red')

        # Добавление предсказанных меток и вероятностей в рабочие данные
        working_data_result = pd.concat([working_data_class, pd.DataFrame(probability)], axis=1)
        working_data_result['mark'] = mark
        if 'TORCH' in model.title:
            working_data_result['mark'] = working_data_result['mark'].map(labels_dict)

        pd.set_option('display.max_columns', None)
        print('working_data_result ', working_data_result)

        draw_result_mlp(working_data_result, curr_form, ui_rm.checkBox_color_marker.isChecked())

        result = QtWidgets.QMessageBox.question(ui.listWidget_well_mlp, 'Сохранение результата',
                                                'Вы хотите сохранить результат модели?',
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if result == QtWidgets.QMessageBox.Yes:
            list_result = [round(p[0], 6) for p in probability]
            new_prof_model_pred = ProfileModelPrediction(
                profile_id = get_profile_id(),
                type_model = 'cls',
                model_id = model.id,
                prediction = json.dumps(list_result)
            )

            session.add(new_prof_model_pred)
            session.commit()
            set_info(f'Результат расчета модели "{model.title}" для профиля {get_profile_name()} сохранен', 'green')
            update_list_model_prediction()


    ui_rm.pushButton_calc_model.clicked.connect(calc_class_model)
    # ui.checkBox_relief.stateChanged.connect(calc_class_model)


    Choose_RegModel.exec_()


def draw_result_mlp(working_data, curr_form, color_marker):
    colors = {}
    for m in session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).all():
        colors[m.title] = m.color

    remove_poly_item()
    list_up = json.loads(curr_form.layer_up.layer_line)  # Получение списка с верхними границами формации
    list_down = json.loads(curr_form.layer_down.layer_line)  # Получение списка со снижными границами формации

    if ui.checkBox_relief.isChecked():
        profile = session.query(Profile).filter(Profile.id == get_profile_id()).first()
        if profile.depth_relief:
            depth = [i * 100 / 40 for i in json.loads(profile.depth_relief)]
            coeff = 512 / (512 + np.max(depth))
            list_up = [int((x + y) * coeff) for x, y in zip(list_up, depth)]
            list_down = [int((x + y) * coeff) for x, y in zip(list_down, depth)]

    col = working_data.columns[-3]
    print('col ', col)
    print(working_data[col])


    previous_element = None
    list_dupl = []


    if color_marker:

        for index, current_element in enumerate(working_data['mark']):
            if current_element == previous_element:
                list_dupl.append(index)
            else:
                if list_dupl:
                    list_dupl.append(list_dupl[-1] + 1)
                    y_up = [list_up[i] for i in list_dupl]
                    y_down = [list_down[i] for i in list_dupl]
                    draw_fill_result(list_dupl, y_up, y_down, colors[previous_element])
                list_dupl = [index]
            previous_element = current_element
        if len(list_dupl) > 0:
            y_up = [list_up[i] for i in list_dupl]
            y_down = [list_down[i] for i in list_dupl]
            draw_fill_result(list_dupl, y_up, y_down, colors[previous_element])


    else:
        for index, probability in enumerate(working_data[col]):
            color = get_color_rainbow(probability)
            if color == previous_element:
                list_dupl.append(index)
            else:
                if list_dupl:
                    list_dupl.append(list_dupl[-1] + 1)
                    y_up = [list_up[i] for i in list_dupl]
                    y_down = [list_down[i] for i in list_dupl]
                    draw_fill_result(list_dupl, y_up, y_down, previous_element)
                list_dupl = [index]
            previous_element = color
        if len(list_dupl) > 0:
            y_up = [list_up[i] for i in list_dupl]
            y_down = [list_down[i] for i in list_dupl]
            draw_fill_result(list_dupl, y_up, y_down, get_color_rainbow(working_data[col].tolist()[-1]))

    ui.graph.clear()
    number = list(range(1, len(working_data[col]) + 1))
    # Создаем кривую и кривую, отфильтрованную с помощью savgol_filter
    curve = pg.PlotCurveItem(x=number, y=working_data[col].tolist(), pen=pg.mkPen(color='#969696'))
    curve_filter = pg.PlotCurveItem(x=number, y=savgol_filter(working_data[col].tolist(), 31, 3),
                                    pen=pg.mkPen(color='red', width=2.4))
    # Добавляем кривую и отфильтрованную кривую на график для всех пластов
    text = pg.TextItem(str(col), anchor=(0, 1))
    ui.graph.addItem(curve)
    ui.graph.addItem(curve_filter)
    ui.graph.addItem(text)
    ui.graph.setYRange(0, 1)

    if ui.checkBox_save_prof_mlp.isChecked():
        try:
            file_name = f'{get_object_name()}_{get_research_name()}_{get_profile_name()}__модель_{get_mlp_title()}.xlsx'
            fn = QFileDialog.getSaveFileName(caption="Сохранить выборку в таблицу", directory=file_name,
                                             filter="Excel Files (*.xlsx)")
            working_data.to_excel(fn[0])
            set_info(f'Таблица сохранена в файл: {fn[0]}', 'green')
        except ValueError:
            pass


def get_color_rainbow(probability):

    rainbow_colors =[
        "#0000FF",  # Синий
        "#0066FF",  # Голубой
        "#00CCFF",  # Светло-голубой
        "#00FFCC",  # Бирюзовый
        "#00FF66",  # Зеленовато-голубой
        "#33FF33",  # Ярко-зеленый
        "#99FF33",  # Желто-зеленый
        "#FFFF00",  # Желтый
        "#FF6600",  # Оранжевый
        "#FF0000"   # Красный
    ]
    try:
        return rainbow_colors[int(probability * 10)]
    except (IndexError, ValueError):
        return '#FF0000'



def calc_object_class():
    """ Расчет объекта по модели """
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

    if ui.checkBox_save_prof_mlp.isChecked():
        model = session.query(TrainedModelClass).filter_by(
            id=ui.listWidget_trained_model_class.currentItem().data(Qt.UserRole)).first()

        with open(model.path_model, 'rb') as f:
            class_model = pickle.load(f)

        list_param_num = get_list_param_numerical(json.loads(model.list_params), model)

    for n, prof in enumerate(session.query(Profile).filter(Profile.research_id == get_research_id()).all()):
        count_measure = len(json.loads(session.query(Profile.signal).filter(Profile.id == prof.id).first()[0]))
        ui.comboBox_profile.setCurrentText(f'{prof.title} ({count_measure} измерений) id{prof.id}')
        update_formation_combobox()
        ui.comboBox_plast.setCurrentText(list_formation[n])
        working_data, curr_form = build_table_test('mlp')

        if ui.checkBox_save_prof_mlp.isChecked():
            if session.query(ProfileModelPrediction).filter_by(
                    profile_id=prof.id, type_model='cls', model_id=model.id).count() == 0:

                working_data_profile = working_data.copy()
                working_sample_profile = working_data_profile[list_param_num].values.tolist()

                try:
                    probability = class_model.predict_proba(working_sample_profile)
                except ValueError:
                    working_sample_profile = [[np.nan if np.isinf(x) else x for x in y] for y in working_sample_profile]
                    data = imputer.fit_transform(working_sample_profile)
                    probability = class_model.predict_proba(data)

                list_result = [round(p[0], 6) for p in probability]
                new_prof_model_pred = ProfileModelPrediction(
                    profile_id=get_profile_id(),
                    type_model='cls',
                    model_id=model.id,
                    prediction=json.dumps(list_result)
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

    def calc_class_model():
        labels = set_marks()
        labels_dict = {value: key for key, value in labels.items()}

        model = session.query(TrainedModelClass).filter_by(
            id=ui.listWidget_trained_model_class.currentItem().data(Qt.UserRole)).first()

        with open(model.path_model, 'rb') as f:
            class_model = pickle.load(f)

        list_cat = list(class_model.classes_)
        if 'TORCH' in model.title:
            list_cat = [labels_dict[i] for i in list_cat]
        list_param_num = get_list_param_numerical(json.loads(model.list_params), model)
        try:
            working_sample = working_data_result_copy[list_param_num].values.tolist()
        except KeyError:
            set_info('Не совпадает количество признаков для данной модели. Выберите нужную модель и '
                     'рассчитайте заново', 'red')
            QMessageBox.critical(MainWindow, 'Ошибка', 'Не совпадает количество признаков для данной модели.')
            return

        try:
            mark = class_model.predict(working_sample)
            probability = class_model.predict_proba(working_sample)
        except ValueError:
            working_sample = [[np.nan if np.isinf(x) else x for x in y] for y in working_sample]
            data = imputer.fit_transform(working_sample)
            # if model.title.startswith('torch_NN'):
            #     try:
            #         probability, mark = class_model.predict(working_sample)
            #         mark = [list_cat[0] if i > 0.5 else list_cat[1] for i in mark]
            #     except ValueError:
            #         set_info('Не совпадает количество признаков для данной модели. Выберите нужную модель и '
            #                  'рассчитайте заново', 'red')
            #         QMessageBox.critical(MainWindow, 'Ошибка', 'Не совпадает количество признаков для данной модели.')
            #         return
            # else:
            try:
                mark = class_model.predict(data)
            except ValueError:
                set_info('Не совпадает количество признаков для данной модели. Выберите нужную модель и '
                         'рассчитайте заново', 'red')
                QMessageBox.critical(MainWindow, 'Ошибка', 'Не совпадает количество признаков для данной модели.')
                return
            probability = class_model.predict_proba(data)

            for i in working_data_result_copy.index:
                p_nan = [working_data_result_copy.columns[ic + 3] for ic, v in
                         enumerate(working_data_result_copy.iloc[i, 3:].tolist()) if
                         np.isnan(v)]
                if len(p_nan) > 0:
                    set_info(
                        f'Внимание для измерения "{i}" отсутствуют параметры "{", ".join(p_nan)}", поэтому расчёт для'
                        f' этого измерения может быть не корректен', 'red')

        # Добавление предсказанных меток и вероятностей в рабочие данные
        working_data_result = pd.concat([working_data_result_copy, pd.DataFrame(probability, columns=list_cat)], axis=1)
        working_data_result['mark'] = mark
        if 'TORCH' in model.title:
            working_data_result['mark'] = working_data_result['mark'].map(labels_dict)
        print('working_data_result ', working_data_result)

        x = list(working_data_result['x_pulc'])
        y = list(working_data_result['y_pulc'])
        if len(set(mark)) == 2 and not ui_rm.checkBox_color_marker.isChecked():
            marker_mlp = session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).order_by(MarkerMLP.id).first()
            z = list(working_data_result[marker_mlp.title])
            color_marker = False
            z_number = string_to_unique_number(list(working_data_result['mark']), 'mlp')
            working_data_result['mark_number'] = z_number
        else:
            z = string_to_unique_number(list(working_data_result['mark']), 'mlp')
            color_marker = True
            working_data_result['mark_number'] = z
        draw_map(x, y, z, f'Classifier {ui.listWidget_trained_model_class.currentItem().text()}', color_marker)
        result1 = QMessageBox.question(MainWindow, 'Сохранение', 'Сохранить результаты расчёта MLP?', QMessageBox.Yes, QMessageBox.No)
        if result1 == QMessageBox.Yes:
            result2 = QMessageBox.question(MainWindow, 'Сохранение', 'Сохранить только результаты расчёта?', QMessageBox.Yes, QMessageBox.No)
            if result2 == QMessageBox.Yes:
                list_col = [i.title for i in session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).all()]
                list_col += ['x_pulc', 'y_pulc', 'mark', 'mark_number']
                working_data_result = working_data_result[list_col]
            else:
                pass
            try:
                file_name = f'{get_object_name()}_{get_research_name()}__модель_{get_mlp_title()}.xlsx'
                fn = QFileDialog.getSaveFileName(caption=f'Сохранить результат MLP "{get_object_name()}_{get_research_name()}" в таблицу', directory=file_name,
                                                 filter="Excel Files (*.xlsx)")
                working_data_result.to_excel(fn[0])
                set_info(f'Таблица сохранена в файл: {fn[0]}', 'green')
            except ValueError:
                pass
        else:
            pass

    ui_rm.pushButton_calc_model.clicked.connect(calc_class_model)
    Choose_RegModel.exec_()


def calc_corr_mlp():
    if not session.query(AnalysisMLP).filter(AnalysisMLP.id == get_MLP_id()).first().up_data:
        update_list_param_mlp()
    data_train, list_param = build_table_train(True, 'mlp')
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


def anova_mlp():
    Anova = QtWidgets.QDialog()
    ui_anova = Ui_Anova()
    ui_anova.setupUi(Anova)
    Anova.show()
    Anova.setAttribute(QtCore.Qt.WA_DeleteOnClose) # атрибут удаления виджета после закрытия

    # ui_anova.graphicsView.setBackground('w')

    data_plot, list_param = build_table_train(True, 'mlp')
    markers = list(set(data_plot['mark']))
    pallet = {i: session.query(MarkerMLP).filter(MarkerMLP.title == i, MarkerMLP.analysis_id == get_MLP_id()).first().color for i in markers}

    figure = plt.figure()
    canvas = FigureCanvas(figure)
    ui_anova.horizontalLayout.addWidget(canvas)


    for i in sorted(data_plot.columns.tolist()[2:]):
        ui_anova.listWidget.addItem(i)

    def draw_graph_anova():
        figure.clear()
        param = ui_anova.listWidget.currentItem().text()
        if ui_anova.radioButton_box.isChecked():
            sns.boxplot(data=data_plot, y=param, x='mark', orient='v', palette=pallet)
        if ui_anova.radioButton_violin.isChecked():
            sns.violinplot(data=data_plot, y=param, x='mark', orient='v', palette=pallet)
        if ui_anova.radioButton_strip.isChecked():
            sns.stripplot(data=data_plot, y=param, x='mark', hue='mark', orient='v', palette=pallet)
        if ui_anova.radioButton_boxen.isChecked():
            sns.boxenplot(data=data_plot, y=param, x='mark', orient='v', palette=pallet)
        figure.tight_layout()
        canvas.draw()

    ui_anova.listWidget.currentItemChanged.connect(draw_graph_anova)
    ui_anova.radioButton_boxen.clicked.connect(draw_graph_anova)
    ui_anova.radioButton_strip.clicked.connect(draw_graph_anova)
    ui_anova.radioButton_violin.clicked.connect(draw_graph_anova)
    ui_anova.radioButton_box.clicked.connect(draw_graph_anova)

    Anova.exec_()


def clear_fake_mlp():
    session.query(MarkupMLP).filter(MarkupMLP.analysis_id == get_MLP_id()).update({'list_fake': None},
                                                                                  synchronize_session='fetch')
    session.commit()
    set_info(f'Выбросы для анализа "{ui.comboBox_mlp_analysis.currentText()}" очищены.', 'green')
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    build_table_train(False, 'mlp')
    update_list_well_markup_mlp()


# def test():
#     data_train, _ = build_table_train_mlp()
#     train = data_train.iloc[:, 2:]
#     mark = data_train['mark'].values.tolist()
#     print(train)
#
#     # Нормализация данных
#     scaler = StandardScaler()
#     train_norm = scaler.fit_transform(train)
#     print(train_norm)
#
#     # Разделение данных на обучающую и тестовую выборки
#     train_train, train_test, mark_train, mark_test = train_test_split(train_norm, mark, test_size=0.20, random_state=1)
#
#     # Создание и тренировка MLP
#     mlp = MLPClassifier(hidden_layer_sizes=(500, 500), max_iter=2000)
#     mlp.fit(train_train, mark_train)
#
#     # Классификация тестовой выборки
#     test_preds = mlp.predict(train_test)
#     print(test_preds)
#
#     # Вывод результатов в виде таблицы
#     print('   Index  True label  Predicted label')
#     for i in range(len(mark_test)):
#         print(f'{i} {mark_test[i]} {test_preds[i]}')
#
#     # Классификация обучающей выборки
#     train_preds = mlp.predict(train_train)
#
#     # Вывод результатов в виде таблицы
#     print('   Index  True label  Predicted label')
#     for i in range(len(mark_train)):
#         print(f'{i} {mark_train[i]} {train_preds[i]}')
#
#     # Оценка точности на обучающей выборке
#     train_accuracy = mlp.score(train_train, mark_train)
#     print(f'Training accuracy: {train_accuracy}')
#
#     # Оценка точности на тестовой выборке
#     test_accuracy = mlp.score(train_test, mark_test)
#     print(f'Test accuracy: {test_accuracy}')


def get_model_param_list():
    try:
        model = session.query(TrainedModelClass).filter_by(id=ui.listWidget_trained_model_class.currentItem().data(Qt.UserRole)).first()
    except AttributeError:
        QMessageBox.critical(MainWindow, 'Не выбрана модель', 'Не выбрана модель.')
        set_info('Не выбрана модель', 'red')
        return

    FormParams = QtWidgets.QDialog()
    ui_p = Ui_Form_ModelParams()
    ui_p.setupUi(FormParams)
    FormParams.show()
    FormParams.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    ui_p.label.setText(ui_p.label.text() + model.title)
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


def update_trained_model_comment():
    """ Изменить комментарий обученной модели """
    try:
        an = session.query(TrainedModelClass).filter_by(id=ui.listWidget_trained_model_class.currentItem().data(Qt.UserRole)).first()
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
        session.query(TrainedModelClass).filter_by(id=an.id).update({'comment': ui_cmt.textEdit.toPlainText()}, synchronize_session='fetch')
        session.commit()
        update_list_trained_models_class()
        FormComment.close()

    ui_cmt.buttonBox.accepted.connect(update_comment)

    FormComment.exec_()


def export_model_class():
    """ Экспорт модели """
    try:
        model = session.query(TrainedModelClass).filter_by(
        id=ui.listWidget_trained_model_class.currentItem().data(Qt.UserRole)).first()
    except FileNotFoundError:
        return set_info('Не выбрана модель', 'red')
    analysis = session.query(AnalysisMLP).filter_by(id=model.analysis_id).first()

    model_parameters = {
        'analysis_title': analysis.title,
        'title': model.title,
        'list_params': model.list_params,
        'comment': model.comment
    }

    # Сохранение словаря с параметрами в файл *.pkl
    with open('model_parameters.pkl', 'wb') as parameters_file:
        pickle.dump(model_parameters, parameters_file)

    try:
        filename = \
            QFileDialog.getSaveFileName(caption='Экспорт модели классификации', directory=f'{model.title}.zip', filter="*.zip")[
                0]
        with zipfile.ZipFile(filename, 'w') as zip:
            zip.write('model_parameters.pkl', 'model_parameters.pkl')
            zip.write(model.path_model, 'model.pkl')

        set_info(f'Модель {model.title} экспортирована в файл {filename}', 'blue')
    except FileNotFoundError:
        pass


def import_model_class():
    """ Импорт модели """
    try:
        filename = QFileDialog.getOpenFileName(caption='Импорт модели классификации', filter="*.zip")[0]

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

    path_model = f'models/classifier/{model_name}.pkl'

    with open(path_model, 'wb') as f:
        pickle.dump(loaded_model, f)

    analysis = session.query(AnalysisMLP).filter_by(title=analysis_title).first()
    if not analysis:
        new_analisys = AnalysisMLP(title=analysis_title)
        session.add(new_analisys)
        session.commit()
        analysis = new_analisys
        update_list_mlp(True)

    new_trained_model = TrainedModelClass(
        analysis_id = analysis.id,
        title=model_name,
        path_model=path_model,
        list_params=list_params,
        comment=comment
    )
    session.add(new_trained_model)
    session.commit()
    try:
        shutil.rmtree('extracted_data')
        os.remove('model_parameters.pkl')
    except FileNotFoundError:
        pass

    update_list_trained_models_class()
    set_info(f'Модель {model_name} импортирована', 'blue')


def add_signal_except_mlp():
    """ Список исключений для параметра Signal """
    check_str = parse_range_exception(ui.lineEdit_signal_except.text())
    if not check_str:
        set_info('Неверный формат диапазона исключений', 'red')
        return
    except_line = '' if check_str == -1 else ui.lineEdit_signal_except.text()
    excetp_signal = session.query(ExceptionMLP).filter_by(analysis_id=get_MLP_id()).first()
    if excetp_signal:
        excetp_signal.except_signal = except_line
    else:
        new_except = ExceptionMLP(
            analysis_id=get_MLP_id(),
            except_signal=except_line
        )
        session.add(new_except)
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    set_color_button_updata()
    set_info('Исключения добавлены', 'green')


def add_crl_except_mlp():
    """ Список исключений для параметра Crl """
    check_str = parse_range_exception(ui.lineEdit_crl_except.text())
    if not check_str:
        set_info('Неверный формат диапазона исключений', 'red')
        return
    except_line = '' if check_str == -1 else ui.lineEdit_crl_except.text()
    excetp_crl = session.query(ExceptionMLP).filter_by(analysis_id=get_MLP_id()).first()
    if excetp_crl:
        excetp_crl.except_crl = except_line
    else:
        new_except = ExceptionMLP(
            analysis_id=get_MLP_id(),
            except_crl=except_line
        )
        session.add(new_except)
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    set_color_button_updata()
    set_info('Исключения добавлены', 'green')


def rename_model_class():
    """Переименовать модель"""
    model = session.query(TrainedModelClass).filter_by(id=ui.listWidget_trained_model_class.currentItem().data(
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
        update_list_trained_models_class()
        RenameModel.close()

    ui_rnm.buttonBox.accepted.connect(rename_model)
    ui_rnm.buttonBox.rejected.connect(RenameModel.close)

    RenameModel.exec_()


def list_param_to_lineEdit():
    model = session.query(TrainedModelClass).filter_by(id=ui.listWidget_trained_model_class.currentItem().data(
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


def get_feature_importance_cls():
    model = session.query(TrainedModelClass).filter_by(id=ui.listWidget_trained_model_class.currentItem().data(
        Qt.UserRole)).first()

    if not model:
        return

    if 'GBC' in model.title or 'LGBM' in model.title or 'RFC' in model.title:
        with open(model.path_model, 'rb') as f:
            class_model = pickle.load(f)

        params = json.loads(model.list_params)

        full_params = get_list_param_numerical(params, model)
        feature_importances = class_model.named_steps['model'].feature_importances_

        # Вывод важности признаков вместе с названиями признаков
        feature_importance_df = pd.DataFrame({
            'Feature': full_params,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        print(feature_importance_df.head(30))


def markup_to_excel_mlp():
    list_col = ['маркер', 'объект', 'профиль', 'интервал', 'измерения', 'выбросы', 'скважина',
                                      'альтитуда', 'удаленность', 'X', 'Y']
    analisis = session.query(AnalysisMLP).filter_by(id=get_MLP_id()).first()
    pd_markup = pd.DataFrame(columns=list_col)
    ui.progressBar.setMaximum(len(analisis.markups))
    n = 1
    for mrp in analisis.markups:
        ui.progressBar.setValue(n)
        mrp_dict = dict()
        mrp_dict['маркер'] = mrp.marker.title
        mrp_dict['объект'] = f'{mrp.profile.research.object.title}_{mrp.profile.research.date_research.year}'
        mrp_dict['профиль'] = mrp.profile.title
        mrp_dict['интервал'] = mrp.formation.title
        mrp_dict['измерения'] = mrp.list_measure
        mrp_dict['выбросы'] = mrp.list_fake
        if not mrp.type_markup:
            mrp_dict['скважина'] = mrp.well.name
            mrp_dict['альтитуда'] = mrp.well.alt
            mrp_dict['X'] = mrp.well.x_coord
            mrp_dict['Y'] = mrp.well.y_coord
            mrp_dict['удаленность'] = closest_point(mrp.well.x_coord, mrp.well.y_coord, json.loads(mrp.profile.x_pulc), json.loads(mrp.profile.y_pulc))[1]

        pd_markup = pd.concat([pd_markup, pd.DataFrame(data=mrp_dict, columns=list_col, index=[0])], axis = 0, ignore_index=True)
        n += 1

    file_name = QFileDialog.getSaveFileName()[0]
    if file_name:
        pd_markup.to_excel(file_name, index=False)




