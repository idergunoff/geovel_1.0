from draw import draw_radarogram, draw_formation, draw_fill, draw_fake, draw_fill_result, remove_poly_item
from func import *
from krige import draw_map
from qt.choose_formation_lda import *
from random_search import push_random_search
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
    build_table_test_no_db('lda', new_lda.id, [])
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
        session.commit()
        set_info(f'Удалена модель MLP - "{mlp_title}"', 'green')
        update_list_mlp()
    elif result == QtWidgets.QMessageBox.No:
        pass


def update_list_mlp(db=False):
    """Обновить список анализов MLP"""
    ui.comboBox_mlp_analysis.clear()
    for i in session.query(AnalysisMLP).order_by(AnalysisMLP.title).all():
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
    for i in session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).order_by(MarkerMLP.title).all():
        item = f'{i.title} id{i.id}'
        ui.comboBox_mark_mlp.addItem(f'{i.title} id{i.id}')
        ui.comboBox_mark_mlp.setItemData(ui.comboBox_mark_mlp.findText(item), QBrush(QColor(i.color)),
                                         Qt.BackgroundRole)
    update_list_well_markup_mlp()
    update_list_param_mlp(False)


def update_list_marker_mlp_db():
    """Обновить список маркеров MLP"""
    ui.comboBox_mark_mlp.clear()
    for i in session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).order_by(MarkerMLP.title).all():
        item = f'{i.title} id{i.id}'
        ui.comboBox_mark_mlp.addItem(f'{i.title} id{i.id}')
        ui.comboBox_mark_mlp.setItemData(ui.comboBox_mark_mlp.findText(item), QBrush(QColor(i.color)),
                                         Qt.BackgroundRole)
    update_list_well_markup_mlp()
    update_list_param_mlp(True)


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
            new_markup_mlp = MarkupLDA(analysis_id=analysis_id, well_id=well_id, profile_id=profile_id,
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
    else:
        well = session.query(Well).filter(Well.id == markup.well_id).first()
        x_prof = json.loads(session.query(Profile.x_pulc).filter(Profile.id == markup.profile_id).first()[0])
        y_prof = json.loads(session.query(Profile.y_pulc).filter(Profile.id == markup.profile_id).first()[0])
        index, _ = closest_point(well.x_coord, well.y_coord, x_prof, y_prof)
        well_dist = ui.spinBox_well_dist_mlp.value()
        start = index - well_dist if index - well_dist > 0 else 0
        stop = index + well_dist if index + well_dist < len(x_prof) else len(x_prof)
        list_measure = list(range(start, stop))
    session.query(MarkupMLP).filter(MarkupMLP.id == get_markup_mlp_id()).update(
        {'marker_id': get_marker_mlp_id(), 'list_measure': json.dumps(list_measure)})
    session.commit()
    set_info(f'Изменена обучающая скважина для MLP - "{well.name} {get_marker_mlp_title()}"', 'green')
    update_list_well_markup_mlp()


def remove_well_markup_mlp():
    markup = session.query(MarkupMLP).filter(MarkupMLP.id == get_markup_mlp_id()).first()
    if not markup:
        return
    skv_name = session.query(Well.name).filter(Well.id == markup.well_id).first()[0]
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
    elif result == QtWidgets.QMessageBox.No:
        pass


def update_list_well_markup_mlp():
    """Обновление списка обучающих скважин MLP"""
    ui.listWidget_well_mlp.clear()
    count_markup, count_measure, count_fake = 0, 0, 0
    for i in session.query(MarkupMLP).filter(MarkupMLP.analysis_id == get_MLP_id()).all():
        try:
            fake = len(json.loads(i.list_fake)) if i.list_fake else 0
            measure = len(json.loads(i.list_measure))
            if i.type_markup == 'intersection':
                inter_name = session.query(Intersection.name).filter(Intersection.id == i.well_id).first()[0]
                item = f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | {inter_name} | {measure - fake} из {measure} | id{i.id}'
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


def add_param_signal_mlp():
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    param = ui.comboBox_signal_mlp.currentText()
    if session.query(ParameterMLP).filter_by(
            analysis_id=get_MLP_id(),
            parameter=param
    ).count() == 0:
        add_param_mlp(param)
        update_list_param_mlp()
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
    update_list_param_mlp()


def add_param_geovel_mlp():
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    param = ui.comboBox_geovel_param_mlp.currentText()
    for m in session.query(MarkupMLP).filter(MarkupMLP.analysis_id == get_MLP_id()).all():
        if not session.query(literal_column(f'Formation.{param}')).filter(Formation.id == m.formation_id).first()[0]:
            set_info(f'Параметр {param} отсутствует для профиля {m.profile.title}', 'red')
            return
    if session.query(ParameterMLP).filter_by(
            analysis_id=get_MLP_id(),
            parameter= param
    ).count() == 0:
        add_param_mlp(param)
        update_list_param_mlp()
    else:
        set_info(f'Параметр {param} уже добавлен', 'red')


def add_all_param_geovel_mlp():
    new_list_param = list_param_geovel.copy()
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
    update_list_param_mlp()


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
    update_list_param_mlp()
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
    update_list_param_mlp()
    set_info(f'В параметры добавлены средние значения разделения на {ui.spinBox_count_distr_mlp.value()} интервалов по '
             f'{ui.comboBox_atrib_distr_mlp.currentText()}', 'green')


def add_all_param_distr_mlp():
    list_distr = ['distr_Abase', 'distr_diff', 'distr_At', 'distr_Vt', 'distr_Pht', 'distr_Wt', 'sep_Abase', 'sep_diff', 'sep_At', 'sep_Vt', 'sep_Pht', 'sep_Wt']
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
    update_list_param_mlp()
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
    update_list_param_mlp()
    set_info(f'В параметры добавлены {ui.spinBox_count_mfcc_mlp.value()} кепстральных коэффициентов '
             f'{ui.comboBox_atrib_mfcc_mlp.currentText()}', 'green')


def add_all_param_mfcc_mlp():
    list_mfcc = ['mfcc_Abase', 'mfcc_diff', 'mfcc_At', 'mfcc_Vt', 'mfcc_Pht', 'mfcc_Wt']
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
    update_list_param_mlp()
    set_info(f'Добавлены коэффициенты mfcc по всем параметрам по {count} интервалам', 'green')


def remove_param_geovel_mlp():
    param = ui.listWidget_param_mlp.currentItem().text().split(' ')[0]
    if param:
        if param.startswith('distr') or param.startswith('sep') or param.startswith('mfcc') or param.startswith('Signal'):
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


def remove_all_param_geovel_mlp():
    session.query(ParameterMLP).filter_by(analysis_id=get_MLP_id()).delete()
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    update_list_param_mlp()


def update_list_param_mlp(db=False):
    data_train, list_param = build_table_train(db, 'mlp')
    list_marker = get_list_marker_mlp()
    ui.listWidget_param_mlp.clear()
    list_param_mlp = data_train.columns.tolist()[2:]
    for param in list_param_mlp:
        groups = []
        for mark in list_marker:
            groups.append(data_train[data_train['mark'] == mark][param].values.tolist())
        F, p = f_oneway(*groups)
        if np.isnan(F).any() or np.isnan(p).any():
            session.query(ParameterMLP).filter_by(analysis_id=get_MLP_id(), parameter=param).delete()
            data_train.drop(param, axis=1, inplace=True)
            session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'data': json.dumps(data_train.to_dict())}, synchronize_session='fetch')
            session.commit()
            set_info(f'Параметр {param} удален', 'red')
            continue
        ui.listWidget_param_mlp.addItem(f'{param} \t\tF={round(F, 2)} p={round(p, 3)}')
        if F < 1 or p > 0.05:
            i_item = ui.listWidget_param_mlp.findItems(f'{param} \t\tF={round(F, 2)} p={round(p, 3)}', Qt.MatchContains)[0]
            i_item.setBackground(QBrush(QColor('red')))
    ui.label_count_param_mlp.setText(f'<i><u>{ui.listWidget_param_mlp.count()}</u></i> параметров')
    set_color_button_updata()
    update_list_trained_models_class()


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
    """ Построить диаграмму рассеяния для модели анализа MLP """
    data_train, list_param = build_table_train(True, 'mlp')
    list_param_mlp = data_train.columns.tolist()[2:]
    colors = {}
    for m in session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).all():
        colors[m.title] = m.color
    training_sample = data_train[list_param_mlp].values.tolist()
    markup = sum(data_train[['mark']].values.tolist(), [])
    list_marker = get_list_marker_mlp()

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
                model_class = ExtraTreesClassifier(n_estimators=ui_cls.spinBox_rfc_n.value(), bootstrap=True, oob_score=True, random_state=0, n_jobs=-1)
                text_model = f'**ETC**: \nn estimators: {ui_cls.spinBox_rfc_n.value()}, '
            else:
                model_class = RandomForestClassifier(n_estimators=ui_cls.spinBox_rfc_n.value(), oob_score=True, random_state=0, n_jobs=-1)
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
                              C=ui_cls.doubleSpinBox_svr_c.value(), random_state=0)
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
                etc = ExtraTreesClassifier(n_estimators=ui_cls.spinBox_rfc_n.value(), bootstrap=True, oob_score=True, random_state=0, n_jobs=-1)
                estimators.append(('etc', etc))
                list_model.append('etc')
            else:
                rfc = RandomForestClassifier(n_estimators=ui_cls.spinBox_rfc_n.value(), oob_score=True, random_state=0, n_jobs=-1)
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
                      probability=True, C=ui_cls.doubleSpinBox_svr_c.value(), random_state=0)
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

        start_time = datetime.datetime.now()
        # Нормализация данных
        scaler = StandardScaler()

        pipe_steps = []
        pipe_steps.append(('scaler', scaler))

        # Разделение данных на обучающую и тестовую выборки
        training_sample_train, training_sample_test, markup_train, markup_test = train_test_split(
            training_sample, markup, test_size=0.20, random_state=1)

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

            tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
            train_tsne = tsne.fit_transform(preds_proba_train)
            data_tsne = pd.DataFrame(train_tsne)
            data_tsne['mark'] = preds_train

        if ui_cls.checkBox_cross_val.isChecked():
            kf = KFold(n_splits=ui_cls.spinBox_n_cross_val.value(), shuffle=True, random_state=0)
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
            sns.scatterplot(data=data_tsne, x=0, y=1, hue='mark', s=200, palette=colors, ax=axes[0])
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
            path_model = f'models/classifier/{model_name}_{round(test_accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y")}.pkl'
            with open(path_model, 'wb') as f:
                pickle.dump(pipe, f)

            new_trained_model = TrainedModelClass(
                analysis_id=get_MLP_id(),
                title=f'{model_name}_{round(test_accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y")}',
                path_model=path_model,
                list_params=json.dumps(list_param),
                comment=text_model
            )
            session.add(new_trained_model)
            session.commit()
            update_list_trained_models_class()
        else:
            pass




    def calc_lof():
        """ Расчет выбросов методом LOF """
        global data_pca, data_tsne, colors, factor_lof

        data_lof = data_train.copy()
        data_lof.drop(['prof_well_index', 'mark'], axis=1, inplace=True)

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
            update_list_well_markup_mlp()
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


    def class_exit():
        Classifier.close()

    ui_cls.pushButton_random_search.clicked.connect(class_exit)
    ui_cls.pushButton_random_search.clicked.connect(push_random_search)
    ui_cls.pushButton_lof.clicked.connect(calc_lof)
    ui_cls.pushButton_calc.clicked.connect(calc_model_class)
    ui_cls.checkBox_rfc_extra.clicked.connect(push_checkbutton_extra)
    ui_cls.checkBox_rfc_ada.clicked.connect(push_checkbutton_ada)
    Classifier.exec_()



def update_list_trained_models_class():
    """ Обновление списка тренерованных моделей """

    models = session.query(TrainedModelClass).filter_by(analysis_id=get_MLP_id()).all()
    ui.listWidget_trained_model_class.clear()
    for model in models:
        item_text = model.title
        item = QListWidgetItem(item_text)
        item.setData(Qt.UserRole, model.id)
        item.setToolTip(model.comment)
        ui.listWidget_trained_model_class.addItem(item)
    ui.listWidget_trained_model_class.setCurrentRow(0)


def remove_trained_model_class():
    """ Удаление модели """
    model = session.query(TrainedModelClass).filter_by(id=ui.listWidget_trained_model_class.currentItem().data(Qt.UserRole)).first()
    os.remove(model.path_model)
    session.delete(model)
    session.commit()
    update_list_trained_models_class()
    set_info(f'Модель {model.title} удалена', 'blue')


def calc_class_profile():
    """  Расчет профиля по выбранной модели классификатора """
    working_data, curr_form = build_table_test('mlp')
    working_data_class = working_data.copy()

    Choose_RegModel = QtWidgets.QDialog()
    ui_rm = Ui_FormRegMod()
    ui_rm.setupUi(Choose_RegModel)
    Choose_RegModel.show()
    Choose_RegModel.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    def calc_class_model():

        model = session.query(TrainedModelClass).filter_by(
            id=ui.listWidget_trained_model_class.currentItem().data(Qt.UserRole)).first()

        with open(model.path_model, 'rb') as f:
            class_model = pickle.load(f)

        list_param_num = get_list_param_numerical(json.loads(model.list_params))
        working_sample = working_data_class[list_param_num].values.tolist()

        list_cat = list(class_model.classes_)

        try:
            mark = class_model.predict(working_sample)
            probability = class_model.predict_proba(working_sample)
        except ValueError:
            data = imputer.fit_transform(working_sample)
            mark = class_model.predict(data)
            probability = class_model.predict_proba(data)

            for i in working_data_class.index:
                p_nan = [working_data_class.columns[ic + 3] for ic, v in enumerate(working_data_class.iloc[i, 3:].tolist()) if
                         np.isnan(v)]
                if len(p_nan) > 0:
                    set_info(f'Внимание для измерения "{i}" отсутствуют параметры "{", ".join(p_nan)}", поэтому расчёт для'
                             f' этого измерения может быть не корректен', 'red')

        # Добавление предсказанных меток и вероятностей в рабочие данные
        working_data_result = pd.concat([working_data_class, pd.DataFrame(probability, columns=list_cat)], axis=1)
        working_data_result['mark'] = mark

        draw_result_mlp(working_data_result, curr_form)


    ui_rm.pushButton_calc_model.clicked.connect(calc_class_model)
    Choose_RegModel.exec_()


def draw_result_mlp(working_data, curr_form):
    colors = {}
    for m in session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).all():
        colors[m.title] = m.color

    remove_poly_item()
    list_up = json.loads(curr_form.layer_up.layer_line)  # Получение списка с верхними границами формации
    list_down = json.loads(curr_form.layer_down.layer_line)  # Получение списка со снижными границами формации

    previous_element = None
    list_dupl = []
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

    ui.graph.clear()
    col = working_data.columns[-3]
    number = list(range(1, len(working_data[col]) + 1))
    # Создаем кривую и кривую, отфильтрованную с помощью savgol_filter
    curve = pg.PlotCurveItem(x=number, y=working_data[col].tolist())
    curve_filter = pg.PlotCurveItem(x=number, y=savgol_filter(working_data[col].tolist(), 31, 3),
                                    pen=pg.mkPen(color='red', width=2.4))
    # Добавляем кривую и отфильтрованную кривую на график для всех пластов
    text = pg.TextItem(col, anchor=(0, 1))
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

def calc_object_class():
    """ Расчет объекта по модели """
    working_data_result = pd.DataFrame()
    list_formation = []
    for n, prof in enumerate(session.query(Profile).filter(Profile.research_id == get_research_id()).all()):
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
                # ui.comboBox_plast.setCurrentText(ui_cf.listWidget_form_lda.currentItem().text())
                list_formation.append(ui_cf.listWidget_form_lda.currentItem().text())
                Choose_Formation.close()
            ui_cf.pushButton_ok_form_lda.clicked.connect(form_mlp_ok)
            Choose_Formation.exec_()
    for n, prof in enumerate(session.query(Profile).filter(Profile.research_id == get_research_id()).all()):
        count_measure = len(json.loads(session.query(Profile.signal).filter(Profile.id == prof.id).first()[0]))
        ui.comboBox_profile.setCurrentText(f'{prof.title} ({count_measure} измерений) id{prof.id}')
        update_formation_combobox()
        ui.comboBox_plast.setCurrentText(list_formation[n])
        working_data, curr_form = build_table_test('mlp')
        working_data_result = pd.concat([working_data_result, working_data], axis=0, ignore_index=True)

    working_data_result_copy = working_data_result.copy()

    Choose_RegModel = QtWidgets.QDialog()
    ui_rm = Ui_FormRegMod()
    ui_rm.setupUi(Choose_RegModel)
    Choose_RegModel.show()
    Choose_RegModel.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    def calc_class_model():

        model = session.query(TrainedModelClass).filter_by(
            id=ui.listWidget_trained_model_class.currentItem().data(Qt.UserRole)).first()

        with open(model.path_model, 'rb') as f:
            class_model = pickle.load(f)

        list_param_num = get_list_param_numerical(json.loads(model.list_params))
        working_sample = working_data_result_copy[list_param_num].values.tolist()

        list_cat = list(class_model.classes_)

        try:
            mark = class_model.predict(working_sample)
            probability = class_model.predict_proba(working_sample)
        except ValueError:
            data = imputer.fit_transform(working_sample)
            mark = class_model.predict(data)
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


    for i in data_plot.columns.tolist()[2:]:
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
    model = session.query(TrainedModelClass).filter_by(
        id=ui.listWidget_trained_model_class.currentItem().data(Qt.UserRole)).first()
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

    filename = \
        QFileDialog.getSaveFileName(caption='Экспорт модели классификации', directory=f'{model.title}.zip', filter="*.zip")[
            0]
    with zipfile.ZipFile(filename, 'w') as zip:
        zip.write('model_parameters.pkl', 'model_parameters.pkl')
        zip.write(model.path_model, 'model.pkl')

    set_info(f'Модель {model.title} экспортирована в файл {filename}', 'blue')


def import_model_class():
    """ Импорт модели """
    filename = QFileDialog.getOpenFileName(caption='Импорт модели классификации', filter="*.zip")[0]

    with zipfile.ZipFile(filename, 'r') as zip:
        zip.extractall('extracted_data')

        with open('extracted_data/model_parameters.pkl', 'rb') as parameters_file:
            loaded_parameters = pickle.load(parameters_file)

        # Загрузка модели из файла *.pkl
        with open('extracted_data/model.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)

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
