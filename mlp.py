from draw import draw_radarogram, draw_formation, draw_fill, draw_fake, draw_fill_result, remove_poly_item
from func import *
from krige import draw_map
from qt.choose_formation_lda import *


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


def add_marker_mlp():
    """Добавить новый маркер MLP"""
    if ui.lineEdit_string.text() == '':
        set_info('Введите название маркера', 'red')
        return
    if session.query(MarkerMLP).filter_by(title=ui.lineEdit_string.text(), analysis_id=get_MLP_id()).count() > 0:
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
            well_dist = ui.spinBox_well_dist.value()
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
            well_dist = ui.spinBox_well_dist.value()
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
        well_dist = ui.spinBox_well_dist.value()
        start = well.i_profile - well_dist if well.i_profile - well_dist > 0 else 0
        stop = well.i_profile + well_dist if well.i_profile + well_dist < len(x_prof) else len(x_prof)
        list_measure = list(range(start, stop))
    else:
        well = session.query(Well).filter(Well.id == markup.well_id).first()
        x_prof = json.loads(session.query(Profile.x_pulc).filter(Profile.id == markup.profile_id).first()[0])
        y_prof = json.loads(session.query(Profile.y_pulc).filter(Profile.id == markup.profile_id).first()[0])
        index, _ = closest_point(well.x_coord, well.y_coord, x_prof, y_prof)
        well_dist = ui.spinBox_well_dist.value()
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
    count_markup, count_measure = 0, 0
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
            # ui.listWidget_well_mlp.setItemData(ui.listWidget_well_mlp.findText(item), QBrush(QColor(i.marker.color)), Qt.BackgroundRole)
        except AttributeError:
            set_info(f'Параметр для профиля {i.profile.title} удален из-за отсутствия одного из параметров', 'red')
            session.delete(i)
            session.commit()
    ui.label_count_markup_mlp.setText(f'<i><u>{count_markup}</u></i> обучающих скважин; <i><u>{count_measure}</u></i> измерений')


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


def add_param_geovel_mlp():
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
    session.commit()
    update_list_param_mlp()
    set_info(f'Добавлены коэффициенты mfcc по всем параметрам по {count} интервалам', 'green')


def remove_param_geovel_mlp():
    param = ui.listWidget_param_mlp.currentItem().text().split(' ')[0]
    if param:
        if param.startswith('distr') or param.startswith('sep') or param.startswith('mfcc'):
            for p in session.query(ParameterMLP).filter(ParameterMLP.analysis_id == get_MLP_id()).all():
                if p.parameter.startswith('_'.join(param.split('_')[:-1])):
                    session.query(ParameterMLP).filter_by(id=p.id).delete()
                    session.commit()
        else:
            session.query(ParameterMLP).filter_by(analysis_id=get_MLP_id(), parameter=param ).delete()
        session.commit()
        update_list_param_mlp()


def remove_all_param_geovel_mlp():
    session.query(ParameterMLP).filter_by(analysis_id=get_MLP_id()).delete()
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
        ui.listWidget_param_mlp.addItem(f'{param} \t\tF={round(F, 2)} p={round(p, 3)}')
        if F < 1 or p > 0.05:
            i_item = ui.listWidget_param_mlp.findItems(f'{param} \t\tF={round(F, 2)} p={round(p, 3)}', Qt.MatchContains)[0]
            i_item.setBackground(QBrush(QColor('red')))
    ui.label_count_param_mlp.setText(f'<i><u>{ui.listWidget_param_mlp.count()}</u></i> параметров')


def draw_MLP():
    """ Построить диаграмму рассеяния для модели анализа MLP """
    data_train, list_param = build_table_train(True, 'mlp')
    list_param_mlp = data_train.columns.tolist()[2:]
    colors = {}
    for m in session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).all():
        colors[m.title] = m.color
    training_sample = data_train[list_param_mlp].values.tolist()
    markup = sum(data_train[['mark']].values.tolist(), [])


    Classifier = QtWidgets.QDialog()
    ui_cls = Ui_ClassifierForm()
    ui_cls.setupUi(Classifier)
    Classifier.show()
    Classifier.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    def calc_mlp_form():
        try:
            # Нормализация данных
            scaler = StandardScaler()
            training_sample_norm = scaler.fit_transform(training_sample)

            # Разделение данных на обучающую и тестовую выборки
            training_sample_train, training_sample_test, markup_train, markup_test = train_test_split(
                training_sample_norm, markup, test_size=0.20, random_state=1
            )
            # Создание и тренировка MLP
            layers = tuple(map(int, ui_cls.lineEdit_layer_mlp.text().split()))
            mlp = MLPClassifier(
                hidden_layer_sizes=layers,
                activation=ui_cls.comboBox_activation_mlp.currentText(),
                solver=ui_cls.comboBox_solvar_mlp.currentText(),
                alpha=ui_cls.doubleSpinBox_alpha_mlp.value(),
                max_iter=5000,
                early_stopping=ui_cls.checkBox_e_stop_mlp.isChecked(),
                validation_fraction=ui_cls.doubleSpinBox_valid_mlp.value(),
                random_state=1
            )
            mlp.fit(training_sample_train, markup_train)

            # Оценка точности на всей обучающей выборке
            train_accuracy = mlp.score(training_sample_norm, markup)
            test_accuracy = mlp.score(training_sample_test, markup_test)

            set_info(f'**MLP**: \nhidden_layer_sizes: ({",".join(map(str, layers))}), '
                     f'\nactivation: {ui_cls.comboBox_activation_mlp.currentText()}, '
                     f'\nsolver: {ui_cls.comboBox_solvar_mlp.currentText()}, '
                     f'\nalpha: {ui_cls.doubleSpinBox_alpha_mlp.value()}, '
                     f'\n{"early stopping, " if ui_cls.checkBox_e_stop_mlp.isChecked() else ""}'
                     f'\nvalidation_fraction: {ui_cls.doubleSpinBox_valid_mlp.value()}, '
                     f'точность на всей обучающей выборке: {train_accuracy}, '
                     f'точность на тестовой выборке: {test_accuracy}', 'blue')
        except ValueError:
            set_info(f'Ошибка в расчетах MLP! Возможно значения одного из параметров отсутствуют в интервале обучающей '
                     f'выборки.', 'red')
            return
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
        preds_proba_train = mlp.predict_proba(training_sample_norm)
        preds_train = mlp.predict(training_sample_norm)
        train_tsne = tsne.fit_transform(preds_proba_train)
        data_tsne = pd.DataFrame(train_tsne)
        data_tsne['mark'] = preds_train
        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot()
        sns.scatterplot(data=data_tsne, x=0, y=1, hue='mark', s=200, palette=colors)
        ax.grid()
        ax.xaxis.grid(True, "minor", linewidth=.25)
        ax.yaxis.grid(True, "minor", linewidth=.25)
        title_graph = f'Диаграмма рассеяния для канонических значений MLP\nдля обучающей выборки и тестовой выборки' \
                      f'\n{get_mlp_title().upper()}, параметров: {ui.listWidget_param_mlp.count()}, количество образцов: ' \
                      f'{str(len(data_tsne.index))}\n' \
                      f'hidden_layer_sizes: ({",".join(map(str, layers))}), ' \
                      f'\nalpha: {ui_cls.doubleSpinBox_alpha_mlp.value()}, ' \
                      f'\n{"early stopping, " if ui_cls.checkBox_e_stop_mlp.isChecked() else ""}' \
                      f'\nvalidation_fraction: {ui_cls.doubleSpinBox_valid_mlp.value()}\n' \
                      f'точность на всей обучающей выборке: {round(train_accuracy, 7)}\n'\
                      f'точность на тестовой выборке: {round(test_accuracy, 7)}'
        plt.title(title_graph, fontsize=16)
        plt.tight_layout()
        fig.show()


    def calc_knn_form():
        try:
            # Разделение данных на обучающую и тестовую выборки
            training_sample_train, training_sample_test, markup_train, markup_test = train_test_split(
                training_sample, markup, test_size=0.20, random_state=1)

            # Создание и тренировка KNN
            n_knn = ui_cls.spinBox_neighbors.value()
            weights_knn = 'distance' if ui_cls.checkBox_knn_weights.isChecked() else 'uniform'
            algorithm_knn = ui_cls.comboBox_knn_algorithm.currentText()
            knn = KNeighborsClassifier(n_neighbors=n_knn, weights=weights_knn, algorithm=algorithm_knn)
            knn.fit(training_sample_train, markup_train)

            # Оценка точности
            train_accuracy = knn.score(training_sample, markup)
            test_accuracy = knn.score(training_sample_test, markup_test)
            set_info(f'**KNN**: \nn_neighbors: {n_knn}, '
                     f'\nweights: {weights_knn}, '
                     f'\nalgorithm: {algorithm_knn}, '
                     f'точность на всей обучающей выборке: {train_accuracy}, '
                     f'точность на тестовой выборке: {test_accuracy}', 'blue')
        except ValueError:
            set_info(f'Ошибка в расчетах KNN! Возможно значения одного из параметров отсутствуют в интервале обучающей '
                     f'выборки.', 'red')
            return
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
        preds_proba_train = knn.predict_proba(training_sample)
        preds_train = knn.predict(training_sample)
        train_tsne = tsne.fit_transform(preds_proba_train)
        data_tsne = pd.DataFrame(train_tsne)
        data_tsne['mark'] = preds_train
        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot()
        sns.scatterplot(data=data_tsne, x=0, y=1, hue='mark', s=200, palette=colors)
        ax.grid()
        ax.xaxis.grid(True, "minor", linewidth=.25)
        ax.yaxis.grid(True, "minor", linewidth=.25)
        title_graph = (f'Диаграмма рассеяния для канонических значений KNN\nдля обучающей выборки и тестовой выборки\n'
                       f'n_neighbors: {n_knn}\n'
                       f'weights: {weights_knn}\n'
                       f'algorithm: {algorithm_knn}\n'
                       f'точность на всей обучающей выборке: {round(train_accuracy, 7)}\n'
                       f'точность на тестовой выборке: {round(test_accuracy, 7)}')
        plt.title(title_graph, fontsize=16)
        plt.tight_layout()
        fig.show()


    def calc_gpc_form():
        try:
            # Разделение данных на обучающую и тестовую выборки
            training_sample_train, training_sample_test, markup_train, markup_test = train_test_split(
                training_sample, markup, test_size=0.20, random_state=1)
            # Создание и тренировка GPC
            gpc_kernel_width = ui_cls.doubleSpinBox_gpc_wigth.value()
            gpc_kernel_scale = ui_cls.doubleSpinBox_gpc_scale.value()
            n_restart_optimization = ui_cls.spinBox_gpc_n_restart.value()
            multi_class = ui_cls.comboBox_gpc_multi.currentText()
            kernel = gpc_kernel_scale * RBF(gpc_kernel_width)
            gpc = GaussianProcessClassifier(
                kernel=kernel,
                n_restarts_optimizer=n_restart_optimization,
                random_state = 0,
                multi_class=multi_class,
                n_jobs=-1
            )
            gpc.fit(training_sample, markup)

        # Оценка точности на обучающей выборке
            train_accuracy = gpc.score(training_sample, markup)
            test_accuracy = gpc.score(training_sample_test, markup_test)
            set_info(f'**GPC**: \nwidth kernal: {gpc_kernel_width}, '
                     f'\nscale kernal: {gpc_kernel_scale}, '
                     f'\nn restart: {n_restart_optimization}, '
                     f'\nmulti_class: {multi_class} ,'
                     f'точность на всей обучающей выборке: {train_accuracy}, '
                     f'точность на тестовой выборке: {test_accuracy}', 'blue')
        except ValueError:
            set_info(f'Ошибка в расчетах GPC! Возможно значения одного из параметров отсутствуют в интервале обучающей '
                     f'выборки.', 'red')
            return
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
        preds_train = gpc.predict(training_sample)
        preds_proba_train = gpc.predict_proba(training_sample)
        train_tsne = tsne.fit_transform(preds_proba_train)
        data_tsne = pd.DataFrame(train_tsne)
        data_tsne['mark'] = preds_train
        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot()
        sns.scatterplot(data=data_tsne, x=0, y=1, hue='mark', s=200, palette=colors)
        ax.grid()
        ax.xaxis.grid(True, "minor", linewidth=.25)
        ax.yaxis.grid(True, "minor", linewidth=.25)
        title_graph = (f'Диаграмма рассеяния для канонических значений GPC\nдля обучающей и тестовой выборки\n'
                       f'scale kernal: {gpc_kernel_scale}\n'
                       f'n restart: {n_restart_optimization}\n'
                       f'multi_class: {multi_class}\n'
                       f'точность на всей обучающей выборке: {round(train_accuracy, 7)}\n'
                       f'точность на тестовой выборке: {round(test_accuracy, 7)}')
        plt.title(title_graph, fontsize=16)
        plt.tight_layout()
        fig.show()

    ui_cls.pushButton_calc_mlp.clicked.connect(calc_mlp_form)
    ui_cls.pushButton_calc_knn.clicked.connect(calc_knn_form)
    ui_cls.pushButton_calc_gpc.clicked.connect(calc_gpc_form)
    Classifier.exec_()


def calc_MLP():
    # Получение обучающих данных для MLP
    data_train, list_param = build_table_train(True, 'mlp')
    list_param_mlp = data_train.columns.tolist()[2:]
    training_sample = data_train[list_param_mlp].values.tolist()
    markup = sum(data_train[['mark']].values.tolist(), [])

    # Подготовка тестовых данных для MLP
    working_data, curr_form = build_table_test('mlp')
    profile_title = session.query(Profile.title).filter_by(id=working_data['prof_index'][0].split('_')[0]).first()[0][0]
    working_data_new = working_data.copy()

    Classifier = QtWidgets.QDialog()
    ui_cls = Ui_ClassifierForm()
    ui_cls.setupUi(Classifier)
    Classifier.show()
    Classifier.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    def calc_mlp_form():
        set_info(f'Процесс расчёта MLP. {ui.comboBox_mlp_analysis.currentText()} по профилю {profile_title}', 'blue')
        try:
            # Нормализация данных
            scaler = StandardScaler()
            training_sample_norm = scaler.fit_transform(training_sample)
            working_sample = scaler.fit_transform(working_data_new.iloc[:, 3:])

            # Создание и тренировка MLP
            layers = tuple(map(int, ui_cls.lineEdit_layer_mlp.text().split()))
            mlp = MLPClassifier(
                hidden_layer_sizes=layers,
                activation=ui_cls.comboBox_activation_mlp.currentText(),
                solver=ui_cls.comboBox_solvar_mlp.currentText(),
                alpha=ui_cls.doubleSpinBox_alpha_mlp.value(),
                max_iter=5000,
                early_stopping=ui_cls.checkBox_e_stop_mlp.isChecked(),
                validation_fraction=ui_cls.doubleSpinBox_valid_mlp.value(),
                random_state=1
            )
            mlp.fit(training_sample_norm, markup)

            # Оценка точности на обучающей выборке
            train_accuracy = mlp.score(training_sample_norm, markup)

            # Вывод информации о параметрах MLP и точности модели
            set_info(f'**MLP**: \nhidden_layer_sizes: ({",".join(map(str, layers))}), '
                     f'\nactivation: {ui_cls.comboBox_activation_mlp.currentText()}, '
                     f'\nsolver: {ui_cls.comboBox_solvar_mlp.currentText()}, '
                     f'\nalpha: {ui_cls.doubleSpinBox_alpha_mlp.value()}, '
                     f'\n{"early stopping, " if ui_cls.checkBox_e_stop_mlp.isChecked() else ""}'
                     f'\nvalidation_fraction: {ui_cls.doubleSpinBox_valid_mlp.value()}, '
                     f'\nточность на обучающей выборке: {train_accuracy}', 'blue')
        except ValueError:
            set_info(f'Ошибка в расчетах MLP! Возможно значения одного из параметров отсутствуют в интервале обучающей '
                     f'выборки.', 'red')
            return

        # Создание и обучение модели t-SNE для отображения результата на графике
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
        preds_proba_train = mlp.predict_proba(training_sample_norm)
        preds_train = mlp.predict(training_sample_norm)
        data_probability = pd.DataFrame(preds_proba_train)
        data_probability['mark'] = preds_train
        data_probability['shape'] = ['train'] * len(preds_train)

        list_cat = list(mlp.classes_)

        try:
            # Предсказание меток для тестовых данных
            new_mark = mlp.predict(working_sample)
            probability = mlp.predict_proba(working_sample)
        except ValueError:
            # Обработка возможных ошибок в расчетах MLP для тестовых данных
            data = imputer.fit_transform(working_sample)
            new_mark = mlp.predict(data)
            probability = mlp.predict_proba(data)
            for i in working_data_new.index:
                p_nan = [working_data_new.columns[ic + 3] for ic, v in enumerate(working_data_new.iloc[i, 3:].tolist()) if
                         np.isnan(v)]
                if len(p_nan) > 0:
                    set_info(f'Внимание для измерения "{i}" отсутствуют параметры "{", ".join(p_nan)}", поэтому расчёт для'
                             f' этого измерения может быть не корректен', 'red')

        # Добавление предсказанных меток и вероятностей в рабочие данные
        working_data = pd.concat([working_data_new, pd.DataFrame(probability, columns=list_cat)], axis=1)
        working_data['mark'] = new_mark

        data_work_probability = pd.DataFrame(probability)
        data_work_probability['mark'] = new_mark
        data_work_probability['shape'] = ['work'] * len(new_mark)
        data_probability = pd.concat([data_probability, data_work_probability], ignore_index=True)

        # Вычисление t-SNE для обучающих и тестовых данных
        data_tsne = pd.DataFrame(tsne.fit_transform(data_probability.iloc[:, :-2]))
        data_tsne['mark'] = data_probability['mark']
        data_tsne['shape'] = data_probability['shape']

        # Формирование заголовка для графика
        title_graph = f'Диаграмма рассеяния для канонических значений MLP\nдля обучающей выборки и тестовой выборки' \
                      f'\n{get_mlp_title().upper()}, параметров: {ui.listWidget_param_mlp.count()}, количество образцов: ' \
                      f'{str(len(data_tsne.index))}\n' \
                      f'hidden_layer_sizes: ({",".join(map(str, layers))}), ' \
                      f'\nalpha: {ui_cls.doubleSpinBox_alpha_mlp.value()}, ' \
                      f'\n{"early stopping, " if ui_cls.checkBox_e_stop_mlp.isChecked() else ""}' \
                      f'\nvalidation_fraction: {ui_cls.doubleSpinBox_valid_mlp.value()}\n' \
                      f'точность на обучающей выборке: {round(train_accuracy, 7)}'
        draw_result_mlp(working_data, data_tsne, curr_form, title_graph)


    def calc_knn_form():
        set_info(f'Процесс расчёта KNN. {ui.comboBox_mlp_analysis.currentText()} по профилю {profile_title}', 'blue')
        try:
            # Создание и тренировка KNN
            n_knn = ui_cls.spinBox_neighbors.value()
            weights_knn = 'distance' if ui_cls.checkBox_knn_weights.isChecked() else 'uniform'
            algorithm_knn = ui_cls.comboBox_knn_algorithm.currentText()
            knn = KNeighborsClassifier(n_neighbors=n_knn, weights=weights_knn, algorithm=algorithm_knn)
            knn.fit(training_sample, markup)

            # Оценка точности на обучающей выборке
            train_accuracy = knn.score(training_sample, markup)

            set_info(f'**KNN**: \nn_neighbors: {n_knn}, '
                     f'\nweights: {weights_knn}, '
                     f'\nalgorithm: {algorithm_knn}, '
                     f'\nточность на обучающей выборке: {train_accuracy}, ', 'blue')
        except ValueError:
            set_info(f'Ошибка в расчетах KNN! Возможно значения одного из параметров отсутствуют в интервале обучающей '
                     f'выборки.', 'red')
            return


        # Создание и обучение модели t-SNE для отображения результата на графике
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
        preds_proba_train = knn.predict_proba(training_sample)
        preds_train = knn.predict(training_sample)
        data_probability = pd.DataFrame(preds_proba_train)
        data_probability['mark'] = preds_train
        data_probability['shape'] = ['train'] * len(preds_train)


        list_cat = list(knn.classes_)


        working_sample = working_data_new.iloc[:, 3:]

        try:
            # Предсказание меток для тестовых данных
            new_mark = knn.predict(working_sample)
            probability = knn.predict_proba(working_sample)
        except ValueError:
            # Обработка возможных ошибок в расчетах KNN для тестовых данных
            data = imputer.fit_transform(working_sample)
            new_mark = knn.predict(data)
            probability = knn.predict_proba(data)
            for i in working_data_new.index:
                p_nan = [working_data_new.columns[ic + 3] for ic, v in enumerate(working_data_new.iloc[i, 3:].tolist()) if
                         np.isnan(v)]
                if len(p_nan) > 0:
                    set_info(f'Внимание для измерения "{i}" отсутствуют параметры "{", ".join(p_nan)}", поэтому расчёт для'
                             f' этого измерения может быть не корректен', 'red')

        # Добавление предсказанных меток и вероятностей в рабочие данные
        working_data = pd.concat([working_data_new, pd.DataFrame(probability, columns=list_cat)], axis=1)
        working_data['mark'] = new_mark

        data_work_probability = pd.DataFrame(probability)
        data_work_probability['mark'] = new_mark
        data_work_probability['shape'] = ['work'] * len(new_mark)
        data_probability = pd.concat([data_probability, data_work_probability], ignore_index=True)

        # Вычисление t-SNE для обучающих и тестовых данных
        data_tsne = pd.DataFrame(tsne.fit_transform(data_probability.iloc[:, :-2]))
        data_tsne['mark'] = data_probability['mark']
        data_tsne['shape'] = data_probability['shape']

        # Формирование заголовка для графика
        title_graph = (f'Диаграмма рассеяния для канонических значений KNN\nдля обучающей выборки и тестовой выборки\n'
                       f'n_neighbors: {n_knn}\n'
                       f'weights: {weights_knn}\n'
                       f'algorithm: {algorithm_knn}\n'
                       f'точность на всей обучающей выборке: {train_accuracy}')
        draw_result_mlp(working_data, data_tsne, curr_form, title_graph)


    def calc_gpc_form():
        set_info(f'Процесс расчёта GPC. {ui.comboBox_mlp_analysis.currentText()} по профилю {profile_title}', 'blue')
        try:
            # Создание и тренировка GPC
            gpc_kernel_width = ui_cls.doubleSpinBox_gpc_wigth.value()
            gpc_kernel_scale = ui_cls.doubleSpinBox_gpc_scale.value()
            n_restart_optimization = ui_cls.spinBox_gpc_n_restart.value()
            multi_class = ui_cls.comboBox_gpc_multi.currentText()
            kernel = gpc_kernel_scale * RBF(gpc_kernel_width)
            gpc = GaussianProcessClassifier(
                kernel=kernel,
                n_restarts_optimizer=n_restart_optimization,
                random_state = 0,
                multi_class=multi_class,
                n_jobs=-1
            )
            gpc.fit(training_sample, markup)

            # Оценка точности на обучающей выборке
            train_accuracy = gpc.score(training_sample, markup)
            set_info(f'**GPC**: \nwidth kernal: {gpc_kernel_width}, '
                     f'\nscale kernal: {gpc_kernel_scale}, '
                     f'\nn restart: {n_restart_optimization}, '
                     f'\nmulti_class: {multi_class} ,'
                     f'\nточность на обучающей выборке: {train_accuracy}', 'blue')
        except ValueError:
            set_info(f'Ошибка в расчетах GPC! Возможно значения одного из параметров отсутствуют в интервале обучающей '
                     f'выборки.', 'red')
            return

        # Создание и обучение модели t-SNE для отображения результата на графике
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)

        preds_proba_train = gpc.predict_proba(training_sample)
        preds_train = gpc.predict(training_sample)
        data_probability = pd.DataFrame(preds_proba_train)
        data_probability['mark'] = preds_train
        data_probability['shape'] = ['train'] * len(preds_train)


        list_cat = list(gpc.classes_)

        # Подготовка тестовых данных для MLP
        working_sample = working_data_new.iloc[:, 3:]

        try:
            # Предсказание меток для тестовых данных
            new_mark = gpc.predict(working_sample)
            probability = gpc.predict_proba(working_sample)
        except ValueError:
            # Обработка возможных ошибок в расчетах MLP для тестовых данных
            data = imputer.fit_transform(working_sample)
            new_mark = gpc.predict(data)
            probability = gpc.predict_proba(data)
            for i in working_data_new.index:
                p_nan = [working_data_new.columns[ic + 3] for ic, v in enumerate(working_data_new.iloc[i, 3:].tolist()) if
                         np.isnan(v)]
                if len(p_nan) > 0:
                    set_info(f'Внимание для измерения "{i}" отсутствуют параметры "{", ".join(p_nan)}", поэтому расчёт для'
                             f' этого измерения может быть не корректен', 'red')

        # Добавление предсказанных меток и вероятностей в рабочие данные
        working_data = pd.concat([working_data_new, pd.DataFrame(probability, columns=list_cat)], axis=1)
        working_data['mark'] = new_mark

        data_work_probability = pd.DataFrame(probability)
        data_work_probability['mark'] = new_mark
        data_work_probability['shape'] = ['work'] * len(new_mark)
        data_probability = pd.concat([data_probability, data_work_probability], ignore_index=True)

        # Вычисление t-SNE для обучающих и тестовых данных
        data_tsne = pd.DataFrame(tsne.fit_transform(data_probability.iloc[:, :-2]))
        data_tsne['mark'] = data_probability['mark']
        data_tsne['shape'] = data_probability['shape']

        # Формирование заголовка для графика
        title_graph = (f'Диаграмма рассеяния для канонических значений GPC\nдля обучающей и тестовой выборки\n'
                       f'scale kernal: {gpc_kernel_scale}\n'
                       f'n restart: {n_restart_optimization}\n'
                       f'multi_class: {multi_class}\n'
                       f'точность на обучающей выборке: {train_accuracy}')

        draw_result_mlp(working_data, data_tsne, curr_form, title_graph)


    ui_cls.pushButton_calc_mlp.clicked.connect(calc_mlp_form)
    ui_cls.pushButton_calc_knn.clicked.connect(calc_knn_form)
    ui_cls.pushButton_calc_gpc.clicked.connect(calc_gpc_form)
    Classifier.exec_()


def draw_result_mlp(working_data, data_tsne, curr_form, title_graph):
    colors = {}
    for m in session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).all():
        colors[m.title] = m.color
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot()

    sns.scatterplot(data=data_tsne, x=0, y=1, hue='mark', style='shape', palette=colors)
    ax.grid()
    ax.xaxis.grid(True, "minor", linewidth=.25)
    ax.yaxis.grid(True, "minor", linewidth=.25)

    plt.title(title_graph, fontsize=16)
    plt.tight_layout()
    fig.show()
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
    if ui.checkBox_save_prof_mlp.isChecked():
        try:
            file_name = f'{get_object_name()}_{get_research_name()}_{get_profile_name()}__модель_{get_mlp_title()}.xlsx'
            fn = QFileDialog.getSaveFileName(caption="Сохранить выборку в таблицу", directory=file_name,
                                             filter="Excel Files (*.xlsx)")
            working_data.to_excel(fn[0])
            set_info(f'Таблица сохранена в файл: {fn[0]}', 'green')
        except ValueError:
            pass


def calc_obj_mlp():
    working_data_result = pd.DataFrame()
    for n, prof in enumerate(session.query(Profile).filter(Profile.research_id == get_research_id()).all()):
        count_measure = len(json.loads(session.query(Profile.signal).filter(Profile.id == prof.id).first()[0]))
        ui.comboBox_profile.setCurrentText(f'{prof.title} ({count_measure} измерений) id{prof.id}')
        update_formation_combobox()
        if len(prof.formations) == 1:
            ui.comboBox_plast.setCurrentText(f'{prof.formations[0].title} id{prof.formations[0].id}')
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
                ui.comboBox_plast.setCurrentText(ui_cf.listWidget_form_lda.currentItem().text())
                Choose_Formation.close()
            ui_cf.pushButton_ok_form_lda.clicked.connect(form_mlp_ok)
            Choose_Formation.exec_()
        working_data, curr_form = build_table_test('mlp')
        working_data_result = pd.concat([working_data_result, working_data], axis=0, ignore_index=True)
    data_train, list_param = build_table_train(True, 'mlp')
    list_param_mlp = data_train.columns.tolist()[2:]
    training_sample = data_train[list_param_mlp].values.tolist()
    markup = sum(data_train[['mark']].values.tolist(), [])

    working_data_result_copy = working_data_result.copy()

    Classifier = QtWidgets.QDialog()
    ui_cls = Ui_ClassifierForm()
    ui_cls.setupUi(Classifier)
    Classifier.show()
    Classifier.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    def calc_mlp_form():
        try:
            # Нормализация данных
            scaler = StandardScaler()
            training_sample_norm = scaler.fit_transform(training_sample)

            # Создание и тренировка MLP
            layers = tuple(map(int, ui_cls.lineEdit_layer_mlp.text().split()))
            mlp = MLPClassifier(
                hidden_layer_sizes=layers,
                activation=ui_cls.comboBox_activation_mlp.currentText(),
                solver=ui_cls.comboBox_solvar_mlp.currentText(),
                alpha=ui_cls.doubleSpinBox_alpha_mlp.value(),
                max_iter=5000,
                early_stopping=ui_cls.checkBox_e_stop_mlp.isChecked(),
                validation_fraction=ui_cls.doubleSpinBox_valid_mlp.value(),
                random_state=1
            )
            mlp.fit(training_sample_norm, markup)

            # Оценка точности на обучающей выборке
            train_accuracy = mlp.score(training_sample_norm, markup)

            # Вывод информации о параметрах MLP и точности модели
            set_info(f'**MLP**: \nhidden_layer_sizes: ({",".join(map(str, layers))}), '
                     f'\nactivation: {ui_cls.comboBox_activation_mlp.currentText()}, '
                     f'\nsolver: {ui_cls.comboBox_solvar_mlp.currentText()}, '
                     f'\nalpha: {ui_cls.doubleSpinBox_alpha_mlp.value()}, '
                     f'\n{"early stopping, " if ui_cls.checkBox_e_stop_mlp.isChecked() else ""}'
                     f'\nvalidation_fraction: {ui_cls.doubleSpinBox_valid_mlp.value()}, '
                     f'\nточность на обучающей выборке: {train_accuracy}', 'blue')
        except ValueError:
            set_info(f'Ошибка в расчетах MLP! Возможно значения одного из параметров отсутствуют в интервале обучающей '
                     f'выборки.', 'red')
            return

        list_cat = list(mlp.classes_)

        # Подготовка тестовых данных для MLP
        set_info(f'Процесс расчёта MLP. {ui.comboBox_mlp_analysis.currentText()} по {get_object_name()} {get_research_name()}', 'blue')
        working_sample = scaler.fit_transform(working_data_result_copy.iloc[:, 3:])

        try:
            # Предсказание меток для тестовых данных
            new_mark = mlp.predict(working_sample)
            probability = mlp.predict_proba(working_sample)
        except ValueError:
            # Обработка возможных ошибок в расчетах MLP для тестовых данных
            data = imputer.fit_transform(working_sample)
            new_mark = mlp.predict(data)
            probability = mlp.predict_proba(data)
            for i in working_data.index:
                p_nan = [working_data.columns[ic + 3] for ic, v in enumerate(working_data.iloc[i, 3:].tolist()) if
                         np.isnan(v)]
                if len(p_nan) > 0:
                    set_info(
                        f'Внимание для измерения "{i}" отсутствуют параметры "{", ".join(p_nan)}", поэтому расчёт для'
                        f' этого измерения может быть не корректен', 'red')

        # Добавление предсказанных меток и вероятностей в рабочие данные
        working_data_result = pd.concat([working_data_result_copy, pd.DataFrame(probability, columns=list_cat)], axis=1)
        working_data_result['mark'] = new_mark
        x = list(working_data_result['x_pulc'])
        y = list(working_data_result['y_pulc'])
        if len(set(new_mark)) == 2 and not ui_cls.checkBox_color_marker.isChecked():
            marker_mlp = session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).order_by(MarkerMLP.id).first()
            print(marker_mlp.title)
            z = list(working_data_result[marker_mlp.title])
            color_marker = False
            z_number = string_to_unique_number(list(working_data_result['mark']), 'mlp')
            working_data_result['mark_number'] = z_number
        else:
            z = string_to_unique_number(list(working_data_result['mark']), 'mlp')
            color_marker = True
            working_data_result['mark_number'] = z
        draw_map(x, y, z, 'Classifier MLP', color_marker)
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


    def calc_knn_form():
        try:
            n_knn = ui_cls.spinBox_neighbors.value()
            weights_knn = 'distance' if ui_cls.checkBox_knn_weights.isChecked() else 'uniform'
            algorithm_knn = ui_cls.comboBox_knn_algorithm.currentText()
            knn = KNeighborsClassifier(n_neighbors=n_knn, weights=weights_knn, algorithm=algorithm_knn)
            knn.fit(training_sample, markup)

            # Оценка точности на обучающей выборке
            train_accuracy = knn.score(training_sample, markup)

            set_info(f'**KNN**: \nn_neighbors: {n_knn}, '
                     f'\nweights: {weights_knn}, '
                     f'\nalgorithm: {algorithm_knn}, '
                     f'\nточность на обучающей выборке: {train_accuracy}, ', 'blue')
        except ValueError:
            set_info(f'Ошибка в расчетах KNN! Возможно значения одного из параметров отсутствуют в интервале обучающей '
                     f'выборки.', 'red')
            return


        list_cat = list(knn.classes_)

        # Подготовка тестовых данных для KNN
        set_info(f'Процесс расчёта KNN. {ui.comboBox_mlp_analysis.currentText()} по {get_object_name()} {get_research_name()}', 'blue')
        working_sample = working_data_result_copy.iloc[:, 3:]

        try:
            # Предсказание меток для тестовых данных
            new_mark = knn.predict(working_sample)
            probability = knn.predict_proba(working_sample)
        except ValueError:
            # Обработка возможных ошибок в расчетах KNN для тестовых данных
            data = imputer.fit_transform(working_sample)
            new_mark = knn.predict(data)
            probability = knn.predict_proba(data)
            for i in working_data_result_copy.index:
                p_nan = [working_data_result_copy.columns[ic + 3] for ic, v in enumerate(working_data_result_copy.iloc[i, 3:].tolist()) if
                         np.isnan(v)]
                if len(p_nan) > 0:
                    set_info(
                        f'Внимание для измерения "{i}" отсутствуют параметры "{", ".join(p_nan)}", поэтому расчёт для'
                        f' этого измерения может быть не корректен', 'red')

        # Добавление предсказанных меток и вероятностей в рабочие данные
        working_data_result = pd.concat([working_data_result_copy, pd.DataFrame(probability, columns=list_cat)], axis=1)
        working_data_result['mark'] = new_mark
        x = list(working_data_result['x_pulc'])
        y = list(working_data_result['y_pulc'])
        if len(set(new_mark)) == 2 and not ui_cls.checkBox_color_marker.isChecked():
            marker_mlp = session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).order_by(MarkerMLP.id).first()
            print(marker_mlp.title)
            z = list(working_data_result[marker_mlp.title])
            color_marker = False
            z_number = string_to_unique_number(list(working_data_result['mark']), 'mlp')
            working_data_result['mark_number'] = z_number
        else:
            z = string_to_unique_number(list(working_data_result['mark']), 'mlp')
            working_data_result['mark_number'] = z
            color_marker = True
        draw_map(x, y, z, 'Classifier KNN', color_marker)
        result1 = QMessageBox.question(MainWindow, 'Сохранение', 'Сохранить результаты расчёта KNN?', QMessageBox.Yes, QMessageBox.No)
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
                fn = QFileDialog.getSaveFileName(caption=f'Сохранить результат KNN "{get_object_name()}_{get_research_name()}" в таблицу', directory=file_name,
                                                 filter="Excel Files (*.xlsx)")
                working_data_result.to_excel(fn[0])
                set_info(f'Таблица сохранена в файл: {fn[0]}', 'green')
            except ValueError:
                pass
        else:
            pass


    def calc_gpc_form():
        try:
            # Создание и тренировка GPC
            gpc_kernel_width = ui_cls.doubleSpinBox_gpc_wigth.value()
            gpc_kernel_scale = ui_cls.doubleSpinBox_gpc_scale.value()
            n_restart_optimization = ui_cls.spinBox_gpc_n_restart.value()
            multi_class = ui_cls.comboBox_gpc_multi.currentText()
            kernel = gpc_kernel_scale * RBF(gpc_kernel_width)
            gpc = GaussianProcessClassifier(
                kernel=kernel,
                n_restarts_optimizer=n_restart_optimization,
                random_state = 0,
                multi_class=multi_class,
                n_jobs=-1
            )
            gpc.fit(training_sample, markup)

            train_accuracy = gpc.score(training_sample, markup)
            set_info(f'**GPC**: \nwidth kernal: {gpc_kernel_width}, '
                     f'\nscale kernal: {gpc_kernel_scale}, '
                     f'\nn restart: {n_restart_optimization}, '
                     f'\nmulti_class: {multi_class} ,'
                     f'\nточность на обучающей выборке: {train_accuracy}', 'blue')
        except ValueError:
            set_info(f'Ошибка в расчетах GPC! Возможно значения одного из параметров отсутствуют в интервале обучающей '
                     f'выборки.', 'red')
            return


        list_cat = list(gpc.classes_)

        # Подготовка тестовых данных для GPC
        set_info(f'Процесс расчёта GPC. {ui.comboBox_mlp_analysis.currentText()} по {get_object_name()} {get_research_name()}', 'blue')
        working_sample = working_data_result_copy.iloc[:, 3:]

        try:
            # Предсказание меток для тестовых данных
            new_mark = gpc.predict(working_sample)
            probability = gpc.predict_proba(working_sample)
        except ValueError:
            # Обработка возможных ошибок в расчетах GPC для тестовых данных
            data = imputer.fit_transform(working_sample)
            new_mark = gpc.predict(data)
            probability = gpc.predict_proba(data)
            for i in working_data_result_copy.index:
                p_nan = [working_data_result_copy.columns[ic + 3] for ic, v in enumerate(working_data_result_copy.iloc[i, 3:].tolist()) if
                         np.isnan(v)]
                if len(p_nan) > 0:
                    set_info(f'Внимание для измерения "{i}" отсутствуют параметры "{", ".join(p_nan)}", поэтому расчёт для'
                        f' этого измерения может быть не корректен', 'red')

        # Добавление предсказанных меток и вероятностей в рабочие данные
        working_data_result = pd.concat([working_data_result_copy, pd.DataFrame(probability, columns=list_cat)], axis=1)
        working_data_result['mark'] = new_mark
        x = list(working_data_result['x_pulc'])
        y = list(working_data_result['y_pulc'])
        if len(set(new_mark)) == 2 and not ui_cls.checkBox_color_marker.isChecked():
            marker_mlp = session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).order_by(MarkerMLP.id).first()
            print(marker_mlp.title)
            z = list(working_data_result[marker_mlp.title])
            color_marker = False
            z_number = string_to_unique_number(list(working_data_result['mark']), 'mlp')
            working_data_result['mark_number'] = z_number
        else:
            z = string_to_unique_number(list(working_data_result['mark']), 'mlp')
            working_data_result['mark_number'] = z
            color_marker = True
        draw_map(x, y, z, 'Classifier GPC', color_marker)
        result1 = QMessageBox.question(MainWindow, 'Сохранение', 'Сохранить результаты расчёта GPC?', QMessageBox.Yes, QMessageBox.No)
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
                fn = QFileDialog.getSaveFileName(caption=f'Сохранить результат GPC "{get_object_name()}_{get_research_name()}" в таблицу', directory=file_name,
                                                 filter="Excel Files (*.xlsx)")
                working_data_result.to_excel(fn[0])
                set_info(f'Таблица сохранена в файл: {fn[0]}', 'green')
            except ValueError:
                pass
        else:
            pass

    ui_cls.pushButton_calc_mlp.clicked.connect(calc_mlp_form)
    ui_cls.pushButton_calc_knn.clicked.connect(calc_knn_form)
    ui_cls.pushButton_calc_gpc.clicked.connect(calc_gpc_form)
    Classifier.exec_()





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
