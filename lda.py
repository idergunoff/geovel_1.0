from draw import draw_radarogram, draw_formation, draw_fill, draw_fake, draw_fill_result, remove_poly_item
from func import *
from build_table import *
from qt.choose_formation_lda import *
from krige import draw_map


def add_lda():
    """Добавить новый анализ LDA"""
    if ui.lineEdit_string.text() == '':
        set_info('Введите название анализа', 'red')
        return
    new_lda = AnalysisLDA(title=ui.lineEdit_string.text())
    session.add(new_lda)
    session.commit()
    update_list_lda()
    set_info(f'Добавлен новый анализ LDA - "{ui.lineEdit_string.text()}"', 'green')


def copy_lda():
    """Скопировать анализ LDA"""
    if ui.lineEdit_string.text() == '':
        set_info('Введите название для копии анализа', 'red')
        QMessageBox.critical(MainWindow, 'Ошибка', 'Введите название для копии анализа в поле в верхней части главного окна')
        return
    old_lda = session.query(AnalysisLDA).filter_by(id=get_LDA_id()).first()
    new_lda = AnalysisLDA(title=ui.lineEdit_string.text())
    session.add(new_lda)
    session.commit()
    for old_marker in old_lda.markers:
        new_marker = MarkerLDA(analysis_id=new_lda.id, title=old_marker.title, color=old_marker.color)
        session.add(new_marker)
        for old_markup in session.query(MarkupLDA).filter_by(analysis_id=get_LDA_id(), marker_id=old_marker.id):
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
    build_table_train_no_db('lda', new_lda.id, [])
    session.commit()
    update_list_lda()
    set_info(f'Скопирован анализ LDA - "{old_lda.title}"', 'green')


def copy_lda_to_mlp():
    """Скопировать анализ LDA в MLP"""
    from mlp import update_list_mlp

    if ui.lineEdit_string.text() == '':
        set_info('Введите название для копии анализа', 'red')
        QMessageBox.critical(MainWindow, 'Ошибка', 'Введите название для копии анализа в поле в верхней части главного окна')
        return
    old_lda = session.query(AnalysisLDA).filter_by(id=get_LDA_id()).first()
    new_mlp = AnalysisMLP(title=ui.lineEdit_string.text())
    session.add(new_mlp)
    session.commit()
    for old_marker in old_lda.markers:
        new_marker = MarkerMLP(analysis_id=new_mlp.id, title=old_marker.title, color=old_marker.color)
        session.add(new_marker)
        for old_markup in session.query(MarkupLDA).filter_by(analysis_id=get_LDA_id(), marker_id=old_marker.id):
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
    build_table_train_no_db('mlp', new_mlp.id, [])
    update_list_mlp()
    set_info(f'Скопирован анализ LDA - "{old_lda.title}"', 'green')



def remove_lda():
    """Удалить анализ LDA"""
    lda_title = get_lda_title()
    result = QtWidgets.QMessageBox.question(ui.listWidget_well_lda, 'Remove markup LDA',
                                            f'Вы уверены, что хотите удалить модель LDA "{lda_title}"?',
                                            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        session.query(ParameterLDA).filter_by(analysis_id=get_LDA_id()).delete()
        session.query(MarkerLDA).filter_by(analysis_id=get_LDA_id()).delete()
        session.query(MarkupLDA).filter_by(analysis_id=get_LDA_id()).delete()
        session.query(AnalysisLDA).filter_by(id=get_LDA_id()).delete()
        session.commit()
        set_info(f'Удалена модель LDA - "{lda_title}"', 'green')
        update_list_lda()
    elif result == QtWidgets.QMessageBox.No:
        pass

def update_list_lda(db=False):
    """Обновить список анализов LDA"""
    ui.comboBox_lda_analysis.clear()
    for i in session.query(AnalysisLDA).order_by(AnalysisLDA.title).all():
        ui.comboBox_lda_analysis.addItem(f'{i.title} id{i.id}')
    if db:
        update_list_marker_lda_db()
    else:
        update_list_marker_lda()


def add_marker_lda():
    """Добавить новый маркер LDA"""
    if ui.lineEdit_string.text() == '':
        set_info('Введите название маркера', 'red')
        return
    if session.query(MarkerLDA).filter_by(title=ui.lineEdit_string.text(), analysis_id=get_LDA_id()).count() > 0:
        session.query(MarkerLDA).filter_by(title=ui.lineEdit_string.text(), analysis_id=get_LDA_id()).update(
            {'color': ui.pushButton_color.text()}, synchronize_session='fetch')
        set_info(f'Изменен цвет маркера LDA - "{ui.lineEdit_string.text()}"', 'green')
    else:
        new_marker = MarkerLDA(title=ui.lineEdit_string.text(), analysis_id=get_LDA_id(), color=ui.pushButton_color.text())
        session.add(new_marker)
        set_info(f'Добавлен новый маркер LDA - "{ui.lineEdit_string.text()}"', 'green')
    session.commit()
    update_list_marker_lda_db()


def remove_marker_lda():
    """Удалить маркер LDA"""
    marker_title = get_marker_title()
    result = QtWidgets.QMessageBox.question(ui.listWidget_well_lda, 'Remove marker LDA',
            f'В модели {session.query(MarkupLDA).filter_by(marker_id=get_marker_id()).count()} скважин отмеченных '
            f'этим маркером. Вы уверены, что хотите удалить маркер LDA "{marker_title}" вместе с обучающими скважинами'
            f' из модели "{get_lda_title()}"?',
                                            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        session.query(MarkupLDA).filter_by(marker_id=get_marker_id()).delete()
        session.query(MarkerLDA).filter_by(id=get_marker_id()).delete()
        session.commit()
        set_info(f'Удалена маркер LDA - "{marker_title}"', 'green')
        update_list_lda()
    elif result == QtWidgets.QMessageBox.No:
        pass


def update_list_marker_lda():
    """Обновить список маркеров LDA"""
    ui.comboBox_mark_lda.clear()
    for i in session.query(MarkerLDA).filter(MarkerLDA.analysis_id == get_LDA_id()).order_by(MarkerLDA.title).all():
        item = f'{i.title} id{i.id}'
        ui.comboBox_mark_lda.addItem(f'{i.title} id{i.id}')
        ui.comboBox_mark_lda.setItemData(ui.comboBox_mark_lda.findText(item), QBrush(QColor(i.color)),
                                         Qt.BackgroundRole)
    update_list_well_markup_lda()
    update_list_param_lda(False)


def update_list_marker_lda_db():
    """Обновить список маркеров LDA"""
    ui.comboBox_mark_lda.clear()
    for i in session.query(MarkerLDA).filter(MarkerLDA.analysis_id == get_LDA_id()).order_by(MarkerLDA.title).all():
        item = f'{i.title} id{i.id}'
        ui.comboBox_mark_lda.addItem(f'{i.title} id{i.id}')
        ui.comboBox_mark_lda.setItemData(ui.comboBox_mark_lda.findText(item), QBrush(QColor(i.color)),
                                         Qt.BackgroundRole)
    update_list_well_markup_lda()
    update_list_param_lda(True)


def add_well_markup_lda():
    """Добавить новую обучающую скважину или пересечение с термограммой для LDA"""
    analysis_id = get_LDA_id()
    well_id = get_well_id()
    profile_id = get_profile_id()
    formation_id = get_formation_id()
    marker_id = get_marker_id()

    if analysis_id and well_id and profile_id and marker_id and formation_id:
        remove_all_param_geovel_lda()
        # for param in get_list_param_lda():
        #     if param.startswith('distr') or param.startswith('sep') or param.startswith('mfcc'):
        #         continue
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
            new_markup_lda = MarkupLDA(analysis_id=analysis_id, well_id=well_id, profile_id=profile_id,
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
            new_markup_lda = MarkupLDA(analysis_id=analysis_id, well_id=well_id, profile_id=profile_id,
                                       marker_id=marker_id, formation_id=formation_id,
                                       list_measure=json.dumps(list_measure))
        session.add(new_markup_lda)
        session.commit()
        set_info(f'Добавлена новая обучающая скважина для LDA - "{get_well_name()} {get_marker_title()}"', 'green')
        update_list_well_markup_lda()
    else:
        set_info('выбраны не все параметры', 'red')


def update_well_markup_lda():
    markup = session.query(MarkupLDA).filter(MarkupLDA.id == get_markup_id()).first()
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
    session.query(MarkupLDA).filter(MarkupLDA.id == get_markup_id()).update(
        {'marker_id': get_marker_id(), 'list_measure': json.dumps(list_measure)})
    session.commit()
    set_info(f'Изменена обучающая скважина для LDA - "{well.name} {get_marker_title()}"', 'green')
    update_list_well_markup_lda()


def remove_well_markup_lda():
    markup = session.query(MarkupLDA).filter(MarkupLDA.id == get_markup_id()).first()
    if not markup:
        return
    if markup.type_markup == 'intersection':
        skv_name = session.query(Intersection.name).filter(Intersection.id == markup.well_id).first()[0]
    else:
        skv_name = session.query(Well.name).filter(Well.id == markup.well_id).first()[0]
    prof_name = session.query(Profile.title).filter(Profile.id == markup.profile_id).first()[0]
    lda_name = session.query(AnalysisLDA.title).filter(AnalysisLDA.id == markup.analysis_id).first()[0]
    result = QtWidgets.QMessageBox.question(ui.listWidget_well_lda, 'Remove markup LDA',
                                            f'Вы уверены, что хотите удалить скважину "{skv_name}" на '
                                            f'профиле "{prof_name}" из обучающей модели LDA-анализа "{lda_name}"?',
                                            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        session.delete(markup)
        session.commit()
        set_info(f'Удалена обучающая скважина для LDA - "{ui.listWidget_well_lda.currentItem().text()}"', 'green')
        update_list_well_markup_lda()
    elif result == QtWidgets.QMessageBox.No:
        pass


def update_list_well_markup_lda():
    """Обновление списка обучающих скважин LDA"""
    ui.listWidget_well_lda.clear()
    count_markup, count_measure = 0, 0
    for i in session.query(MarkupLDA).filter(MarkupLDA.analysis_id == get_LDA_id()).all():
        fake = len(json.loads(i.list_fake)) if i.list_fake else 0
        measure = len(json.loads(i.list_measure))
        try:
            if i.type_markup == 'intersection':
                inter_name = session.query(Intersection.name).filter(Intersection.id == i.well_id).first()[0]
                item = f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | {inter_name} | {measure - fake} из {measure} | id{i.id}'
            else:
                item = f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | {i.well.name} | {measure - fake} из {measure} | id{i.id}'
            ui.listWidget_well_lda.addItem(item)
            i_item = ui.listWidget_well_lda.findItems(item, Qt.MatchContains)[0]
            i_item.setBackground(QBrush(QColor(i.marker.color)))
            count_markup += 1
            count_measure += measure - fake
            # ui.listWidget_well_lda.setItemData(ui.listWidget_well_lda.findText(item), QBrush(QColor(i.marker.color)), Qt.BackgroundRole)
        except AttributeError:
            set_info(f'Параметр для профиля {i.profile.title} удален из-за отсутствия одного из параметров', 'red')
            session.delete(i)
            session.commit()
    ui.label_count_markup_lda.setText(f'<i><u>{count_markup}</u></i> обучающих скважин; <i><u>{count_measure}</u></i> измерений')


def choose_marker_lda():
    # Функция выбора маркера LDA
    # Выбирает маркер, на основе сохраненных данных из базы данных, и затем обновляет все соответствующие виджеты
    # пользовательского интерфейса

    # Получение информации о маркере из БД по его ID
    markup = session.query(MarkupLDA).filter(MarkupLDA.id == get_markup_id()).first()
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


def add_param_geovel_lda():
    session.query(AnalysisLDA).filter_by(id=get_LDA_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    param = ui.comboBox_geovel_param_lda.currentText()
    for m in session.query(MarkupLDA).filter(MarkupLDA.analysis_id == get_LDA_id()).all():
        if not session.query(literal_column(f'Formation.{param}')).filter(Formation.id == m.formation_id).first()[0]:
            set_info(f'Параметр {param} отсутствует для профиля {m.profile.title}', 'red')
            return
    if session.query(ParameterLDA).filter_by(
            analysis_id=get_LDA_id(),
            parameter= param
    ).count() == 0:
        add_param_lda(param)
        update_list_param_lda()
    else:
        set_info(f'Параметр {param} уже добавлен', 'red')


def add_all_param_geovel_lda():
    new_list_param = list_param_geovel.copy()
    for param in list_param_geovel:
        for m in session.query(MarkupLDA).filter(MarkupLDA.analysis_id == get_LDA_id()).all():
            if not session.query(literal_column(f'Formation.{param}')).filter(Formation.id == m.formation_id).first()[0]:
                if param in new_list_param:
                    set_info(f'Параметр {param} отсутствует для профиля {m.profile.title}', 'red')
                    new_list_param.remove(param)
    for param in new_list_param:
        if session.query(ParameterLDA).filter(ParameterLDA.analysis_id == get_LDA_id()).filter(
                ParameterLDA.parameter == param).count() > 0:
            set_info(f'Параметр {param} уже добавлен', 'red')
            continue
        add_param_lda(param)
    session.query(AnalysisLDA).filter_by(id=get_LDA_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    update_list_param_lda()


def add_param_distr_lda():
    for param in session.query(ParameterLDA).filter(ParameterLDA.analysis_id == get_LDA_id()).all():
        if param.parameter.startswith(f'distr_{ui.comboBox_atrib_distr_lda.currentText()}'):
            session.query(ParameterLDA).filter_by(id=param.id).update({
                'parameter': f'distr_{ui.comboBox_atrib_distr_lda.currentText()}_{ui.spinBox_count_distr_lda.value()}'
            }, synchronize_session='fetch')
            session.commit()
            update_list_param_lda()
            set_info(f'В параметры добавлены {ui.spinBox_count_distr_lda.value()} интервалов распределения по '
                     f'{ui.comboBox_atrib_distr_lda.currentText()}', 'green')
            return
    session.query(AnalysisLDA).filter_by(id=get_LDA_id()).update({'up_data': False}, synchronize_session='fetch')
    add_param_lda('distr')
    update_list_param_lda()
    set_info(f'В параметры добавлены {ui.spinBox_count_distr_lda.value()} интервалов распределения по '
             f'{ui.comboBox_atrib_distr_lda.currentText()}', 'green')


def add_param_sep_lda():
    for param in session.query(ParameterLDA).filter(ParameterLDA.analysis_id == get_LDA_id()).all():
        if param.parameter.startswith(f'sep_{ui.comboBox_atrib_distr_lda.currentText()}'):
            session.query(ParameterLDA).filter_by(id=param.id).update({
                'parameter': f'sep_{ui.comboBox_atrib_distr_lda.currentText()}_{ui.spinBox_count_distr_lda.value()}'
            }, synchronize_session='fetch')
            session.commit()
            update_list_param_lda()
            set_info(f'В параметры добавлены средние значения разделения на {ui.spinBox_count_distr_lda.value()} интервалов по '
                f'{ui.comboBox_atrib_distr_lda.currentText()}', 'green')
            return
    add_param_lda('sep')
    session.query(AnalysisLDA).filter_by(id=get_LDA_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    update_list_param_lda()
    set_info(f'В параметры добавлены средние значения разделения на {ui.spinBox_count_distr_lda.value()} интервалов по '
             f'{ui.comboBox_atrib_distr_lda.currentText()}', 'green')


def add_all_param_distr_lda():
    list_distr = ['distr_Abase', 'distr_diff', 'distr_At', 'distr_Vt', 'distr_Pht', 'distr_Wt', 'sep_Abase', 'sep_diff', 'sep_At', 'sep_Vt', 'sep_Pht', 'sep_Wt']
    count = ui.spinBox_count_distr_lda.value()
    for param in session.query(ParameterLDA).filter(ParameterLDA.analysis_id == get_LDA_id()).all():
        if param.parameter.startswith('distr') or param.parameter.startswith('sep'):
            session.query(ParameterLDA).filter_by(id=param.id).delete()
            session.commit()
    for distr_param in list_distr:
        new_param = f'{distr_param}_{count}'
        new_param_lda = ParameterLDA(analysis_id=get_LDA_id(), parameter=new_param)
        session.add(new_param_lda)
    session.query(AnalysisLDA).filter_by(id=get_LDA_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    update_list_param_lda()
    set_info(f'Добавлены все параметры распределения по {count} интервалам', 'green')


def add_param_mfcc_lda():
    for param in session.query(ParameterLDA).filter(ParameterLDA.analysis_id == get_LDA_id()).all():
        if param.parameter.startswith(f'mfcc_{ui.comboBox_atrib_mfcc_lda.currentText()}'):
            session.query(ParameterLDA).filter_by(id=param.id).update({
                'parameter': f'mfcc_{ui.comboBox_atrib_mfcc_lda.currentText()}_{ui.spinBox_count_mfcc.value()}'
            }, synchronize_session='fetch')
            session.commit()
            update_list_param_lda()
            set_info(f'В параметры добавлены {ui.spinBox_count_mfcc.value()} кепстральных коэффициентов '
                f'{ui.comboBox_atrib_mfcc_lda.currentText()}', 'green')
            return
    session.query(AnalysisLDA).filter_by(id=get_LDA_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    add_param_lda('mfcc')
    update_list_param_lda()
    set_info(f'В параметры добавлены {ui.spinBox_count_mfcc.value()} кепстральных коэффициентов '
                f'{ui.comboBox_atrib_mfcc_lda.currentText()}', 'green')


def add_all_param_mfcc_lda():
    list_mfcc = ['mfcc_Abase', 'mfcc_diff', 'mfcc_At', 'mfcc_Vt', 'mfcc_Pht', 'mfcc_Wt']
    count = ui.spinBox_count_mfcc.value()
    for param in session.query(ParameterLDA).filter(ParameterLDA.analysis_id == get_LDA_id()).all():
        if param.parameter.startswith('mfcc'):
            session.query(ParameterLDA).filter_by(id=param.id).delete()
            session.commit()
    for mfcc_param in list_mfcc:
        new_param = f'{mfcc_param}_{count}'
        new_param_lda = ParameterLDA(analysis_id=get_LDA_id(), parameter=new_param)
        session.add(new_param_lda)
    session.query(AnalysisLDA).filter_by(id=get_LDA_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    update_list_param_lda()
    set_info(f'Добавлены коэффициенты mfcc по всем параметрам по {count} интервалам', 'green')


def remove_param_geovel_lda():
    param = ui.listWidget_param_lda.currentItem().text().split(' ')[0]
    if param:
        if param.startswith('distr') or param.startswith('sep') or param.startswith('mfcc'):
            for p in session.query(ParameterLDA).filter(ParameterLDA.analysis_id == get_LDA_id()).all():
                if p.parameter.startswith('_'.join(param.split('_')[:-1])):
                    session.query(ParameterLDA).filter_by(id=p.id).delete()
                    session.commit()
        else:
            session.query(ParameterLDA).filter_by(analysis_id=get_LDA_id(), parameter=param ).delete()
        session.commit()
        ui.listWidget_param_lda.takeItem(ui.listWidget_param_lda.currentRow())
        session.query(AnalysisLDA).filter_by(id=get_LDA_id()).update({'up_data': False}, synchronize_session='fetch')
        session.commit()
        set_color_button_updata_lda()


def remove_all_param_geovel_lda():
    session.query(ParameterLDA).filter_by(analysis_id=get_LDA_id()).delete()
    session.query(AnalysisLDA).filter_by(id=get_LDA_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    update_list_param_lda()


def update_list_param_lda(db=False):
    try:
        data_train, list_param = build_table_train(db)
    except TypeError:
        return
    list_marker = get_list_marker()
    ui.listWidget_param_lda.clear()
    list_param_lda = data_train.columns.tolist()[2:]
    for param in list_param_lda:
        groups = []
        for mark in list_marker:
            groups.append(data_train[data_train['mark'] == mark][param].values.tolist())
        F, p = f_oneway(*groups)
        if np.isnan(F) or np.isnan(p):
            session.query(ParameterLDA).filter_by(analysis_id=get_LDA_id(), parameter=param).delete()
            data_train.drop(param, axis=1, inplace=True)
            session.query(AnalysisLDA).filter_by(id=get_LDA_id()).update({'data': json.dumps(data_train.to_dict())}, synchronize_session='fetch')
            session.commit()
            set_info(f'Параметр {param} удален', 'red')
            continue
        ui.listWidget_param_lda.addItem(f'{param} \t\tF={round(F, 2)} p={round(p, 3)}')
        if F < 1 or p > 0.05:
            i_item = ui.listWidget_param_lda.findItems(f'{param} \t\tF={round(F, 2)} p={round(p, 3)}', Qt.MatchContains)[0]
            i_item.setBackground(QBrush(QColor('red')))
    ui.label_count_param_lda.setText(f'<i><u>{ui.listWidget_param_lda.count()}</u></i> параметров')
    set_color_button_updata_lda()


def set_color_button_updata_lda():
    lda = session.query(AnalysisLDA).filter(AnalysisLDA.id == get_LDA_id()).first()
    btn_color = 'background-color: rgb(191, 255, 191);' if lda.up_data else 'background-color: rgb(255, 185, 185);'
    ui.pushButton_updata_lda.setStyleSheet(btn_color)


def draw_LDA():
    """ Построить диаграмму рассеяния для модели анализа LDA """
    data_train, list_param = build_table_train(True)
    list_param_lda = data_train.columns.tolist()[2:]
    colors = {}
    for m in session.query(MarkerLDA).filter(MarkerLDA.analysis_id == get_LDA_id()).all():
        colors[m.title] = m.color
    training_sample = data_train[list_param_lda].values.tolist()
    markup = sum(data_train[['mark']].values.tolist(), [])
    clf = LinearDiscriminantAnalysis()
    try:
        # selector = RFE(clf, n_features_to_select=0.5, step=1)
        # selector = selector.fit(training_sample, markup)
        # print(selector.support_)
        trans_coef = clf.fit(training_sample, markup).transform(training_sample)
    except ValueError:
        set_info('Ошибка в расчетах LDA! Возможно значения одного из параметров отсутствуют в интервале обучающей '
                 'выборки.', 'red')
        return
    data_trans_coef = pd.DataFrame(trans_coef)
    data_trans_coef['mark'] = data_train['mark'].values.tolist()

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot()
    if ui.listWidget_param_lda.count() < 3:
        sns.scatterplot(data=data_trans_coef, x=0, y=100, hue='mark', palette=colors)
    else:
        sns.scatterplot(data=data_trans_coef, x=0, y=1, hue='mark', palette=colors)
    ax.grid()
    ax.xaxis.grid(True, "minor", linewidth=.25)
    ax.yaxis.grid(True, "minor", linewidth=.25)
    title_graph = f'Диаграмма рассеяния для канонических значений для обучающей выборки' \
                  f'\n{get_lda_title().upper() }\nПараметры: {" ".join(list_param)}\nКоличество образцов: {str(len(data_trans_coef.index))}'
    plt.title(title_graph, fontsize=16)
    fig.show()


def calc_verify_lda():
    data_train, list_param = build_table_train(True)
    list_param_lda = data_train.columns.tolist()[2:]
    # colors = [m.color for m in session.query(MarkerLDA).filter(MarkerLDA.analysis_id == get_LDA_id()).all()]
    training_sample = data_train[list_param_lda].values.tolist()
    markup = sum(data_train[['mark']].values.tolist(), [])
    clf = LinearDiscriminantAnalysis()
    try:
        clf.fit(training_sample, markup)
    except ValueError:
        set_info(f'Ошибка в расчетах LDA! Возможно значения одного из параметров отсутствуют в интервале обучающей выборки.', 'red')
        return
    n, k = 0, 0
    ui.progressBar.setMaximum(len(data_train.index))
    for i in data_train.index:
        new_mark = clf.predict([data_train.loc[i].loc[list_param_lda].tolist()])[0]
        if data_train['mark'][i] != new_mark:
            prof_id = data_train['prof_well_index'][i].split('_')[0]
            well_id = data_train['prof_well_index'][i].split('_')[1]
            ix = int(data_train['prof_well_index'][i].split('_')[2])
            old_list_fake = session.query(MarkupLDA.list_fake).filter(
                MarkupLDA.analysis_id == get_LDA_id(),
                MarkupLDA.profile_id == prof_id,
                MarkupLDA.well_id == well_id
            ).first()[0]
            if old_list_fake:
                new_list_fake = json.loads(old_list_fake)
                new_list_fake.append(ix)
            else:
                new_list_fake = [ix]
            session.query(MarkupLDA).filter(
                MarkupLDA.analysis_id == get_LDA_id(),
                MarkupLDA.profile_id == prof_id,
                MarkupLDA.well_id == well_id
            ).update({'list_fake': json.dumps(new_list_fake)}, synchronize_session='fetch')
            session.commit()
            n += 1
        k += 1
        ui.progressBar.setValue(k)
    session.commit()
    set_info(f'Из обучающей выборки удалено {n} измерений.', 'blue')
    update_list_well_markup_lda()
    db = True if n == 0 else False
    update_list_param_lda(db)


def reset_verify_lda():
    session.query(MarkupLDA).filter(MarkupLDA.analysis_id == get_LDA_id()).update({'list_fake': None},
                                                                                  synchronize_session='fetch')
    session.commit()
    set_info(f'Выбросы для анализа "{ui.comboBox_lda_analysis.currentText()}" очищены.', 'green')
    update_list_well_markup_lda()
    update_list_param_lda()


def calc_LDA():
    colors = {}
    for m in session.query(MarkerLDA).filter(MarkerLDA.analysis_id == get_LDA_id()).all():
        colors[m.title] = m.color
    # colors['test'] = '#999999'
    working_data, data_trans_coef, curr_form = get_working_data_lda()
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot()
    if ui.listWidget_param_lda.count() < 3:
        sns.scatterplot(data=data_trans_coef, x=0, y=100, hue='mark', style='shape', palette=colors)
    else:
        sns.scatterplot(data=data_trans_coef, x=0, y=1, hue='mark', style='shape', palette=colors)
    ax.grid()
    ax.xaxis.grid(True, "minor", linewidth=.25)
    ax.yaxis.grid(True, "minor", linewidth=.25)
    title_graph = f'Диаграмма рассеяния для канонических значений для обучающей выборки' \
                  f'\n{get_lda_title().upper()}\nКоличество образцов: {str(len(data_trans_coef.index))}'
    plt.title(title_graph, fontsize=16)
    # title_graph = f'Диаграмма рассеяния для канонических значений для обучающей выборки' \
    #               f'\n{ui.comboBox_class_lda.currentText().split(". ")[1]}' \
    #               f'\nПараметры: {" ".join(list_param)}' f'\nКоличество образцов: {str(len(data_trans_coef.index))}'
    # plt.title(title_graph, fontsize=16)
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
    if ui.checkBox_save_prof_lda.isChecked():
        try:
            file_name = f'{get_object_name()}_{get_research_name()}_{get_profile_name()}__модель_{get_lda_title()}.xlsx'
            fn = QFileDialog.getSaveFileName(caption="Сохранить выборку в таблицу", directory=file_name,
                                             filter="Excel Files (*.xlsx)")
            working_data.to_excel(fn[0])
            set_info(f'Таблица сохранена в файл: {fn[0]}', 'green')
        except ValueError:
            pass


def calc_obj_lda():
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

            def form_lda_ok():
                # ui.comboBox_plast.setCurrentText(ui_cf.listWidget_form_lda.currentItem().text())
                list_formation.append(ui_cf.listWidget_form_lda.currentItem().text())
                Choose_Formation.close()
            ui_cf.pushButton_ok_form_lda.clicked.connect(form_lda_ok)
            Choose_Formation.exec_()
    for n, prof in enumerate(session.query(Profile).filter(Profile.research_id == get_research_id()).all()):
        count_measure = len(json.loads(session.query(Profile.signal).filter(Profile.id == prof.id).first()[0]))
        ui.comboBox_profile.setCurrentText(f'{prof.title} ({count_measure} измерений) id{prof.id}')
        update_formation_combobox()
        ui.comboBox_plast.setCurrentText(list_formation[n])
        try:
            working_data, curr_form = build_table_test()
        except TypeError:
            set_info('Для выбранной залежи не хватает параметров выбранных в модели анализа.', 'red')
            return
        working_data_result = pd.concat([working_data_result, working_data], axis=0, ignore_index=True)
    data_train, list_param = build_table_train(True)
    list_param_lda = data_train.columns.tolist()[2:]
    training_sample = data_train[list_param_lda].values.tolist()
    markup = sum(data_train[['mark']].values.tolist(), [])
    clf = LinearDiscriminantAnalysis()
    try:
        trans_coef = clf.fit(training_sample, markup).transform(training_sample)
    except ValueError:
        ui.label_info.setText(
            f'Ошибка в расчетах LDA! Возможно значения одного из параметров отсутствуют в интервале обучающей выборки.')
        ui.label_info.setStyleSheet('color: red')
        return
    data_trans_coef = pd.DataFrame(trans_coef)
    data_trans_coef['mark'] = data_train['mark'].values.tolist()
    data_trans_coef['shape'] = ['train'] * len(data_trans_coef)
    list_cat = list(clf.classes_)

    try:
        new_mark = clf.predict(working_data_result.iloc[:, 3:])
        probability = clf.predict_proba(working_data_result.iloc[:, 3:])
    except ValueError:
        data = imputer.fit_transform(working_data_result.iloc[:, 3:])
        new_mark = clf.predict(data)
        probability = clf.predict_proba(data)
        for i in working_data_result.index:
            p_nan = [working_data_result.columns[ic + 3] for ic, v in enumerate(working_data_result.iloc[i, 3:].tolist()) if
                     np.isnan(v)]
            if len(p_nan) > 0:
                profile_title = session.query(Profile.title).filter_by(id=working_data_result['prof_index'][i].split('_')[0]).first()[0][0]
                set_info(f'Внимание для измерения "{i}" на профиле "{profile_title}" отсутствуют параметры "{", ".join(p_nan)}", поэтому расчёт для'
                         f' этого измерения может быть не корректен', 'red')
    working_data_result = pd.concat([working_data_result, pd.DataFrame(probability, columns=list_cat)], axis=1)
    working_data_result['mark'] = new_mark
    x = list(working_data_result['x_pulc'])
    y = list(working_data_result['y_pulc'])
    if len(set(new_mark)) == 2 and not ui.checkBox_color_marker.isChecked():
        marker_lda = session.query(MarkerLDA).filter(MarkerLDA.analysis_id == get_LDA_id()).order_by(MarkerLDA.id).first()
        print(marker_lda.title)
        z = list(working_data_result[marker_lda.title])
        color_marker = False
        z_number = string_to_unique_number(list(working_data_result['mark']), 'lda')
        working_data_result['mark_number'] = z_number
    else:
        z = string_to_unique_number(list(working_data_result['mark']), 'lda')
        color_marker = True
        working_data_result['mark_number'] = z
    draw_map(x, y, z, 'lda', color_marker)
    result1 = QMessageBox.question(MainWindow, 'Сохранение', 'Сохранить результаты расчёта LDA?', QMessageBox.Yes, QMessageBox.No)
    if result1 == QMessageBox.Yes:
        result2 = QMessageBox.question(MainWindow, 'Сохранение', 'Сохранить только результаты расчёта?', QMessageBox.Yes, QMessageBox.No)
        if result2 == QMessageBox.Yes:
            list_col = [i.title for i in session.query(MarkerLDA).filter(MarkerLDA.analysis_id == get_LDA_id()).all()]
            list_col += ['x_pulc', 'y_pulc', 'mark', 'mark_number']
            working_data_result = working_data_result[list_col]
        else:
            pass
        try:
            file_name = f'{get_object_name()}_{get_research_name()}__модель_{get_lda_title()}.xlsx'
            fn = QFileDialog.getSaveFileName(caption=f'Сохранить результат GPC "{get_object_name()}_{get_research_name()}" в таблицу', directory=file_name,
                                             filter="Excel Files (*.xlsx)")
            working_data_result.to_excel(fn[0])
            set_info(f'Таблица сохранена в файл: {fn[0]}', 'green')
        except ValueError:
            pass
    else:
        pass


def calc_corr_lda():
    if not session.query(AnalysisLDA).filter(AnalysisLDA.id == get_LDA_id()).first().up_data:
        update_list_param_lda()
    data_train, list_param = build_table_train(True, 'lda')
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


def anova_lda():
    Anova = QtWidgets.QDialog()
    ui_anova = Ui_Anova()
    ui_anova.setupUi(Anova)
    Anova.show()
    Anova.setAttribute(QtCore.Qt.WA_DeleteOnClose) # атрибут удаления виджета после закрытия

    # ui_anova.graphicsView.setBackground('w')

    data_plot, _ = build_table_train(True, 'lda')

    figure = plt.figure()
    canvas = FigureCanvas(figure)
    ui_anova.horizontalLayout.addWidget(canvas)

    for i in session.query(ParameterLDA).filter_by(analysis_id=get_LDA_id()).all():
        ui_anova.listWidget.addItem(f'{i.parameter} id{i.id}')

    def draw_graph_anova():
        figure.clear()
        param = ui_anova.listWidget.currentItem().text().split(' id')[0]
        if ui_anova.radioButton_box.isChecked():
            sns.boxplot(data=data_plot, y=param, x='mark', orient='v')
        if ui_anova.radioButton_violin.isChecked():
            sns.violinplot(data=data_plot, y=param, x='mark', orient='v')
        if ui_anova.radioButton_strip.isChecked():
            sns.stripplot(data=data_plot, y=param, x='mark', hue='mark', orient='v')
        if ui_anova.radioButton_boxen.isChecked():
            sns.boxenplot(data=data_plot, y=param, x='mark', orient='v')
        figure.tight_layout()
        canvas.draw()

    ui_anova.listWidget.currentItemChanged.connect(draw_graph_anova)
    ui_anova.radioButton_boxen.clicked.connect(draw_graph_anova)
    ui_anova.radioButton_strip.clicked.connect(draw_graph_anova)
    ui_anova.radioButton_violin.clicked.connect(draw_graph_anova)
    ui_anova.radioButton_box.clicked.connect(draw_graph_anova)

    Anova.exec_()
