from draw import draw_radarogram, draw_formation, draw_fill, draw_fake, draw_fill_result, remove_poly_item
from func import *
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
    for i in session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).order_by(MarkerMLP.id).all():
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


def add_param_signal_mlp():
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    param = ui.comboBox_signal_mlp.currentText()
    if session.query(ParameterMLP).filter_by(
            analysis_id=get_MLP_id(),
            parameter=param
    ).count() == 0:
        add_param_mlp(param)
        # update_list_param_mlp()
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
    # update_list_param_mlp()
    set_color_button_updata()


def add_param_crl_mlp():
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    if session.query(ParameterMLP).filter_by(
            analysis_id=get_MLP_id(),
            parameter='CRL'
    ).count() == 0:
        add_param_mlp('CRL')
        # update_list_param_mlp()
        set_color_button_updata()
    else:
        set_info(f'Параметр CRL уже добавлен', 'red')


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
        set_color_button_updata()
        # update_list_param_mlp()
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
    set_color_button_updata()
    # update_list_param_mlp()


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
    # update_list_param_mlp()
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
    # update_list_param_mlp()
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
    set_color_button_updata()
    # update_list_param_mlp()
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
    # update_list_param_mlp()
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
    set_color_button_updata()
    # update_list_param_mlp()
    set_info(f'Добавлены коэффициенты mfcc по всем параметрам по {count} интервалам', 'green')


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


def remove_all_param_geovel_mlp():
    session.query(ParameterMLP).filter_by(analysis_id=get_MLP_id()).delete()
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    update_list_param_mlp()


def update_list_param_mlp(db=False):
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
            print(working_data)
            working_sample = [[np.nan if np.isinf(x) else x for x in y] for y in working_sample]

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
            working_sample = [[np.nan if np.isinf(x) else x for x in y] for y in working_sample]
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
