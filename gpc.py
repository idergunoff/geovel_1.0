from draw import draw_radarogram, draw_formation, draw_fill, draw_fake, draw_fill_result, remove_poly_item
from func import *
from krige import draw_map
from qt.choose_formation_lda import *


def add_gpc():
    """Добавить новый анализ GPC"""
    if ui.lineEdit_string.text() == '':
        set_info('Введите название анализа', 'red')
        return
    new_gpc = AnalysisGPC(title=ui.lineEdit_string.text())
    session.add(new_gpc)
    session.commit()
    update_list_gpc()
    set_info(f'Добавлен новый анализ GPC - "{ui.lineEdit_string.text()}"', 'green')


def copy_gpc():
    """Скопировать анализ GPC"""
    if ui.lineEdit_string.text() == '':
        set_info('Введите название для копии анализа', 'red')
        return
    old_gpc = session.query(AnalysisGPC).filter_by(id=get_GPC_id()).first()
    new_gpc = AnalysisGPC(title=ui.lineEdit_string.text())
    session.add(new_gpc)
    session.commit()
    for old_marker in old_gpc.markers:
        new_marker = MarkerGPC(analysis_id=new_gpc.id, title=old_marker.title, color=old_marker.color)
        session.add(new_marker)
        for old_markup in session.query(MarkupGPC).filter_by(analysis_id=get_GPC_id(), marker_id=old_marker.id):
            new_markup = MarkupGPC(
                analysis_id=new_gpc.id,
                well_id=old_markup.well_id,
                profile_id=old_markup.profile_id,
                formation_id=old_markup.formation_id,
                marker_id=new_marker.id,
                list_measure=old_markup.list_measure
            )
            session.add(new_markup)
    session.commit()
    update_list_gpc()
    set_info(f'Скопирован анализ GPC - "{old_gpc.title}"', 'green')


def remove_gpc():
    """Удалить анализ GPC"""
    gpc_title = get_gpc_title()
    result = QtWidgets.QMessageBox.question(ui.listWidget_well_gpc, 'Remove markup GPC',
                                            f'Вы уверены, что хотите удалить модель GPC "{gpc_title}"?',
                                            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        session.query(ParameterGPC).filter_by(analysis_id=get_GPC_id()).delete()
        session.query(MarkerGPC).filter_by(analysis_id=get_GPC_id()).delete()
        session.query(MarkupGPC).filter_by(analysis_id=get_GPC_id()).delete()
        session.query(AnalysisGPC).filter_by(id=get_GPC_id()).delete()
        session.commit()
        set_info(f'Удалена модель GPC - "{gpc_title}"', 'green')
        update_list_gpc()
    elif result == QtWidgets.QMessageBox.No:
        pass


def update_list_gpc(db=False):
    """Обновить список анализов GPC"""
    ui.comboBox_gpc_analysis.clear()
    for i in session.query(AnalysisGPC).order_by(AnalysisGPC.title).all():
        ui.comboBox_gpc_analysis.addItem(f'{i.title} id{i.id}')
    if db:
        update_list_marker_gpc_db()
    else:
        update_list_marker_gpc()



def add_marker_gpc():
    """Добавить новый маркер GPC"""
    if ui.lineEdit_string.text() == '':
        set_info('Введите название маркера', 'red')
        return
    if session.query(MarkerGPC).filter_by(title=ui.lineEdit_string.text(), analysis_id=get_GPC_id()).count() > 0:
        session.query(MarkerGPC).filter_by(title=ui.lineEdit_string.text(), analysis_id=get_GPC_id()).update(
            {'color': ui.pushButton_color.text()}, synchronize_session='fetch')
        set_info(f'Изменен цвет маркера GPC - "{ui.lineEdit_string.text()}"', 'green')
    else:
        new_marker = MarkerGPC(title=ui.lineEdit_string.text(), analysis_id=get_GPC_id(), color=ui.pushButton_color.text())
        session.add(new_marker)
        set_info(f'Добавлен новый маркер GPC - "{ui.lineEdit_string.text()}"', 'green')
    session.commit()
    update_list_marker_gpc(True)


def remove_marker_gpc():
    """Удалить маркер GPC"""
    marker_title = get_marker_gpc_title()
    result = QtWidgets.QMessageBox.question(ui.listWidget_well_gpc, 'Remove marker GPC',
                                            f'В модели {session.query(MarkupGPC).filter_by(marker_id=get_marker_gpc_id()).count()} скважин отмеченных '
                                            f'этим маркером. Вы уверены, что хотите удалить маркер GPC "{marker_title}" вместе с обучающими скважинами'
                                            f' из модели "{get_gpc_title()}"?',
                                            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        session.query(MarkupGPC).filter_by(marker_id=get_marker_gpc_id()).delete()
        session.query(MarkerGPC).filter_by(id=get_marker_gpc_id()).delete()
        session.commit()
        set_info(f'Удалена маркер GPC - "{marker_title}"', 'green')
        update_list_gpc()
    elif result == QtWidgets.QMessageBox.No:
        pass


def update_list_marker_gpc():
    """Обновить список маркеров GPC"""
    ui.comboBox_mark_gpc.clear()
    for i in session.query(MarkerGPC).filter(MarkerGPC.analysis_id == get_GPC_id()).order_by(MarkerGPC.title).all():
        item = f'{i.title} id{i.id}'
        ui.comboBox_mark_gpc.addItem(f'{i.title} id{i.id}')
        ui.comboBox_mark_gpc.setItemData(ui.comboBox_mark_gpc.findText(item), QBrush(QColor(i.color)),
                                         Qt.BackgroundRole)
    update_list_well_markup_gpc()
    update_list_param_gpc(False)


def update_list_marker_gpc_db():
    """Обновить список маркеров GPC"""
    ui.comboBox_mark_gpc.clear()
    for i in session.query(MarkerGPC).filter(MarkerGPC.analysis_id == get_GPC_id()).order_by(MarkerGPC.title).all():
        item = f'{i.title} id{i.id}'
        ui.comboBox_mark_gpc.addItem(f'{i.title} id{i.id}')
        ui.comboBox_mark_gpc.setItemData(ui.comboBox_mark_gpc.findText(item), QBrush(QColor(i.color)),
                                         Qt.BackgroundRole)
    update_list_well_markup_gpc()
    update_list_param_gpc(True)


def add_well_markup_gpc():
    """Добавить новую обучающую скважину для GPC"""
    analysis_id = get_GPC_id()
    well_id = get_well_id()
    profile_id = get_profile_id()
    formation_id = get_formation_id()
    marker_id = get_marker_gpc_id()

    if analysis_id and well_id and profile_id and marker_id and formation_id:
        for param in get_list_param_gpc():
            if not session.query(literal_column(f'Formation.{param}')).filter(Formation.id == formation_id).first()[0]:
                set_info(f'Параметр {param} отсутствует для профиля {get_profile_name()}', 'red')
                return
        well = session.query(Well).filter(Well.id == well_id).first()
        x_prof = json.loads(session.query(Profile.x_pulc).filter(Profile.id == profile_id).first()[0])
        y_prof = json.loads(session.query(Profile.y_pulc).filter(Profile.id == profile_id).first()[0])
        index, _ = closest_point(well.x_coord, well.y_coord, x_prof, y_prof)
        well_dist = ui.spinBox_well_dist.value()
        start = index - well_dist if index - well_dist > 0 else 0
        stop = index + well_dist if index + well_dist < len(x_prof) else len(x_prof)
        list_measure = list(range(start, stop))
        new_markup_gpc = MarkupGPC(analysis_id=analysis_id, well_id=well_id, profile_id=profile_id,
                                   marker_id=marker_id, formation_id=formation_id,
                                   list_measure=json.dumps(list_measure))
        session.add(new_markup_gpc)
        session.commit()
        set_info(f'Добавлена новая обучающая скважина для GPC - "{get_well_name()} {get_marker_gpc_title()}"', 'green')
        update_list_well_markup_gpc()
    else:
        set_info('выбраны не все параметры', 'red')


def update_well_markup_gpc():
    markup = session.query(MarkupGPC).filter(MarkupGPC.id == get_markup_gpc_id()).first()
    if not markup:
        return
    well = session.query(Well).filter(Well.id == markup.well_id).first()
    x_prof = json.loads(session.query(Profile.x_pulc).filter(Profile.id == markup.profile_id).first()[0])
    y_prof = json.loads(session.query(Profile.y_pulc).filter(Profile.id == markup.profile_id).first()[0])
    index, _ = closest_point(well.x_coord, well.y_coord, x_prof, y_prof)
    well_dist = ui.spinBox_well_dist.value()
    start = index - well_dist if index - well_dist > 0 else 0
    stop = index + well_dist if index + well_dist < len(x_prof) else len(x_prof)
    list_measure = list(range(start, stop))
    session.query(MarkupGPC).filter(MarkupGPC.id == get_markup_gpc_id()).update(
        {'marker_id': get_marker_gpc_id(), 'list_measure': json.dumps(list_measure)})
    session.commit()
    set_info(f'Изменена обучающая скважина для GPC - "{well.name} {get_marker_gpc_title()}"', 'green')
    update_list_well_markup_gpc()


def remove_well_markup_gpc():
    markup = session.query(MarkupGPC).filter(MarkupGPC.id == get_markup_gpc_id()).first()
    if not markup:
        return
    skv_name = session.query(Well.name).filter(Well.id == markup.well_id).first()[0]
    prof_name = session.query(Profile.title).filter(Profile.id == markup.profile_id).first()[0]
    gpc_name = session.query(AnalysisGPC.title).filter(AnalysisGPC.id == markup.analysis_id).first()[0]
    result = QtWidgets.QMessageBox.question(ui.listWidget_well_gpc, 'Remove markup GPC',
                                            f'Вы уверены, что хотите удалить скважину "{skv_name}" на '
                                            f'профиле "{prof_name}" из обучающей модели GPC-анализа "{gpc_name}"?',
                                            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        session.delete(markup)
        session.commit()
        set_info(f'Удалена обучающая скважина для GPC - "{ui.listWidget_well_gpc.currentItem().text()}"', 'green')
        update_list_well_markup_gpc()
    elif result == QtWidgets.QMessageBox.No:
        pass


def update_list_well_markup_gpc():
    """Обновление списка обучающих скважин GPC"""
    ui.listWidget_well_gpc.clear()
    for i in session.query(MarkupGPC).filter(MarkupGPC.analysis_id == get_GPC_id()).all():
        fake = len(json.loads(i.list_fake)) if i.list_fake else 0
        measure = len(json.loads(i.list_measure))
        item = f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | {i.well.name} | {measure - fake} из {measure} | id{i.id}'
        ui.listWidget_well_gpc.addItem(item)
        i_item = ui.listWidget_well_gpc.findItems(item, Qt.MatchContains)[0]
        i_item.setBackground(QBrush(QColor(i.marker.color)))
        # ui.listWidget_well_gpc.setItemData(ui.listWidget_well_gpc.findText(item), QBrush(QColor(i.marker.color)), Qt.BackgroundRole)


def choose_marker_gpc():
    # Функция выбора маркера GPC
    # Выбирает маркер, на основе сохраненных данных из базы данных, и затем обновляет все соответствующие виджеты
    # пользовательского интерфейса

    # Получение информации о маркере из БД по его ID
    markup = session.query(MarkupGPC).filter(MarkupGPC.id == get_markup_gpc_id()).first()
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
    draw_well(markup.well_id)
    list_measure = json.loads(markup.list_measure)  # Получение списка измерений
    list_fake = json.loads(markup.list_fake) if markup.list_fake else []  # Получение списка пропущенных измерений
    list_up = json.loads(markup.formation.layer_up.layer_line)  # Получение списка с верхними границами формации
    list_down = json.loads(markup.formation.layer_down.layer_line)  # Получение списка со снижными границами формации
    y_up = [list_up[i] for i in list_measure]  # Создание списка верхних границ для отображения
    y_down = [list_down[i] for i in list_measure]  # Создание списка нижних границ для отображения
    # Обновление маркера с конкретными данными о верхней и нижней границах и цветом
    draw_fill(list_measure, y_up, y_down, markup.marker.color)
    draw_fake(list_fake, list_up, list_down)


def add_param_geovel_gpc():
    param = ui.comboBox_geovel_param_gpc.currentText()
    for m in session.query(MarkupGPC).filter(MarkupGPC.analysis_id == get_GPC_id()).all():
        if not session.query(literal_column(f'Formation.{param}')).filter(Formation.id == m.formation_id).first()[0]:
            set_info(f'Параметр {param} отсутствует для профиля {m.profile.title}', 'red')
            return
    if session.query(ParameterGPC).filter_by(
            analysis_id=get_GPC_id(),
            parameter= param
    ).count() == 0:
        add_param_gpc(param)
        update_list_param_gpc()
    else:
        set_info(f'Параметр {param} уже добавлен', 'red')


def add_all_param_geovel_gpc():
    new_list_param = list_param_geovel.copy()
    for param in list_param_geovel:
        for m in session.query(MarkupGPC).filter(MarkupGPC.analysis_id == get_GPC_id()).all():
            if not session.query(literal_column(f'Formation.{param}')).filter(Formation.id == m.formation_id).first()[0]:
                if param in new_list_param:
                    set_info(f'Параметр {param} отсутствует для профиля {m.profile.title}', 'red')
                    new_list_param.remove(param)
    for param in new_list_param:
        if session.query(ParameterGPC).filter(ParameterGPC.analysis_id == get_GPC_id()).filter(
                ParameterGPC.parameter == param).count() > 0:
            set_info(f'Параметр {param} уже добавлен', 'red')
            continue
        add_param_gpc(param)
    update_list_param_gpc()


def add_param_distr_gpc():
    for param in session.query(ParameterGPC).filter(ParameterGPC.analysis_id == get_GPC_id()).all():
        if param.parameter.startswith(f'distr_{ui.comboBox_atrib_distr_gpc.currentText()}'):
            session.query(ParameterGPC).filter_by(id=param.id).update({
                'parameter': f'distr_{ui.comboBox_atrib_distr_gpc.currentText()}_{ui.spinBox_count_distr_gpc.value()}'
            }, synchronize_session='fetch')
            session.commit()
            update_list_param_gpc()
            set_info(f'В параметры добавлены {ui.spinBox_count_distr_gpc.value()} интервалов распределения по '
                     f'{ui.comboBox_atrib_distr_gpc.currentText()}', 'green')
            return
    add_param_gpc('distr')
    update_list_param_gpc()
    set_info(f'В параметры добавлены {ui.spinBox_count_distr_gpc.value()} интервалов распределения по '
             f'{ui.comboBox_atrib_distr_gpc.currentText()}', 'green')


def add_param_sep_gpc():
    for param in session.query(ParameterGPC).filter(ParameterGPC.analysis_id == get_GPC_id()).all():
        if param.parameter.startswith(f'sep_{ui.comboBox_atrib_distr_gpc.currentText()}'):
            session.query(ParameterGPC).filter_by(id=param.id).update({
                'parameter': f'sep_{ui.comboBox_atrib_distr_gpc.currentText()}_{ui.spinBox_count_distr_gpc.value()}'
            }, synchronize_session='fetch')
            session.commit()
            update_list_param_gpc()
            set_info(f'В параметры добавлены средние значения разделения на {ui.spinBox_count_distr_gpc.value()} интервалов по '
                     f'{ui.comboBox_atrib_distr_gpc.currentText()}', 'green')
            return
    add_param_gpc('sep')
    update_list_param_gpc()
    set_info(f'В параметры добавлены средние значения разделения на {ui.spinBox_count_distr_gpc.value()} интервалов по '
             f'{ui.comboBox_atrib_distr_gpc.currentText()}', 'green')


def add_all_param_distr_gpc():
    list_distr = ['distr_Abase', 'distr_diff', 'distr_At', 'distr_Vt', 'distr_Pht', 'distr_Wt', 'sep_Abase', 'sep_diff', 'sep_At', 'sep_Vt', 'sep_Pht', 'sep_Wt']
    count = ui.spinBox_count_distr_gpc.value()
    for param in session.query(ParameterGPC).filter(ParameterGPC.analysis_id == get_GPC_id()).all():
        if param.parameter.startswith('distr') or param.parameter.startswith('sep'):
            session.query(ParameterGPC).filter_by(id=param.id).delete()
            session.commit()
    for distr_param in list_distr:
        new_param = f'{distr_param}_{count}'
        new_param_gpc = ParameterGPC(analysis_id=get_GPC_id(), parameter=new_param)
        session.add(new_param_gpc)
    session.commit()
    update_list_param_gpc()
    set_info(f'Добавлены все параметры распределения по {count} интервалам', 'green')


def add_param_mfcc_gpc():
    for param in session.query(ParameterGPC).filter(ParameterGPC.analysis_id == get_GPC_id()).all():
        if param.parameter.startswith(f'mfcc_{ui.comboBox_atrib_mfcc_gpc.currentText()}'):
            session.query(ParameterGPC).filter_by(id=param.id).update({
                'parameter': f'mfcc_{ui.comboBox_atrib_mfcc_gpc.currentText()}_{ui.spinBox_count_mfcc_gpc.value()}'
            }, synchronize_session='fetch')
            session.commit()
            update_list_param_gpc()
            set_info(f'В параметры добавлены {ui.spinBox_count_mfcc_gpc.value()} кепстральных коэффициентов '
                     f'{ui.comboBox_atrib_mfcc_gpc.currentText()}', 'green')
            return
    add_param_gpc('mfcc')
    update_list_param_gpc()
    set_info(f'В параметры добавлены {ui.spinBox_count_mfcc_gpc.value()} кепстральных коэффициентов '
             f'{ui.comboBox_atrib_mfcc_gpc.currentText()}', 'green')


def add_all_param_mfcc_gpc():
    list_mfcc = ['mfcc_Abase', 'mfcc_diff', 'mfcc_At', 'mfcc_Vt', 'mfcc_Pht', 'mfcc_Wt']
    count = ui.spinBox_count_mfcc_gpc.value()
    for param in session.query(ParameterGPC).filter(ParameterGPC.analysis_id == get_GPC_id()).all():
        if param.parameter.startswith('mfcc'):
            session.query(ParameterGPC).filter_by(id=param.id).delete()
            session.commit()
    for mfcc_param in list_mfcc:
        new_param = f'{mfcc_param}_{count}'
        new_param_gpc = ParameterGPC(analysis_id=get_GPC_id(), parameter=new_param)
        session.add(new_param_gpc)
    session.commit()
    update_list_param_gpc()
    set_info(f'Добавлены коэффициенты mfcc по всем параметрам по {count} интервалам', 'green')


def remove_param_geovel_gpc():
    param = ui.listWidget_param_gpc.currentItem().text().split(' ')[0]
    if param:
        if param.startswith('distr') or param.startswith('sep') or param.startswith('mfcc'):
            for p in session.query(ParameterGPC).filter(ParameterGPC.analysis_id == get_GPC_id()).all():
                if p.parameter.startswith('_'.join(param.split('_')[:-1])):
                    session.query(ParameterGPC).filter_by(id=p.id).delete()
                    session.commit()
        else:
            session.query(ParameterGPC).filter_by(analysis_id=get_GPC_id(), parameter=param ).delete()
        session.commit()
        update_list_param_gpc()


def remove_all_param_geovel_gpc():
    session.query(ParameterGPC).filter_by(analysis_id=get_GPC_id()).delete()
    session.commit()
    update_list_param_gpc()


def update_list_param_gpc(db=False):
    data_train, list_param = build_table_train(db, 'gpc')
    list_marker = get_list_marker_gpc()
    ui.listWidget_param_gpc.clear()
    list_param_gpc = data_train.columns.tolist()[2:]
    for param in list_param_gpc:
        groups = []
        for mark in list_marker:
            groups.append(data_train[data_train['mark'] == mark][param].values.tolist())
        F, p = f_oneway(*groups)
        ui.listWidget_param_gpc.addItem(f'{param} \t\tF={round(F, 2)} p={round(p, 3)}')
        if F < 1 or p > 0.05:
            i_item = ui.listWidget_param_gpc.findItems(f'{param} \t\tF={round(F, 2)} p={round(p, 3)}', Qt.MatchContains)[0]
            i_item.setBackground(QBrush(QColor('red')))


def draw_GPC():
    """ Построить диаграмму рассеяния для модели анализа GPC """
    data_train, list_param = build_table_train(True, 'gpc')
    list_param_gpc = data_train.columns.tolist()[2:]
    colors = {}
    for m in session.query(MarkerGPC).filter(MarkerGPC.analysis_id == get_GPC_id()).all():
        colors[m.title] = m.color
    training_sample = data_train[list_param_gpc].values.tolist()
    markup = sum(data_train[['mark']].values.tolist(), [])

    try:
        # # Нормализация данных
        # scaler = StandardScaler()
        # training_sample_norm = scaler.fit_transform(training_sample)
        # Разделение данных на обучающую и тестовую выборки
        # training_sagpce_train, training_sample_test, markup_train, markup_test = train_test_split(
        #     training_sample_norm, markup, test_size=0.20, random_state=1
        # )
        # Создание и тренировка GPC
        gpc_kernel_width = ui.doubleSpinBox_gpc_wigth.value()
        gpc_kernel_scale = ui.doubleSpinBox_gpc_scale.value()
        n_restart_optimization = ui.spinBox_gpc_n_restart.value()
        multi_class = ui.comboBox_gpc_multi.currentText()
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
        training_sagpce_train, training_sample_test, markup_train, markup_test = train_test_split(
            training_sample, markup, test_size=0.2, random_state=1)
        test_accuracy = gpc.score(training_sample_test, markup_test)
        set_info(f'точность на всей обучающей выборке: {train_accuracy}, '
                 f'точность на тестовой выборке: {test_accuracy}', 'blue')
    except ValueError:
        set_info(f'Ошибка в расчетах GPC! Возможно значения одного из параметров отсутствуют в интервале обучающей '
                 f'выборки.', 'red')
        return
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    preds_train = gpc.predict(training_sample)
    print(preds_train)
    preds_proba_train = gpc.predict_proba(training_sample)
    print(preds_proba_train)
    train_tsne = tsne.fit_transform(preds_proba_train)
    data_tsne = pd.DataFrame(train_tsne)
    data_tsne['mark'] = preds_train
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot()
    sns.scatterplot(data=data_tsne, x=0, y=1, hue='mark', s=200, palette=colors)
    ax.grid()
    ax.xaxis.grid(True, "minor", linewidth=.25)
    ax.yaxis.grid(True, "minor", linewidth=.25)
    title_graph = f'Диаграмма рассеяния для канонических значений для обучающей выборки\n' \
        f'точность на всей обучающей выборке: {round(train_accuracy, 7)}\n' \
        f'точность на тестовой выборке: {round(test_accuracy, 7)}'
    plt.title(title_graph, fontsize=16)
    plt.tight_layout()
    fig.show()


def calc_verify_gpc():
    data_train, list_param = build_table_train(True, 'gpc')
    list_param_gpc = data_train.columns.tolist()[2:]
    # colors = [m.color for m in session.query(MarkerGPC).filter(MarkerGPC.analysis_id == get_GPC_id()).all()]
    training_sample = data_train[list_param_gpc].values.tolist()
    markup = sum(data_train[['mark']].values.tolist(), [])
    clf = LinearDiscriminantAnalysis()
    try:
        clf.fit(training_sample, markup)
    except ValueError:
        set_info(f'Ошибка в расчетах GPC! Возможно значения одного из параметров отсутствуют в интервале обучающей выборки.', 'red')
        return
    n, k = 0, 0
    ui.progressBar.setMaximum(len(data_train.index))
    for i in data_train.index:
        new_mark = clf.predict([data_train.loc[i].loc[list_param_gpc].tolist()])[0]
        if data_train['mark'][i] != new_mark:
            prof_id = data_train['prof_well_index'][i].split('_')[0]
            well_id = data_train['prof_well_index'][i].split('_')[1]
            ix = int(data_train['prof_well_index'][i].split('_')[2])
            old_list_fake = session.query(MarkupGPC.list_fake).filter(
                MarkupGPC.analysis_id == get_GPC_id(),
                MarkupGPC.profile_id == prof_id,
                MarkupGPC.well_id == well_id
            ).first()[0]
            if old_list_fake:
                new_list_fake = json.loads(old_list_fake)
                new_list_fake.append(ix)
            else:
                new_list_fake = [ix]
            session.query(MarkupGPC).filter(
                MarkupGPC.analysis_id == get_GPC_id(),
                MarkupGPC.profile_id == prof_id,
                MarkupGPC.well_id == well_id
            ).update({'list_fake': json.dumps(new_list_fake)}, synchronize_session='fetch')
            session.commit()
            n += 1
        k += 1
        ui.progressBar.setValue(k)
    session.commit()
    set_info(f'Из обучающей выборки удалено {n} измерений.', 'blue')
    update_list_well_markup_gpc()
    db = True if n == 0 else False
    update_list_param_gpc(db)


def reset_verify_gpc():
    session.query(MarkupGPC).filter(MarkupGPC.analysis_id == get_GPC_id()).update({'list_fake': None},
                                                                                  synchronize_session='fetch')
    session.commit()
    set_info(f'Выбросы для анализа "{ui.comboBox_gpc_analysis.currentText()}" очищены.', 'green')
    update_list_well_markup_gpc()
    update_list_param_gpc()


def calc_GPC():
    colors = {}
    for m in session.query(MarkerGPC).filter(MarkerGPC.analysis_id == get_GPC_id()).all():
        colors[m.title] = m.color
    # colors['test'] = '#999999'
    working_data, data_tsne, curr_form, title_graph = get_working_data_gpc()
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot()

    sns.scatterplot(data=data_tsne, x=0, y=1, hue='mark', style='shape', s=200, palette=colors)
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
    if ui.checkBox_save_prof_gpc.isChecked():
        try:
            file_name = f'{get_object_name()}_{get_research_name()}_{get_profile_name()}__модель_{get_gpc_title()}.xlsx'
            fn = QFileDialog.getSaveFileName(caption="Сохранить выборку в таблицу", directory=file_name,
                                             filter="Excel Files (*.xlsx)")
            working_data.to_excel(fn[0])
            set_info(f'Таблица сохранена в файл: {fn[0]}', 'green')
        except ValueError:
            pass


def calc_obj_gpc():
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

            def form_gpc_ok():
                ui.comboBox_plast.setCurrentText(ui_cf.listWidget_form_lda.currentItem().text())
                Choose_Formation.close()
            ui_cf.pushButton_ok_form_lda.clicked.connect(form_gpc_ok)
            Choose_Formation.exec_()
        working_data, curr_form = build_table_test('gpc')
        working_data_result = pd.concat([working_data_result, working_data], axis=0, ignore_index=True)
    data_train, list_param = build_table_train(True, 'gpc')
    list_param_gpc = data_train.columns.tolist()[2:]
    training_sample = data_train[list_param_gpc].values.tolist()
    markup = sum(data_train[['mark']].values.tolist(), [])

    try:
        # Создание и тренировка GPC
        gpc_kernel_width = ui.doubleSpinBox_gpc_wigth.value()
        gpc_kernel_scale = ui.doubleSpinBox_gpc_scale.value()
        n_restart_optimization = ui.spinBox_gpc_n_restart.value()
        multi_class = ui.comboBox_gpc_multi.currentText()
        kernel = gpc_kernel_scale * RBF(gpc_kernel_width)
        gpc = GaussianProcessClassifier(
            kernel=kernel,
            n_restarts_optimizer=n_restart_optimization,
            random_state = 0,
            multi_class=multi_class,
            n_jobs=-1
        )
        gpc.fit(training_sample, markup)
    except ValueError:
        set_info(f'Ошибка в расчетах GPC! Возможно значения одного из параметров отсутствуют в интервале обучающей '
                 f'выборки.', 'red')
        return


    list_cat = list(gpc.classes_)

    # Подготовка тестовых данных для GPC
    set_info(f'Процесс расчёта GPC. {ui.comboBox_gpc_analysis.currentText()} по {get_object_name()} {get_research_name()}', 'blue')
    working_sample = working_data_result.iloc[:, 3:]

    try:
        # Предсказание меток для тестовых данных
        new_mark = gpc.predict(working_sample)
        probability = gpc.predict_proba(working_sample)
    except ValueError:
        # Обработка возможных ошибок в расчетах GPC для тестовых данных
        data = imputer.fit_transform(working_sample)
        new_mark = gpc.predict(data)
        probability = gpc.predict_proba(data)
        for i in working_data_result.index:
            p_nan = [working_data_result.columns[ic + 3] for ic, v in enumerate(working_data_result.iloc[i, 3:].tolist()) if
                     np.isnan(v)]
            if len(p_nan) > 0:
                set_info(
                    f'Внимание для измерения "{i}" отсутствуют параметры "{", ".join(p_nan)}", поэтому расчёт для'
                    f' этого измерения может быть не корректен', 'red')

    # Добавление предсказанных меток и вероятностей в рабочие данные
    working_data_result = pd.concat([working_data_result, pd.DataFrame(probability, columns=list_cat)], axis=1)
    working_data_result['mark'] = new_mark
    x = list(working_data_result['x_pulc'])
    y = list(working_data_result['y_pulc'])
    # if len(set(new_mark)) == 2:
    #     z = list(working_data_result[list(set(new_mark))[0]])
    # else:
    #     z = string_to_unique_number(list(working_data_result['mark']), 'gpc')
    z = string_to_unique_number(list(working_data_result['mark']), 'gpc')
    draw_map(x, y, z, 'gpc')
    try:
        file_name = f'{get_object_name()}_{get_research_name()}__модель_{get_gpc_title()}.xlsx'
        fn = QFileDialog.getSaveFileName(caption=f'Сохранить результат GPC "{get_object_name()}_{get_research_name()}" в таблицу', directory=file_name,
                                         filter="Excel Files (*.xlsx)")
        working_data_result.to_excel(fn[0])
        set_info(f'Таблица сохранена в файл: {fn[0]}', 'green')
    except ValueError:
        pass
