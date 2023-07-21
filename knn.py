from draw import draw_radarogram, draw_formation, draw_fill, draw_fake, draw_fill_result, remove_poly_item
from func import *
from krige import draw_map
from qt.choose_formation_lda import *


def add_knn():
    """Добавить новый анализ KNN"""
    if ui.lineEdit_string.text() == '':
        set_info('Введите название анализа', 'red')
        return
    new_knn = AnalysisKNN(title=ui.lineEdit_string.text())
    session.add(new_knn)
    session.commit()
    update_list_knn()
    set_info(f'Добавлен новый анализ KNN - "{ui.lineEdit_string.text()}"', 'green')


def copy_knn():
    """Скопировать анализ KNN"""
    if ui.lineEdit_string.text() == '':
        set_info('Введите название для копии анализа', 'red')
        return
    old_knn = session.query(AnalysisKNN).filter_by(id=get_KNN_id()).first()
    new_knn = AnalysisKNN(title=ui.lineEdit_string.text())
    session.add(new_knn)
    session.commit()
    for old_marker in old_knn.markers:
        new_marker = MarkerKNN(analysis_id=new_knn.id, title=old_marker.title, color=old_marker.color)
        session.add(new_marker)
        for old_markup in session.query(MarkupKNN).filter_by(analysis_id=get_KNN_id(), marker_id=old_marker.id):
            new_markup = MarkupKNN(
                analysis_id=new_knn.id,
                well_id=old_markup.well_id,
                profile_id=old_markup.profile_id,
                formation_id=old_markup.formation_id,
                marker_id=new_marker.id,
                list_measure=old_markup.list_measure
            )
            session.add(new_markup)
    session.commit()
    update_list_knn()
    set_info(f'Скопирован анализ KNN - "{old_knn.title}"', 'green')


def remove_knn():
    """Удалить анализ KNN"""
    knn_title = get_knn_title()
    result = QtWidgets.QMessageBox.question(ui.listWidget_well_knn, 'Remove markup KNN',
                                            f'Вы уверены, что хотите удалить модель KNN "{knn_title}"?',
                                            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        session.query(ParameterKNN).filter_by(analysis_id=get_KNN_id()).delete()
        session.query(MarkerKNN).filter_by(analysis_id=get_KNN_id()).delete()
        session.query(MarkupKNN).filter_by(analysis_id=get_KNN_id()).delete()
        session.query(AnalysisKNN).filter_by(id=get_KNN_id()).delete()
        session.commit()
        set_info(f'Удалена модель KNN - "{knn_title}"', 'green')
        update_list_knn()
    elif result == QtWidgets.QMessageBox.No:
        pass


def update_list_knn(db=False):
    """Обновить список анализов KNN"""
    ui.comboBox_knn_analysis.clear()
    for i in session.query(AnalysisKNN).order_by(AnalysisKNN.title).all():
        ui.comboBox_knn_analysis.addItem(f'{i.title} id{i.id}')
    if db:
        update_list_marker_knn_db()
    else:
        update_list_marker_knn()



def add_marker_knn():
    """Добавить новый маркер KNN"""
    if ui.lineEdit_string.text() == '':
        set_info('Введите название маркера', 'red')
        return
    if session.query(MarkerKNN).filter_by(title=ui.lineEdit_string.text(), analysis_id=get_KNN_id()).count() > 0:
        session.query(MarkerKNN).filter_by(title=ui.lineEdit_string.text(), analysis_id=get_KNN_id()).update(
            {'color': ui.pushButton_color.text()}, synchronize_session='fetch')
        set_info(f'Изменен цвет маркера KNN - "{ui.lineEdit_string.text()}"', 'green')
    else:
        new_marker = MarkerKNN(title=ui.lineEdit_string.text(), analysis_id=get_KNN_id(), color=ui.pushButton_color.text())
        session.add(new_marker)
        set_info(f'Добавлен новый маркер KNN - "{ui.lineEdit_string.text()}"', 'green')
    session.commit()
    update_list_marker_knn(True)


def remove_marker_knn():
    """Удалить маркер KNN"""
    marker_title = get_marker_knn_title()
    result = QtWidgets.QMessageBox.question(ui.listWidget_well_knn, 'Remove marker KNN',
                                            f'В модели {session.query(MarkupKNN).filter_by(marker_id=get_marker_knn_id()).count()} скважин отмеченных '
                                            f'этим маркером. Вы уверены, что хотите удалить маркер KNN "{marker_title}" вместе с обучающими скважинами'
                                            f' из модели "{get_knn_title()}"?',
                                            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        session.query(MarkupKNN).filter_by(marker_id=get_marker_knn_id()).delete()
        session.query(MarkerKNN).filter_by(id=get_marker_knn_id()).delete()
        session.commit()
        set_info(f'Удалена маркер KNN - "{marker_title}"', 'green')
        update_list_knn()
    elif result == QtWidgets.QMessageBox.No:
        pass


def update_list_marker_knn():
    """Обновить список маркеров KNN"""
    ui.comboBox_mark_knn.clear()
    for i in session.query(MarkerKNN).filter(MarkerKNN.analysis_id == get_KNN_id()).order_by(MarkerKNN.title).all():
        item = f'{i.title} id{i.id}'
        ui.comboBox_mark_knn.addItem(f'{i.title} id{i.id}')
        ui.comboBox_mark_knn.setItemData(ui.comboBox_mark_knn.findText(item), QBrush(QColor(i.color)),
                                         Qt.BackgroundRole)
    update_list_well_markup_knn()
    update_list_param_knn(False)


def update_list_marker_knn_db():
    """Обновить список маркеров KNN"""
    ui.comboBox_mark_knn.clear()
    for i in session.query(MarkerKNN).filter(MarkerKNN.analysis_id == get_KNN_id()).order_by(MarkerKNN.title).all():
        item = f'{i.title} id{i.id}'
        ui.comboBox_mark_knn.addItem(f'{i.title} id{i.id}')
        ui.comboBox_mark_knn.setItemData(ui.comboBox_mark_knn.findText(item), QBrush(QColor(i.color)),
                                         Qt.BackgroundRole)
    update_list_well_markup_knn()
    update_list_param_knn(True)


def add_well_markup_knn():
    """Добавить новую обучающую скважину для KNN"""
    analysis_id = get_KNN_id()
    well_id = get_well_id()
    profile_id = get_profile_id()
    formation_id = get_formation_id()
    marker_id = get_marker_knn_id()

    if analysis_id and well_id and profile_id and marker_id and formation_id:
        for param in get_list_param_knn():
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
        new_markup_knn = MarkupKNN(analysis_id=analysis_id, well_id=well_id, profile_id=profile_id,
                                   marker_id=marker_id, formation_id=formation_id,
                                   list_measure=json.dumps(list_measure))
        session.add(new_markup_knn)
        session.commit()
        set_info(f'Добавлена новая обучающая скважина для KNN - "{get_well_name()} {get_marker_knn_title()}"', 'green')
        update_list_well_markup_knn()
    else:
        set_info('выбраны не все параметры', 'red')


def update_well_markup_knn():
    markup = session.query(MarkupKNN).filter(MarkupKNN.id == get_markup_knn_id()).first()
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
    session.query(MarkupKNN).filter(MarkupKNN.id == get_markup_knn_id()).update(
        {'marker_id': get_marker_knn_id(), 'list_measure': json.dumps(list_measure)})
    session.commit()
    set_info(f'Изменена обучающая скважина для KNN - "{well.name} {get_marker_knn_title()}"', 'green')
    update_list_well_markup_knn()


def remove_well_markup_knn():
    markup = session.query(MarkupKNN).filter(MarkupKNN.id == get_markup_knn_id()).first()
    if not markup:
        return
    skv_name = session.query(Well.name).filter(Well.id == markup.well_id).first()[0]
    prof_name = session.query(Profile.title).filter(Profile.id == markup.profile_id).first()[0]
    knn_name = session.query(AnalysisKNN.title).filter(AnalysisKNN.id == markup.analysis_id).first()[0]
    result = QtWidgets.QMessageBox.question(ui.listWidget_well_knn, 'Remove markup KNN',
                                            f'Вы уверены, что хотите удалить скважину "{skv_name}" на '
                                            f'профиле "{prof_name}" из обучающей модели KNN-анализа "{knn_name}"?',
                                            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        session.delete(markup)
        session.commit()
        set_info(f'Удалена обучающая скважина для KNN - "{ui.listWidget_well_knn.currentItem().text()}"', 'green')
        update_list_well_markup_knn()
    elif result == QtWidgets.QMessageBox.No:
        pass


def update_list_well_markup_knn():
    """Обновление списка обучающих скважин KNN"""
    ui.listWidget_well_knn.clear()
    for i in session.query(MarkupKNN).filter(MarkupKNN.analysis_id == get_KNN_id()).all():
        fake = len(json.loads(i.list_fake)) if i.list_fake else 0
        measure = len(json.loads(i.list_measure))
        item = f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | {i.well.name} | {measure - fake} из {measure} | id{i.id}'
        ui.listWidget_well_knn.addItem(item)
        i_item = ui.listWidget_well_knn.findItems(item, Qt.MatchContains)[0]
        i_item.setBackground(QBrush(QColor(i.marker.color)))
        # ui.listWidget_well_knn.setItemData(ui.listWidget_well_knn.findText(item), QBrush(QColor(i.marker.color)), Qt.BackgroundRole)


def choose_marker_knn():
    # Функция выбора маркера KNN
    # Выбирает маркер, на основе сохраненных данных из базы данных, и затем обновляет все соответствующие виджеты
    # пользовательского интерфейса

    # Получение информации о маркере из БД по его ID
    markup = session.query(MarkupKNN).filter(MarkupKNN.id == get_markup_knn_id()).first()
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


def add_param_geovel_knn():
    param = ui.comboBox_geovel_param_knn.currentText()
    for m in session.query(MarkupKNN).filter(MarkupKNN.analysis_id == get_KNN_id()).all():
        if not session.query(literal_column(f'Formation.{param}')).filter(Formation.id == m.formation_id).first()[0]:
            set_info(f'Параметр {param} отсутствует для профиля {m.profile.title}', 'red')
            return
    if session.query(ParameterKNN).filter_by(
            analysis_id=get_KNN_id(),
            parameter= param
    ).count() == 0:
        add_param_knn(param)
        update_list_param_knn()
    else:
        set_info(f'Параметр {param} уже добавлен', 'red')


def add_all_param_geovel_knn():
    new_list_param = list_param_geovel.copy()
    for param in list_param_geovel:
        for m in session.query(MarkupKNN).filter(MarkupKNN.analysis_id == get_KNN_id()).all():
            if not session.query(literal_column(f'Formation.{param}')).filter(Formation.id == m.formation_id).first()[0]:
                if param in new_list_param:
                    set_info(f'Параметр {param} отсутствует для профиля {m.profile.title}', 'red')
                    new_list_param.remove(param)
    for param in new_list_param:
        if session.query(ParameterKNN).filter(ParameterKNN.analysis_id == get_KNN_id()).filter(
                ParameterKNN.parameter == param).count() > 0:
            set_info(f'Параметр {param} уже добавлен', 'red')
            continue
        add_param_knn(param)
    update_list_param_knn()


def add_param_distr_knn():
    for param in session.query(ParameterKNN).filter(ParameterKNN.analysis_id == get_KNN_id()).all():
        if param.parameter.startswith(f'distr_{ui.comboBox_atrib_distr_knn.currentText()}'):
            session.query(ParameterKNN).filter_by(id=param.id).update({
                'parameter': f'distr_{ui.comboBox_atrib_distr_knn.currentText()}_{ui.spinBox_count_distr_knn.value()}'
            }, synchronize_session='fetch')
            session.commit()
            update_list_param_knn()
            set_info(f'В параметры добавлены {ui.spinBox_count_distr_knn.value()} интервалов распределения по '
                     f'{ui.comboBox_atrib_distr_knn.currentText()}', 'green')
            return
    add_param_knn('distr')
    update_list_param_knn()
    set_info(f'В параметры добавлены {ui.spinBox_count_distr_knn.value()} интервалов распределения по '
             f'{ui.comboBox_atrib_distr_knn.currentText()}', 'green')


def add_param_sep_knn():
    for param in session.query(ParameterKNN).filter(ParameterKNN.analysis_id == get_KNN_id()).all():
        if param.parameter.startswith(f'sep_{ui.comboBox_atrib_distr_knn.currentText()}'):
            session.query(ParameterKNN).filter_by(id=param.id).update({
                'parameter': f'sep_{ui.comboBox_atrib_distr_knn.currentText()}_{ui.spinBox_count_distr_knn.value()}'
            }, synchronize_session='fetch')
            session.commit()
            update_list_param_knn()
            set_info(f'В параметры добавлены средние значения разделения на {ui.spinBox_count_distr_knn.value()} интервалов по '
                     f'{ui.comboBox_atrib_distr_knn.currentText()}', 'green')
            return
    add_param_knn('sep')
    update_list_param_knn()
    set_info(f'В параметры добавлены средние значения разделения на {ui.spinBox_count_distr_knn.value()} интервалов по '
             f'{ui.comboBox_atrib_distr_knn.currentText()}', 'green')


def add_all_param_distr_knn():
    list_distr = ['distr_Abase', 'distr_diff', 'distr_At', 'distr_Vt', 'distr_Pht', 'distr_Wt', 'sep_Abase', 'sep_diff', 'sep_At', 'sep_Vt', 'sep_Pht', 'sep_Wt']
    count = ui.spinBox_count_distr_knn.value()
    for param in session.query(ParameterKNN).filter(ParameterKNN.analysis_id == get_KNN_id()).all():
        if param.parameter.startswith('distr') or param.parameter.startswith('sep'):
            session.query(ParameterKNN).filter_by(id=param.id).delete()
            session.commit()
    for distr_param in list_distr:
        new_param = f'{distr_param}_{count}'
        new_param_knn = ParameterKNN(analysis_id=get_KNN_id(), parameter=new_param)
        session.add(new_param_knn)
    session.commit()
    update_list_param_knn()
    set_info(f'Добавлены все параметры распределения по {count} интервалам', 'green')


def add_param_mfcc_knn():
    for param in session.query(ParameterKNN).filter(ParameterKNN.analysis_id == get_KNN_id()).all():
        if param.parameter.startswith(f'mfcc_{ui.comboBox_atrib_mfcc_knn.currentText()}'):
            session.query(ParameterKNN).filter_by(id=param.id).update({
                'parameter': f'mfcc_{ui.comboBox_atrib_mfcc_knn.currentText()}_{ui.spinBox_count_mfcc_knn.value()}'
            }, synchronize_session='fetch')
            session.commit()
            update_list_param_knn()
            set_info(f'В параметры добавлены {ui.spinBox_count_mfcc_knn.value()} кепстральных коэффициентов '
                     f'{ui.comboBox_atrib_mfcc_knn.currentText()}', 'green')
            return
    add_param_knn('mfcc')
    update_list_param_knn()
    set_info(f'В параметры добавлены {ui.spinBox_count_mfcc_knn.value()} кепстральных коэффициентов '
             f'{ui.comboBox_atrib_mfcc_knn.currentText()}', 'green')


def add_all_param_mfcc_knn():
    list_mfcc = ['mfcc_Abase', 'mfcc_diff', 'mfcc_At', 'mfcc_Vt', 'mfcc_Pht', 'mfcc_Wt']
    count = ui.spinBox_count_mfcc_knn.value()
    for param in session.query(ParameterKNN).filter(ParameterKNN.analysis_id == get_KNN_id()).all():
        if param.parameter.startswith('mfcc'):
            session.query(ParameterKNN).filter_by(id=param.id).delete()
            session.commit()
    for mfcc_param in list_mfcc:
        new_param = f'{mfcc_param}_{count}'
        new_param_knn = ParameterKNN(analysis_id=get_KNN_id(), parameter=new_param)
        session.add(new_param_knn)
    session.commit()
    update_list_param_knn()
    set_info(f'Добавлены коэффициенты mfcc по всем параметрам по {count} интервалам', 'green')


def remove_param_geovel_knn():
    param = ui.listWidget_param_knn.currentItem().text().split(' ')[0]
    if param:
        if param.startswith('distr') or param.startswith('sep') or param.startswith('mfcc'):
            for p in session.query(ParameterKNN).filter(ParameterKNN.analysis_id == get_KNN_id()).all():
                if p.parameter.startswith('_'.join(param.split('_')[:-1])):
                    session.query(ParameterKNN).filter_by(id=p.id).delete()
                    session.commit()
        else:
            session.query(ParameterKNN).filter_by(analysis_id=get_KNN_id(), parameter=param ).delete()
        session.commit()
        update_list_param_knn()


def remove_all_param_geovel_knn():
    session.query(ParameterKNN).filter_by(analysis_id=get_KNN_id()).delete()
    session.commit()
    update_list_param_knn()


def update_list_param_knn(db=False):
    data_train, list_param = build_table_train(db, 'knn')
    list_marker = get_list_marker_knn()
    ui.listWidget_param_knn.clear()
    list_param_knn = data_train.columns.tolist()[2:]
    for param in list_param_knn:
        groups = []
        for mark in list_marker:
            groups.append(data_train[data_train['mark'] == mark][param].values.tolist())
        F, p = f_oneway(*groups)
        ui.listWidget_param_knn.addItem(f'{param} \t\tF={round(F, 2)} p={round(p, 3)}')
        if F < 1 or p > 0.05:
            i_item = ui.listWidget_param_knn.findItems(f'{param} \t\tF={round(F, 2)} p={round(p, 3)}', Qt.MatchContains)[0]
            i_item.setBackground(QBrush(QColor('red')))


def draw_KNN():
    """ Построить диаграмму рассеяния для модели анализа KNN """
    data_train, list_param = build_table_train(True, 'knn')
    list_param_knn = data_train.columns.tolist()[2:]
    colors = {}
    for m in session.query(MarkerKNN).filter(MarkerKNN.analysis_id == get_KNN_id()).all():
        colors[m.title] = m.color
    training_sample = data_train[list_param_knn].values.tolist()
    markup = sum(data_train[['mark']].values.tolist(), [])

    try:
        # # Нормализация данных
        # scaler = StandardScaler()
        # training_sample_norm = scaler.fit_transform(training_sample)
        # Разделение данных на обучающую и тестовую выборки
        # training_saknne_train, training_sample_test, markup_train, markup_test = train_test_split(
        #     training_sample_norm, markup, test_size=0.20, random_state=1
        # )
        # Создание и тренировка KNN
        n_knn = ui.spinBox_neighbors.value()
        weights_knn = 'distance' if ui.checkBox_knn_weights.isChecked() else 'uniform'
        algorithm_knn = ui.comboBox_knn_algorithm.currentText()
        knn = KNeighborsClassifier(n_neighbors=n_knn, weights=weights_knn, algorithm=algorithm_knn)
        knn.fit(training_sample, markup)
        for i in knn.kneighbors(training_sample, return_distance=True):
            print(i)
        print(knn.kneighbors_graph(training_sample))


    # Оценка точности на обучающей выборке
    #     train_accuracy = knn.score(training_sample_norm, markup)
    #     training_saknne_train, training_sample_test, markup_train, markup_test = train_test_split(
    #         training_sample_norm, markup, test_size=ui.doubleSpinBox_valid_knn.value(), random_state=1)
    #     test_accuracy = knn.score(training_sample_test, markup_test)
    #     set_info(f'hidden_layer_sizes - ({",".join(map(str, layers))}), '
    #              f'activation - {ui.comboBox_activation_knn.currentText()}, '
    #              f'solver - {ui.comboBox_solvar_knn.currentText()}, '
    #              f'alpha - {ui.doubleSpinBox_alpha_knn.value()}, '
    #              f'{"early stopping, " if ui.checkBox_e_stop_knn.isChecked() else ""}'
    #              f'validation_fraction - {ui.doubleSpinBox_valid_knn.value()}, '
    #              f'точность на всей обучающей выборке: {train_accuracy}, '
    #              f'точность на тестовой выборке: {test_accuracy}', 'blue')
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
    title_graph = f'Диаграмма рассеяния для канонических значений для обучающей выборки' \
                  # f'\n{get_knn_title().upper()}, параметров: {ui.listWidget_param_knn.count()}, количество образцов: ' \
                  # f'{str(len(data_tsne.index))}\n' \
                  # f'hidden_layer_sizes - ({",".join(map(str, layers))}), ' \
                  # f'alpha - {ui.doubleSpinBox_alpha_knn.value()}, ' \
                  # f'{"early stopping, " if ui.checkBox_e_stop_knn.isChecked() else ""}' \
                  # f'validation_fraction - {ui.doubleSpinBox_valid_knn.value()}\n' \
                  # f'точность на всей обучающей выборке: {round(train_accuracy, 7)}\n' \
                  # f'точность на тестовой выборке: {round(test_accuracy, 7)}'
    plt.title(title_graph, fontsize=16)
    plt.tight_layout()
    fig.show()


def calc_verify_knn():
    data_train, list_param = build_table_train(True, 'knn')
    list_param_knn = data_train.columns.tolist()[2:]
    # colors = [m.color for m in session.query(MarkerKNN).filter(MarkerKNN.analysis_id == get_KNN_id()).all()]
    training_sample = data_train[list_param_knn].values.tolist()
    markup = sum(data_train[['mark']].values.tolist(), [])
    clf = LinearDiscriminantAnalysis()
    try:
        clf.fit(training_sample, markup)
    except ValueError:
        set_info(f'Ошибка в расчетах KNN! Возможно значения одного из параметров отсутствуют в интервале обучающей выборки.', 'red')
        return
    n, k = 0, 0
    ui.progressBar.setMaximum(len(data_train.index))
    for i in data_train.index:
        new_mark = clf.predict([data_train.loc[i].loc[list_param_knn].tolist()])[0]
        if data_train['mark'][i] != new_mark:
            prof_id = data_train['prof_well_index'][i].split('_')[0]
            well_id = data_train['prof_well_index'][i].split('_')[1]
            ix = int(data_train['prof_well_index'][i].split('_')[2])
            old_list_fake = session.query(MarkupKNN.list_fake).filter(
                MarkupKNN.analysis_id == get_KNN_id(),
                MarkupKNN.profile_id == prof_id,
                MarkupKNN.well_id == well_id
            ).first()[0]
            if old_list_fake:
                new_list_fake = json.loads(old_list_fake)
                new_list_fake.append(ix)
            else:
                new_list_fake = [ix]
            session.query(MarkupKNN).filter(
                MarkupKNN.analysis_id == get_KNN_id(),
                MarkupKNN.profile_id == prof_id,
                MarkupKNN.well_id == well_id
            ).update({'list_fake': json.dumps(new_list_fake)}, synchronize_session='fetch')
            session.commit()
            n += 1
        k += 1
        ui.progressBar.setValue(k)
    session.commit()
    set_info(f'Из обучающей выборки удалено {n} измерений.', 'blue')
    update_list_well_markup_knn()
    db = True if n == 0 else False
    update_list_param_knn(db)


def reset_verify_knn():
    session.query(MarkupKNN).filter(MarkupKNN.analysis_id == get_KNN_id()).update({'list_fake': None},
                                                                                  synchronize_session='fetch')
    session.commit()
    set_info(f'Выбросы для анализа "{ui.comboBox_knn_analysis.currentText()}" очищены.', 'green')
    update_list_well_markup_knn()
    update_list_param_knn()


def calc_KNN():
    colors = {}
    for m in session.query(MarkerKNN).filter(MarkerKNN.analysis_id == get_KNN_id()).all():
        colors[m.title] = m.color
    # colors['test'] = '#999999'
    working_data, data_tsne, curr_form, title_graph = get_working_data_knn()
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
    if ui.checkBox_save_prof_knn.isChecked():
        try:
            file_name = f'{get_object_name()}_{get_research_name()}_{get_profile_name()}__модель_{get_knn_title()}.xlsx'
            fn = QFileDialog.getSaveFileName(caption="Сохранить выборку в таблицу", directory=file_name,
                                             filter="Excel Files (*.xlsx)")
            working_data.to_excel(fn[0])
            set_info(f'Таблица сохранена в файл: {fn[0]}', 'green')
        except ValueError:
            pass


# def calc_obj_knn():
#     working_data_result = pd.DataFrame()
#     for n, prof in enumerate(session.query(Profile).filter(Profile.research_id == get_research_id()).all()):
#         count_measure = len(json.loads(session.query(Profile.signal).filter(Profile.id == prof.id).first()[0]))
#         ui.comboBox_profile.setCurrentText(f'{prof.title} ({count_measure} измерений) id{prof.id}')
#         update_formation_combobox()
#         if len(prof.formations) == 1:
#             ui.comboBox_plast.setCurrentText(f'{prof.formations[0].title} id{prof.formations[0].id}')
#         elif len(prof.formations) > 1:
#             Choose_Formation = QtWidgets.QDialog()
#             ui_cf = Ui_FormationLDA()
#             ui_cf.setupUi(Choose_Formation)
#             Choose_Formation.show()
#             Choose_Formation.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия
#             for f in prof.formations:
#                 ui_cf.listWidget_form_lda.addItem(f'{f.title} id{f.id}')
#             ui_cf.listWidget_form_lda.setCurrentRow(0)
#
#             def form_knn_ok():
#                 ui.comboBox_plast.setCurrentText(ui_cf.listWidget_form_lda.currentItem().text())
#                 Choose_Formation.close()
#             ui_cf.pushButton_ok_form_lda.clicked.connect(form_knn_ok)
#             Choose_Formation.exec_()
#         working_data, curr_form = build_table_test('knn')
#         working_data_result = pd.concat([working_data_result, working_data], axis=0, ignore_index=True)
#     data_train, list_param = build_table_train(True, 'knn')
#     list_param_knn = data_train.columns.tolist()[2:]
#     training_sample = data_train[list_param_knn].values.tolist()
#     markup = sum(data_train[['mark']].values.tolist(), [])
#
#     try:
#         # Нормализация данных
#         scaler = StandardScaler()
#         training_sample_norm = scaler.fit_transform(training_sample)
#
#         # Создание и тренировка KNN
#         layers = tuple(map(int, ui.lineEdit_layer_knn.text().split()))
#         knn = KNNClassifier(
#             hidden_layer_sizes=layers,
#             activation=ui.comboBox_activation_knn.currentText(),
#             solver=ui.comboBox_solvar_knn.currentText(),
#             alpha=ui.doubleSpinBox_alpha_knn.value(),
#             max_iter=5000,
#             early_stopping=ui.checkBox_e_stop_knn.isChecked(),
#             validation_fraction=ui.doubleSpinBox_valid_knn.value(),
#             random_state=1
#         )
#         knn.fit(training_sample_norm, markup)
#
#         # Оценка точности на обучающей выборке
#         train_accuracy = knn.score(training_sample_norm, markup)
#
#         # Разделение данных на обучающую и тестовую выборки с использованием заданного значения test_size
#         training_saknne_train, training_sample_test, markup_train, markup_test = train_test_split(
#             training_sample_norm, markup, test_size=ui.doubleSpinBox_valid_knn.value(), random_state=1)
#
#         # Оценка точности на тестовой выборке
#         test_accuracy = knn.score(training_sample_test, markup_test)
#
#         # Вывод информации о параметрах KNN и точности модели
#         set_info(f'hidden_layer_sizes - ({",".join(map(str, layers))}), '
#                  f'activation - {ui.comboBox_activation_knn.currentText()}, '
#                  f'solver - {ui.comboBox_solvar_knn.currentText()}, '
#                  f'alpha - {ui.doubleSpinBox_alpha_knn.value()}, '
#                  f'{"early stopping, " if ui.checkBox_e_stop_knn.isChecked() else ""}'
#                  f'validation_fraction - {ui.doubleSpinBox_valid_knn.value()}, '
#                  f'точность на всей обучающей выборке: {train_accuracy}, '
#                  f'точность на тестовой выборке: {test_accuracy}', 'blue')
#     except ValueError:
#         set_info(f'Ошибка в расчетах KNN! Возможно значения одного из параметров отсутствуют в интервале обучающей '
#                  f'выборки.', 'red')
#         return
#
#
#     list_cat = list(knn.classes_)
#
#     # Подготовка тестовых данных для KNN
#     set_info(f'Процесс расчёта KNN. {ui.comboBox_lda_analysis.currentText()} по {get_object_name()} {get_research_name()}', 'blue')
#     working_sample = scaler.fit_transform(working_data_result.iloc[:, 3:])
#
#     try:
#         # Предсказание меток для тестовых данных
#         new_mark = knn.predict(working_sample)
#         probability = knn.predict_proba(working_sample)
#     except ValueError:
#         # Обработка возможных ошибок в расчетах KNN для тестовых данных
#         data = imputer.fit_transform(working_sample)
#         new_mark = knn.predict(data)
#         probability = knn.predict_proba(data)
#         for i in working_data.index:
#             p_nan = [working_data.columns[ic + 3] for ic, v in enumerate(working_data.iloc[i, 3:].tolist()) if
#                      np.isnan(v)]
#             if len(p_nan) > 0:
#                 set_info(
#                     f'Внимание для измерения "{i}" отсутствуют параметры "{", ".join(p_nan)}", поэтому расчёт для'
#                     f' этого измерения может быть не корректен', 'red')
#
#     # Добавление предсказанных меток и вероятностей в рабочие данные
#     working_data_result = pd.concat([working_data_result, pd.DataFrame(probability, columns=list_cat)], axis=1)
#     working_data_result['mark'] = new_mark
#     x = list(working_data_result['x_pulc'])
#     y = list(working_data_result['y_pulc'])
#     # if len(set(new_mark)) == 2:
#     #     z = list(working_data_result[list(set(new_mark))[0]])
#     # else:
#     #     z = string_to_unique_number(list(working_data_result['mark']), 'knn')
#     z = string_to_unique_number(list(working_data_result['mark']), 'knn')
#     draw_map(x, y, z, 'knn')
#     try:
#         file_name = f'{get_object_name()}_{get_research_name()}__модель_{get_knn_title()}.xlsx'
#         fn = QFileDialog.getSaveFileName(caption=f'Сохранить результат KNN "{get_object_name()}_{get_research_name()}" в таблицу', directory=file_name,
#                                          filter="Excel Files (*.xlsx)")
#         working_data_result.to_excel(fn[0])
#         set_info(f'Таблица сохранена в файл: {fn[0]}', 'green')
#     except ValueError:
#         pass
