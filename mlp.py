import pandas as pd

from draw import draw_radarogram, draw_formation, draw_fill, draw_fake, draw_fill_result, remove_poly_item
from func import *
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
                list_measure=old_markup.list_measure
            )
            session.add(new_markup)
    session.commit()
    update_list_mlp()
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
    update_list_marker_mlp(db)


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
    update_list_marker_mlp(True)


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


def update_list_marker_mlp(db=False):
    """Обновить список маркеров MLP"""
    ui.comboBox_mark_mlp.clear()
    for i in session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).order_by(MarkerMLP.title).all():
        item = f'{i.title} id{i.id}'
        ui.comboBox_mark_mlp.addItem(f'{i.title} id{i.id}')
        ui.comboBox_mark_mlp.setItemData(ui.comboBox_mark_mlp.findText(item), QBrush(QColor(i.color)),
                                         Qt.BackgroundRole)
    update_list_well_markup_mlp()
    update_list_param_mlp(db)


def add_well_markup_mlp():
    """Добавить новую обучающую скважину для MLP"""
    analysis_id = get_MLP_id()
    well_id = get_well_id()
    profile_id = get_profile_id()
    formation_id = get_formation_id()
    marker_id = get_marker_mlp_id()

    if analysis_id and well_id and profile_id and marker_id and formation_id:
        for param in get_list_param_mlp():
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
    for i in session.query(MarkupMLP).filter(MarkupMLP.analysis_id == get_MLP_id()).all():
        fake = len(json.loads(i.list_fake)) if i.list_fake else 0
        measure = len(json.loads(i.list_measure))
        item = f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | {i.well.name} | {measure - fake} из {measure} | id{i.id}'
        ui.listWidget_well_mlp.addItem(item)
        i_item = ui.listWidget_well_mlp.findItems(item, Qt.MatchContains)[0]
        i_item.setBackground(QBrush(QColor(i.marker.color)))
        # ui.listWidget_well_mlp.setItemData(ui.listWidget_well_mlp.findText(item), QBrush(QColor(i.marker.color)), Qt.BackgroundRole)


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


def draw_MLP():
    """ Построить диаграмму рассеяния для модели анализа MLP """
    data_train, list_param = build_table_train(True, 'mlp')
    list_param_mlp = data_train.columns.tolist()[2:]
    colors = {}
    for m in session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).all():
        colors[m.title] = m.color
    training_sample = data_train[list_param_mlp].values.tolist()
    markup = sum(data_train[['mark']].values.tolist(), [])

    try:
        # Нормализация данных
        scaler = StandardScaler()
        training_sample_norm = scaler.fit_transform(training_sample)
        # Разделение данных на обучающую и тестовую выборки
        # training_samlpe_train, training_sample_test, markup_train, markup_test = train_test_split(
        #     training_sample_norm, markup, test_size=0.20, random_state=1
        # )
        # Создание и тренировка MLP
        layers = tuple(map(int, ui.lineEdit_layer_mlp.text().split()))
        mlp = MLPClassifier(
            hidden_layer_sizes=layers,
            activation=ui.comboBox_activation_mlp.currentText(),
            solver=ui.comboBox_solvar_mlp.currentText(),
            alpha=ui.doubleSpinBox_alpha_mlp.value(),
            max_iter=5000,
            early_stopping=ui.checkBox_e_stop_mlp.isChecked(),
            validation_fraction=ui.doubleSpinBox_valid_mlp.value(),
            random_state=1
        )
        mlp.fit(training_sample_norm, markup)
        # Оценка точности на обучающей выборке
        train_accuracy = mlp.score(training_sample_norm, markup)
        training_samlpe_train, training_sample_test, markup_train, markup_test = train_test_split(
            training_sample_norm, markup, test_size=ui.doubleSpinBox_valid_mlp.value(), random_state=1)
        test_accuracy = mlp.score(training_sample_test, markup_test)
        set_info(f'hidden_layer_sizes - ({",".join(map(str, layers))}), '
                 f'activation - {ui.comboBox_activation_mlp.currentText()}, '
                 f'solver - {ui.comboBox_solvar_mlp.currentText()}, '
                 f'alpha - {ui.doubleSpinBox_alpha_mlp.value()}, '
                 f'{"early stopping, " if ui.checkBox_e_stop_mlp.isChecked() else ""}'
                 f'validation_fraction - {ui.doubleSpinBox_valid_mlp.value()}, '
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
    fig = plt.figure(figsize=(10, 10), dpi=80)
    ax = plt.subplot()
    sns.scatterplot(data=data_tsne, x=0, y=1, hue='mark', palette=colors)
    ax.grid()
    ax.xaxis.grid(True, "minor", linewidth=.25)
    ax.yaxis.grid(True, "minor", linewidth=.25)
    title_graph = f'Диаграмма рассеяния для канонических значений для обучающей выборки' \
                  f'\n{get_mlp_title().upper()}, параметров: {ui.listWidget_param_mlp.count()}, количество образцов: ' \
                  f'{str(len(data_tsne.index))}\n' \
                  f'hidden_layer_sizes - ({",".join(map(str, layers))}), '\
                  f'alpha - {ui.doubleSpinBox_alpha_mlp.value()}, '\
                  f'{"early stopping, " if ui.checkBox_e_stop_mlp.isChecked() else ""}'\
                  f'validation_fraction - {ui.doubleSpinBox_valid_mlp.value()}\n'\
                  f'точность на всей обучающей выборке: {round(train_accuracy, 7)}\n'\
                  f'точность на тестовой выборке: {round(test_accuracy, 7)}'
    plt.title(title_graph, fontsize=16)
    plt.tight_layout()
    fig.show()


def calc_verify_mlp():
    data_train, list_param = build_table_train(True, 'mlp')
    list_param_mlp = data_train.columns.tolist()[2:]
    # colors = [m.color for m in session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).all()]
    training_sample = data_train[list_param_mlp].values.tolist()
    markup = sum(data_train[['mark']].values.tolist(), [])
    clf = LinearDiscriminantAnalysis()
    try:
        clf.fit(training_sample, markup)
    except ValueError:
        set_info(f'Ошибка в расчетах MLP! Возможно значения одного из параметров отсутствуют в интервале обучающей выборки.', 'red')
        return
    n, k = 0, 0
    ui.progressBar.setMaximum(len(data_train.index))
    for i in data_train.index:
        new_mark = clf.predict([data_train.loc[i].loc[list_param_mlp].tolist()])[0]
        if data_train['mark'][i] != new_mark:
            prof_id = data_train['prof_well_index'][i].split('_')[0]
            well_id = data_train['prof_well_index'][i].split('_')[1]
            ix = int(data_train['prof_well_index'][i].split('_')[2])
            old_list_fake = session.query(MarkupMLP.list_fake).filter(
                MarkupMLP.analysis_id == get_MLP_id(),
                MarkupMLP.profile_id == prof_id,
                MarkupMLP.well_id == well_id
            ).first()[0]
            if old_list_fake:
                new_list_fake = json.loads(old_list_fake)
                new_list_fake.append(ix)
            else:
                new_list_fake = [ix]
            session.query(MarkupMLP).filter(
                MarkupMLP.analysis_id == get_MLP_id(),
                MarkupMLP.profile_id == prof_id,
                MarkupMLP.well_id == well_id
            ).update({'list_fake': json.dumps(new_list_fake)}, synchronize_session='fetch')
            session.commit()
            n += 1
        k += 1
        ui.progressBar.setValue(k)
    session.commit()
    set_info(f'Из обучающей выборки удалено {n} измерений.', 'blue')
    update_list_well_markup_mlp()
    db = True if n == 0 else False
    update_list_param_mlp(db)


def reset_verify_mlp():
    session.query(MarkupMLP).filter(MarkupMLP.analysis_id == get_MLP_id()).update({'list_fake': None},
                                                                                  synchronize_session='fetch')
    session.commit()
    set_info(f'Выбросы для анализа "{ui.comboBox_mlp_analysis.currentText()}" очищены.', 'green')
    update_list_well_markup_mlp()
    update_list_param_mlp()


def calc_MLP():
    colors = {}
    for m in session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).all():
        colors[m.title] = m.color
    # colors['test'] = '#999999'
    working_data, data_tsne, curr_form, title_graph = get_working_data_mlp()
    fig = plt.figure(figsize=(10, 10), dpi=80)
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
    training_samlpe = data_train[list_param_mlp].values.tolist()
    markup = sum(data_train[['mark']].values.tolist(), [])
    clf = LinearDiscriminantAnalysis()
    try:
        trans_coef = clf.fit(training_sample, markup).transform(training_sample)
    except ValueError:
        ui.label_info.setText(
            f'Ошибка в расчетах MLP! Возможно значения одного из параметров отсутствуют в интервале обучающей выборки.')
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
    try:
        file_name = f'{get_object_name()}_{get_research_name()}__модель_{get_mlp_title()}.xlsx'
        fn = QFileDialog.getSaveFileName(caption=f'Сохранить результат MLP "{get_object_name()}_{get_research_name()}" в таблицу', directory=file_name,
                                         filter="Excel Files (*.xlsx)")
        working_data_result.to_excel(fn[0])
        set_info(f'Таблица сохранена в файл: {fn[0]}', 'green')
    except ValueError:
        pass



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