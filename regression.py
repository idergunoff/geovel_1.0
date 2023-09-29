from draw import draw_radarogram, draw_formation, draw_fill, draw_fake
from func import *
from krige import draw_map
from qt.formation_ai_form import *
from qt.choose_formation_lda import *
from qt.choose_regmod import *


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
    for i in session.query(AnalysisReg).order_by(AnalysisReg.title).all():
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


def update_list_well_markup_reg():
    """Обновить список обучающих скважин"""
    ui.listWidget_well_regmod.clear()
    count_markup, count_measure = 0, 0
    for i in session.query(MarkupReg).filter(MarkupReg.analysis_id == get_regmod_id()).all():
        try:
            fake = len(json.loads(i.list_fake)) if i.list_fake else 0
            measure = len(json.loads(i.list_measure))
            if i.type_markup == 'intersection':
                inter_name = session.query(Intersection.name).filter(Intersection.id == i.well_id).first()[0]
                item = (f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | {inter_name.split("_")[0]} | '
                        f'{measure - fake} | {i.target_value} | id{i.id}')
            else:
                item = (f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | {i.well.name} | '
                        f'{measure - fake} | {i.target_value} | id{i.id}')
            ui.listWidget_well_regmod.addItem(item)
            count_markup += 1
            count_measure += measure - fake
        except AttributeError:
            set_info(f'Параметр для профиля {i.profile.title} удален из-за отсутствия одного из параметров', 'red')
            session.delete(i)
            session.commit()
    ui.label_count_markup_reg.setText(f'<i><u>{count_markup}</u></i> обучающих скважин; <i><u>{count_measure}</u></i> измерений')
    update_list_param_regmod(db=True)


def remove_well_markup_reg():
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
    # Функция выбора маркера регрессионной модели
    # Выбирает маркер, на основе сохраненных данных из базы данных, и затем обновляет все соответствующие виджеты
    # пользовательского интерфейса

    # Получение информации о маркере из БД по его ID
    markup = session.query(MarkupReg).filter(MarkupReg.id == get_markup_regmod_id()).first()
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


def update_well_markup_reg():
    markup = session.query(MarkupReg).filter(MarkupReg.id == get_markup_regmod_id()).first()
    if not markup:
        return
    x_prof = json.loads(session.query(Profile.x_pulc).filter(Profile.id == markup.profile_id).first()[0])
    target_value = ui.doubleSpinBox_target_val.value()
    well_dist = ui.spinBox_well_dist_reg.value()
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
        {'target_value': target_value, 'list_measure': json.dumps(list_measure)})
    session.commit()
    set_info(f'Изменена обучающая скважина для регрессионной модели - "{well.name}"', 'green')
    update_list_well_markup_reg()



def add_param_geovel_reg():
    session.query(AnalysisReg).filter_by(id=get_regmod_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    param = ui.comboBox_geovel_param_reg.currentText()
    for m in session.query(MarkupReg).filter(MarkupReg.analysis_id == get_regmod_id()).all():
        if not session.query(literal_column(f'Formation.{param}')).filter(Formation.id == m.formation_id).first()[0]:
            set_info(f'Параметр {param} отсутствует для профиля {m.profile.title}', 'red')
            return
    if session.query(ParameterReg).filter_by(
            analysis_id=get_regmod_id(),
            parameter= param
    ).count() == 0:
        add_param_regmod(param)
        update_list_param_regmod()
    else:
        set_info(f'Параметр {param} уже добавлен', 'red')


def add_all_param_geovel_reg():
    new_list_param = list_param_geovel.copy()
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
    session.query(AnalysisReg).filter_by(id=get_regmod_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    update_list_param_regmod()


def add_param_distr_reg():
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
    session.query(AnalysisReg).filter_by(id=get_regmod_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    update_list_param_regmod()
    set_info(f'В параметры добавлены {ui.spinBox_count_distr_reg.value()} интервалов распределения по '
             f'{ui.comboBox_atrib_distr_reg.currentText()}', 'green')


def add_param_sep_reg():
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
    session.query(AnalysisReg).filter_by(id=get_regmod_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    update_list_param_regmod()
    set_info(f'В параметры добавлены средние значения разделения на {ui.spinBox_count_distr_reg.value()} интервалов по '
             f'{ui.comboBox_atrib_distr_reg.currentText()}', 'green')


def add_all_param_distr_reg():
    list_distr = ['distr_Abase', 'distr_diff', 'distr_At', 'distr_Vt', 'distr_Pht', 'distr_Wt', 'sep_Abase', 'sep_diff', 'sep_At', 'sep_Vt', 'sep_Pht', 'sep_Wt']
    count = ui.spinBox_count_distr_reg.value()
    for param in session.query(ParameterReg).filter(ParameterReg.analysis_id == get_regmod_id()).all():
        if param.parameter.startswith('distr') or param.parameter.startswith('sep'):
            session.query(ParameterReg).filter_by(id=param.id).delete()
            session.commit()
    for distr_param in list_distr:
        new_param = f'{distr_param}_{count}'
        new_param_reg = ParameterReg(analysis_id=get_regmod_id(), parameter=new_param)
        session.add(new_param_reg)
    session.query(AnalysisReg).filter_by(id=get_regmod_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    update_list_param_regmod()
    set_info(f'Добавлены все параметры распределения по {count} интервалам', 'green')


def add_param_mfcc_reg():
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
    session.query(AnalysisReg).filter_by(id=get_regmod_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    update_list_param_regmod()
    set_info(f'В параметры добавлены {ui.spinBox_count_mfcc_reg.value()} кепстральных коэффициентов '
             f'{ui.comboBox_atrib_mfcc_reg.currentText()}', 'green')


def add_all_param_mfcc_reg():
    list_mfcc = ['mfcc_Abase', 'mfcc_diff', 'mfcc_At', 'mfcc_Vt', 'mfcc_Pht', 'mfcc_Wt']
    count = ui.spinBox_count_mfcc_reg.value()
    for param in session.query(ParameterReg).filter(ParameterReg.analysis_id == get_regmod_id()).all():
        if param.parameter.startswith('mfcc'):
            session.query(ParameterReg).filter_by(id=param.id).delete()
            session.commit()
    for mfcc_param in list_mfcc:
        new_param = f'{mfcc_param}_{count}'
        new_param_mlp = ParameterReg(analysis_id=get_regmod_id(), parameter=new_param)
        session.add(new_param_mlp)
    session.query(AnalysisReg).filter_by(id=get_regmod_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    update_list_param_regmod()
    set_info(f'Добавлены коэффициенты mfcc по всем параметрам по {count} интервалам', 'green')


def remove_param_geovel_reg():
    try:
        param = ui.listWidget_param_reg.currentItem().text().split(' ')[0]
    except AttributeError:
        set_info('Выберите параметр', 'red')
        return
    if param:
        if param.startswith('distr') or param.startswith('sep') or param.startswith('mfcc'):
            for p in session.query(ParameterReg).filter(ParameterReg.analysis_id == get_regmod_id()).all():
                if p.parameter.startswith('_'.join(param.split('_')[:-1])):
                    session.query(ParameterReg).filter_by(id=p.id).delete()
                    session.commit()
        else:
            session.query(ParameterReg).filter_by(analysis_id=get_regmod_id(), parameter=param ).delete()
        session.commit()
        ui.listWidget_param_reg.takeItem(ui.listWidget_param_reg.currentRow())
        session.query(AnalysisReg).filter_by(id=get_regmod_id()).update({'up_data': False}, synchronize_session='fetch')
        session.commit()
        set_color_button_updata_regmod()


def remove_all_param_geovel_reg():
    session.query(ParameterReg).filter_by(analysis_id=get_regmod_id()).delete()
    session.query(AnalysisReg).filter_by(id=get_regmod_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()
    update_list_param_regmod()


def update_list_param_regmod(db=False):
    data_train, list_param = build_table_train(db, 'regmod')
    ui.listWidget_param_reg.clear()
    list_param_reg = data_train.columns.tolist()[2:]
    for param in list_param_reg:
        ui.listWidget_param_reg.addItem(f'{param}')
    ui.label_count_param_regmod.setText(f'<i><u>{ui.listWidget_param_reg.count()}</u></i> параметров')
    update_list_trained_models_regmod()
    set_color_button_updata_regmod()


def set_color_button_updata_regmod():
    reg = session.query(AnalysisReg).filter(AnalysisReg.id == get_regmod_id()).first()
    btn_color = 'background-color: rgb(191, 255, 191);' if reg.up_data else 'background-color: rgb(255, 185, 185);'
    ui.pushButton_updata_regmod.setStyleSheet(btn_color)


def train_regression_model():
    """ Расчет модели """
    data_train, list_param = build_table_train(True, 'regmod')
    training_sample = data_train[list_param].values.tolist()
    target = sum(data_train[['target_value']].values.tolist(), [])

    Form_Regmod = QtWidgets.QDialog()
    ui_frm = Ui_Form_formation_ai()
    ui_frm.setupUi(Form_Regmod)
    Form_Regmod.show()
    Form_Regmod.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    def calc_regression_model():
        start_time = datetime.datetime.now()
        model = ui_frm.comboBox_model_ai.currentText()

        x_train, x_test, y_train, y_test = train_test_split(
            training_sample, target, test_size=0.2, random_state=42
        )

        if model == 'LinearRegression':
            model_regression = LinearRegression(fit_intercept=ui_frm.checkBox_fit_intercept.isChecked())
            model_name = 'LR'

        if model == 'DecisionTreeRegressor':
            spl = 'random' if ui_frm.checkBox_splitter_rnd.isChecked() else 'best'
            model_regression = DecisionTreeRegressor(splitter=spl)
            model_name = 'DTR'

        if model == 'KNeighborsRegressor':
            model_regression = KNeighborsRegressor(
                n_neighbors=ui_frm.spinBox_neighbors.value(),
                weights='distance' if ui_frm.checkBox_knn_weights.isChecked() else 'uniform',
                algorithm=ui_frm.comboBox_knn_algorithm.currentText()
            )
            model_name = 'KNNR'

        if model == 'SVR':
            model_regression = SVR(kernel=ui_frm.comboBox_svr_kernel.currentText(), C=ui_frm.doubleSpinBox_svr_c.value())
            model_name = 'SVR'

        if model == 'MLPRegressor':
            layers = tuple(map(int, ui_frm.lineEdit_layer_mlp.text().split()))
            model_regression = MLPRegressor(
                hidden_layer_sizes=layers,
                activation=ui_frm.comboBox_activation_mlp.currentText(),
                solver=ui_frm.comboBox_solvar_mlp.currentText(),
                alpha=ui_frm.doubleSpinBox_alpha_mlp.value(),
                max_iter=5000,
                early_stopping=ui_frm.checkBox_e_stop_mlp.isChecked(),
                validation_fraction=ui_frm.doubleSpinBox_valid_mlp.value()
            )
            model_name = 'MLPR'

        if model == 'GradientBoostingRegressor':
            model_regression = GradientBoostingRegressor(
                n_estimators=ui_frm.spinBox_n_estimators.value(),
                learning_rate=ui_frm.doubleSpinBox_learning_rate.value(),
            )
            model_name = 'GBR'

        if model == 'ElasticNet':
            model_regression = ElasticNet(
                alpha=ui_frm.doubleSpinBox_alpha.value(),
                l1_ratio=ui_frm.doubleSpinBox_l1_ratio.value()
            )
            model_name = 'EN'

        if model == 'Lasso':
            model_regression = Lasso(alpha=ui_frm.doubleSpinBox_alpha.value())
            model_name = 'Lss'

        model_regression.fit(x_train, y_train)

        y_pred = model_regression.predict(x_test)

        accuracy = model_regression.score(x_test, y_test)
        mse = mean_squared_error(y_test, y_pred)
        train_time = datetime.datetime.now() - start_time
        set_info(f'Модель {model}:\n точность: {accuracy} '
                 f' Mean Squared Error\n top: {mse}, \n время обучения: {train_time}', 'blue')

        result = QtWidgets.QMessageBox.question(
            MainWindow,
            'Сохранение модели',
            f'Сохранить модель {model}?',
            QtWidgets.QMessageBox.Yes,
            QtWidgets.QMessageBox.No)
        if result == QtWidgets.QMessageBox.Yes:
            # Сохранение модели в файл с помощью pickle
            path_model = f'models/{model_name}_{round(accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y")}.pkl'
            with open(path_model, 'wb') as f:
                pickle.dump(model_regression, f)

            new_trained_model = TrainedModelReg(
                analysis_id=get_regmod_id(),
                title=f'{model_name}_{round(accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y")}',
                path_model=path_model,
                list_params=json.dumps(list_param)
            )
            session.add(new_trained_model)
            session.commit()
            update_list_trained_models_regmod()
        else:
            pass

    ui_frm.pushButton_calc_model.clicked.connect(calc_regression_model)
    Form_Regmod.exec_()


def update_list_trained_models_regmod():
    """  Обновление списка моделей """
    models = session.query(TrainedModelReg).filter(TrainedModelReg.analysis_id == get_regmod_id()).all()
    ui.listWidget_trained_model_reg.clear()
    for model in models:
        item_text = model.title
        item = QListWidgetItem(item_text)
        item.setData(Qt.UserRole, model.id)
        ui.listWidget_trained_model_reg.addItem(item)
    ui.listWidget_trained_model_reg.setCurrentRow(0)


def remove_trained_model_regmod():
    """ Удаление модели """
    model = session.query(TrainedModelReg).filter_by(id=ui.listWidget_trained_model_reg.currentItem().data(Qt.UserRole)).first()
    os.remove(model.path_model)
    session.delete(model)
    session.commit()
    update_list_trained_models_regmod()
    set_info(f'Модель {model.title} удалена', 'blue')


def calc_profile_model_regmod():
    model = session.query(TrainedModelReg).filter_by(id=ui.listWidget_trained_model_reg.currentItem().data(Qt.UserRole)).first()
    with open(model.path_model, 'rb') as f:
        reg_model = pickle.load(f)

    working_data, curr_form = build_table_test('regmod')

    working_sample = working_data[json.loads(model.list_params)].values.tolist()
    try:
        y_pred = reg_model.predict(working_sample)
    except ValueError:
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


def calc_object_model_regmod():
    working_data_result = pd.DataFrame()
    list_formation = []
    for n, prof in enumerate(session.query(Profile).filter(Profile.research_id == get_research_id()).all()):
        count_measure = len(json.loads(session.query(Profile.signal).filter(Profile.id == prof.id).first()[0]))
        ui.comboBox_profile.setCurrentText(f'{prof.title} ({count_measure} измерений) id{prof.id}')
        set_info(f'Профиль {prof.title} ({count_measure} измерений)', 'blue')
        update_formation_combobox()
        if len(prof.formations) == 1:
            list_formation.append(f'{prof.formations[0].title} id{prof.formations[0].id}')
            # ui.comboBox_plast.setCurrentText(f'{prof.formations[0].title} id{prof.formations[0].id}')
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
                list_formation.append(ui_cf.listWidget_form_lda.currentItem().text())
                # ui.comboBox_plast.setCurrentText(ui_cf.listWidget_form_lda.currentItem().text())
                Choose_Formation.close()

            ui_cf.pushButton_ok_form_lda.clicked.connect(form_mlp_ok)
            Choose_Formation.exec_()
    for n, prof in enumerate(session.query(Profile).filter(Profile.research_id == get_research_id()).all()):
        count_measure = len(json.loads(session.query(Profile.signal).filter(Profile.id == prof.id).first()[0]))
        ui.comboBox_profile.setCurrentText(f'{prof.title} ({count_measure} измерений) id{prof.id}')
        update_formation_combobox()
        ui.comboBox_plast.setCurrentText(list_formation[n])
        working_data, curr_form = build_table_test('regmod')
        working_data_result = pd.concat([working_data_result, working_data], axis=0, ignore_index=True)
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

        working_sample = working_data_result_copy[json.loads(model.list_params)].values.tolist()

        try:
            y_pred = reg_model.predict(working_sample)
        except ValueError:
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

    def calc_correlation():
        model = session.query(TrainedModelReg).filter_by(id=ui.listWidget_trained_model_reg.currentItem().data(Qt.UserRole)).first()
        with open(model.path_model, 'rb') as f:
            reg_model = pickle.load(f)

        working_sample = working_data_result_copy[json.loads(model.list_params)].values.tolist()

        try:
            y_pred = reg_model.predict(working_sample)
        except ValueError:
            data = imputer.fit_transform(working_sample)
            y_pred = reg_model.predict(data)

            for i in working_data_result_copy.index:
                p_nan = [working_data_result_copy.columns[ic + 3] for ic, v in enumerate(working_data_result_copy.iloc[i, 3:].tolist()) if
                         np.isnan(v)]
                if len(p_nan) > 0:
                    set_info(f'Внимание для измерения "{i}" отсутствуют параметры "{", ".join(p_nan)}", поэтому расчёт для'
                             f' этого измерения может быть не корректен', 'red')


        working_data_result_copy['value'] = y_pred
        data_corr = working_data_result_copy.iloc[:, 3:]
        list_param = list(data_corr.columns)
        fig = plt.figure(figsize=(14, 12), dpi=70)
        ax = plt.subplot()
        sns.heatmap(data_corr.corr(), xticklabels=list_param, yticklabels=list_param, cmap='seismic', annot=True, linewidths=0.25, center=0)
        plt.title(f'Корреляция параметров по {len(data_corr.index)} измерениям', fontsize=22)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        fig.show()


    ui_rm.pushButton_calc_model.clicked.connect(calc_regmod)
    ui_rm.pushButton_corr.clicked.connect(calc_correlation)
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


def calc_corr_regmod():
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
