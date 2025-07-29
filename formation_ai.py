import datetime
import time

from draw import draw_radarogram, draw_formation
from layer import add_param_in_new_formation
from func import *
from qt.formation_ai_form import *

list_comb_data = ['t/b/t_i/b_i', 'top/bottom', 'top_i/bottom_i', 'top', 'bottom', 'top_i', 'bottom_i']
for i in list_comb_data:
    ui.comboBox_comb_data.addItem(i)


def update_combobox_model_ai():
    """ Обновить список моделей """
    ui.comboBox_model_ai.clear()
    for i in session.query(ModelFormationAI).order_by(ModelFormationAI.title).all():
        ui.comboBox_model_ai.addItem(f'{i.title}')
        ui.comboBox_model_ai.setItemData(ui.comboBox_model_ai.count() - 1, {'id': i.id})
    update_list_formation_ai()

def get_model_id():
    if ui.comboBox_model_ai.count() > 0:
        return ui.comboBox_model_ai.currentData()['id']


def update_list_formation_ai():
    """ Обновить список пластов """
    ui.listWidget_formation_ai.clear()
    for i in session.query(FormationAI).filter_by(model_id=get_model_id()).all():
        try:
            item_text = (f'{i.formation.title} ({i.formation.profile.title}, {i.formation.profile.research.object.title}, '
                         f'{i.formation.profile.research.date_research.year})')
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, i.id)
            ui.listWidget_formation_ai.addItem(item)
        except AttributeError:
            session.query(FormationAI).filter_by(id=i.id).delete()
            session.commit()


def add_model_ai():
    """ Добавить модель """
    new_model = ModelFormationAI(title=ui.lineEdit_string.text())
    session.add(new_model)
    session.commit()
    update_combobox_model_ai()
    set_info(f'Модель "{new_model.title}" добавлена', 'green')


def remove_model_ai():
    """ Удалить модель """
    model_title = ui.comboBox_model_ai.currentText()
    result = QtWidgets.QMessageBox.question(
        MainWindow,
        'Удаление модели',
        f'Вы уверены, что хотите удалить модель "{model_title}" со всеми пластами?',
        QtWidgets.QMessageBox.Yes,
        QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        session.query(FormationAI).filter_by(model_id=get_model_id()).delete()
        session.query(ModelFormationAI).filter_by(id=get_model_id()).delete()
        session.commit()
        set_info(f'Модель "{model_title}" удалена', 'green')
        update_combobox_model_ai()
    else:
        pass


def add_formation_ai():
    """ Добавить пласт для обучения """
    if get_formation_id():
        new_formation = FormationAI(formation_id=get_formation_id(), model_id=get_model_id())
        session.add(new_formation)
        session.commit()
        update_list_formation_ai()


def clear_formation_ai():
    """ Очистка списка пластов """
    session.query(FormationAI).filter_by(model_id=get_model_id()).delete()
    session.commit()
    update_list_formation_ai()


def remove_formation_ai():
    """ Удаление пласта из списка """
    item = ui.listWidget_formation_ai.currentItem()
    if item:
        session.query(FormationAI).filter_by(id=item.data(Qt.UserRole)).delete()
        session.commit()
        update_list_formation_ai()


# def calc_model_ai():
#     """ Расчет модели """
#     list_input, list_target, comb_data = [], [], ui.comboBox_comb_data.currentText()
#     for i in range(ui.listWidget_formation_ai.count()):
#         item = ui.listWidget_formation_ai.item(i)
#         formation_ai = session.query(FormationAI).filter_by(id=item.data(Qt.UserRole)).first()
#         formation = session.query(Formation).filter_by(id=formation_ai.formation_id).first()
#         signal = json.loads(formation.profile.signal)
#         for s in signal:
#             list_input.append(s)
#         line_top, line_bottom = json.loads(formation.layer_up.layer_line), json.loads(formation.layer_down.layer_line)
#         for i in range(len(line_top)):
#             list_current_target = []
#             if comb_data == 'top':
#                 list_target.append(signal[i][line_top[i]])
#             elif comb_data == 'top/bottom' or comb_data == 't/b/t_i/b_i':
#                 list_current_target.append(signal[i][line_top[i]])
#             if comb_data == 'bottom':
#                 list_target.append(signal[i][line_bottom[i]])
#             elif  comb_data == 'top/bottom' or comb_data == 't/b/t_i/b_i':
#                 list_current_target.append(signal[i][line_bottom[i]])
#             if comb_data == 'top_i':
#                 list_target.append(line_top[i])
#             elif  comb_data == 'top_i/bottom_i' or comb_data == 't/b/t_i/b_i':
#                 list_current_target.append(line_top[i])
#             if comb_data == 'bottom_i':
#                 list_target.append(line_bottom[i])
#             elif  comb_data == 'top_i/bottom_i' or comb_data == 't/b/t_i/b_i':
#                 list_current_target.append(line_bottom[i])
#             if comb_data == 'top/bottom' or comb_data == 'top_i/bottom_i' or comb_data == 't/b/t_i/b_i':
#                 list_target.append(list_current_target)
#
#     input_data = np.array(list_input)
#     target_data = np.array(list_target)
#
#     Form_AI = QtWidgets.QDialog()
#     ui_fai = Ui_Form_formation_ai()
#     ui_fai.setupUi(Form_AI)
#     Form_AI.show()
#     Form_AI.setAttribute(QtCore.Qt.WA_DeleteOnClose)
#     def calc_model_formation_ai():
#         model = ui_fai.comboBox_model_ai.currentText()
#
#         X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size=0.2)
#
#         if model == 'LinearRegression':
#             model_ai = LinearRegression(fit_intercept=ui_fai.checkBox_fit_intercept.isChecked())
#             model_ai.fit(X_train, y_train)
#             y_pred = model_ai.predict(X_test)
#             set_info(f'LinearRegression: {model_ai.score(X_test, y_test)}', 'blue')
#             mse = mean_squared_error(y_test, y_pred)
#             set_info(f"Mean Squared Error, LinearRegression: {mse}", 'blue')
#
#         if model == 'DecisionTreeRegressor':
#             spl = 'random' if ui_fai.checkBox_splitter_rnd.isChecked() else 'best'
#             model_ai = DecisionTreeRegressor(splitter=spl)
#             model_ai.fit(X_train, y_train)
#             y_pred = model_ai.predict(X_test)
#             set_info(f'DecisionTreeRegressor: {model_ai.score(X_test, y_test)}', 'blue')
#             mse = mean_squared_error(y_test, y_pred)
#             set_info(f"Mean Squared Error, DecisionTreeRegressor: {mse}", 'blue')
#
#         if model == 'KNeighborsRegressor':
#             model_ai = KNeighborsRegressor(
#                 n_neighbors=ui_fai.spinBox_neighbors.value(),
#                 weights='distance' if ui_fai.checkBox_knn_weights.isChecked() else 'uniform',
#                 algorithm=ui_fai.comboBox_knn_algorithm.currentText()
#             )
#             model_ai.fit(X_train, y_train)
#             y_pred = model_ai.predict(X_test)
#             set_info(f'KNeighborsRegressor: {model_ai.score(X_test, y_test)}', 'blue')
#             mse = mean_squared_error(y_test, y_pred)
#             set_info(f"Mean Squared Error, KNeighborsRegressor: {mse}", 'blue')
#
#         if model == 'SVR':
#             try:
#                 model_ai = SVR(kernel=ui_fai.comboBox_svr_kernel.currentText(), C=ui_fai.doubleSpinBox_svr_c.value())
#                 model_ai.fit(X_train, y_train)
#                 y_pred = (model_ai.predict(X_test))
#                 set_info(f'SVR: {model_ai.score(X_test, y_test)}', 'blue')
#                 mse = mean_squared_error(y_test, y_pred)
#                 set_info(f"Mean Squared Error, SVR: {mse}", 'blue')
#             except ValueError:
#                 set_info('ValueError', 'red')
#
#         if model == 'MLPRegressor':
#             try:
#                 layers = tuple(map(int, ui_fai.lineEdit_layer_mlp.text().split()))
#                 model_ai = MLPRegressor(
#                     hidden_layer_sizes=layers,
#                     activation=ui_fai.comboBox_activation_mlp.currentText(),
#                     solver=ui_fai.comboBox_solvar_mlp.currentText(),
#                     alpha=ui_fai.doubleSpinBox_alpha_mlp.value(),
#                     max_iter=5000,
#                     early_stopping=ui_fai.checkBox_e_stop_mlp.isChecked(),
#                     validation_fraction=ui_fai.doubleSpinBox_valid_mlp.value()
#                 )
#                 model_ai.fit(X_train, y_train)
#                 y_pred = model_ai.predict(X_test)
#                 set_info(f'MLPRegressor: {model_ai.score(X_test, y_test)}', 'blue')
#                 mse = mean_squared_error(y_test, y_pred)
#                 set_info(f"Mean Squared Error, MLPRegressor: {mse}", 'blue')
#             except ValueError:
#                 set_info('ValueError', 'red')
#
#         if model == 'GradientBoostingRegressor':
#             try:
#                 model_ai = GradientBoostingRegressor(
#                     n_estimators=ui_fai.spinBox_n_estimators.value(),
#                     learning_rate=ui_fai.doubleSpinBox_learning_rate.value(),
#                 )
#                 model_ai.fit(X_train, y_train)
#                 y_pred = model_ai.predict(X_test)
#                 set_info(f'GradientBoostingRegressor: {model_ai.score(X_test, y_test)}', 'blue')
#                 mse = mean_squared_error(y_test, y_pred)
#                 set_info(f"Mean Squared Error, GradientBoostingRegressor: {mse}", 'blue')
#             except ValueError:
#                 set_info('ValueError', 'red')
#
#         if model == 'ElasticNet':
#             try:
#                 model_ai = ElasticNet(
#                     alpha=ui_fai.doubleSpinBox_alpha.value(),
#                     l1_ratio=ui_fai.doubleSpinBox_l1_ratio.value()
#                 )
#                 model_ai.fit(X_train, y_train)
#                 y_pred = model_ai.predict(X_test)
#                 set_info(f'ElasticNet: {model_ai.score(X_test, y_test)}', 'blue')
#                 mse = mean_squared_error(y_test, y_pred)
#                 set_info(f"Mean Squared Error, ElasticNet: {mse}", 'blue')
#             except ValueError:
#                 set_info('ValueError', 'red')
#
#         if model == 'Lasso':
#             try:
#                 model_ai = Lasso(alpha=ui_fai.doubleSpinBox_alpha.value())
#                 model_ai.fit(X_train, y_train)
#                 y_pred = model_ai.predict(X_test)
#                 set_info(f'Lasso: {model_ai.score(X_test, y_test)}', 'blue')
#                 mse = mean_squared_error(y_test, y_pred)
#                 set_info(f"Mean Squared Error, Lasso: {mse}", 'blue')
#             except ValueError:
#                 set_info('ValueError', 'red')
#         if comb_data == 'top_i/bottom_i' or comb_data == 't/b/t_i/b_i' or comb_data == 'top_i' or comb_data == 'bottom_i':
#             result = QtWidgets.QMessageBox.question(
#                 MainWindow,
#                 'Расчет модели для профиля',
#                 f'Расчитать модель {ui_fai.comboBox_model_ai.currentText()} для текущего профиля?',
#                 QtWidgets.QMessageBox.Yes,
#                 QtWidgets.QMessageBox.No)
#             if result == QtWidgets.QMessageBox.Yes:
#                 prof = session.query(Profile).filter_by(id=get_profile_id()).first()
#                 test_signal = np.array(json.loads(prof.signal))
#                 y_pred_signal = model_ai.predict(test_signal)
#                 if comb_data == 'top_i/bottom_i':
#                     top_model, bottom_model = [pred[0] for pred in y_pred_signal], [pred[1] for pred in y_pred_signal]
#                     new_top = Layers(profile_id=get_profile_id(), layer_line=json.dumps(top_model), layer_title=f'top_{model}')
#                     new_bottom = Layers(profile_id=get_profile_id(), layer_line=json.dumps(bottom_model), layer_title=f'bottom_{model}')
#                     session.add(new_top)
#                     session.add(new_bottom)
#                     session.commit()
#                 if comb_data == 't/b/t_i/b_i':
#                     top_model, bottom_model = [int(pred[2]) for pred in y_pred_signal], [int(pred[3]) for pred in y_pred_signal]
#                     top_model_a, bottom_model_a = [pred[0] for pred in y_pred_signal], [pred[1] for pred in y_pred_signal]
#                     new_top = Layers(profile_id=get_profile_id(), layer_title=f'top_{model}')
#                     new_bottom = Layers(profile_id=get_profile_id(), layer_title=f'bottom_{model}')
#                     session.add(new_top)
#                     session.add(new_bottom)
#                     session.commit()
#                     for n, i in enumerate(test_signal):
#                         if top_model_a[n] - 3 < test_signal[n][top_model[n]] < top_model_a[n] + 3:
#                             new_point = PointsOfLayer(layer_id=new_top.id, point_x=n, point_y=top_model[n])
#                             session.add(new_point)
#                         if bottom_model_a[n] - 3 < test_signal[n][bottom_model[n]] < bottom_model_a[n] + 3:
#                             new_point = PointsOfLayer(layer_id=new_bottom.id, point_x=n, point_y=bottom_model[n])
#                             session.add(new_point)
#                     session.commit()
#                 if comb_data == 'top_i':
#                     result = list(map(int, savgol_filter(y_pred_signal.tolist(), 31, 3)))
#                     new_top = Layers(
#                         profile_id=get_profile_id(),
#                         layer_line=json.dumps(result),
#                         layer_title=f'top_{model}'
#                     )
#                     session.add(new_top)
#                     session.commit()
#                 if comb_data == 'bottom_i':
#                     result = list(map(int, savgol_filter(y_pred_signal.tolist(), 31, 3)))
#                     new_bottom = Layers(
#                         profile_id=get_profile_id(),
#                         layer_line=json.dumps(result),
#                         layer_title=f'bottom_{model}'
#                     )
#                     session.add(new_bottom)
#                     session.commit()
#             if result == QtWidgets.QMessageBox.No:
#                 pass
#
#
#
#     ui_fai.pushButton_calc_model.clicked.connect(calc_model_formation_ai)
#     Form_AI.exec_()

def choose_formation_ai():
    try:
        formation_ai = session.query(FormationAI).filter_by(id=ui.listWidget_formation_ai.currentItem().data(Qt.UserRole)).first()
        ui.comboBox_object.setCurrentText(f'{formation_ai.formation.profile.research.object.title} id{formation_ai.formation.profile.research.object_id}')
        update_research_combobox()
        ui.comboBox_research.setCurrentText(
            f'{formation_ai.formation.profile.research.date_research.strftime("%m.%Y")} id{formation_ai.formation.profile.research_id}')
        update_profile_combobox()
        count_measure = len(json.loads(session.query(Profile.signal).filter(Profile.id == formation_ai.formation.profile_id).first()[0]))
        ui.comboBox_profile.setCurrentText(f'{formation_ai.formation.profile.title} ({count_measure} измерений) id{formation_ai.formation.profile_id}')
        draw_radarogram()
        ui.comboBox_plast.setCurrentText(f'{formation_ai.formation.title} id{formation_ai.formation_id}')
        draw_formation()
    except AttributeError:
        pass


def calc_model_ai():
    """ Расчет модели """
    list_input, list_target_top, list_target_bottom = [], [], []
    ui.progressBar.setMaximum(ui.listWidget_formation_ai.count())
    for i in range(ui.listWidget_formation_ai.count()):
        item = ui.listWidget_formation_ai.item(i)
        formation_ai = session.query(FormationAI).filter_by(id=item.data(Qt.UserRole)).first()
        formation = session.query(Formation).filter_by(id=formation_ai.formation_id).first()
        signal = json.loads(formation.profile.signal)
        for s in signal:
            diff_s = calc_atrib_measure(s, 'diff')
            At = calc_atrib_measure(s, 'At')
            # Vt = calc_atrib_measure(s, 'Vt')
            Pht = calc_atrib_measure(s, 'Pht')
            Wt = calc_atrib_measure(s, 'Wt')
            list_input.append(s + diff_s + At + Pht + Wt)
        line_top, line_bottom = json.loads(formation.layer_up.layer_line), json.loads(formation.layer_down.layer_line)
        list_target_top += line_top
        list_target_bottom += line_bottom
        ui.progressBar.setValue(i + 1)
    input_data, target_data_top, target_data_bottom = np.array(list_input), np.array(list_target_top), np.array(list_target_bottom)
    show_regressor_train_form(input_data, target_data_top, target_data_bottom)


def show_regressor_train_form(input_data, target_data_top, target_data_bottom):
    Regressor = QtWidgets.QDialog()
    ui_r = Ui_RegressorForm()
    ui_r.setupUi(Regressor)
    Regressor.show()
    Regressor.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    ui_r.spinBox_pca.setMaximum(len(input_data[0]))
    ui_r.spinBox_pca.setValue(len(input_data[0]) // 2)

    def choice_model_regressor(model):
        """ Выбор модели регрессии """
        if model == 'MLPR':
            model_reg_top = MLPRegressor(
                hidden_layer_sizes=tuple(map(int, ui_r.lineEdit_layer_mlp.text().split())),
                activation=ui_r.comboBox_activation_mlp.currentText(),
                solver=ui_r.comboBox_solvar_mlp.currentText(),
                alpha=ui_r.doubleSpinBox_alpha_mlp.value(),
                max_iter=5000,
                early_stopping=ui_r.checkBox_e_stop_mlp.isChecked(),
                validation_fraction=ui_r.doubleSpinBox_valid_mlp.value(),
                random_state=0
            )
            model_reg_bottom = MLPRegressor(
                hidden_layer_sizes=tuple(map(int, ui_r.lineEdit_layer_mlp.text().split())),
                activation=ui_r.comboBox_activation_mlp.currentText(),
                solver=ui_r.comboBox_solvar_mlp.currentText(),
                alpha=ui_r.doubleSpinBox_alpha_mlp.value(),
                max_iter=5000,
                early_stopping=ui_r.checkBox_e_stop_mlp.isChecked(),
                validation_fraction=ui_r.doubleSpinBox_valid_mlp.value(),
                random_state=0
            )
            text_model = (f'**MLPR**: \nhidden_layer_sizes: '
                          f'({",".join(map(str, tuple(map(int, ui_r.lineEdit_layer_mlp.text().split()))))}), '
                          f'\nactivation: {ui_r.comboBox_activation_mlp.currentText()}, '
                          f'\nsolver: {ui_r.comboBox_solvar_mlp.currentText()}, '
                          f'\nalpha: {round(ui_r.doubleSpinBox_alpha_mlp.value(), 2)}, '
                          f'\n{"early stopping, " if ui_r.checkBox_e_stop_mlp.isChecked() else ""}'
                          f'\nvalidation_fraction: {round(ui_r.doubleSpinBox_valid_mlp.value(), 2)}, ')

        elif model == 'KNNR':
            n_knn = ui_r.spinBox_neighbors.value()
            weights_knn = 'distance' if ui_r.checkBox_knn_weights.isChecked() else 'uniform'
            model_reg_top = KNeighborsRegressor(n_neighbors=n_knn, weights=weights_knn, algorithm='auto')
            model_reg_bottom = KNeighborsRegressor(n_neighbors=n_knn, weights=weights_knn, algorithm='auto')
            text_model = f'**KNNR**: \nn_neighbors: {n_knn}, \nweights: {weights_knn}, '

        elif model == 'GBR':
            est = ui_r.spinBox_n_estimators_gbr.value()
            l_rate = ui_r.doubleSpinBox_learning_rate_gbr.value()
            model_reg_top = GradientBoostingRegressor(n_estimators=est, learning_rate=l_rate, random_state=0)
            model_reg_bottom = GradientBoostingRegressor(n_estimators=est, learning_rate=l_rate, random_state=0)
            text_model = f'**GBR**: \nn estimators: {round(est, 2)}, \nlearning rate: {round(l_rate, 2)}, '

        elif model == 'XGB':
            model_reg_top = XGBRegressor(n_estimators=ui_r.spinBox_n_estimators_xgb.value(),
                                     learning_rate=ui_r.doubleSpinBox_learning_rate_xgb.value(),
                                     max_depth=ui_r.spinBox_depth_xgb.value(),
                                     alpha=ui_r.doubleSpinBox_alpha_xgb.value(), booster='gbtree', random_state=0)
            model_reg_bottom = XGBRegressor(n_estimators=ui_r.spinBox_n_estimators_xgb.value(),
                                         learning_rate=ui_r.doubleSpinBox_learning_rate_xgb.value(),
                                         max_depth=ui_r.spinBox_depth_xgb.value(),
                                         alpha=ui_r.doubleSpinBox_alpha_xgb.value(), booster='gbtree', random_state=0)
            text_model = f'**XGB**: \nn estimators: {ui_r.spinBox_n_estimators_xgb.value()}, ' \
                         f'\nlearning_rate: {ui_r.doubleSpinBox_learning_rate_xgb.value()}, ' \
                         f'\nmax_depth: {ui_r.spinBox_depth_xgb.value()} \nalpha: {ui_r.doubleSpinBox_alpha_xgb.value()}'

        elif model == 'LGBM':
            model_reg_top = lgb.LGBMRegressor(
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

            model_reg_bottom = lgb.LGBMRegressor(
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

        elif model == 'LR':
            model_reg_top = LinearRegression(fit_intercept=ui_r.checkBox_fit_intercept.isChecked())
            model_reg_bottom = LinearRegression(fit_intercept=ui_r.checkBox_fit_intercept.isChecked())
            text_model = f'**LR**: \nfit_intercept: {"on" if ui_r.checkBox_fit_intercept.isChecked() else "off"}, '

        elif model == 'DTR':
            spl = 'random' if ui_r.checkBox_splitter_rnd.isChecked() else 'best'
            model_reg_top = DecisionTreeRegressor(splitter=spl, random_state=0)
            model_reg_bottom = DecisionTreeRegressor(splitter=spl, random_state=0)
            text_model = f'**DTR**: \nsplitter: {spl}, '

        elif model == 'RFR':
            if ui_r.checkBox_rfr_ada.isChecked():
                model_reg_top = AdaBoostRegressor(n_estimators=ui_r.spinBox_rfr_n.value(), random_state=0)
                model_reg_bottom = AdaBoostRegressor(n_estimators=ui_r.spinBox_rfr_n.value(), random_state=0)
                text_model = f'**ABR**: \nn estimators: {ui_r.spinBox_rfr_n.value()}, '
            elif ui_r.checkBox_rfr_extra.isChecked():
                model_reg_top = ExtraTreesRegressor(n_estimators=ui_r.spinBox_rfr_n.value(), bootstrap=True, oob_score=True, random_state=0, n_jobs=-1)
                model_reg_bottom = ExtraTreesRegressor(n_estimators=ui_r.spinBox_rfr_n.value(), bootstrap=True, oob_score=True, random_state=0, n_jobs=-1)
                text_model = f'**ETR**: \nn estimators: {ui_r.spinBox_rfr_n.value()}, '
            else:
                model_reg_top = RandomForestRegressor(n_estimators=ui_r.spinBox_rfr_n.value(), oob_score=True, random_state=0, n_jobs=-1)
                model_reg_bottom = RandomForestRegressor(n_estimators=ui_r.spinBox_rfr_n.value(), oob_score=True, random_state=0, n_jobs=-1)
                text_model = f'**RFR**: \nn estimators: {ui_r.spinBox_rfr_n.value()}, '

        elif model == 'GPR':
            gpc_kernel_width = ui_r.doubleSpinBox_gpc_wigth.value()
            gpc_kernel_scale = ui_r.doubleSpinBox_gpc_scale.value()
            n_restart_optimization = ui_r.spinBox_gpc_n_restart.value()
            kernel = gpc_kernel_scale * RBF(gpc_kernel_width)
            model_reg_top = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=n_restart_optimization,
                random_state=0
            )
            model_reg_bottom = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=n_restart_optimization,
                random_state=0
            )
            text_model = (f'**GPR**: \nwidth kernal: {round(gpc_kernel_width, 2)}, \nscale kernal: {round(gpc_kernel_scale, 2)}, '
                          f'\nn restart: {n_restart_optimization} ,')

        elif model == 'SVR':
            model_reg_top = SVR(kernel=ui_r.comboBox_svr_kernel.currentText(), C=ui_r.doubleSpinBox_svr_c.value())
            model_reg_bottom = SVR(kernel=ui_r.comboBox_svr_kernel.currentText(), C=ui_r.doubleSpinBox_svr_c.value())
            text_model = (f'**SVR**: \nkernel: {ui_r.comboBox_svr_kernel.currentText()}, '
                          f'\nC: {round(ui_r.doubleSpinBox_svr_c.value(), 2)}, ')

        elif model == 'EN':
            model_reg_top = ElasticNet(
                alpha=ui_r.doubleSpinBox_alpha.value(),
                l1_ratio=ui_r.doubleSpinBox_l1_ratio.value(),
                random_state=0
            )
            model_reg_bottom = ElasticNet(
                alpha=ui_r.doubleSpinBox_alpha.value(),
                l1_ratio=ui_r.doubleSpinBox_l1_ratio.value(),
                random_state=0
            )
            text_model = (f'**EN**: \nalpha: {round(ui_r.doubleSpinBox_alpha.value(), 2)}, '
                          f'\nl1_ratio: {round(ui_r.doubleSpinBox_l1_ratio.value(), 2)}, ')

        elif model == 'LSS':
            model_reg_top = Lasso(alpha=ui_r.doubleSpinBox_alpha.value(), random_state=0)
            model_reg_bottom = Lasso(alpha=ui_r.doubleSpinBox_alpha.value(), random_state=0)
            text_model = f'**LSS**: \nalpha: {round(ui_r.doubleSpinBox_alpha.value(), 2)}, '

        else:
            n_knn = ui_r.spinBox_neighbors.value()
            weights_knn = 'distance' if ui_r.checkBox_knn_weights.isChecked() else 'uniform'
            model_reg_top = KNeighborsRegressor(n_neighbors=n_knn, weights=weights_knn, algorithm='auto')
            model_reg_bottom = KNeighborsRegressor(n_neighbors=n_knn, weights=weights_knn, algorithm='auto')
            text_model = f'**KNNR**: \nn_neighbors: {n_knn}, \nweights: {weights_knn}, '

        return model_reg_top, model_reg_bottom, text_model


    def build_stacking_voting_model():
        """ Построить модель стекинга """
        estimators_top, estimators_bottom, list_model = [], [], []

        if ui_r.checkBox_stv_mlpr.isChecked():
            mlpr_top = MLPRegressor(
                hidden_layer_sizes=tuple(map(int, ui_r.lineEdit_layer_mlp.text().split())),
                activation=ui_r.comboBox_activation_mlp.currentText(),
                solver=ui_r.comboBox_solvar_mlp.currentText(),
                alpha=ui_r.doubleSpinBox_alpha_mlp.value(),
                max_iter=5000,
                early_stopping=ui_r.checkBox_e_stop_mlp.isChecked(),
                validation_fraction=ui_r.doubleSpinBox_valid_mlp.value(),
                random_state=0
            )
            mlpr_bottom = MLPRegressor(
                hidden_layer_sizes=tuple(map(int, ui_r.lineEdit_layer_mlp.text().split())),
                activation=ui_r.comboBox_activation_mlp.currentText(),
                solver=ui_r.comboBox_solvar_mlp.currentText(),
                alpha=ui_r.doubleSpinBox_alpha_mlp.value(),
                max_iter=5000,
                early_stopping=ui_r.checkBox_e_stop_mlp.isChecked(),
                validation_fraction=ui_r.doubleSpinBox_valid_mlp.value(),
                random_state=0
            )
            estimators_top.append(('mlpr', mlpr_top))
            estimators_bottom.append(('mlpr', mlpr_bottom))
            list_model.append('mlpr')

        if ui_r.checkBox_stv_knnr.isChecked():
            n_knn = ui_r.spinBox_neighbors.value()
            weights_knn = 'distance' if ui_r.checkBox_knn_weights.isChecked() else 'uniform'
            knnr_top = KNeighborsRegressor(n_neighbors=n_knn, weights=weights_knn, algorithm='auto')
            knnr_bottom = KNeighborsRegressor(n_neighbors=n_knn, weights=weights_knn, algorithm='auto')
            estimators_top.append(('knnr', knnr_top))
            estimators_bottom.append(('knnr', knnr_bottom))
            list_model.append('knnr')

        if ui_r.checkBox_stv_gbr.isChecked():
            est = ui_r.spinBox_n_estimators.value()
            l_rate = ui_r.doubleSpinBox_learning_rate.value()
            gbr_top = GradientBoostingRegressor(n_estimators=est, learning_rate=l_rate, random_state=0)
            gbr_bottom = GradientBoostingRegressor(n_estimators=est, learning_rate=l_rate, random_state=0)
            estimators_top.append(('gbr', gbr_top))
            estimators_bottom.append(('gbr', gbr_bottom))
            list_model.append('gbr')

        if ui_r.checkBox_stv_dtr.isChecked():
            spl = 'random' if ui_r.checkBox_splitter_rnd.isChecked() else 'best'
            dtr_top = DecisionTreeRegressor(splitter=spl, random_state=0)
            dtr_bottom = DecisionTreeRegressor(splitter=spl, random_state=0)
            estimators_top.append(('dtr', dtr_top))
            estimators_bottom.append(('dtr', dtr_bottom))
            list_model.append('dtr')

        if ui_r.checkBox_stv_rfr.isChecked():
            if ui_r.checkBox_rfr_ada.isChecked():
                abr_top = AdaBoostRegressor(n_estimators=ui_r.spinBox_rfr_n.value(), random_state=0)
                abr_bottom = AdaBoostRegressor(n_estimators=ui_r.spinBox_rfr_n.value(), random_state=0)
                estimators_top.append(('abr', abr_top))
                estimators_bottom.append(('abr', abr_bottom))
                list_model.append('abr')
            elif ui_r.checkBox_rfr_extra.isChecked():
                etr_top = ExtraTreesRegressor(n_estimators=ui_r.spinBox_rfr_n.value(), bootstrap=True, oob_score=True, random_state=0, n_jobs=-1)
                etr_bottom = ExtraTreesRegressor(n_estimators=ui_r.spinBox_rfr_n.value(), bootstrap=True, oob_score=True, random_state=0, n_jobs=-1)
                estimators_top.append(('etr', etr_top))
                estimators_bottom.append(('etr', etr_bottom))
                list_model.append('etr')
            else:
                rfr_top = RandomForestRegressor(n_estimators=ui_r.spinBox_rfr_n.value(), oob_score=True, random_state=0, n_jobs=-1)
                rfr_bottom = RandomForestRegressor(n_estimators=ui_r.spinBox_rfr_n.value(), oob_score=True, random_state=0, n_jobs=-1)
                estimators_top.append(('rfr', rfr_top))
                estimators_bottom.append(('rfr', rfr_bottom))
                list_model.append('rfr')

        if ui_r.checkBox_stv_gpr.isChecked():
            gpc_kernel_width = ui_r.doubleSpinBox_gpc_wigth.value()
            gpc_kernel_scale = ui_r.doubleSpinBox_gpc_scale.value()
            n_restart_optimization = ui_r.spinBox_gpc_n_restart.value()
            kernel = gpc_kernel_scale * RBF(gpc_kernel_width)
            gpr_top = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=n_restart_optimization,
                random_state=0
            )
            gpr_bottom = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=n_restart_optimization,
                random_state=0
            )
            estimators_top.append(('gpr', gpr_top))
            estimators_bottom.append(('gpr', gpr_bottom))
            list_model.append('gpr')

        if ui_r.checkBox_stv_svr.isChecked():
            svr_top = SVR(kernel=ui_r.comboBox_svr_kernel.currentText(),
                      C=ui_r.doubleSpinBox_svr_c.value())
            svr_bottom = SVR(kernel=ui_r.comboBox_svr_kernel.currentText(),
                      C=ui_r.doubleSpinBox_svr_c.value())
            estimators_top.append(('svr', svr_top))
            estimators_bottom.append(('svr', svr_bottom))
            list_model.append('svr')

        if ui_r.checkBox_stv_lr.isChecked():
            lr_top = LinearRegression(fit_intercept=ui_r.checkBox_fit_intercept.isChecked())
            lr_bottom = LinearRegression(fit_intercept=ui_r.checkBox_fit_intercept.isChecked())
            estimators_top.append(('lr', lr_top))
            estimators_bottom.append(('lr', lr_bottom))
            list_model.append('lr')

        if ui_r.checkBox_stv_en.isChecked():
            en_top = ElasticNet(
                alpha=ui_r.doubleSpinBox_alpha.value(),
                l1_ratio=ui_r.doubleSpinBox_l1_ratio.value(),
                random_state=0
            )
            en_bottom = ElasticNet(
                alpha=ui_r.doubleSpinBox_alpha.value(),
                l1_ratio=ui_r.doubleSpinBox_l1_ratio.value(),
                random_state=0
            )
            estimators_top.append(('en', en_top))
            estimators_bottom.append(('en', en_bottom))
            list_model.append('en')

        if ui_r.checkBox_stv_lasso.isChecked():
            lss_top = Lasso(alpha=ui_r.doubleSpinBox_alpha.value(), random_state=0)
            lss_bottom = Lasso(alpha=ui_r.doubleSpinBox_alpha.value(), random_state=0)
            estimators_top.append(('lss', lss_top))
            estimators_bottom.append(('lss', lss_bottom))
            list_model.append('lss')

        final_model_top, final_model_bottom, final_text_model = choice_model_regressor(ui_r.buttonGroup.checkedButton().text())
        list_model_text = ', '.join(list_model)

        if ui_r.buttonGroup_stack_vote.checkedButton().text() == 'Voting':
            model_class_top = VotingRegressor(estimators=estimators_top, n_jobs=-1)
            model_class_bottom = VotingRegressor(estimators=estimators_bottom, n_jobs=-1)
            text_model = f'**Voting**: \n({list_model_text})\n'
            model_name = 'VOT'
        else:
            model_class_top = StackingRegressor(estimators=estimators_top, final_estimator=final_model_top, n_jobs=-1)
            model_class_bottom = StackingRegressor(estimators=estimators_bottom, final_estimator=final_model_bottom, n_jobs=-1)
            text_model = f'**Stacking**:\nFinal estimator: {final_text_model}\n({list_model_text})\n'
            model_name = 'STACK'
        return model_class_top, model_class_bottom, text_model, model_name


    def calc_model_reg():
        """ Создание и тренировка модели """

        start_time = datetime.datetime.now()
        # Нормализация данных
        scaler_top = StandardScaler()
        scaler_bottom = StandardScaler()

        pipe_steps_top, pipe_steps_bottom = [], []
        pipe_steps_top.append(('scaler', scaler_top))
        pipe_steps_bottom.append(('scaler', scaler_bottom))

        # Разделение данных на обучающую и тестовую выборки
        x_train_top, x_test_top, y_train_top, y_test_top = train_test_split(
            input_data, target_data_top, test_size=0.2, random_state=0
        )

        x_train_bottom, x_test_bottom, y_train_bottom, y_test_bottom = train_test_split(
            input_data, target_data_bottom, test_size=0.2, random_state=0
        )

        if ui_r.checkBox_pca.isChecked():
            n_comp = 'mle' if ui_r.checkBox_pca_mle.isChecked() else ui_r.spinBox_pca.value()
            pca_top = PCA(n_components=n_comp, random_state=0)
            pca_bottom = PCA(n_components=n_comp, random_state=0)
            pipe_steps_top.append(('pca', pca_top))
            pipe_steps_bottom.append(('pca', pca_bottom))
        text_pca = f'\nPCA: n_components={n_comp}' if ui_r.checkBox_pca.isChecked() else ''

        if ui_r.checkBox_stack_vote.isChecked():
            model_reg_top, model_reg_bottom, text_model, model_name = build_stacking_voting_model()
        else:
            model_name = ui_r.buttonGroup.checkedButton().text()
            model_reg_top, model_reg_bottom, text_model = choice_model_regressor(model_name)
            if model_name == 'RFR':
                if ui_r.checkBox_rfr_ada.isChecked():
                    model_name = 'ABR'
                if ui_r.checkBox_rfr_extra.isChecked():
                    model_name = 'ETR'

        text_model += text_pca

        pipe_steps_top.append(('model', model_reg_top))
        pipe_steps_bottom.append(('model', model_reg_bottom))
        pipe_top, pipe_bottom = Pipeline(pipe_steps_top), Pipeline(pipe_steps_bottom)

        if ui_r.checkBox_cross_val.isChecked():
            kf_top = KFold(n_splits=ui_r.spinBox_n_cross_val.value(), shuffle=True, random_state=0)
            kf_bottom = KFold(n_splits=ui_r.spinBox_n_cross_val.value(), shuffle=True, random_state=0)
            list_train, list_test, n_cross = [], [], 1
            for train_index, test_index in kf_top.split(input_data):
                list_train.append(train_index.tolist())
                list_test.append(test_index.tolist())
                n_cross += 1
            scores_cv_top = cross_val_score(pipe_top, input_data, target_data_top, cv=kf_top, n_jobs=-1)
            scores_cv_bottom = cross_val_score(pipe_bottom, input_data, target_data_bottom, cv=kf_bottom, n_jobs=-1)
            n_max = np.argmax(scores_cv_top)
            train_index, test_index = list_train[n_max], list_test[n_max]

            x_train_top = [input_data[i] for i in train_index]
            x_test_top = [input_data[i] for i in test_index]
            x_train_bottom = x_train_top.copy()
            x_test_bottom = x_test_top.copy()

            y_train_top = [target_data_top[i] for i in train_index]
            y_test_top = [target_data_top[i] for i in test_index]
            y_train_bottom = [target_data_bottom[i] for i in train_index]
            y_test_bottom = [target_data_bottom[i] for i in test_index]


        cv_text = (
            f'\nКРОСС-ВАЛИДАЦИЯ TOP\nОценки на каждом разбиении:\n {" / ".join(str(round(val, 2)) for val in scores_cv_top)}'
            f'\nСредн.: {round(scores_cv_top.mean(), 2)} '
            f'Станд. откл.: {round(scores_cv_top.std(), 2)}'
            f'\nКРОСС-ВАЛИДАЦИЯ BOTTOM\nОценки на каждом разбиении:\n {" / ".join(str(round(val, 2)) for val in scores_cv_bottom)}'
            f'\nСредн.: {round(scores_cv_bottom.mean(), 2)} '
            f'Станд. откл.: {round(scores_cv_bottom.std(), 2)}'
        ) if ui_r.checkBox_cross_val.isChecked() else ''

        pipe_top.fit(x_train_top, y_train_top)
        pipe_bottom.fit(x_train_bottom, y_train_bottom)
        y_pred_top = pipe_top.predict(x_test_top)
        y_pred_bottom = pipe_bottom.predict(x_test_bottom)

        accuracy_top = round(pipe_top.score(x_test_top, y_test_top), 5)
        mse_top = round(mean_squared_error(y_test_top, y_pred_top), 5)
        accuracy_bottom = round(pipe_bottom.score(x_test_bottom, y_test_bottom), 5)
        mse_bottom = round(mean_squared_error(y_test_bottom, y_pred_bottom), 5)

        train_time = datetime.datetime.now() - start_time

        set_info(f'Модель {model_name}:\n точность top/bottom: {accuracy_top}/{accuracy_bottom} '
                 f' Mean Squared Error:\n {mse_top}/{mse_bottom}, \n время обучения: {train_time}', 'blue')
        y_remain_top = [round(y_test_top[i] - y_pred_top[i], 5) for i in range(len(y_pred_top))]
        y_remain_bottom = [round(y_test_bottom[i] - y_pred_bottom[i], 5) for i in range(len(y_pred_bottom))]


        data_graph = pd.DataFrame({
            'y_test_top': y_test_top,
            'y_pred_top': y_pred_top,
            'y_remain_top': y_remain_top,
            'y_test_bottom': y_test_bottom,
            'y_pred_bottom': y_pred_bottom,
            'y_remain_bottom': y_remain_bottom
        })

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

        fig.suptitle(f'Модель {model_name}:\n точность top/bottom: {accuracy_top}/{accuracy_bottom} '
                     f' Mean Squared Error:\n {mse_top}/{mse_bottom}, \n время обучения: {train_time}' + cv_text)
        sns.scatterplot(data=data_graph, x='y_test_top', y='y_pred_top', ax=axes[0, 0])
        sns.regplot(data=data_graph, x='y_test_top', y='y_pred_top', ax=axes[0, 0])
        sns.scatterplot(data=data_graph, x='y_test_bottom', y='y_pred_bottom', ax=axes[1, 0])
        sns.regplot(data=data_graph, x='y_test_bottom', y='y_pred_bottom', ax=axes[1, 0])

        sns.scatterplot(data=data_graph, x='y_test_top', y='y_remain_top', ax=axes[0, 1])
        sns.regplot(data=data_graph, x='y_test_top', y='y_remain_top', ax=axes[0, 1])
        sns.scatterplot(data=data_graph, x='y_test_bottom', y='y_remain_bottom', ax=axes[1, 1])
        sns.regplot(data=data_graph, x='y_test_bottom', y='y_remain_bottom', ax=axes[1, 1])

        if ui_r.checkBox_cross_val.isChecked():
            axes[0, 2].bar(range(len(scores_cv_top)), scores_cv_top)
            axes[0, 2].set_title('Кросс-валидация_TOP')
            axes[1, 2].bar(range(len(scores_cv_bottom)), scores_cv_bottom)
            axes[1, 2].set_title('Кросс-валидация_BOTTOM')
        else:
            sns.histplot(data=data_graph, x='y_remain_top', kde=True, ax=axes[0, 2])
            sns.histplot(data=data_graph, x='y_remain_bottom', kde=True, ax=axes[1, 2])
        fig.tight_layout()
        fig.show()

        if not ui_r.checkBox_save_model.isChecked():
            return
        result = QtWidgets.QMessageBox.question(
            MainWindow,
            'Сохранение модели',
            f'Сохранить модель "{model_name}"?',
            QtWidgets.QMessageBox.Yes,
            QtWidgets.QMessageBox.No)
        if result == QtWidgets.QMessageBox.Yes:
            # Сохранение модели в файл с помощью pickle
            path_model_top = f'models/reg_form/{model_name}_TOP_{round(accuracy_top, 3)}_{datetime.datetime.now().strftime("%d%m%y")}.pkl'
            path_model_bottom = f'models/reg_form/{model_name}_BOTTOM_{round(accuracy_bottom, 3)}_{datetime.datetime.now().strftime("%d%m%y")}.pkl'

            with open(path_model_top, 'wb') as f:
                pickle.dump(pipe_top, f)
            with open(path_model_bottom, 'wb') as f:
                pickle.dump(pipe_bottom, f)
            new_trained_model = TrainedModel(
                title=f'{model_name}_{round(accuracy_top, 3)}_{datetime.datetime.now().strftime("%d%m%y")}',
                path_top=path_model_top,
                path_bottom=path_model_bottom
            )
            session.add(new_trained_model)
            session.commit()
            update_list_trained_models()
        else:
            pass


    def calc_lof():
        """ Расчет выбросов методом LOF """
        global data_pca, data_tsne, colors, factor_lof

        data_lof = input_data.copy()
        # data_lof.drop(['prof_well_index', 'target_value'], axis=1, inplace=True)

        scaler = StandardScaler()
        training_sample_lof = scaler.fit_transform(data_lof)
        n_LOF = ui_r.spinBox_lof_neighbor.value()

        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42, n_jobs=-1)
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


        def calc_lof_in_window():
            global data_pca, data_tsne, colors, factor_lof
            colors, data_pca, data_tsne, factor_lof, label_lof = calc_lof_model(ui_lof.spinBox_lof_n.value(), training_sample_lof)
            ui_lof.checkBox_samples.setChecked(False)
            draw_lof_tsne(data_tsne, ui_lof)
            draw_lof_pca(data_pca, ui_lof)
            draw_lof_bar(colors, factor_lof, label_lof, ui_lof)

            set_title_lof_form(label_lof)


        def calc_clean_model():
            _, _, _, _, label_lof = calc_lof_model(ui_lof.spinBox_lof_n.value(), training_sample_lof)
            input_data_clean = input_data.copy()
            target_data_top_clean = target_data_top.copy()
            target_data_bottom_clean = target_data_bottom.copy()
            lof_index = [i for i, x in enumerate(label_lof) if x == -1]
            input_data_new = np.delete(input_data_clean, lof_index, axis=0)
            target_data_top_new = np.delete(target_data_top_clean, lof_index, axis=0)
            target_data_bottom_new = np.delete(target_data_bottom_clean, lof_index, axis=0)
            Regressor.close()
            Form_LOF.close()
            show_regressor_train_form(input_data_new, target_data_top_new, target_data_bottom_new)


        # ui_lof.spinBox_lof_n.valueChanged.connect(calc_lof_in_window)
        ui_lof.pushButton_clean_lof.clicked.connect(calc_clean_model)
        ui_lof.pushButton_lof.clicked.connect(calc_lof_in_window)

        Form_LOF.exec_()


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

    ui_r.pushButton_lof.clicked.connect(calc_lof)
    # ui_r.checkBox_rfr_extra.clicked.connect(push_checkbutton_extra)
    # ui_r.checkBox_rfr_ada.clicked.connect(push_checkbutton_ada)
    ui_r.pushButton_calc.clicked.connect(calc_model_reg)
    Regressor.exec_()


def update_list_trained_models():
    ui.listWidget_trained_model.clear()
    for model in session.query(TrainedModel).all():
        item_text = model.title
        item = QListWidgetItem(item_text)
        item.setData(Qt.UserRole, model.id)
        ui.listWidget_trained_model.addItem(item)



def calc_model_profile():
    model = session.query(TrainedModel).filter_by(id=ui.listWidget_trained_model.currentItem().data(Qt.UserRole)).first()
    with open(model.path_top, 'rb') as f:
        model_ai_top = pickle.load(f)
    with open(model.path_bottom, 'rb') as f:
        model_ai_bottom = pickle.load(f)
    prof = session.query(Profile).filter_by(id=get_profile_id()).first()
    list_test_signals = []
    ui.progressBar.setMaximum(len(json.loads(prof.signal)))
    for n, s in enumerate(json.loads(prof.signal)):
        diff_s = calc_atrib_measure(s, 'diff')
        At = calc_atrib_measure(s, 'At')
        # Vt = calc_atrib_measure(s, 'Vt')
        Pht = calc_atrib_measure(s, 'Pht')
        Wt = calc_atrib_measure(s, 'Wt')
        list_test_signals.append(s + diff_s + At + Pht + Wt)
        ui.progressBar.setValue(n+1)
    test_signal = np.array(list_test_signals)
    y_pred_signal_top = model_ai_top.predict(test_signal)
    y_pred_signal_bottom = model_ai_bottom.predict(test_signal)
    result_top = list(map(int, savgol_filter(savgol_filter(savgol_filter(y_pred_signal_top.tolist(), 31, 3), 31, 3), 31, 3)))
    result_bottom = list(map(int, savgol_filter(savgol_filter(savgol_filter(y_pred_signal_bottom.tolist(), 31, 3), 31, 3), 31, 3)))
    new_top = Layers(
        profile_id=get_profile_id(),
        layer_line=json.dumps(result_top),
        layer_title=f'top_{model.title}'
    )
    new_bottom = Layers(
        profile_id=get_profile_id(),
        layer_line=json.dumps(result_bottom),
        layer_title=f'bottom_{model.title}'
    )
    session.add(new_top)
    session.add(new_bottom)
    session.commit()
    update_layers()
    new_formation = Formation(title=model.title, profile_id=get_profile_id(), up=new_top.id, down=new_bottom.id)
    session.add(new_formation)
    session.commit()
    add_param_in_new_formation(new_formation.id, new_formation.profile_id)
    update_formation_combobox()


def calc_model_object():
    model = session.query(TrainedModel).filter_by(id=ui.listWidget_trained_model.currentItem().data(Qt.UserRole)).first()
    for n, prof in enumerate(session.query(Profile).filter(Profile.research_id == get_research_id()).all()):
        count_measure = len(json.loads(session.query(Profile.signal).filter(Profile.id == prof.id).first()[0]))
        ui.comboBox_profile.setCurrentText(f'{prof.title} ({count_measure} измерений) id{prof.id}')
        set_info(f'Профиль {prof.title} ({count_measure} измерений)', 'blue')
        if session.query(Formation).filter_by(title=model.title, profile_id=prof.id).count() > 0:
            continue
        calc_model_profile()


def remove_trained_model():
    """ Удаление модели """
    model = session.query(TrainedModel).filter_by(id=ui.listWidget_trained_model.currentItem().data(Qt.UserRole)).first()
    os.remove(model.path_top)
    os.remove(model.path_bottom)
    session.delete(model)
    session.commit()
    update_list_trained_models()
    set_info(f'Модель {model.title} удалена', 'blue')


def export_model_formation_ai():
    """ Экспорт модели """
    try:
        model = session.query(TrainedModel).filter_by(id=ui.listWidget_trained_model.currentItem().data(Qt.UserRole)).first()
    except AttributeError:
        return set_info('Модель не выбрана', 'red')

    model_parameters = {
        'title': model.title
    }
    # Сохранение словаря с параметрами в файл *.pkl
    with open('model_parameters.pkl', 'wb') as parameters_file:
        pickle.dump(model_parameters, parameters_file)

    try:
        filename = QFileDialog.getSaveFileName(caption='Экспорт модели', directory=f'{model.title}.zip', filter="*.zip")[0]
        with zipfile.ZipFile(filename, 'w') as zip:
            zip.write('model_parameters.pkl', 'model_parameters.pkl')
            zip.write(model.path_top, 'model_top.pkl')
            zip.write(model.path_bottom, 'model_bottom.pkl')

        set_info(f'Модель {model.title} экспортирована в файл {filename}', 'blue')
    except FileNotFoundError:
        pass


def import_model_formation_ai():
    """ Импорт модели """
    try:
        filename = QFileDialog.getOpenFileName(caption='Импорт модели', filter="*.zip")[0]

        with zipfile.ZipFile(filename, 'r') as zip:
            zip.extractall('extracted_data')

            with open('extracted_data/model_parameters.pkl', 'rb') as parameters_file:
                loaded_parameters = pickle.load(parameters_file)

            # Загрузка модели кровли из файла *.pkl
            with open('extracted_data/model_top.pkl', 'rb') as model_top_file:
                loaded_model_top = pickle.load(model_top_file)

            # Загрузка модели подошвы из файла *.pkl
            with open('extracted_data/model_bottom.pkl', 'rb') as model_bottom_file:
                loaded_model_bottom = pickle.load(model_bottom_file)
    except FileNotFoundError:
        return

    model_name = loaded_parameters['title']

    path_model_top = f'models/reg_form/{model_name}_TOP.pkl'
    path_model_bottom = f'models/reg_form/{model_name}_BOTTOM.pkl'

    with open(path_model_top, 'wb') as f:
        pickle.dump(loaded_model_top, f)
    with open(path_model_bottom, 'wb') as f:
        pickle.dump(loaded_model_bottom, f)

    new_trained_model = TrainedModel(
        title=model_name,
        path_top=path_model_top,
        path_bottom=path_model_bottom
    )
    session.add(new_trained_model)
    session.commit()
    try:
        shutil.rmtree('extracted_data')
        os.remove('model_parameters.pkl')
    except FileNotFoundError:
        pass

    update_list_trained_models()
    set_info(f'Модель "{model_name}" добавлена', 'blue')
