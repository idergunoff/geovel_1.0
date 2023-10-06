import datetime
import time

from draw import draw_radarogram, draw_formation
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
    set_info(f'Модель {new_model.title} добавлена', 'green')


def remove_model_ai():
    """ Удалить модель """
    model_title = ui.comboBox_model_ai.currentText()
    result = QtWidgets.QMessageBox.question(
        MainWindow,
        'Удаление модели',
        f'Вы уверены, что хотите удалить модель {model_title} со всеми пластами?',
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
    formation_ai = session.query(FormationAI).filter_by(id=ui.listWidget_formation_ai.currentItem().data(Qt.UserRole)).first()
    ui.comboBox_object.setCurrentText(f'{formation_ai.profile.research.object.title} id{formation_ai.profile.research.object_id}')
    update_research_combobox()
    ui.comboBox_research.setCurrentText(
        f'{formation_ai.profile.research.date_research.strftime("%m.%Y")} id{formation_ai.profile.research_id}')
    update_profile_combobox()
    count_measure = len(json.loads(session.query(Profile.signal).filter(Profile.id == formation_ai.profile_id).first()[0]))
    ui.comboBox_profile.setCurrentText(f'{formation_ai.profile.title} ({count_measure} измерений) id{formation_ai.profile_id}')
    draw_radarogram()
    ui.comboBox_plast.setCurrentText(f'{formation_ai.formation.title} id{formation_ai.formation_id}')
    draw_formation()


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

    Form_AI = QtWidgets.QDialog()
    ui_fai = Ui_Form_formation_ai()
    ui_fai.setupUi(Form_AI)
    Form_AI.show()
    Form_AI.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    def calc_model_formation_ai():
        start_time = datetime.datetime.now()
        model = ui_fai.comboBox_model_ai.currentText()

        x_train_top, x_test_top, y_train_top, y_test_top = train_test_split(
            input_data, target_data_top, test_size=0.2, random_state=42
        )
        x_train_bottom, x_test_bottom, y_train_bottom, y_test_bottom = train_test_split(
            input_data, target_data_bottom, test_size=0.2, random_state=42
        )

        if model == 'LinearRegression':
            model_ai_top = LinearRegression(fit_intercept=ui_fai.checkBox_fit_intercept.isChecked())
            model_ai_bottom = LinearRegression(fit_intercept=ui_fai.checkBox_fit_intercept.isChecked())
            model_name = 'LR'

        if model == 'DecisionTreeRegressor':
            spl = 'random' if ui_fai.checkBox_splitter_rnd.isChecked() else 'best'
            model_ai_top = DecisionTreeRegressor(splitter=spl)
            model_ai_bottom = DecisionTreeRegressor(splitter=spl)
            model_name = 'DTR'

        if model == 'KNeighborsRegressor':
            model_ai_top = KNeighborsRegressor(
                n_neighbors=ui_fai.spinBox_neighbors.value(),
                weights='distance' if ui_fai.checkBox_knn_weights.isChecked() else 'uniform',
                algorithm=ui_fai.comboBox_knn_algorithm.currentText()
            )
            model_ai_bottom = KNeighborsRegressor(
                n_neighbors=ui_fai.spinBox_neighbors.value(),
                weights='distance' if ui_fai.checkBox_knn_weights.isChecked() else 'uniform',
                algorithm=ui_fai.comboBox_knn_algorithm.currentText()
            )
            model_name = 'KNNR'

        if model == 'SVR':
            model_ai_top = SVR(kernel=ui_fai.comboBox_svr_kernel.currentText(), C=ui_fai.doubleSpinBox_svr_c.value())
            model_ai_bottom = SVR(kernel=ui_fai.comboBox_svr_kernel.currentText(), C=ui_fai.doubleSpinBox_svr_c.value())
            model_name = 'SVR'


        if model == 'MLPRegressor':
            layers = tuple(map(int, ui_fai.lineEdit_layer_mlp.text().split()))
            model_ai_top = MLPRegressor(
                hidden_layer_sizes=layers,
                activation=ui_fai.comboBox_activation_mlp.currentText(),
                solver=ui_fai.comboBox_solvar_mlp.currentText(),
                alpha=ui_fai.doubleSpinBox_alpha_mlp.value(),
                max_iter=5000,
                early_stopping=ui_fai.checkBox_e_stop_mlp.isChecked(),
                validation_fraction=ui_fai.doubleSpinBox_valid_mlp.value()
            )
            model_ai_bottom = MLPRegressor(
                hidden_layer_sizes=layers,
                activation=ui_fai.comboBox_activation_mlp.currentText(),
                solver=ui_fai.comboBox_solvar_mlp.currentText(),
                alpha=ui_fai.doubleSpinBox_alpha_mlp.value(),
                max_iter=5000,
                early_stopping=ui_fai.checkBox_e_stop_mlp.isChecked(),
                validation_fraction=ui_fai.doubleSpinBox_valid_mlp.value()
            )
            model_name = 'MLPR'

        if model == 'GradientBoostingRegressor':
            model_ai_top = GradientBoostingRegressor(
                n_estimators=ui_fai.spinBox_n_estimators.value(),
                learning_rate=ui_fai.doubleSpinBox_learning_rate.value(),
            )
            model_ai_bottom = GradientBoostingRegressor(
                n_estimators=ui_fai.spinBox_n_estimators.value(),
                learning_rate=ui_fai.doubleSpinBox_learning_rate.value(),
            )
            model_name = 'GBR'

        if model == 'ElasticNet':
            model_ai_top = ElasticNet(
                alpha=ui_fai.doubleSpinBox_alpha.value(),
                l1_ratio=ui_fai.doubleSpinBox_l1_ratio.value()
            )
            model_ai_bottom = ElasticNet(
                alpha=ui_fai.doubleSpinBox_alpha.value(),
                l1_ratio=ui_fai.doubleSpinBox_l1_ratio.value()
            )
            model_name = 'EN'

        if model == 'Lasso':
            model_ai_top = Lasso(alpha=ui_fai.doubleSpinBox_alpha.value())
            model_ai_bottom = Lasso(alpha=ui_fai.doubleSpinBox_alpha.value())
            model_name = 'LASSO'

        model_ai_top.fit(x_train_top, y_train_top)
        model_ai_bottom.fit(x_train_bottom, y_train_bottom)

        y_pred_top = model_ai_top.predict(x_test_top)
        y_pred_bottom = model_ai_bottom.predict(x_test_bottom)

        mse_top = mean_squared_error(y_test_top, y_pred_top)
        mse_bottom = mean_squared_error(y_test_bottom, y_pred_bottom)
        train_time = datetime.datetime.now() - start_time

        accuracy_top = model_ai_top.score(x_test_top, y_test_top)
        accuracy_bottom = model_ai_bottom.score(x_test_bottom, y_test_bottom)

        set_info(f'Модель {model}:'
                 f'\n точность для top: {accuracy_top} '
                 f'\n точность для bottom: {accuracy_bottom}\n'
                 f' Mean Squared Error\n top: {mse_top} \n bottom: {mse_bottom}, '
                 f'\n время обучения: {train_time}', 'blue')

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

        fig, axes = plt.subplots(nrows=2, ncols=3)
        fig.set_size_inches(15, 10)
        fig.suptitle(f'Модель {model}:\n точность top: {accuracy_top}, bottom: {accuracy_bottom} '
                     f' Mean Squared Error:\n top: {mse_top}, bottom: {mse_bottom}, \n время обучения: {train_time}')
        sns.scatterplot(data=data_graph, x='y_test_top', y='y_pred_top', ax=axes[0, 0])
        sns.regplot(data=data_graph, x='y_test_top', y='y_pred_top', line_kws={'color': 'red'}, ax=axes[0, 0])
        sns.scatterplot(data=data_graph, x='y_test_top', y='y_remain_top', ax=axes[0, 1])
        sns.regplot(data=data_graph, x='y_test_top', y='y_remain_top', line_kws={'color': 'red'}, ax=axes[0, 1])
        sns.histplot(data=data_graph, x='y_remain_top', kde=True, ax=axes[0, 2])
        axes[0, 2].lines[0].set_color("red")
        sns.scatterplot(data=data_graph, x='y_test_bottom', y='y_pred_bottom', ax=axes[1, 0])
        sns.regplot(data=data_graph, x='y_test_bottom', y='y_pred_bottom', line_kws={'color': 'red'}, ax=axes[1, 0])
        sns.scatterplot(data=data_graph, x='y_test_bottom', y='y_remain_bottom', ax=axes[1, 1])
        sns.regplot(data=data_graph, x='y_test_bottom', y='y_remain_bottom', line_kws={'color': 'red'}, ax=axes[1, 1])
        sns.histplot(data=data_graph, x='y_remain_bottom', kde=True, ax=axes[1, 2])
        axes[1, 2].lines[0].set_color("red")
        fig.tight_layout()
        fig.show()

        result = QtWidgets.QMessageBox.question(
                    MainWindow,
                    'Сохранение модели',
                    f'Сохранить модель {model}?',
                    QtWidgets.QMessageBox.Yes,
                    QtWidgets.QMessageBox.No)
        if result == QtWidgets.QMessageBox.Yes:
            # Сохранение модели в файл с помощью pickle
            path_model_top = f'models/{model_name}_{round(accuracy_top, 3)}_{datetime.datetime.now().strftime("%d%m%y")}_top.pkl'
            path_model_bottom = f'models/{model_name}_{round(accuracy_bottom, 3)}_{datetime.datetime.now().strftime("%d%m%y")}_bottom.pkl'
            with open(path_model_top, 'wb') as f:
                pickle.dump(model_ai_top, f)
            with open(path_model_bottom, 'wb') as f:
                pickle.dump(model_ai_bottom, f)
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

        # if comb_data == 'top_i/bottom_i' or comb_data == 't/b/t_i/b_i' or comb_data == 'top_i' or comb_data == 'bottom_i':
        #     result = QtWidgets.QMessageBox.question(
        #         MainWindow,
        #         'Расчет модели для профиля',
        #         f'Расчитать модель {ui_fai.comboBox_model_ai.currentText()} для текущего профиля?',
        #         QtWidgets.QMessageBox.Yes,
        #         QtWidgets.QMessageBox.No)
        #     if result == QtWidgets.QMessageBox.Yes:
        #         prof = session.query(Profile).filter_by(id=get_profile_id()).first()
        #         test_signal = np.array(json.loads(prof.signal))
        #         y_pred_signal = model_ai.predict(test_signal)
        #         if comb_data == 'top_i/bottom_i':
        #             top_model, bottom_model = [pred[0] for pred in y_pred_signal], [pred[1] for pred in y_pred_signal]
        #             new_top = Layers(profile_id=get_profile_id(), layer_line=json.dumps(top_model), layer_title=f'top_{model}')
        #             new_bottom = Layers(profile_id=get_profile_id(), layer_line=json.dumps(bottom_model), layer_title=f'bottom_{model}')
        #             session.add(new_top)
        #             session.add(new_bottom)
        #             session.commit()
        #         if comb_data == 't/b/t_i/b_i':
        #             top_model, bottom_model = [int(pred[2]) for pred in y_pred_signal], [int(pred[3]) for pred in y_pred_signal]
        #             top_model_a, bottom_model_a = [pred[0] for pred in y_pred_signal], [pred[1] for pred in y_pred_signal]
        #             new_top = Layers(profile_id=get_profile_id(), layer_title=f'top_{model}')
        #             new_bottom = Layers(profile_id=get_profile_id(), layer_title=f'bottom_{model}')
        #             session.add(new_top)
        #             session.add(new_bottom)
        #             session.commit()
        #             for n, i in enumerate(test_signal):
        #                 if top_model_a[n] - 3 < test_signal[n][top_model[n]] < top_model_a[n] + 3:
        #                     new_point = PointsOfLayer(layer_id=new_top.id, point_x=n, point_y=top_model[n])
        #                     session.add(new_point)
        #                 if bottom_model_a[n] - 3 < test_signal[n][bottom_model[n]] < bottom_model_a[n] + 3:
        #                     new_point = PointsOfLayer(layer_id=new_bottom.id, point_x=n, point_y=bottom_model[n])
        #                     session.add(new_point)
        #             session.commit()
        #         if comb_data == 'top_i':
        #             result = list(map(int, savgol_filter(y_pred_signal.tolist(), 31, 3)))
        #             new_top = Layers(
        #                 profile_id=get_profile_id(),
        #                 layer_line=json.dumps(result),
        #                 layer_title=f'top_{model}'
        #             )
        #             session.add(new_top)
        #             session.commit()
        #         if comb_data == 'bottom_i':
        #             result = list(map(int, savgol_filter(y_pred_signal.tolist(), 31, 3)))
        #             new_bottom = Layers(
        #                 profile_id=get_profile_id(),
        #                 layer_line=json.dumps(result),
        #                 layer_title=f'bottom_{model}'
        #             )
        #             session.add(new_bottom)
        #             session.commit()
        #     if result == QtWidgets.QMessageBox.No:
        #         pass



    ui_fai.pushButton_calc_model.clicked.connect(calc_model_formation_ai)
    Form_AI.exec_()


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
    result_top = list(map(int, savgol_filter(y_pred_signal_top.tolist(), 31, 3)))
    result_bottom = list(map(int, savgol_filter(y_pred_signal_bottom.tolist(), 31, 3)))
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


def remove_trained_model():
    """ Удаление модели """
    model = session.query(TrainedModel).filter_by(id=ui.listWidget_trained_model.currentItem().data(Qt.UserRole)).first()
    os.remove(model.path_top)
    os.remove(model.path_bottom)
    session.delete(model)
    session.commit()
    update_list_trained_models()
    set_info(f'Модель {model.title} удалена', 'blue')




