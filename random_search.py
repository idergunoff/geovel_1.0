from scipy.stats import randint, uniform
from func import *


def push_random_search():
    """ Старт окна RandomizedSearchCV """
    global dict_param_distr
    RandomSearch = QtWidgets.QDialog()
    ui_rs = Ui_RandomSearchForm()
    ui_rs.setupUi(RandomSearch)
    RandomSearch.show()
    RandomSearch.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    data_train, list_param = build_table_train(True, 'mlp')
    list_param_mlp = data_train.columns.tolist()[2:]
    colors = {}
    for m in session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).all():
        colors[m.title] = m.color
    training_sample = data_train[list_param_mlp].values.tolist()
    markup = sum(data_train[['mark']].values.tolist(), [])

    ui_rs.spinBox_pca.setMaximum(len(list_param_mlp))
    ui_rs.spinBox_pca_lim.setMaximum(len(list_param_mlp))
    ui_rs.spinBox_pca.setValue(len(list_param_mlp) // 2)
    ui_rs.spinBox_pca_lim.setValue((len(list_param_mlp) // 3) * 2)

    def add_text(parameter, p_value):
        ui_rs.textEdit.append(f'<p><b>{parameter}</b>: {p_value}</p>')


    ### KNN ###

    def show_dict_param_distr_knn():
        global dict_param_distr
        ui_rs.textEdit.clear()
        add_text('knn__weights', dict_param_distr['knn__weights'])
        if ui_rs.checkBox_neighbors_lim.isChecked():
            add_text('knn__n_neighbors',
                     f'randint({ui_rs.spinBox_neighbors.value()}, {ui_rs.spinBox_neighbors_lim.value()})')
        else:
            add_text('knn__n_neighbors', dict_param_distr['knn__n_neighbors'])


    def click_knn():
        global dict_param_distr
        ui_rs.checkBox_neighbors_lim.setChecked(False)
        n_knn = ui_rs.spinBox_neighbors.value()
        weights_knn = 'distance' if ui_rs.checkBox_knn_weights.isChecked() else 'uniform'
        dict_param_distr = {'knn__n_neighbors': [n_knn], 'knn__weights': [weights_knn]}
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_knn()


    def add_neighbors():
        global dict_param_distr
        if not ui_rs.radioButton_knn.isChecked():
            return
        n_knn = ui_rs.spinBox_neighbors.value()
        if ui_rs.checkBox_neighbors_lim.isChecked():
            dict_param_distr['knn__n_neighbors'] = [n_knn]
            ui_rs.checkBox_neighbors_lim.setChecked(False)
        else:
            if not n_knn in dict_param_distr['knn__n_neighbors']:
                dict_param_distr['knn__n_neighbors'].append(n_knn)
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_knn()


    def lim_neighbors():
        global dict_param_distr
        if not ui_rs.radioButton_knn.isChecked():
            return
        n_knn = ui_rs.spinBox_neighbors.value()
        if ui_rs.checkBox_neighbors_lim.isChecked():
            n_knn_lim = ui_rs.spinBox_neighbors_lim.value()
            dict_param_distr['knn__n_neighbors'] = randint(n_knn, n_knn_lim)
        else:
            dict_param_distr['knn__n_neighbors'] = [n_knn]
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_knn()


    def correct_lim_neighbors():
        global dict_param_distr
        if not ui_rs.radioButton_knn.isChecked() or not ui_rs.checkBox_neighbors_lim.isChecked():
            return
        n_knn = ui_rs.spinBox_neighbors.value()
        n_knn_lim = ui_rs.spinBox_neighbors_lim.value()
        dict_param_distr['knn__n_neighbors'] = randint(n_knn, n_knn_lim)
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_knn()


    def add_weights():
        global dict_param_distr
        if not ui_rs.radioButton_knn.isChecked():
            return
        weights_knn = 'distance' if ui_rs.checkBox_knn_weights.isChecked() else 'uniform'
        if not weights_knn in dict_param_distr['knn__weights']:
            dict_param_distr['knn__weights'].append(weights_knn)
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_knn()


    ### SVC ###
    def show_dict_param_distr_svc():
        global dict_param_distr
        ui_rs.textEdit.clear()
        add_text('svc__kernel', dict_param_distr['svc__kernel'])
        if ui_rs.checkBox_svc_lim.isChecked():
            add_text('svc__C',
                     f'uniform({ui_rs.doubleSpinBox_svr_c.value()}, {ui_rs.doubleSpinBox_svr_c_lim.value()})')
        else:
            add_text('svc__C', dict_param_distr['svc__C'])


    def click_svc():
        global dict_param_distr
        ui_rs.checkBox_svc_lim.setChecked(False)
        dict_param_distr = {'svc__kernel': [ui_rs.comboBox_svr_kernel.currentText()], 'svc__C': [ui_rs.doubleSpinBox_svr_c.value()]}
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_svc()


    def add_kernel():
        global dict_param_distr
        if not ui_rs.radioButton_svc.isChecked():
            return
        if not ui_rs.comboBox_svr_kernel.currentText() in dict_param_distr['svc__kernel']:
            dict_param_distr['svc__kernel'].append(ui_rs.comboBox_svr_kernel.currentText())
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_svc()


    def add_c():
        global dict_param_distr
        if not ui_rs.radioButton_svc.isChecked():
            return
        svc_c = ui_rs.doubleSpinBox_svr_c.value()
        if ui_rs.checkBox_svc_lim.isChecked():
            dict_param_distr['svc__C'] = [svc_c]
            ui_rs.checkBox_svc_lim.setChecked(False)
        else:
            if not svc_c in dict_param_distr['svc__C']:
                dict_param_distr['svc__C'].append(svc_c)
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_svc()



    def lim_c():
        global dict_param_distr
        if not ui_rs.radioButton_svc.isChecked():
            return
        svc_c = ui_rs.doubleSpinBox_svr_c.value()
        if ui_rs.checkBox_svc_lim.isChecked():
            svc_c_lim = ui_rs.doubleSpinBox_svr_c_lim.value()
            dict_param_distr['svc__C'] = uniform(svc_c, svc_c_lim)
        else:
            dict_param_distr['svc__C'] = [svc_c]
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_svc()


    def correct_lim_c():
        global dict_param_distr
        if not ui_rs.radioButton_svc.isChecked() or not ui_rs.checkBox_svc_lim.isChecked():
            return
        svc_c = ui_rs.doubleSpinBox_svr_c.value()
        svc_c_lim = ui_rs.doubleSpinBox_svr_c_lim.value()
        dict_param_distr['svc__C'] = uniform(svc_c, svc_c_lim)
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_svc()


    ### QDA ###

    def show_dict_param_distr_qda():
        global dict_param_distr
        ui_rs.textEdit.clear()
        if ui_rs.checkBox_reg_param_lim.isChecked():
            add_text('qda__reg_param',
                     f'uniform({round(ui_rs.doubleSpinBox_qda_reg_param.value(), 2)}, '
                     f'{round(ui_rs.doubleSpinBox_qda_reg_param_lim.value(), 2)})')
        else:
            add_text('qda__reg_param', dict_param_distr['qda__reg_param'])


    def click_qda():
        global dict_param_distr
        ui_rs.checkBox_reg_param_lim.setChecked(False)
        dict_param_distr = {'qda__reg_param': [round(ui_rs.doubleSpinBox_qda_reg_param.value(), 2)]}
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_qda()


    def add_reg_param():
        global dict_param_distr
        if not ui_rs.radioButton_qda.isChecked():
            return
        reg_param = round(ui_rs.doubleSpinBox_qda_reg_param.value(), 2)
        if ui_rs.checkBox_reg_param_lim.isChecked():
            dict_param_distr['qda__reg_param'] = [reg_param]
            ui_rs.checkBox_reg_param_lim.setChecked(False)
        else:
            if not reg_param in dict_param_distr['qda__reg_param']:
                dict_param_distr['qda__reg_param'].append(reg_param)
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_qda()


    def lim_reg_param():
        global dict_param_distr
        if not ui_rs.radioButton_qda.isChecked():
            return
        reg_param = round(ui_rs.doubleSpinBox_qda_reg_param.value(), 2)
        if ui_rs.checkBox_reg_param_lim.isChecked():
            reg_param_lim = round(ui_rs.doubleSpinBox_qda_reg_param_lim.value(), 2)
            dict_param_distr['qda__reg_param'] = uniform(reg_param, reg_param_lim)
        else:
            dict_param_distr['qda__reg_param'] = [reg_param]
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_qda()


    def correct_lim_reg_param():
        global dict_param_distr
        if not ui_rs.radioButton_qda.isChecked() or not ui_rs.checkBox_reg_param_lim.isChecked():
            return
        reg_param = round(ui_rs.doubleSpinBox_qda_reg_param.value(), 2)
        reg_param_lim = round(ui_rs.doubleSpinBox_qda_reg_param_lim.value(), 2)
        dict_param_distr['qda__reg_param'] = uniform(reg_param, reg_param_lim)
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_qda()


### GNB ###

    def show_dict_param_distr_gnb():
        global dict_param_distr
        ui_rs.textEdit.clear()
        if ui_rs.checkBox_var_smooth_lim.isChecked():
            add_text('gnb__var_smoothing', f'uniform(10 ** (-{ui_rs.spinBox_gnb_var_smooth.value()}), '
                                       f'10 ** (-{ui_rs.spinBox_gnb_var_smooth_lim.value()}))')
        else:
            add_text('gnb__var_smoothing', dict_param_distr['gnb__var_smoothing'])


    def click_gnb():
        global dict_param_distr
        ui_rs.checkBox_var_smooth_lim.setChecked(False)
        dict_param_distr = {'gnb__var_smoothing': [10 ** (-ui_rs.spinBox_gnb_var_smooth.value())]}
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_gnb()


    def add_var_smooth():
        global dict_param_distr
        if not ui_rs.radioButton_gnb.isChecked():
            return
        var_smooth = 10 ** (-ui_rs.spinBox_gnb_var_smooth.value())
        if ui_rs.checkBox_var_smooth_lim.isChecked():
            dict_param_distr['gnb__var_smoothing'] = [var_smooth]
            ui_rs.checkBox_var_smooth_lim.setChecked(False)
        else:
            if not var_smooth in dict_param_distr['gnb__var_smoothing']:
                dict_param_distr['gnb__var_smoothing'].append(var_smooth)
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_gnb()


    def lim_var_smooth():
        global dict_param_distr
        if not ui_rs.radioButton_gnb.isChecked():
            return
        var_smooth = 10 ** (-ui_rs.spinBox_gnb_var_smooth.value())
        if ui_rs.checkBox_var_smooth_lim.isChecked():
            var_smooth_lim = 10 ** (-ui_rs.spinBox_gnb_var_smooth_lim.value())
            dict_param_distr['gnb__var_smoothing'] = uniform(var_smooth, var_smooth_lim)
        else:
            dict_param_distr['gnb__var_smoothing'] = [var_smooth]
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_gnb()


    def correct_lim_var_smooth():
        global dict_param_distr
        if not ui_rs.radioButton_gnb.isChecked() or not ui_rs.checkBox_var_smooth_lim.isChecked():
            return
        var_smooth = 10 ** (-ui_rs.spinBox_gnb_var_smooth.value())
        var_smooth_lim = 10 ** (-ui_rs.spinBox_gnb_var_smooth_lim.value())
        dict_param_distr['gnb__var_smoothing'] = uniform(var_smooth, var_smooth_lim)
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_gnb()


### RFC ###

    def show_dict_param_distr_rfc():
        global dict_param_distr
        ui_rs.textEdit.clear()
        if ui_rs.checkBox_rfc_n_lim.isChecked():
            add_text('rfc__n_estimators', f'randint({ui_rs.spinBox_rfc_n.value()}, '
                                       f'{ui_rs.spinBox_rfc_n_lim.value()})')
        else:
            add_text('rfc__n_estimators', dict_param_distr['rfc__n_estimators'])


    def click_rfc():
        global dict_param_distr
        ui_rs.checkBox_rfc_n_lim.setChecked(False)
        dict_param_distr = {'rfc__n_estimators': [ui_rs.spinBox_rfc_n.value()]}
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_rfc()


    def add_rfc_n():
        global dict_param_distr
        if not ui_rs.radioButton_rfc.isChecked():
            return
        n_est = ui_rs.spinBox_rfc_n.value()
        if ui_rs.checkBox_rfc_n_lim.isChecked():
            dict_param_distr['rfc__n_estimators'] = [n_est]
            ui_rs.checkBox_rfc_n_lim.setChecked(False)
        else:
            if not n_est in dict_param_distr['rfc__n_estimators']:
                dict_param_distr['rfc__n_estimators'].append(n_est)
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_rfc()


    def lim_rfc_n():
        global dict_param_distr
        if not ui_rs.radioButton_rfc.isChecked():
            return
        n_est = ui_rs.spinBox_rfc_n.value()
        if ui_rs.checkBox_rfc_n_lim.isChecked():
            n_est_lim = ui_rs.spinBox_rfc_n_lim.value()
            dict_param_distr['rfc__n_estimators'] = randint(n_est, n_est_lim)
        else:
            dict_param_distr['rfc__n_estimators'] = [n_est]
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_rfc()


    def correct_lim_rfc_n():
        global dict_param_distr
        if not ui_rs.radioButton_rfc.isChecked() or not ui_rs.checkBox_rfc_n_lim.isChecked():
            return
        n_est = ui_rs.spinBox_rfc_n.value()
        n_est_lim = ui_rs.spinBox_rfc_n_lim.value()
        dict_param_distr['rfc__n_estimators'] = randint(n_est, n_est_lim)
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_rfc()


### GBC ###

    def show_dict_param_distr_gbc():
        global dict_param_distr
        ui_rs.textEdit.clear()
        if ui_rs.checkBox_lerning_rate_lim.isChecked():
            add_text('gbc__learning_rate', f'uniform({round(ui_rs.doubleSpinBox_learning_rate.value(), 2)}, '
                                       f'{round(ui_rs.doubleSpinBox_learning_rate_lim.value(), 2)})')
        else:
            add_text('gbc__learning_rate', dict_param_distr['gbc__learning_rate'])
        if ui_rs.checkBox_n_estimator_lim.isChecked():
            add_text('gbc__n_estimators', f'randint({ui_rs.spinBox_n_estimators.value()}, '
                                       f'{ui_rs.spinBox_n_estimators_lim.value()})')
        else:
            add_text('gbc__n_estimators', dict_param_distr['gbc__n_estimators'])


    def click_gbc():
        global dict_param_distr
        ui_rs.checkBox_lerning_rate_lim.setChecked(False)
        ui_rs.checkBox_n_estimator_lim.setChecked(False)
        dict_param_distr = {'gbc__learning_rate': [ui_rs.doubleSpinBox_learning_rate.value()],
                            'gbc__n_estimators': [ui_rs.spinBox_n_estimators.value()]}
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_gbc()


    def add_lr():
        global dict_param_distr
        if not ui_rs.radioButton_gbc.isChecked():
            return
        lr = round(ui_rs.doubleSpinBox_learning_rate.value(), 2)
        if ui_rs.checkBox_lerning_rate_lim.isChecked():
            dict_param_distr['gbc__learning_rate'] = [lr]
            ui_rs.checkBox_lerning_rate_lim.setChecked(False)
        else:
            if not lr in dict_param_distr['gbc__learning_rate']:
                dict_param_distr['gbc__learning_rate'].append(lr)
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_gbc()


    def lim_lr():
        global dict_param_distr
        if not ui_rs.radioButton_gbc.isChecked():
            return
        lr = round(ui_rs.doubleSpinBox_learning_rate.value(), 2)
        if ui_rs.checkBox_lerning_rate_lim.isChecked():
            lr_lim = round(ui_rs.doubleSpinBox_learning_rate_lim.value(), 2)
            dict_param_distr['gbc__learning_rate'] = uniform(lr, lr_lim)
        else:
            dict_param_distr['gbc__learning_rate'] = [lr]
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_gbc()


    def correct_lim_lr():
        global dict_param_distr
        if not ui_rs.radioButton_gbc.isChecked() or not ui_rs.checkBox_lerning_rate_lim.isChecked():
            return
        lr = round(ui_rs.doubleSpinBox_learning_rate.value(), 2)
        lr_lim = round(ui_rs.doubleSpinBox_learning_rate_lim.value(), 2)
        dict_param_distr['gbc__learning_rate'] = uniform(lr, lr_lim)
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_gbc()


    def add_gbc_n():
        global dict_param_distr
        if not ui_rs.radioButton_gbc.isChecked():
            return
        n_est = ui_rs.spinBox_n_estimators.value()
        if ui_rs.checkBox_n_estimator_lim.isChecked():
            dict_param_distr['gbc__n_estimators'] = [n_est]
        else:
            if not n_est in dict_param_distr['gbc__n_estimators']:
                dict_param_distr['gbc__n_estimators'].append(n_est)
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_gbc()


    def lim_gbc_n():
        global dict_param_distr
        if not ui_rs.radioButton_gbc.isChecked():
            return
        n_est = ui_rs.spinBox_n_estimators.value()
        if ui_rs.checkBox_n_estimator_lim.isChecked():
            n_est_lim = ui_rs.spinBox_n_estimators_lim.value()
            dict_param_distr['gbc__n_estimators'] = randint(n_est, n_est_lim)
        else:
            dict_param_distr['gbc__n_estimators'] = [n_est]
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_gbc()


    def correct_lim_gbc_n():
        global dict_param_distr
        if not ui_rs.radioButton_gbc.isChecked() or not ui_rs.checkBox_n_estimator_lim.isChecked():
            return
        n_est = ui_rs.spinBox_n_estimators.value()
        n_est_lim = ui_rs.spinBox_n_estimators_lim.value()
        dict_param_distr['gbc__n_estimators'] = randint(n_est, n_est_lim)
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_gbc()


### MLP ###

    def show_dict_param_distr_mlp():
        global dict_param_distr
        ui_rs.textEdit.clear()
        if ui_rs.checkBox_random_hidden_layer.isChecked():
            add_text('mlp__hidden_layer_sizes', f'hidden_layer({ui_rs.spinBox_min_neuro.value()}, '
                                                f'{ui_rs.spinBox_max_neuro.value()}), sizes('
                                                f'{ui_rs.spinBox_min_layer.value()}, '
                                                f'{ui_rs.spinBox_max_layer.value()})')
        else:
            add_text('mlp__hidden_layer_sizes', dict_param_distr['mlp__hidden_layer_sizes'])
        add_text('mlp__activation', dict_param_distr['mlp__activation'])
        add_text('mlp__solver', dict_param_distr['mlp__solver'])
        if ui_rs.checkBox_mlp_alpha_lim.isChecked():
            add_text('mlp__alpha', f'uniform({round(ui_rs.doubleSpinBox_alpha_mlp.value(), 5)}, '
                                   f'{round(ui_rs.doubleSpinBox_alpha_mlp_lim.value(), 5)})')
        else:
            add_text('mlp__alpha', dict_param_distr['mlp__alpha'])


    def click_mlp():
        global dict_param_distr
        ui_rs.checkBox_mlp_alpha_lim.setChecked(False)
        ui_rs.checkBox_random_hidden_layer.setChecked(False)
        dict_param_distr = {'mlp__hidden_layer_sizes': [tuple(map(int, ui_rs.lineEdit_layer_mlp.text().split()))],
                            'mlp__activation': [ui_rs.comboBox_activation_mlp.currentText()],
                            'mlp__solver': [ui_rs.comboBox_solvar_mlp.currentText()],
                            'mlp__alpha': [round(ui_rs.doubleSpinBox_alpha_mlp.value(), 5)]}
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_mlp()


    def add_hidden_layer():
        global dict_param_distr
        if ui_rs.lineEdit_layer_mlp.text() == '' or not ui_rs.radioButton_mlp.isChecked():
            return
        if ui_rs.checkBox_random_hidden_layer.isChecked():
            ui_rs.checkBox_random_hidden_layer.setChecked(False)
            dict_param_distr['mlp__hidden_layer_sizes'] = [tuple(map(int, ui_rs.lineEdit_layer_mlp.text().split()))]
        else:
            if tuple(map(int, ui_rs.lineEdit_layer_mlp.text().split())) not in dict_param_distr['mlp__hidden_layer_sizes']:
                dict_param_distr['mlp__hidden_layer_sizes'].append(tuple(map(int, ui_rs.lineEdit_layer_mlp.text().split())))
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_mlp()


    def add_random_hidden_layer():
        global dict_param_distr
        if not ui_rs.radioButton_mlp.isChecked():
            return
        if ui_rs.checkBox_random_hidden_layer.isChecked():
            min_neuro = ui_rs.spinBox_min_neuro.value()
            max_neuro = ui_rs.spinBox_max_neuro.value()
            min_layer = ui_rs.spinBox_min_layer.value()
            max_layer = ui_rs.spinBox_max_layer.value()
            n_iter = ui_rs.spinBox_n_iter.value()
            dict_param_distr['mlp__hidden_layer_sizes'] = [(randint.rvs(min_neuro, max_neuro),) * randint.rvs(min_layer, max_layer) for _ in range(n_iter * 2)]
        else:
            dict_param_distr['mlp__hidden_layer_sizes'] = [tuple(map(int, ui_rs.lineEdit_layer_mlp.text().split()))]
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_mlp()


    def add_activation():
        global dict_param_distr
        if not ui_rs.radioButton_mlp.isChecked():
            return
        if ui_rs.comboBox_activation_mlp.currentText() not in dict_param_distr['mlp__activation']:
            dict_param_distr['mlp__activation'].append(ui_rs.comboBox_activation_mlp.currentText())
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_mlp()


    def add_solver():
        global dict_param_distr
        if not ui_rs.radioButton_mlp.isChecked():
            return
        if ui_rs.comboBox_solvar_mlp.currentText() not in dict_param_distr['mlp__solver']:
            dict_param_distr['mlp__solver'].append(ui_rs.comboBox_solvar_mlp.currentText())
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_mlp()


    def add_alpha():
        global dict_param_distr
        if not ui_rs.radioButton_mlp.isChecked():
            return
        alpha = round(ui_rs.doubleSpinBox_alpha_mlp.value(), 5)
        if ui_rs.checkBox_mlp_alpha_lim.isChecked():
            dict_param_distr['mlp__alpha'] = [alpha]
            ui_rs.checkBox_mlp_alpha_lim.setChecked(False)
        else:
            if alpha not in dict_param_distr['mlp__alpha']:
                dict_param_distr['mlp__alpha'].append(alpha)
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_mlp()


    def lim_alpha():
        global dict_param_distr
        if not ui_rs.radioButton_mlp.isChecked():
            return
        alpha = round(ui_rs.doubleSpinBox_alpha_mlp.value(), 5)
        if ui_rs.checkBox_mlp_alpha_lim.isChecked():
            alpha_lim = round(ui_rs.doubleSpinBox_alpha_mlp_lim.value(), 5)
            dict_param_distr['mlp__alpha'] = uniform(alpha, alpha_lim)
        else:
            dict_param_distr['mlp__alpha'] = [alpha]
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_mlp()


    def correct_alpha():
        global dict_param_distr
        if not ui_rs.radioButton_mlp.isChecked() or not ui_rs.checkBox_mlp_alpha_lim.isChecked():
            return
        alpha = round(ui_rs.doubleSpinBox_alpha_mlp.value(), 5)
        alpha_lim = round(ui_rs.doubleSpinBox_alpha_mlp_lim.value(), 5)
        dict_param_distr['mlp__alpha'] = uniform(alpha, alpha_lim)
        show_dict_param_distr_pca() if ui_rs.checkBox_pca.isChecked() else show_dict_param_distr_mlp()

### PCA ###

    def show_dict_param_distr_pca():
        global dict_param_distr
        if ui_rs.radioButton_mlp.isChecked():
            show_dict_param_distr_mlp()
        if ui_rs.radioButton_gbc.isChecked():
            show_dict_param_distr_gbc()
        if ui_rs.radioButton_qda.isChecked():
            show_dict_param_distr_qda()
        if ui_rs.radioButton_svc.isChecked():
            show_dict_param_distr_svc()
        if ui_rs.radioButton_knn.isChecked():
            show_dict_param_distr_knn()
        if ui_rs.radioButton_gnb.isChecked():
            show_dict_param_distr_gnb()
        if ui_rs.radioButton_rfc.isChecked():
            show_dict_param_distr_rfc()
        if ui_rs.checkBox_pca.isChecked():
            if ui_rs.checkBox_pca_lim.isChecked():
                add_text('pca__n_components', f'randint({ui_rs.spinBox_pca.value()}, '
                                           f'{ui_rs.spinBox_pca_lim.value()})')
            else:
                add_text('pca__n_components', dict_param_distr['pca__n_components'])


    def click_pca():
        global dict_param_distr
        ui_rs.checkBox_pca_lim.setChecked(False)
        ui_rs.checkBox_pca_mle.setChecked(False)
        if ui_rs.checkBox_pca.isChecked():
            dict_param_distr['pca__n_components'] = [ui_rs.spinBox_pca.value()]
        else:
            del dict_param_distr['pca__n_components']
        show_dict_param_distr_pca()


    def add_pca():
        global dict_param_distr
        if not ui_rs.checkBox_pca.isChecked():
            return
        n_comp = 'mle' if ui_rs.checkBox_pca_mle.isChecked() else ui_rs.spinBox_pca.value()
        if n_comp not in dict_param_distr['pca__n_components']:
            dict_param_distr['pca__n_components'].append(n_comp)
        show_dict_param_distr_pca()


    def lim_pca():
        global dict_param_distr
        if not ui_rs.checkBox_pca.isChecked():
            return
        n_comp = ui_rs.spinBox_pca.value()
        if ui_rs.checkBox_pca_lim.isChecked():
            n_comp_lim = ui_rs.spinBox_pca_lim.value()
            dict_param_distr['pca__n_components'] = randint(n_comp, n_comp_lim)
        else:
            n_comp = 'mle' if ui_rs.checkBox_pca_mle.isChecked() else ui_rs.spinBox_pca.value()
            dict_param_distr['pca__n_components'] = [n_comp]
        show_dict_param_distr_pca()


    def correct_lim_pca():
        global dict_param_distr
        if not ui_rs.checkBox_pca.isChecked() or not ui_rs.checkBox_pca_lim.isChecked():
            return
        n_comp = ui_rs.spinBox_pca.value()
        n_comp_lim = ui_rs.spinBox_pca_lim.value()
        dict_param_distr['pca__n_components'] = randint(n_comp, n_comp_lim)
        show_dict_param_distr_pca()


    def push_checkbutton_extra():
        if ui_rs.checkBox_rfc_ada.isChecked():
            ui_rs.checkBox_rfc_ada.setChecked(False)

    def push_checkbutton_ada():
        if ui_rs.checkBox_rfc_extra.isChecked():
            ui_rs.checkBox_rfc_extra.setChecked(False)


    def calc_random_search():
        """ Создание и тренировка модели """
        global dict_param_distr

        start_time = datetime.datetime.now()
        # Нормализация данных
        scaler = StandardScaler()

        pipe_steps = []
        pipe_steps.append(('scaler', scaler))

        if ui_rs.checkBox_pca.isChecked():
            pca = PCA(random_state=0)
            pipe_steps.append(('pca', pca))

        if ui_rs.radioButton_mlp.isChecked():
            mlp = MLPClassifier(
                max_iter=5000,
                early_stopping=True,
                validation_fraction=0.2,
                random_state=0)
            pipe_steps.append(('mlp', mlp))
        if ui_rs.radioButton_knn.isChecked():
            knn = KNeighborsClassifier(algorithm='auto')
            pipe_steps.append(('knn', knn))
        if ui_rs.radioButton_gbc.isChecked():
            gbc = GradientBoostingClassifier(random_state=0)
            pipe_steps.append(('gbc', gbc))
        if ui_rs.radioButton_gnb.isChecked():
            gnb = GaussianNB()
            pipe_steps.append(('gnb', gnb))
        if ui_rs.radioButton_rfc.isChecked():
            if ui_rs.checkBox_rfc_ada.isChecked():
                rfc = AdaBoostClassifier(random_state=0)
                pipe_steps.append(('rfc', rfc))
            elif ui_rs.checkBox_rfc_extra.isChecked():
                rfc = ExtraTreesClassifier(random_state=0, n_jobs=-1)
                pipe_steps.append(('rfc', rfc))
            else:
                rfc = RandomForestClassifier(random_state=0, n_jobs=-1)
                pipe_steps.append(('rfc', rfc))
        if ui_rs.radioButton_qda.isChecked():
            qda = QuadraticDiscriminantAnalysis()
            pipe_steps.append(('qda', qda))
        if ui_rs.radioButton_svc.isChecked():
            svc = SVC(random_state=0, probability=True)
            pipe_steps.append(('svc', svc))

        pipe = Pipeline(pipe_steps)

        # Создание экземпляра RandomizedSearchCV
        kf = KFold(n_splits=ui_rs.spinBox_n_cross_val.value(), shuffle=True, random_state=0)
        random_search = RandomizedSearchCV(
            pipe,
            param_distributions=dict_param_distr,
            n_iter=ui_rs.spinBox_n_iter.value(),
            cv=kf,
            scoring='accuracy',
            verbose=6,
            random_state=0,
            n_jobs=-1)

        # Выполнение поиска
        random_search.fit(training_sample, markup)

        time_search = datetime.datetime.now() - start_time

        # Лучшие найденные параметры и оценка

        print("Лучшие параметры:", random_search.best_params_)
        print("Лучшая оценка:", random_search.best_score_)
        print("Количество cv:", random_search.n_splits_)
        print("Лучший индекс:", random_search.best_index_)
        print("Качество:", random_search.scorer_)
        # print(random_search.cv_results_)
        # Получение результатов из RandomizedSearchCV
        results = random_search.cv_results_

        # Создание DataFrame из результатов
        df_results = pd.DataFrame(results)
        df_graph = pd.DataFrame()
        df_graph['mean_test_score'] = df_results['mean_test_score']
        df_graph['mean_fit_time'] = df_results['mean_fit_time']
        if ui_rs.checkBox_random_hidden_layer.isChecked() and ui_rs.radioButton_mlp.isChecked():
            list_layers, list_percep = [], []
            for i in df_results['param_mlp__hidden_layer_sizes']:
                list_layers.append(len(i))
                list_percep.append(i[0])
            df_graph['n_layers'] = list_layers
            df_graph['n_perceptron'] = list_percep
        else:
            for col in df_results.columns:
                if not col.startswith('param_'):
                    continue
                # Конвертируем значения в числовой формат, игнорируя ошибки
                numeric_values = pd.to_numeric(df_results[col], errors='coerce')
                if not numeric_values.isnull().any():
                    df_graph[col] = numeric_values



        if len(df_graph.columns) > 3:
            col_1, col_2 = df_graph.columns[2], df_graph.columns[3]
            fig, axes = plt.subplots(nrows=2, ncols=2)
            fig.set_size_inches(15, 10)
            fig.suptitle(f'RandomizedSearchCV Results\n'
                         f'Лучшие параметры: {random_search.best_params_}\n'
                         f'Лучшая оценка: {random_search.best_score_}\n'
                         f'Время поиска: {time_search}')
            sns.scatterplot(df_graph, x=col_1, y='mean_test_score', hue='mean_fit_time', size='mean_fit_time', sizes=(5, 250), palette='brg', ax=axes[0, 0])
            sns.regplot(df_graph, x=col_1, y='mean_test_score', color='red', scatter=False, ax=axes[0, 0])
            axes[0, 0].set_xlabel(col_1)
            axes[0, 0].set_ylabel('Mean Test Score (Accuracy)')
            axes[0, 0].grid(True)
            plt.grid(True)
            sns.scatterplot(df_graph, x=col_2, y='mean_test_score', hue='mean_fit_time', size='mean_fit_time', sizes=(5, 250), palette='brg', ax=axes[1, 0])
            sns.regplot(df_graph, x=col_2, y='mean_test_score', color='red', scatter=False, ax=axes[1, 0])
            axes[1, 0].set_xlabel(col_2)
            axes[1, 0].set_ylabel('Mean Test Score (Accuracy)')
            axes[1, 0].grid(True)
            sns.histplot(df_graph, x='mean_test_score', kde=True, ax=axes[0, 1])
            axes[0, 1].set_xlabel('Mean Test Score (Accuracy)')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].grid(True)

            sns.scatterplot(df_graph, x=col_1, y=col_2, hue='mean_test_score', size='mean_test_score', sizes=(5, 250), palette='brg', ax=axes[1, 1])

            # df_graph = df_graph.drop_duplicates(subset=[col_1, col_2])
            # heatmap_data = df_graph.pivot(index=col_1, columns=col_2, values='mean_test_score').round(2)
            # sns.heatmap(heatmap_data, cmap='jet', ax=axes[1, 1])
            axes[1, 1].set_ylabel(col_1)
            axes[1, 1].set_xlabel(col_2)
            axes[1, 1].grid(True)
        else:
            col = df_graph.columns[2]
            fig, axes = plt.subplots(nrows=1, ncols=2)
            fig.set_size_inches(15, 5)
            fig.suptitle(f'RandomizedSearchCV Results\n'
                         f'Лучшие параметры: {random_search.best_params_}\n'
                         f'Лучшая оценка: {random_search.best_score_}\n'
                         f'Время поиска: {time_search}')
            sns.scatterplot(df_graph, x=col, y='mean_test_score', hue='mean_fit_time', size='mean_fit_time', sizes=(5, 250), palette='brg', ax=axes[0])
            sns.regplot(df_graph, x=col, y='mean_test_score', color='red', scatter=False, ax=axes[0])
            axes[0].set_xlabel(col)
            axes[0].set_ylabel('Mean Test Score (Accuracy)')
            axes[0].grid(True)

            sns.histplot(df_graph, x='mean_test_score', kde=True, ax=axes[1])
            axes[1].set_xlabel('Mean Test Score (Accuracy)')
            axes[1].set_ylabel('Count')
            axes[1].grid(True)
        fig.tight_layout()
        fig.show()


    click_knn()

    ui_rs.pushButton_calc.clicked.connect(calc_random_search)
    ui_rs.checkBox_rfc_ada.clicked.connect(push_checkbutton_ada)
    ui_rs.checkBox_rfc_extra.clicked.connect(push_checkbutton_extra)

    ui_rs.radioButton_knn.clicked.connect(click_knn)
    ui_rs.toolButton_add_neighbors.clicked.connect(add_neighbors)
    ui_rs.checkBox_neighbors_lim.clicked.connect(lim_neighbors)
    ui_rs.spinBox_neighbors_lim.valueChanged.connect(correct_lim_neighbors)
    ui_rs.spinBox_neighbors.valueChanged.connect(correct_lim_neighbors)
    ui_rs.toolButton_add_weight.clicked.connect(add_weights)

    ui_rs.radioButton_svc.clicked.connect(click_svc)
    ui_rs.toolButton_add_kernel.clicked.connect(add_kernel)
    ui_rs.toolButton_add_c.clicked.connect(add_c)
    ui_rs.checkBox_svc_lim.clicked.connect(lim_c)
    ui_rs.doubleSpinBox_svr_c.valueChanged.connect(correct_lim_c)
    ui_rs.doubleSpinBox_svr_c_lim.valueChanged.connect(correct_lim_c)

    ui_rs.radioButton_qda.clicked.connect(click_qda)
    ui_rs.toolButton_add_reg_param.clicked.connect(add_reg_param)
    ui_rs.checkBox_reg_param_lim.clicked.connect(lim_reg_param)
    ui_rs.doubleSpinBox_qda_reg_param.valueChanged.connect(correct_lim_reg_param)
    ui_rs.doubleSpinBox_qda_reg_param_lim.valueChanged.connect(correct_lim_reg_param)

    ui_rs.radioButton_gnb.clicked.connect(click_gnb)
    ui_rs.toolButton_add_var_smooth.clicked.connect(add_var_smooth)
    ui_rs.checkBox_var_smooth_lim.clicked.connect(lim_var_smooth)
    ui_rs.spinBox_gnb_var_smooth.valueChanged.connect(correct_lim_var_smooth)
    ui_rs.spinBox_gnb_var_smooth_lim.valueChanged.connect(correct_lim_var_smooth)

    ui_rs.radioButton_rfc.clicked.connect(click_rfc)
    ui_rs.toolButton_add_rfc_n.clicked.connect(add_rfc_n)
    ui_rs.checkBox_rfc_n_lim.clicked.connect(lim_rfc_n)
    ui_rs.spinBox_rfc_n.valueChanged.connect(correct_lim_rfc_n)
    ui_rs.spinBox_rfc_n_lim.valueChanged.connect(correct_lim_rfc_n)

    ui_rs.radioButton_gbc.clicked.connect(click_gbc)
    ui_rs.toolButton_add_lr.clicked.connect(add_lr)
    ui_rs.checkBox_lerning_rate_lim.clicked.connect(lim_lr)
    ui_rs.doubleSpinBox_learning_rate.valueChanged.connect(correct_lim_lr)
    ui_rs.doubleSpinBox_learning_rate_lim.valueChanged.connect(correct_lim_lr)
    ui_rs.toolButton_add_gbc_n.clicked.connect(add_gbc_n)
    ui_rs.checkBox_n_estimator_lim.clicked.connect(lim_gbc_n)
    ui_rs.spinBox_n_estimators.valueChanged.connect(correct_lim_gbc_n)
    ui_rs.spinBox_n_estimators_lim.valueChanged.connect(correct_lim_gbc_n)

    ui_rs.radioButton_mlp.clicked.connect(click_mlp)
    ui_rs.toolButton_add_hidden_layer.clicked.connect(add_hidden_layer)
    ui_rs.checkBox_random_hidden_layer.clicked.connect(add_random_hidden_layer)
    ui_rs.toolButton_add_activate.clicked.connect(add_activation)
    ui_rs.toolButton_add_solver.clicked.connect(add_solver)
    ui_rs.toolButton_add_alpha.clicked.connect(add_alpha)
    ui_rs.checkBox_mlp_alpha_lim.clicked.connect(lim_alpha)
    ui_rs.doubleSpinBox_alpha_mlp.valueChanged.connect(correct_alpha)
    ui_rs.doubleSpinBox_alpha_mlp_lim.valueChanged.connect(correct_alpha)

    ui_rs.checkBox_pca.clicked.connect(click_pca)
    ui_rs.toolButton_add_pca.clicked.connect(add_pca)
    ui_rs.checkBox_pca_lim.clicked.connect(lim_pca)
    ui_rs.spinBox_pca.valueChanged.connect(correct_lim_pca)
    ui_rs.spinBox_pca_lim.valueChanged.connect(correct_lim_pca)


    RandomSearch.exec_()