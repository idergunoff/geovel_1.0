from scipy.stats import randint, uniform
from scipy.stats._distn_infrastructure import rv_discrete_frozen, rv_continuous_frozen
from func import *


def push_random_search():
    """ Старт окна RandomizedSearchCV """
    global dict_param_distr
    RandomSearch = QtWidgets.QDialog()
    ui_rs = Ui_RandomSearchForm()
    ui_rs.setupUi(RandomSearch)
    RandomSearch.show()
    RandomSearch.setAttribute(QtCore.Qt.WA_DeleteOnClose)

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
        show_dict_param_distr_knn()


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
        show_dict_param_distr_knn()


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
        show_dict_param_distr_knn()


    def correct_lim_neighbors():
        global dict_param_distr
        if not ui_rs.radioButton_knn.isChecked() or not ui_rs.checkBox_neighbors_lim.isChecked():
            return
        n_knn = ui_rs.spinBox_neighbors.value()
        n_knn_lim = ui_rs.spinBox_neighbors_lim.value()
        dict_param_distr['knn__n_neighbors'] = randint(n_knn, n_knn_lim)
        show_dict_param_distr_knn()


    def add_weights():
        global dict_param_distr
        if not ui_rs.radioButton_knn.isChecked():
            return
        weights_knn = 'distance' if ui_rs.checkBox_knn_weights.isChecked() else 'uniform'
        if not weights_knn in dict_param_distr['knn__weights']:
            dict_param_distr['knn__weights'].append(weights_knn)
        show_dict_param_distr_knn()


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
        show_dict_param_distr_svc()


    def add_kernel():
        global dict_param_distr
        if not ui_rs.radioButton_svc.isChecked():
            return
        if not ui_rs.comboBox_svr_kernel.currentText() in dict_param_distr['svc__kernel']:
            dict_param_distr['svc__kernel'].append(ui_rs.comboBox_svr_kernel.currentText())
        show_dict_param_distr_svc()


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
        show_dict_param_distr_svc()



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
        show_dict_param_distr_svc()


    def correct_lim_c():
        global dict_param_distr
        if not ui_rs.radioButton_svc.isChecked() or not ui_rs.checkBox_svc_lim.isChecked():
            return
        svc_c = ui_rs.doubleSpinBox_svr_c.value()
        svc_c_lim = ui_rs.doubleSpinBox_svr_c_lim.value()
        dict_param_distr['svc__C'] = uniform(svc_c, svc_c_lim)
        show_dict_param_distr_svc()


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
        show_dict_param_distr_qda()


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
        show_dict_param_distr_qda()


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
        show_dict_param_distr_qda()


    def correct_lim_reg_param():
        global dict_param_distr
        if not ui_rs.radioButton_qda.isChecked() or not ui_rs.checkBox_reg_param_lim.isChecked():
            return
        reg_param = round(ui_rs.doubleSpinBox_qda_reg_param.value(), 2)
        reg_param_lim = round(ui_rs.doubleSpinBox_qda_reg_param_lim.value(), 2)
        dict_param_distr['qda__reg_param'] = uniform(reg_param, reg_param_lim)
        show_dict_param_distr_qda()


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
        show_dict_param_distr_gnb()


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
        show_dict_param_distr_gnb()


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
        show_dict_param_distr_gnb()


    def correct_lim_var_smooth():
        global dict_param_distr
        if not ui_rs.radioButton_gnb.isChecked() or not ui_rs.checkBox_var_smooth_lim.isChecked():
            return
        var_smooth = 10 ** (-ui_rs.spinBox_gnb_var_smooth.value())
        var_smooth_lim = 10 ** (-ui_rs.spinBox_gnb_var_smooth_lim.value())
        dict_param_distr['gnb__var_smoothing'] = uniform(var_smooth, var_smooth_lim)
        show_dict_param_distr_gnb()


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
        show_dict_param_distr_rfc()


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
        show_dict_param_distr_rfc()


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
        show_dict_param_distr_rfc()


    def correct_lim_rfc_n():
        global dict_param_distr
        if not ui_rs.radioButton_rfc.isChecked() or not ui_rs.checkBox_rfc_n_lim.isChecked():
            return
        n_est = ui_rs.spinBox_rfc_n.value()
        n_est_lim = ui_rs.spinBox_rfc_n_lim.value()
        dict_param_distr['rfc__n_estimators'] = randint(n_est, n_est_lim)
        show_dict_param_distr_rfc()


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
        show_dict_param_distr_gbc()


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
        show_dict_param_distr_gbc()


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
        show_dict_param_distr_gbc()


    def correct_lim_lr():
        global dict_param_distr
        if not ui_rs.radioButton_gbc.isChecked() or not ui_rs.checkBox_lerning_rate_lim.isChecked():
            return
        lr = round(ui_rs.doubleSpinBox_learning_rate.value(), 2)
        lr_lim = round(ui_rs.doubleSpinBox_learning_rate_lim.value(), 2)
        dict_param_distr['gbc__learning_rate'] = uniform(lr, lr_lim)
        show_dict_param_distr_gbc()


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
        show_dict_param_distr_gbc()


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
        show_dict_param_distr_gbc()


    def correct_lim_gbc_n():
        global dict_param_distr
        if not ui_rs.radioButton_gbc.isChecked() or not ui_rs.checkBox_n_estimator_lim.isChecked():
            return
        n_est = ui_rs.spinBox_n_estimators.value()
        n_est_lim = ui_rs.spinBox_n_estimators_lim.value()
        dict_param_distr['gbc__n_estimators'] = randint(n_est, n_est_lim)
        show_dict_param_distr_gbc()


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
            add_text('mlp__alpha', f'uniform({ui_rs.doubleSpinBox_alpha_mlp.value()}, '
                                   f'{ui_rs.doubleSpinBox_alpha_mlp_lim.value()})')
        else:
            add_text('mlp__alpha', dict_param_distr['mlp__alpha'])


    def click_mlp():
        global dict_param_distr
        ui_rs.checkBox_mlp_alpha_lim.setChecked(False)
        ui_rs.checkBox_random_hidden_layer.setChecked(False)
        dict_param_distr = {'mlp__hidden_layer_sizes': [tuple(map(int, ui_rs.lineEdit_layer_mlp.text().split()))],
                            'mlp__activation': [ui_rs.comboBox_activation_mlp.currentText()],
                            'mlp__solver': [ui_rs.comboBox_solvar_mlp.currentText()],
                            'mlp__alpha': [ui_rs.doubleSpinBox_alpha_mlp.value()]}
        show_dict_param_distr_mlp()


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
        show_dict_param_distr_mlp()


    def add_random_hidden_layer():
        global dict_param_distr
        if not ui_rs.radioButton_mlp.isChecked() or not ui_rs.checkBox_random_hidden_layer.isChecked():
            return
        min_neuro = ui_rs.spinBox_min_neuro.value()
        max_neuro = ui_rs.spinBox_max_neuro.value()
        min_layer = ui_rs.spinBox_min_layer.value()
        max_layer = ui_rs.spinBox_max_layer.value()
        n_iter = ui_rs.spinBox_n_iter.value()
        dict_param_distr['mlp__hidden_layer_sizes'] = [(randint.rvs(min_neuro, max_neuro),) * randint.rvs(min_layer, max_layer) for _ in range(n_iter ** 2)]
        show_dict_param_distr_mlp()


    click_knn()

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
    ui_rs.toolButton_add_activate.clicked.connect(add_activate)
    ui_rs.toolButton_add_solver.clicked.connect(add_solver)
    ui_rs.toolButton_add_alpha.clicked.connect(add_alpha)
    ui_rs.checkBox_mlp_alpha_lim.clicked.connect(lim_alpha)
    ui_rs.doubleSpinBox_alpha_mlp.valueChanged.connect(correct_alpha)
    ui_rs.doubleSpinBox_alpha_mlp_lim.valueChanged.connect(correct_alpha)


    RandomSearch.exec_()