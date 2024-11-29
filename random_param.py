from calc_additional_features import calc_wavelet_features, calc_fractal_features, calc_entropy_features, \
    calc_nonlinear_features, calc_morphology_features, calc_frequency_features, calc_envelope_feature, \
    calc_autocorr_feature, calc_emd_feature, calc_hht_features

from calc_profile_features import calc_wavelet_features_profile, calc_fractal_features_profile, calc_entropy_features_profile, \
    calc_nonlinear_features_profile, calc_morphology_features_profile, calc_frequency_features_profile, calc_envelope_feature_profile, \
    calc_autocorr_feature_profile, calc_emd_feature_profile, calc_hht_features_profile
from func import *

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, dropout_rate, activation_fn):
        super(Model, self).__init__()

        layers = []
        current_input_dim = input_dim

        # Словарь функций активации для простоты использования
        activation_dict = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU()
        }

        if activation_fn not in activation_dict:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

        activation_layer = activation_dict[activation_fn]

        for units in hidden_units:
            layers.append(nn.Linear(current_input_dim, units))
            layers.append(activation_layer)
            layers.append(nn.Dropout(dropout_rate))
            current_input_dim = units

        layers.append(nn.Linear(current_input_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()
        return self.model(x).squeeze(1).float()

def check_spinbox(spinbox1, spinbox2):
    spinbox1.setMaximum(spinbox2.value())
    spinbox2.setMinimum(spinbox1.value())


def random_combination(lst, n):
    """
    Возвращает случайную комбинацию n элементов из списка lst.
    """
    result = []
    remaining = lst[:]  # Создаем копию исходного списка

    for i in range(n):
        if not remaining:
            break  # Если оставшийся список пуст, выходим из цикла

        index = random.randint(0, len(remaining) - 1)
        result.append(remaining.pop(index))

    return result

def push_random_param():
    RandomParam = QtWidgets.QDialog()
    ui_rp = Ui_RandomParam()
    ui_rp.setupUi(RandomParam)
    RandomParam.show()
    RandomParam.setAttribute(Qt.WA_DeleteOnClose)

    m_width, m_height = get_width_height_monitor()
    RandomParam.resize(int(m_width / 1.5), int(m_height / 1.5))

    Classifier = QtWidgets.QDialog()
    ui_cls = Ui_ClassifierForm()
    ui_cls.setupUi(Classifier)
    Classifier.show()
    Classifier.setAttribute(Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия


    def push_checkbutton_smote():
        if ui_cls.checkBox_adasyn.isChecked():
            ui_cls.checkBox_adasyn.setChecked(False)

    def push_checkbutton_adasyn():
        if ui_cls.checkBox_smote.isChecked():
            ui_cls.checkBox_smote.setChecked(False)

    def choice_model_classifier(model):
        """ Выбор модели классификатора """
        if model == 'MLPC':
            model_class = MLPClassifier(
                hidden_layer_sizes=tuple(map(int, ui_cls.lineEdit_layer_mlp.text().split())),
                activation=ui_cls.comboBox_activation_mlp.currentText(),
                solver=ui_cls.comboBox_solvar_mlp.currentText(),
                alpha=ui_cls.doubleSpinBox_alpha_mlp.value(),
                learning_rate_init=ui_cls.doubleSpinBox_lr_mlp.value(),
                max_iter=5000,
                early_stopping=ui_cls.checkBox_e_stop_mlp.isChecked(),
                validation_fraction=ui_cls.doubleSpinBox_valid_mlp.value(),
                random_state=0
            )
            text_model = (f'**MLP**: \nhidden_layer_sizes: '
                          f'({",".join(map(str, tuple(map(int, ui_cls.lineEdit_layer_mlp.text().split()))))}), '
                          f'\nactivation: {ui_cls.comboBox_activation_mlp.currentText()}, '
                          f'\nsolver: {ui_cls.comboBox_solvar_mlp.currentText()}, '
                          f'\nalpha: {round(ui_cls.doubleSpinBox_alpha_mlp.value(), 2)}, '
                          f'\nlearning_rate: {round(ui_cls.doubleSpinBox_lr_mlp.value(), 5)}, '
                          f'\n{"early stopping, " if ui_cls.checkBox_e_stop_mlp.isChecked() else ""}'
                          f'\nvalidation_fraction: {round(ui_cls.doubleSpinBox_valid_mlp.value(), 2)}, ')

        elif model == 'KNNC':
            n_knn = ui_cls.spinBox_neighbors.value()
            weights_knn = 'distance' if ui_cls.checkBox_knn_weights.isChecked() else 'uniform'
            model_class = KNeighborsClassifier(n_neighbors=n_knn, weights=weights_knn, algorithm='auto')
            text_model = f'**KNN**: \nn_neighbors: {n_knn}, \nweights: {weights_knn}, '
        elif model == 'GBC':
            est = ui_cls.spinBox_n_estimators.value()
            l_rate = ui_cls.doubleSpinBox_learning_rate.value()
            model_class = GradientBoostingClassifier(n_estimators=est, learning_rate=l_rate, random_state=0)
            text_model = f'**GBC**: \nn estimators: {round(est, 2)}, \nlearning rate: {round(l_rate, 2)}, '
        elif model == 'G-NB':
            model_class = GaussianNB(var_smoothing=10 ** (-ui_cls.spinBox_gnb_var_smooth.value()))
            text_model = f'**G-NB**: \nvar smoothing: 1E-{str(ui_cls.spinBox_gnb_var_smooth.value())}, '

        elif model == 'DTC':
            spl = 'random' if ui_cls.checkBox_splitter_rnd.isChecked() else 'best'
            model_class = DecisionTreeClassifier(splitter=spl, random_state=0)
            text_model = f'**DTC**: \nsplitter: {spl}, '
        elif model == 'RFC':
            model_class = RandomForestClassifier(n_estimators=ui_cls.spinBox_rfc_n.value(), class_weight='balanced',
                                                 bootstrap=True, oob_score=True, random_state=0, n_jobs=-1)
            text_model = f'**RFC**: \nn estimators: {ui_cls.spinBox_rfc_n.value()}, '

        elif model == 'ABC':
            model_class = AdaBoostClassifier(n_estimators=ui_cls.spinBox_rfc_n.value(), random_state=0)
            text_model = f'**ABC**: \nn estimators: {ui_cls.spinBox_rfc_n.value()}, '

        elif model == 'ETC':
            model_class = ExtraTreesClassifier(n_estimators=ui_cls.spinBox_rfc_n.value(), class_weight='balanced',
                                               bootstrap=True, oob_score=True, random_state=0, n_jobs=-1)
            text_model = f'**ETC**: \nn estimators: {ui_cls.spinBox_rfc_n.value()}, '

        elif model == 'GPC':
            gpc_kernel_width = ui_cls.doubleSpinBox_gpc_wigth.value()
            gpc_kernel_scale = ui_cls.doubleSpinBox_gpc_scale.value()
            n_restart_optimization = ui_cls.spinBox_gpc_n_restart.value()
            multi_class = ui_cls.comboBox_gpc_multi.currentText()
            kernel = gpc_kernel_scale * RBF(gpc_kernel_width)
            model_class = GaussianProcessClassifier(
                kernel=kernel,
                n_restarts_optimizer=n_restart_optimization,
                random_state=0,
                multi_class=multi_class,
                n_jobs=-1
            )
            text_model = (
                f'**GPC**: \nwidth kernal: {round(gpc_kernel_width, 2)}, \nscale kernal: {round(gpc_kernel_scale, 2)}, '
                f'\nn restart: {n_restart_optimization}, \nmulti_class: {multi_class} ,')

        elif model == 'QDA':
            model_class = QuadraticDiscriminantAnalysis(reg_param=ui_cls.doubleSpinBox_qda_reg_param.value())
            text_model = f'**QDA**: \nreg_param: {round(ui_cls.doubleSpinBox_qda_reg_param.value(), 2)}, '

        elif model == 'SVC':
            model_class = SVC(kernel=ui_cls.comboBox_svr_kernel.currentText(), probability=True,
                              C=ui_cls.doubleSpinBox_svr_c.value(), random_state=0, class_weight='balanced')
            text_model = (f'**SVC**: \nkernel: {ui_cls.comboBox_svr_kernel.currentText()}, '
                          f'\nC: {round(ui_cls.doubleSpinBox_svr_c.value(), 2)}, ')

        elif model == 'LGBM':
            model_class = lgb.LGBMClassifier(
                objective='binary',
                verbosity=-1,
                boosting_type='gbdt',
                reg_alpha=ui_cls.doubleSpinBox_l1_lgbm.value(),
                reg_lambda=ui_cls.doubleSpinBox_l2_lgbm.value(),
                num_leaves=ui_cls.spinBox_lgbm_num_leaves.value(),
                colsample_bytree=ui_cls.doubleSpinBox_lgbm_feature.value(),
                subsample=ui_cls.doubleSpinBox_lgbm_subsample.value(),
                subsample_freq=ui_cls.spinBox_lgbm_sub_freq.value(),
                min_child_samples=ui_cls.spinBox_lgbm_child.value(),
                learning_rate=ui_cls.doubleSpinBox_lr_lgbm.value(),
                n_estimators=ui_cls.spinBox_estim_lgbm.value(),
            )

            text_model = f'**LGBM**: \nlambda_1: {ui_cls.doubleSpinBox_l1_lgbm.value()}, ' \
                         f'\nlambda_2: {ui_cls.doubleSpinBox_l2_lgbm.value()}, ' \
                         f'\nnum_leaves: {ui_cls.spinBox_lgbm_num_leaves.value()}, ' \
                         f'\nfeature_fraction: {ui_cls.doubleSpinBox_lgbm_feature.value()}, ' \
                         f'\nsubsample: {ui_cls.doubleSpinBox_lgbm_subsample.value()}, ' \
                         f'\nsubsample_freq: {ui_cls.spinBox_lgbm_sub_freq.value()}, ' \
                         f'\nmin_child_samples: {ui_cls.spinBox_lgbm_child.value()}, ' \
                         f'\nlearning_rate: {ui_cls.doubleSpinBox_lr_lgbm.value()}, ' \
                         f'\nn_estimators: {ui_cls.spinBox_estim_lgbm.value()}'

        else:
            model_class = QuadraticDiscriminantAnalysis()
            text_model = ''

        return model_class, text_model

    def set_marks():
        list_cat = [i.title for i in session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).all()]
        labels = {}
        labels[list_cat[0]] = 1
        labels[list_cat[1]] = 0
        if len(list_cat) > 2:
            for index, i in enumerate(list_cat[2:]):
                labels[i] = index
        return labels

    def build_torch_model(pipe_steps, x_train):
        labels = set_marks()
        output_dim = 1

        epochs = ui_cls.spinBox_epochs_torch.value()
        learning_rate = ui_cls.doubleSpinBox_lr_torch.value()
        hidden_units = list(map(int, ui_cls.lineEdit_layers_torch.text().split()))
        dropout_rate = ui_cls.doubleSpinBox_dropout_torch.value()
        weight_decay = ui_cls.doubleSpinBox_decay_torch.value()

        if ui_cls.comboBox_activation_torch.currentText() == 'ReLU':
            activation_function = 'relu'
        elif ui_cls.comboBox_activation_torch.currentText() == 'Sigmoid':
            activation_function = 'sigmoid'
        elif ui_cls.comboBox_activation_torch.currentText() == 'Tanh':
            activation_function = 'tanh'

        if ui_cls.comboBox_optimizer_torch.currentText() == 'Adam':
            optimizer = torch.optim.Adam
        elif ui_cls.comboBox_optimizer_torch.currentText() == 'SGD':
            optimizer = torch.optim.SGD

        if ui_cls.comboBox_loss_torch.currentText() == 'CrossEntropy':
            loss_function = nn.CrossEntropyLoss
            output_dim = 2
        elif ui_cls.comboBox_loss_torch.currentText() == 'BCEWithLogitsLoss':
            loss_function = nn.BCEWithLogitsLoss
            output_dim = 1
        elif ui_cls.comboBox_loss_torch.currentText() == 'BCELoss':
            loss_function = nn.BCELoss
            output_dim = 1

        patience = 0
        early_stopping_flag = False
        if ui_cls.checkBox_estop_torch.isChecked():
            early_stopping_flag = True
            patience = ui_cls.spinBox_stop_patience.value()

        early_stopping = EarlyStopping(
            monitor='valid_loss',
            patience=patience,
            threshold=1e-4,
            threshold_mode='rel',
            lower_is_better=True,
        )

        model = Model(x_train.shape[1], output_dim, hidden_units, dropout_rate, activation_function)

        net = NeuralNetClassifier(
            model,
            max_epochs=epochs,
            lr=learning_rate,
            optimizer=optimizer,
            criterion=loss_function,
            optimizer__weight_decay=weight_decay,
            iterator_train__batch_size=32,
            callbacks=[early_stopping] if early_stopping_flag else None,
            train_split=ValidSplit(cv=5),
            verbose=0
        )

        pipe_steps.append(('model', net))
        pipeline = Pipeline(pipe_steps)

        text_model = '*** TORCH NN *** \n' + 'learning_rate: ' + str(learning_rate) + '\nhidden units: ' + str(
            hidden_units) + '\nweight decay: ' + str(weight_decay) + '\ndropout rate: ' + str(dropout_rate) + \
                           '\nactivation_func: ' + activation_function  + '\noptimizer: ' + \
                            ui_cls.comboBox_optimizer_torch.currentText() + '\ncriterion: ' + \
                            ui_cls.comboBox_loss_torch.currentText() + '\nepochs: ' + str(epochs)

        return pipeline, text_model


    def check_checkbox_ts():
        push = True if ui_rp.checkBox_ts_all.isChecked() else False
        ui_rp.checkBox_ts_at.setChecked(push)
        ui_rp.checkBox_ts_a.setChecked(push)
        ui_rp.checkBox_ts_vt.setChecked(push)
        ui_rp.checkBox_ts_pht.setChecked(push)
        ui_rp.checkBox_ts_wt.setChecked(push)
        ui_rp.checkBox_ts_diff.setChecked(push)
        ui_rp.checkBox_ts_crlnf.setChecked(push)
        ui_rp.checkBox_ts_crl.setChecked(push)


    def check_checkbox_attr():
        push = True if ui_rp.checkBox_attr_all.isChecked() else False
        ui_rp.checkBox_attr_a.setChecked(push)
        ui_rp.checkBox_attr_at.setChecked(push)
        ui_rp.checkBox_attr_vt.setChecked(push)
        ui_rp.checkBox_attr_pht.setChecked(push)
        ui_rp.checkBox_attr_wt.setChecked(push)
        ui_rp.checkBox_attr_crl.setChecked(push)
        ui_rp.checkBox_form_t.setChecked(push)
        ui_rp.checkBox_grid.setChecked(push)
        ui_rp.checkBox_stat.setChecked(push)
        ui_rp.checkBox_crl_stat.setChecked(push)
        ui_rp.checkBox_wvt.setChecked(push)
        ui_rp.checkBox_fract.setChecked(push)
        ui_rp.checkBox_entr.setChecked(push)
        ui_rp.checkBox_mrph.setChecked(push)
        ui_rp.checkBox_env.setChecked(push)
        ui_rp.checkBox_nnl.setChecked(push)
        ui_rp.checkBox_atc.setChecked(push)
        ui_rp.checkBox_freq.setChecked(push)
        ui_rp.checkBox_emd.setChecked(push)
        ui_rp.checkBox_hht.setChecked(push)
        ui_rp.checkBox_wvt_prof.setChecked(push)
        ui_rp.checkBox_fract_prof.setChecked(push)
        ui_rp.checkBox_entr_prof.setChecked(push)
        ui_rp.checkBox_mrph_prof.setChecked(push)
        ui_rp.checkBox_env_prof.setChecked(push)
        ui_rp.checkBox_nnl_prof.setChecked(push)
        ui_rp.checkBox_atc_prof.setChecked(push)
        ui_rp.checkBox_freq_prof.setChecked(push)
        ui_rp.checkBox_emd_prof.setChecked(push)
        ui_rp.checkBox_hht_prof.setChecked(push)
        ui_rp.checkBox_land.setChecked(push)
        ui_rp.checkBox_xy.setChecked(push)


    def get_MLP_test_id():
        return ui_rp.comboBox_test_analysis.currentText().split(' id')[-1]


    def build_list_param():
        list_param_all = []

        n_distr = str(random.randint(ui_rp.spinBox_distr_up.value(), ui_rp.spinBox_distr_down.value()))
        n_sep = str(random.randint(ui_rp.spinBox_sep_up.value(), ui_rp.spinBox_sep_down.value()))
        n_mfcc = str(random.randint(ui_rp.spinBox_mfcc_up.value(), ui_rp.spinBox_mfcc_down.value()))
        n_sig_top = random.randint(ui_rp.spinBox_top_skip_up.value(), ui_rp.spinBox_top_skip_down.value())
        n_sig_bot = random.randint(ui_rp.spinBox_bot_skip_up.value(), ui_rp.spinBox_bot_skip_down.value())

        if ui_rp.checkBox_width.isChecked():
            if n_sig_top + n_sig_bot > 505:
                n_sig_bot = 0
            else:
                n_sig_bot = 511 - (n_sig_top + n_sig_bot)
        else:
            if n_sig_bot + n_sig_top > 505:
                n_sig_bot = random.randint(0, 505 - n_sig_top)

        def get_n(ts: str) -> str:
            if ts == 'distr':
                return str(n_distr)
            elif ts == 'sep':
                return str(n_sep)
            elif ts == 'mfcc':
                return str(n_mfcc)
            elif ts == 'sig':
                return f'{n_sig_top}_{n_sig_bot}'

        list_ts = []
        if ui_rp.checkBox_distr.isChecked():
            list_ts.append('distr')
        if ui_rp.checkBox_sep.isChecked():
            list_ts.append('sep')
        if ui_rp.checkBox_mfcc.isChecked():
            list_ts.append('mfcc')
        if ui_rp.checkBox_sig.isChecked():
            list_ts.append('sig')

        list_param_attr = []

        if ui_rp.checkBox_ts_a.isChecked():
            for i in list_ts:
                list_param_attr.append(f'{i}_Abase_{get_n(i)}')
        if ui_rp.checkBox_ts_at.isChecked():
            for i in list_ts:
                list_param_attr.append(f'{i}_At_{get_n(i)}')
        if ui_rp.checkBox_ts_vt.isChecked():
            for i in list_ts:
                list_param_attr.append(f'{i}_Vt_{get_n(i)}')
        if ui_rp.checkBox_ts_pht.isChecked():
            for i in list_ts:
                list_param_attr.append(f'{i}_Pht_{get_n(i)}')
        if ui_rp.checkBox_ts_wt.isChecked():
            for i in list_ts:
                list_param_attr.append(f'{i}_Wt_{get_n(i)}')
        if ui_rp.checkBox_ts_diff.isChecked():
            for i in list_ts:
                list_param_attr.append(f'{i}_diff_{get_n(i)}')
        if ui_rp.checkBox_ts_crlnf.isChecked():
            if ui_rp.checkBox_sig.isChecked():
                list_param_attr.append(f'sig_CRLNF_{n_sig_top}_{n_sig_bot}')
        if ui_rp.checkBox_ts_crl.isChecked():
            for i in list_ts:
                list_param_attr.append(f'{i}_CRL_{get_n(i)}')




        if ui_rp.checkBox_attr_a.isChecked():
            list_param_attr += ['A_top', 'A_bottom', 'dA', 'A_sum', 'A_mean', 'A_max', 'A_T_max', 'A_Sn', 'A_wmf',
                                'A_Qf', 'A_Sn_wmf']
        if ui_rp.checkBox_attr_at.isChecked():
            list_param_attr += ['At_top', 'dAt', 'At_sum', 'At_mean', 'At_max', 'At_T_max', 'At_Sn',
                                'At_wmf', 'At_Qf', 'At_Sn_wmf']
        if ui_rp.checkBox_attr_vt.isChecked():
            list_param_attr += ['Vt_top', 'dVt', 'Vt_sum', 'Vt_mean', 'Vt_max', 'Vt_T_max', 'Vt_Sn', 'Vt_wmf',
                                'Vt_Qf', 'Vt_Sn_wmf']
        if ui_rp.checkBox_attr_pht.isChecked():
            list_param_attr += ['Pht_top', 'dPht', 'Pht_sum', 'Pht_mean', 'Pht_max', 'Pht_T_max', 'Pht_Sn',
                                'Pht_wmf', 'Pht_Qf', 'Pht_Sn_wmf']
        if ui_rp.checkBox_attr_wt.isChecked():
            list_param_attr += ['Wt_top', 'Wt_mean', 'Wt_sum', 'Wt_max', 'Wt_T_max', 'Wt_Sn', 'Wt_wmf',
                                'Wt_Qf', 'Wt_Sn_wmf']
        if ui_rp.checkBox_attr_crl.isChecked():
            list_param_attr += ['CRL_top', 'CRL_sum', 'CRL_mean', 'CRL_max', 'CRL_T_max', 'CRL_Sn', 'CRL_wmf',
                                'CRL_Qf', 'CRL_Sn_wmf']

        if ui_rp.checkBox_stat.isChecked():
            list_param_attr += ['skew', 'kurt', 'std', 'k_var']
        if ui_rp.checkBox_crl_stat.isChecked():
            list_param_attr += ['CRL_skew', 'CRL_kurt', 'CRL_std', 'CRL_k_var']

        if ui_rp.checkBox_form_t.isChecked():
            list_param_attr += ['T_top', 'T_bottom', 'dT']
        if ui_rp.checkBox_grid.isChecked():
            list_param_attr += ['width', 'top', 'speed', 'speed_cover']
        if ui_rp.checkBox_land.isChecked():
            list_param_attr += ['land']
        if ui_rp.checkBox_xy.isChecked():
            list_param_attr += ['X', 'Y']
        if ui_rp.checkBox_wvt.isChecked():
            list_param_attr += list_wavelet_features
        if ui_rp.checkBox_fract.isChecked():
            list_param_attr += list_fractal_features
        if ui_rp.checkBox_entr.isChecked():
            list_param_attr += list_entropy_features
        if ui_rp.checkBox_nnl.isChecked():
            list_param_attr += list_nonlinear_features
        if ui_rp.checkBox_mrph.isChecked():
            list_param_attr += list_morphology_feature
        if ui_rp.checkBox_freq.isChecked():
            list_param_attr += list_frequency_feature
        if ui_rp.checkBox_env.isChecked():
            list_param_attr += list_envelope_feature
        if ui_rp.checkBox_atc.isChecked():
            list_param_attr += list_autocorr_feature
        if ui_rp.checkBox_emd.isChecked():
            list_param_attr += list_emd_feature
        if ui_rp.checkBox_hht.isChecked():
            list_param_attr += list_hht_feature
        if ui_rp.checkBox_wvt_prof.isChecked():
            for iwvt in list_wavelet_features:
                list_param_attr.append(f'prof_{iwvt}')
        if ui_rp.checkBox_fract_prof.isChecked():
            for ifract in list_fractal_features:
                if ifract == 'fractal_dim':
                    continue
                list_param_attr.append(f'prof_{ifract}')
        if ui_rp.checkBox_entr_prof.isChecked():
            for ient in list_entropy_features:
                list_param_attr.append(f'prof_{ient}')
        if ui_rp.checkBox_nnl_prof.isChecked():
            for innl in list_nonlinear_features:
                list_param_attr.append(f'prof_{innl}')
        if ui_rp.checkBox_mrph_prof.isChecked():
            for imrph in list_morphology_feature:
                list_param_attr.append(f'prof_{imrph}')
        if ui_rp.checkBox_freq_prof.isChecked():
            for ifreq in list_frequency_feature:
                list_param_attr.append(f'prof_{ifreq}')
        if ui_rp.checkBox_env_prof.isChecked():
            for ienv in list_envelope_feature:
                list_param_attr.append(f'prof_{ienv}')
        if ui_rp.checkBox_atc_prof.isChecked():
            for iatc in list_autocorr_feature:
                list_param_attr.append(f'prof_{iatc}')
        if ui_rp.checkBox_emd_prof.isChecked():
            for iemd in list_emd_feature:
                list_param_attr.append(f'prof_{iemd}')
        if ui_rp.checkBox_hht_prof.isChecked():
            for ihht in list_hht_feature:
                if ihht == 'hht_marg_spec_min':
                    continue
                list_param_attr.append(f'prof_{ihht}')

        n_param = random.randint(1, len(list_param_attr))
        list_param_all = random_combination(list_param_attr, n_param)

        print(len(list_param_all), list_param_all)
        return list_param_all


    def build_table_random_param(analisis_id: int, list_param: list) -> (pd.DataFrame, list):

        locals_dict = inspect.currentframe().f_back.f_locals #

        data_train = pd.DataFrame(columns=['prof_well_index', 'mark'])

        # Получаем размеченные участки
        markups = session.query(MarkupMLP).filter_by(analysis_id=analisis_id).all()

        ui.progressBar.setMaximum(len(markups))

        for nm, markup in enumerate(tqdm(markups)):
            # Получение списка фиктивных меток и границ слоев из разметки
            list_fake = json.loads(markup.list_fake) if markup.list_fake else []
            list_up = json.loads(markup.formation.layer_up.layer_line)
            list_down = json.loads(markup.formation.layer_down.layer_line)

            for measure in json.loads(markup.list_measure):
                # Пропустить измерение, если оно является фиктивным

                if measure in list_fake:
                    continue
                if not str(markup.profile.id) + '_Abase_' + str(measure) in locals_dict:
                    if not str(markup.profile.id) + '_signal' in locals():
                        locals()[str(markup.profile.id) + '_signal'] = json.loads(
                            session.query(Profile.signal).filter(Profile.id == markup.profile_id).first()[0])
                    # Загрузка сигнала из профиля
                    locals_dict.update(
                        {str(markup.profile.id) + '_Abase_' + str(measure):
                              locals()[str(markup.profile.id) + '_signal'][measure]}
                    )
                if not str(markup.profile.id) + '_diff_' + str(measure) in locals_dict:
                    locals_dict.update(
                        {str(markup.profile.id) + '_diff_' + str(measure):
                             calc_atrib_measure(locals_dict[str(markup.profile.id) + '_Abase_' + str(measure)], 'diff')}
                    )
                if not str(markup.profile.id) + '_At_' + str(measure) in locals_dict:
                    locals_dict.update(
                        {str(markup.profile.id) + '_At_' + str(measure):
                             calc_atrib_measure(locals_dict[str(markup.profile.id) + '_Abase_' + str(measure)], 'At')}
                    )
                if not str(markup.profile.id) + '_Vt_' + str(measure) in locals_dict:
                    locals_dict.update(
                        {str(markup.profile.id) + '_Vt_' + str(measure):
                             calc_atrib_measure(locals_dict[str(markup.profile.id) + '_Abase_' + str(measure)], 'Vt')}
                    )
                if not str(markup.profile.id) + '_Pht_' + str(measure) in locals_dict:
                    locals_dict.update(
                        {str(markup.profile.id) + '_Pht_' + str(measure):
                             calc_atrib_measure(locals_dict[str(markup.profile.id) + '_Abase_' + str(measure)], 'Pht')}
                    )
                if not str(markup.profile.id) + '_Wt_' + str(measure) in locals_dict:
                    locals_dict.update(
                        {str(markup.profile.id) + '_Wt_' + str(measure):
                             calc_atrib_measure(locals_dict[str(markup.profile.id) + '_Abase_' + str(measure)], 'Wt')}
                    )

                # if ui_rp.checkBox_ts_crl.isChecked():
                if not str(markup.profile.id) + '_CRL_' + str(measure) in locals_dict:
                    if not str(markup.profile.id) + '_CRL' in locals():
                        locals()[str(markup.profile.id) + '_CRL'] = calc_CRL_filter(
                            locals()[str(markup.profile.id) + '_signal'])

                    locals_dict.update(
                        {str(markup.profile.id) + '_CRL_' + str(measure):
                            locals()[str(markup.profile.id) + '_CRL'][measure]}
                        )
                # if ui_rp.checkBox_ts_crlnf.isChecked():
                if not str(markup.profile.id) + '_CRL_NF_' + str(measure) in locals_dict:
                    if not str(markup.profile.id) + '_CRLNF' in locals():
                        locals()[str(markup.profile.id) + '_CRLNF'] = calc_CRL(
                            locals()[str(markup.profile.id) + '_signal'])
                    locals_dict.update(
                        {str(markup.profile.id) + '_CRL_NF_' + str(measure):
                            locals()[str(markup.profile.id) + '_CRLNF'][measure]}
                    )

                for i in list_param:
                    if i == 'X':
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                   json.loads(session.query(Profile.x_pulc).filter(
                                       Profile.id == markup.profile_id
                                   ).first()[0])}
                            )
                    if i == 'Y':
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                   json.loads(session.query(Profile.y_pulc).filter(
                                       Profile.id == markup.profile_id
                                   ).first()[0])}
                            )
                    if i in list_param_geovel:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                   json.loads(session.query(literal_column(f'Formation.{i}')).filter(
                                       Formation.id == markup.formation_id
                                   ).first()[0])}
                            )
                    if i in list_wavelet_features:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_wavelet_features(markup.formation_id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                   json.loads(session.query(literal_column(f'wavelet_feature.{i}')).filter(
                                       WaveletFeature.formation_id == markup.formation_id
                                   ).first()[0])}
                            )
                    if i[5:] in list_wavelet_features:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_wavelet_features_profile(markup.profile.id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                   json.loads(session.query(literal_column(f'wavelet_feature_profile.{i[5:]}')).filter(
                                       WaveletFeatureProfile.profile_id == markup.profile.id
                                   ).first()[0])}
                            )
                    if i in list_fractal_features:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_fractal_features(markup.formation_id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                   json.loads(session.query(literal_column(f'fractal_feature.{i}')).filter(
                                       FractalFeature.formation_id == markup.formation_id
                                   ).first()[0])}
                            )

                    if i[5:] in list_fractal_features:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_fractal_features_profile(markup.profile.id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                     json.loads(session.query(literal_column(f'fractal_feature_profile.{i[5:]}')).filter(
                                         FractalFeatureProfile.profile_id == markup.profile.id
                                     ).first()[0])}
                            )
                    if i in list_entropy_features:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_entropy_features(markup.formation_id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                   json.loads(session.query(literal_column(f'entropy_feature.{i}')).filter(
                                       EntropyFeature.formation_id == markup.formation_id
                                   ).first()[0])}
                            )
                    if i[5:] in list_entropy_features:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_entropy_features_profile(markup.profile.id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                   json.loads(session.query(literal_column(f'entropy_feature_profile.{i[5:]}')).filter(
                                       EntropyFeatureProfile.profile_id == markup.profile.id
                                   ).first()[0])}
                            )
                    if i in list_nonlinear_features:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_nonlinear_features(markup.formation_id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                   json.loads(session.query(literal_column(f'nonlinear_feature.{i}')).filter(
                                       NonlinearFeature.formation_id == markup.formation_id
                                   ).first()[0])}
                            )
                    if i[5:] in list_nonlinear_features:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_nonlinear_features_profile(markup.profile.id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                   json.loads(session.query(literal_column(f'nonlinear_feature_profile.{i[5:]}')).filter(
                                       NonlinearFeatureProfile.profile_id == markup.profile.id
                                   ).first()[0])}
                            )
                    if i in list_morphology_feature:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_morphology_features(markup.formation_id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                   json.loads(session.query(literal_column(f'morphology_feature.{i}')).filter(
                                       MorphologyFeature.formation_id == markup.formation_id
                                   ).first()[0])}
                            )
                    if i[5:] in list_morphology_feature:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_morphology_features_profile(markup.profile.id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                   json.loads(session.query(literal_column(f'morphology_feature_profile.{i[5:]}')).filter(
                                       MorphologyFeatureProfile.profile_id == markup.profile.id
                                   ).first()[0])}
                            )
                    if i in list_frequency_feature:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_frequency_features(markup.formation_id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                   json.loads(session.query(literal_column(f'frequency_feature.{i}')).filter(
                                       FrequencyFeature.formation_id == markup.formation_id
                                   ).first()[0])}
                            )
                    if i[5:] in list_frequency_feature:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_frequency_features_profile(markup.profile.id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                   json.loads(session.query(literal_column(f'frequency_feature_profile.{i[5:]}')).filter(
                                       FrequencyFeatureProfile.profile_id == markup.profile.id
                                   ).first()[0])}
                            )
                    if i in list_envelope_feature:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_envelope_feature(markup.formation_id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                   json.loads(session.query(literal_column(f'envelope_feature.{i}')).filter(
                                       EnvelopeFeature.formation_id == markup.formation_id
                                   ).first()[0])}
                            )
                    if i[5:] in list_envelope_feature:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_envelope_feature_profile(markup.profile.id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                   json.loads(session.query(literal_column(f'envelope_feature_profile.{i[5:]}')).filter(
                                       EnvelopeFeatureProfile.profile_id == markup.profile.id
                                   ).first()[0])}
                            )
                    if i in list_autocorr_feature:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_autocorr_feature(markup.formation_id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                   json.loads(session.query(literal_column(f'autocorr_feature.{i}')).filter(
                                       AutocorrFeature.formation_id == markup.formation_id
                                   ).first()[0])}
                            )
                    if i[5:] in list_autocorr_feature:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_autocorr_feature_profile(markup.profile.id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                   json.loads(session.query(literal_column(f'autocorr_feature_profile.{i[5:]}')).filter(
                                       AutocorrFeatureProfile.profile_id == markup.profile.id
                                   ).first()[0])}
                            )
                    if i in list_emd_feature:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_emd_feature(markup.formation_id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                   json.loads(session.query(literal_column(f'emd_feature.{i}')).filter(
                                       EMDFeature.formation_id == markup.formation_id
                                   ).first()[0])}
                            )
                    if i[5:] in list_emd_feature:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_emd_feature_profile(markup.profile.id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                   json.loads(session.query(literal_column(f'emd_feature_profile.{i[5:]}')).filter(
                                       EMDFeatureProfile.profile_id == markup.profile.id
                                   ).first()[0])}
                            )
                    if i in list_hht_feature:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_hht_features(markup.formation_id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                   json.loads(session.query(literal_column(f'hht_feature.{i}')).filter(
                                       HHTFeature.formation_id == markup.formation_id
                                   ).first()[0])}
                            )
                    if i[5:] in list_hht_feature:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_hht_features_profile(markup.profile.id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                   json.loads(session.query(literal_column(f'hht_feature_profile.{i[5:]}')).filter(
                                       HHTFeatureProfile.profile_id == markup.profile.id
                                   ).first()[0])}
                            )

            # Обработка каждого измерения в разметке
            for measure in json.loads(markup.list_measure):
                # Пропустить измерение, если оно является фиктивным
                if measure in list_fake:
                    continue

                dict_value = {}
                dict_value['prof_well_index'] = f'{markup.profile_id}_{markup.well_id}_{measure}'
                dict_value['mark'] = markup.marker.title

                # Обработка каждого параметра в списке параметров
                for param in list_param:
                    if param.startswith('sig') or param.startswith('distr') or param.startswith('sep') or param.startswith('mfcc'):
                        if param.startswith('sig'):
                            p, atr, up, down = param.split('_')[0], param.split('_')[1], int(param.split('_')[2]), 511 - int(param.split('_')[3])

                        else:
                            p, atr, n = param.split('_')[0], param.split('_')[1], int(param.split('_')[2])

                        if atr == 'CRL':
                            sig_measure = locals_dict[str(markup.profile.id) + '_CRL_' + str(measure)]
                        elif atr == 'CRLNF':
                            sig_measure = locals_dict[str(markup.profile.id) + '_CRL_NF_' + str(measure)]
                        else:
                            sig_measure = locals_dict[str(markup.profile.id) + '_' + atr + '_' + str(measure)]

                        if p == 'sig':
                            for i_sig in range(len(sig_measure[up:down])):
                                dict_value[f'{p}_{atr}_{up + i_sig + 1}'] = sig_measure[i_sig]
                        elif p == 'distr':
                            distr = get_distribution(sig_measure[list_up[measure]: list_down[measure]], n)
                            for num in range(n):
                                dict_value[f'{p}_{atr}_{num + 1}'] = distr[num]
                        elif p == 'sep':
                            sep = get_interpolate_list(sig_measure[list_up[measure]: list_down[measure]], n)
                            for num in range(n):
                                dict_value[f'{p}_{atr}_{num + 1}'] = sep[num]
                        elif p == 'mfcc':
                            mfcc = get_mfcc(sig_measure[list_up[measure]: list_down[measure]], n)
                            for num in range(n):
                                dict_value[f'{p}_{atr}_{num + 1}'] = mfcc[num]

                    else:
                        # Загрузка значения параметра из списка значений
                        dict_value[param] = locals_dict[str(markup.profile.id) + '_' + param][measure]
                        # dict_value[param] = json.loads(session.query(literal_column(f'Formation.{param}')).filter(
                        # Formation.id == markup.formation_id).first()[0])[measure]

                # Добавление данных в обучающую выборку
                data_train = pd.concat([data_train, pd.DataFrame([dict_value])], ignore_index=True)

            ui.progressBar.setValue(nm + 1)

        new_list_param = []
        for param in list_param:
            if param.startswith('sig') or param.startswith('distr') or param.startswith('sep') or param.startswith('mfcc'):
                if param.startswith('sig'):
                    p, atr, up, down = param.split('_')[0], param.split('_')[1], int(param.split('_')[2]), 511 - int(param.split('_')[3])
                    for i_sig in range(up, down):
                        new_list_param.append(f'sig_{atr}_{i_sig + 1}')
                else:
                    p, atr, n = param.split('_')[0], param.split('_')[1], int(param.split('_')[2])
                    for i_sig in range(n):
                        new_list_param.append(f'{p}_{atr}_{i_sig + 1}')
            else:
                new_list_param.append(param)


        return data_train, new_list_param

    def test_classif_estimator(estimators, data_val, list_param, labels, filename):
        """

        """

        list_estimator_roc = []
        list_estimator_percent = []
        list_estimator_recall, list_estimator_precision = [], []
        list_estimator_f1 = []
        for class_model in estimators:
            time_start = datetime.datetime.now()
            data_test = data_val.copy()
            working_sample = data_test.iloc[:, 2:].values.tolist()
            list_cat = list(class_model['model'].classes_)
            if ui_cls.buttonGroup.checkedButton().text() == 'TORCH':
                list_cat = [labels[item] for item in list_cat]

            try:
                mark = class_model.predict(working_sample)
                probability = class_model.predict_proba(working_sample)
            except ValueError:
                working_sample = [[np.nan if np.isinf(x) else x for x in y] for y in working_sample]

                data = imputer.fit_transform(working_sample)
                try:
                    mark = class_model.predict(data)
                except ValueError:
                    set_info('Не совпадает количество признаков для данной модели. Выберите нужную модель и '
                             'рассчитайте заново', 'red')
                    QMessageBox.critical(MainWindow, 'Ошибка', 'Не совпадает количество признаков для данной модели.')
                    return
                probability = class_model.predict_proba(data)

                for i in data_test.index:
                    p_nan = [data_test.columns[ic + 3] for ic, v in enumerate(data_test.iloc[i, 3:].tolist()) if
                             np.isnan(v)]
                    if len(p_nan) > 0:
                        set_info(
                            f'Внимание для измерения "{i}" отсутствуют параметры "{", ".join(p_nan)}", поэтому расчёт для'
                            f' этого измерения может быть не корректен', 'red')

            predict_df = pd.DataFrame(probability.tolist(), columns=list_cat)
            predict_df.index = data_test.index
            data_test[list_cat[0]] = predict_df[list_cat[0]]
            data_test[list_cat[1]] = predict_df[list_cat[1]]
            if ui_cls.buttonGroup.checkedButton().text() == 'TORCH':
                data_test['mark_probability'] = [labels[item] for item in mark]
            else:
                data_test['mark_probability'] = mark
            data_test['совпадение'] = data_test['mark'].eq(data_test['mark_probability']).astype(int)
            correct_matches = data_test['совпадение'].sum()
            y_prob = np.array([i[0] for i in probability])
            if ui_cls.buttonGroup.checkedButton().text() == 'TORCH':
                y_val = data_test['mark'].replace({list_cat[0]: 1, list_cat[1]: 0}).to_list()
                y_pred = data_test['mark_probability'].replace({list_cat[0]: 1, list_cat[1]: 0}).to_list()
            else:
                y_val = data_test['mark'].to_list()
                y_pred = data_test['mark_probability'].to_list()
            fpr, tpr, thresholds = roc_curve(y_val, y_prob, pos_label=list_cat[0])
            roc_auc = auc(fpr, tpr)
            # print('roc_auc ', roc_auc)
            list_estimator_roc.append(roc_auc)
            recall = recall_score(y_val, y_pred, pos_label=list_cat[0])
            precision = precision_score(y_val, y_pred, pos_label=list_cat[0])
            f1 = f1_score(y_val, y_pred, pos_label=list_cat[0])
            list_estimator_recall.append(recall)
            list_estimator_precision.append(precision)
            list_estimator_f1.append(f1)

            ui_rp.textEdit_test_result.setTextColor(Qt.darkGreen)
            ui_rp.textEdit_test_result.append(f"{datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n"
                      f"Тестирование модели {class_model}\n"
                      f"Количество параметров: {len(list_param)}\n"
                      f"ROC AUC: {roc_auc:.3f}"
                      f'\nRecall: {recall:.3f}\nPrecision: {precision:.3f}\nF1: {f1:.3f}'
                      f'\nВсего совпало: {correct_matches}/{len(data_test)}\n')

            with open(filename, 'a') as f:
                print(f"{datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n"
                      f"Тестирование модели {class_model}\n"
                      f"Количество параметров: {len(list_param)}\n"
                      f"ROC AUC: {roc_auc:.3f}"
                      f'\nRecall: {recall:.3f}\nPrecision: {precision:.3f}\nF1: {f1:.3f}'
                      f'\nВсего совпало: {correct_matches}/{len(data_test)}\n', file=f)

            index = 0
            while index + 1 < len(data_test):
                comp, total = 0, 0
                nulls, ones = 0, 0
                while index + 1 < len(data_test) and \
                        data_test.loc[index, 'prof_well_index'].split('_')[0] == \
                        data_test.loc[index + 1, 'prof_well_index'].split('_')[0] and \
                        data_test.loc[index, 'prof_well_index'].split('_')[1] == \
                        data_test.loc[index + 1, 'prof_well_index'].split('_')[1]:
                    if data_test.loc[index, 'совпадение'] == 1:
                        comp += 1
                    nulls = nulls + data_test.loc[index, list_cat[1]]
                    ones = ones + data_test.loc[index, list_cat[0]]
                    total += 1
                    index += 1
                if data_test.loc[index, 'prof_well_index'].split('_')[1] == \
                        data_test.loc[index - 1, 'prof_well_index'].split('_')[1]:
                    if data_test.loc[index, 'совпадение'] == 1:
                        comp += 1
                    total += 1

                profile = session.query(Profile).filter(
                    Profile.id == data_test.loc[index, 'prof_well_index'].split('_')[0]).first()
                well = session.query(Well).filter(
                    Well.id == data_test.loc[index, 'prof_well_index'].split('_')[1]).first()

                color_text = Qt.black
                if comp / total < 0.5:
                    color_text = Qt.red
                if 0.9 > comp / total >= 0.5:
                    color_text = Qt.darkYellow

                ui_rp.textEdit_test_result.setTextColor(color_text)
                ui_rp.textEdit_test_result.insertPlainText(
                    f'{profile.research.object.title} - {profile.title} | {well.name} |'
                    f'  {list_cat[0]} {ones / total:.3f} | {list_cat[1]} {nulls / total:.3f} | {comp}/{total}')

                with open(filename, 'a') as f:
                    print(f'{profile.research.object.title} - {profile.title} | {well.name} |'
                          f'  {list_cat[0]} {ones / total:.3f} | {list_cat[1]} {nulls / total:.3f} | {comp}/{total}',
                          file=f)
                index += 1

            data_test.reset_index(drop=True, inplace=True)
            percent = correct_matches / len(data_test) * 100
            # print('percent ', percent)
            result_percent = correct_matches / len(data_test)
            list_estimator_percent.append(result_percent)
            time_end = datetime.datetime.now() - time_start

            color_text = Qt.green
            if percent < 80:
                color_text = Qt.darkYellow
            if percent < 50:
                color_text = Qt.red
            ui_rp.textEdit_test_result.setTextColor(color_text)
            ui_rp.textEdit_test_result.insertPlainText(
                f'Всего совпало: {correct_matches}/{len(data_test)} - {percent:.1f}%\nВремя выполнения: {time_end}\n')

            with open(filename, 'a') as f:
                print(
                f'Всего совпало: {correct_matches}/{len(data_test)} - {percent:.1f}%\nВремя выполнения: {time_end}\n',
                file=f)
        return list_estimator_roc, list_estimator_percent, list_estimator_recall, list_estimator_precision, list_estimator_f1, filename


    def result_analysis(results, filename, reverse=True):
        if reverse is True:
            sorted_result = sorted(results, key=lambda x: x[0], reverse=True)
        else:
            sorted_result = sorted(results, key=lambda x: x[0], reverse=False)
        for item in sorted_result[:20]:
            print(item)

        ## file console

        twenty_percent = int(len(sorted_result) * 0.2)
        sorted_result = sorted_result[:twenty_percent]
        result_param = [item[7] for item in sorted_result] #################???????????????????
        flattened_list = [item for sublist in result_param for item in sublist]
        processed_list = []
        for s in flattened_list:
            parts = s.split('_')
            processed_parts = [re.sub(r'\d+', '', part) for part in parts]
            new_string = '_'.join(processed_parts)
            if new_string.endswith('_') or new_string.endswith('__'):
                new_string = new_string[:-1]
            processed_list.append(new_string)

        param_count = Counter(processed_list)
        part_list = int(len(param_count) * 0.2)
        common_param = param_count.most_common(part_list)
        for param in common_param:
            print(f'{param[0]}: {param[1]}')

        if reverse is True:
            test_result = '\nНаиболее часто встречающиеся параметры для лучших моделей:\n'
        else:
            test_result = '\nНаиболее часто встречающиеся параметры для худших моделей:\n'
        ui_rp.textEdit_test_result.setTextColor(Qt.black)
        ui_rp.textEdit_test_result.insertPlainText(test_result)
        for param in common_param:
            ui_rp.textEdit_test_result.insertPlainText(f'{param[0]}: {param[1]}\n')

        with open(filename, 'a') as f:
            print(test_result, file=f)
            for param in common_param:
                print(f'{param[0]}: {param[1]}', file=f)

        print(test_result)
        for param in common_param:
            print(f'{param[0]}: {param[1]}')


    def start_random_param():
        """
            Запуск подбора параметров.
            Собирается талица из рандомного набора параметров, выбранных из отмеченных в окне.
            Собирается пайплан для указанной модели, после чего она проходит кросс-валидацию.
            С выхода получаем 5 разных обученных моделей.
            Все модлели передаются в test_classif_estimator для оценки, на выходе получаем списки
            roc_auc_score, percent, precision, recall, f1 по результатам 5 моделей.
            Все логи и результаты записываются в файл .txt, который в дальнейшем обрабатывается
            для отображения результатов в программе "random_param_view".
        """

        start_time = datetime.datetime.now()
        filename, _ = QFileDialog.getSaveFileName(caption="Сохранить результаты подбора параметров?",
                                                  filter="TXT (*.txt)")
        labels = set_marks()
        labels_dict = {value: key for key, value in labels.items()}

        with open(filename, 'w') as f:
            print(f"START SEARCH PARAMS\n{datetime.datetime.now()}\n\n", file=f)
        results = []
        for i in range(ui_rp.spinBox_n_iter.value()):
            pipe_steps = []
            text_scaler = ''
            if ui_cls.checkBox_stdscaler.isChecked():
                std_scaler = StandardScaler()
                pipe_steps.append(('std_scaler', std_scaler))
                text_scaler += '\nStandardScaler'
            if ui_cls.checkBox_robscaler.isChecked():
                robust_scaler = RobustScaler()
                pipe_steps.append(('robust_scaler', robust_scaler))
                text_scaler += '\nRobustScaler'
            if ui_cls.checkBox_mnmxscaler.isChecked():
                minmax_scaler = MinMaxScaler()
                pipe_steps.append(('minmax_scaler', minmax_scaler))
                text_scaler += '\nMinMaxScaler'
            if ui_cls.checkBox_mxabsscaler.isChecked():
                maxabs_scaler = MaxAbsScaler()
                pipe_steps.append(('maxabs_scaler', maxabs_scaler))
                text_scaler += '\nMaxAbsScaler'

            model_name = ui_cls.buttonGroup.checkedButton().text()
            list_param = build_list_param()
            data_train, new_list_param = build_table_random_param(get_MLP_id(), list_param)

            # Замена inf на 0
            data_train[new_list_param] = data_train[new_list_param].replace([np.inf, -np.inf], 0)

            list_col = data_train.columns.tolist()
            data_train = pd.DataFrame(imputer.fit_transform(data_train), columns=list_col)
            if model_name == 'TORCH':
                y_train = data_train['mark'].replace(labels).values
                y_train = np.array(y_train).astype(np.float32)
            else:
                y_train = data_train['mark'].values
            data_test, new_list_param = build_table_random_param(get_MLP_test_id(), list_param)

            # Замена inf на 0
            data_test[new_list_param] = data_test[new_list_param].replace([np.inf, -np.inf], 0)

            list_col = data_test.columns.tolist()
            data_test = pd.DataFrame(imputer.fit_transform(data_test), columns=list_col)

            if model_name == 'TORCH':
                pipe, text_model = build_torch_model(pipe_steps, data_train.iloc[:, 2:])
                text_model += text_scaler
            else:
                model_class, text_model = choice_model_classifier(model_name)
                text_model += text_scaler
                pipe_steps.append(('model', model_class))
                pipe = Pipeline(pipe_steps)

            ui_rp.textEdit_test_result.setTextColor(Qt.black)
            ui_rp.textEdit_test_result.insertPlainText(
                f"Итерация #{i}\nКоличество параметров: {len(list_param)}\n"
                      f"Выбранные параметры: \n{list_param}\n")

            with open(filename, 'a') as f:
                print(f"Итерация #{i}\nКоличество параметров: {len(list_param)}\n"
                      f"Выбранные параметры: \n{list_param}\n", file=f)

            print(f"Итерация #{i}\nКоличество параметров: {len(list_param)}\n"
                  f"Выбранные параметры: \n{list_param}\n")

            data_train = data_train.iloc[:, 2:].values
            kv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            cv_results = cross_validate(pipe, data_train, y_train, cv=kv, scoring='accuracy',
                                        return_estimator=True)
            estimators = cv_results['estimator']
            print('estimators \n', estimators)

            list_roc, list_percent, list_recall, list_precision, list_f1, filename = test_classif_estimator(estimators,
                                                                    data_test, list_param, labels_dict, filename)
            roc = np.array(list_roc).mean()
            percent = np.array(list_percent).mean()
            recall = np.array(list_recall).mean()
            precision = np.array(list_precision).mean()
            f1 = np.array(list_f1).mean()
            results_list = [roc, list_roc, percent, recall, precision, f1, list_percent, list_param]
            results.append(results_list)

            with open(filename, 'a') as f:
                print(f'\n!!!RESULT!!!\nroc mean: {roc}\npercent mean: {percent}\n'
                      f'recall mean: {recall}\nprecision mean: {precision}\nf1 mean: {f1}\n', file=f)
            print(f'\n!!!RESULT!!!\nroc mean: {roc}\npercent mean: {percent}\n'
                  f'recall mean: {recall}\nprecision mean: {precision}\nf1 mean: {f1}\n')

        result_analysis(results, filename, reverse=True)
        result_analysis(results, filename, reverse=False)
        draw_result_rnd_prm(results)

        end_time = datetime.datetime.now() - start_time
        with open(filename, 'a') as f:
            print(f'\nВремя выполнения: {end_time}', file=f)
        ui_rp.textEdit_test_result.setTextColor(Qt.red)
        ui_rp.textEdit_test_result.insertPlainText(f'Время выполнения: {end_time}\n')
        print(f'Время выполнения: {end_time}')


    def get_test_MLP_id():
        return ui_rp.comboBox_test_analysis.currentText().split(' id')[-1]


    def update_list_test_well():
        ui_rp.listWidget_test_point.clear()
        count_markup, count_measure, count_fake = 0, 0, 0
        for i in session.query(MarkupMLP).filter(MarkupMLP.analysis_id == get_test_MLP_id()).all():
            try:
                fake = len(json.loads(i.list_fake)) if i.list_fake else 0
                measure = len(json.loads(i.list_measure))
                if i.type_markup == 'intersection':
                    try:
                        inter_name = session.query(Intersection.name).filter(Intersection.id == i.well_id).first()[0]
                    except TypeError:
                        session.query(MarkupMLP).filter(MarkupMLP.id == i.id).delete()
                        session.commit()
                        continue
                    item = f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | {inter_name} | {measure - fake} из {measure} | id{i.id}'
                elif i.type_markup == 'profile':
                    item = f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | | {measure - fake} из {measure} | id{i.id}'
                else:
                    item = f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | {i.well.name} | {measure - fake} из {measure} | id{i.id}'
                ui_rp.listWidget_test_point.addItem(item)
                i_item = ui_rp.listWidget_test_point.findItems(item, Qt.MatchContains)[0]
                i_item.setBackground(QBrush(QColor(i.marker.color)))
                count_markup += 1
                count_measure += measure - fake
                count_fake += fake
            except AttributeError:
                session.delete(i)
                session.commit()


    def update_test_analysis_combobox():
        ui_rp.comboBox_test_analysis.clear()
        for i in session.query(AnalysisMLP.title, AnalysisMLP.id).order_by(AnalysisMLP.title).all():
            ui_rp.comboBox_test_analysis.addItem(f'{i.title} id{i.id}')
            update_list_test_well()


    def draw_result_rnd_prm(data):
        fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(20, 14))

        sorted_result_best = sorted(data, key=lambda x: x[0], reverse=True)
        sorted_result_low = sorted(data, key=lambda x: x[0], reverse=False)

        common_param_best, result_best = find_common_param(sorted_result_best)
        common_param_low, result_low = find_common_param(sorted_result_low)

        roc_values_best = [item[0] for item in result_best]
        percent_values_best = [item[2] for item in result_low]
        roc_values_low = [item[0] for item in result_low]
        percent_values_low = [item[2] for item in result_low]

        n_groups_best = len(result_best)
        n_groups_low = len(result_low)

        index_best = np.arange(n_groups_best)
        index_low = np.arange(n_groups_low)

        bar_width = 0.35

        bars_roc_best = ax[0].bar(index_best - bar_width / 2, roc_values_best, bar_width, label='ROC')
        bars_perc_best = ax[0].bar(index_best + bar_width / 2, percent_values_best, bar_width, label='Percent')
        bars_roc_low = ax[2].bar(index_low - bar_width / 2, roc_values_low, bar_width, label='ROC')
        bars_perc_low = ax[2].bar(index_low + bar_width / 2, percent_values_low, bar_width, label='Percent')

        ax[0].set_xlabel('Итерации')
        ax[0].set_ylabel('Значения')
        ax[0].set_title('Средние ROC and % значения по итерациям')
        ax[0].set_xticks(index_best)
        ax[0].set_xticklabels([f'{i + 1}' for i in range(n_groups_best)])
        ax[0].legend()

        ax[2].set_xlabel('Итерации')
        ax[2].set_ylabel('Значения')
        ax[2].set_title('Средние ROC and % значения по итерациям')
        ax[2].set_xticks(index_low)
        ax[2].set_xticklabels([f'{i + 1}' for i in range(n_groups_low)])
        ax[2].legend()

        def add_labels(bars, ax):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

        add_labels(bars_roc_best, ax[0])
        add_labels(bars_perc_best, ax[0])
        add_labels(bars_roc_low, ax[2])
        add_labels(bars_perc_low, ax[2])

        labels_best, counts_best = zip(*common_param_best)
        labels_low, counts_low = zip(*common_param_low)

        ax[1].bar(labels_best, counts_best, color='#aad5ff')
        ax[1].set_xticklabels(labels_best, rotation=90)
        ax[1].set_xlabel('Параметры')
        ax[1].set_ylabel('Вхождения')
        ax[1].set_title('Наиболее часто встречающиеся параметры')

        ax[3].bar(labels_low, counts_low, color='#ff9f98')
        ax[3].set_xticklabels(labels_low, rotation=90)
        ax[3].set_xlabel('Параметры')
        ax[3].set_ylabel('Вхождения')
        ax[3].set_title('Наименее часто встречающиеся параметры')

        fig.tight_layout()
        plt.show()

    def find_common_param(sorted_result, percent_models=0.2, percent_params=0.4):
        count_model = int(len(sorted_result) * percent_models)
        percent_result = sorted_result[:count_model]
        result_param = [item[7] for item in percent_result]
        flattened_list = [item for sublist in result_param for item in sublist]
        processed_list = []
        for s in flattened_list:
            parts = s.split('_')
            processed_parts = [re.sub(r'\d+', '', part) for part in parts]
            new_string = '_'.join(processed_parts)
            if new_string.endswith('_') or new_string.endswith('__'):
                new_string = new_string[:-1]
            processed_list.append(new_string)
        param_count = Counter(processed_list)
        part_list = int(len(param_count) * percent_params)
        common_param = param_count.most_common(part_list)
        return common_param, percent_result

    update_test_analysis_combobox()
    # update_list_test_well()

    ui_rp.comboBox_test_analysis.activated.connect(update_list_test_well)
    ui_rp.pushButton_start.clicked.connect(start_random_param)

    ui_rp.checkBox_ts_all.clicked.connect(check_checkbox_ts)
    ui_rp.checkBox_attr_all.clicked.connect(check_checkbox_attr)

    ui_rp.spinBox_distr_up.valueChanged.connect(lambda: check_spinbox(ui_rp.spinBox_distr_up, ui_rp.spinBox_distr_down))
    ui_rp.spinBox_distr_down.valueChanged.connect(lambda: check_spinbox(ui_rp.spinBox_distr_up, ui_rp.spinBox_distr_down))

    ui_rp.spinBox_sep_up.valueChanged.connect(lambda: check_spinbox(ui_rp.spinBox_sep_up, ui_rp.spinBox_sep_down))
    ui_rp.spinBox_sep_down.valueChanged.connect(lambda: check_spinbox(ui_rp.spinBox_sep_up, ui_rp.spinBox_sep_down))

    ui_rp.spinBox_mfcc_up.valueChanged.connect(lambda: check_spinbox(ui_rp.spinBox_mfcc_up, ui_rp.spinBox_mfcc_down))
    ui_rp.spinBox_mfcc_down.valueChanged.connect(lambda: check_spinbox(ui_rp.spinBox_mfcc_up, ui_rp.spinBox_mfcc_down))

    ui_rp.spinBox_top_skip_up.valueChanged.connect(lambda: check_spinbox(ui_rp.spinBox_top_skip_up, ui_rp.spinBox_top_skip_down))
    ui_rp.spinBox_top_skip_down.valueChanged.connect(lambda: check_spinbox(ui_rp.spinBox_top_skip_up, ui_rp.spinBox_top_skip_down))

    ui_rp.spinBox_bot_skip_up.valueChanged.connect(lambda: check_spinbox(ui_rp.spinBox_bot_skip_up, ui_rp.spinBox_bot_skip_down))
    ui_rp.spinBox_bot_skip_down.valueChanged.connect(lambda: check_spinbox(ui_rp.spinBox_bot_skip_up, ui_rp.spinBox_bot_skip_down))

    RandomParam.exec_()

    Classifier.exec_()
