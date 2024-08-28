import json
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from func import *
from build_table import *
from random_search import push_random_search
from random_param import push_random_param


def train_classifier(data_train: pd.DataFrame, list_param: list, list_param_save: list, colors: dict, mark: str, point_name: str, type_case: str):
    """
    Обучить классификатор

    :param data_train: обучающая выборка
    :param list_param: список параметров
    :param list_param_save: список параметров для сохранения
    :param colors: цвета маркеров
    :param mark: название столбца в data_train с маркерами
    :param point_name: название столбца с названиями точек
    :param type_case: тип классификатора ('georadar', 'geochem' или 'exploration')
    """
    list_nan_param, count_nan = set(), 0
    for i in data_train.index:
        for param in list_param:
            if pd.isna(data_train[param][i]):
                count_nan += 1
                list_nan_param.add(param)
            if data_train[param][i] == np.inf or data_train[param][i] == -np.inf:
                data_train[param][i] = 0
                count_nan += 1
                list_nan_param.add(param)
    if count_nan > 0:
        list_col = data_train.columns.tolist()
        data_train = pd.DataFrame(imputer.fit_transform(data_train), columns=list_col)
        set_info(f'Заполнены пропуски в {count_nan} образцах в параметрах {", ".join(list_nan_param)}', 'red')

    training_sample = np.array(data_train[list_param].values.tolist())
    markup = np.array(sum(data_train[[mark]].values.tolist(), []))
    list_marker = get_list_marker_mlp(type_case)

    Classifier = QtWidgets.QDialog()
    ui_cls = Ui_ClassifierForm()
    ui_cls.setupUi(Classifier)
    Classifier.show()
    Classifier.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    max_pca = min(len(list_param), len(data_train.index))
    max_pca -= int(round(max_pca/5, 0))
    ui_cls.spinBox_pca.setMaximum(max_pca)
    ui_cls.spinBox_pca.setValue(max_pca // 2)
    if len (list_param) > len(data_train.index):
        ui_cls.checkBox_pca_mle.hide()

    text_label = f'Тренеровочный сэмпл: {len(training_sample)}, '
    for i_mark in range(len(list_marker)):
        text_label += f'{list_marker[i_mark]}-{list(markup).count(list_marker[i_mark])}, '

    ui_cls.label.setText(text_label)

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
                          f'\nalpha: {round(ui_cls.doubleSpinBox_alpha_mlp.value(), 5)}, '
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
            # model_class = CategoricalNB()
            # text_model = f'**C-CB**:, '
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
            model_class = ExtraTreesClassifier(n_estimators=ui_cls.spinBox_rfc_n.value(), class_weight='balanced', bootstrap=True, oob_score=True, random_state=0, n_jobs=-1)
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
            text_model = (f'**GPC**: \nwidth kernal: {round(gpc_kernel_width, 2)}, \nscale kernal: {round(gpc_kernel_scale, 2)}, '
                          f'\nn restart: {n_restart_optimization}, \nmulti_class: {multi_class} ,')

        elif model == 'QDA':
            model_class = QuadraticDiscriminantAnalysis(reg_param=ui_cls.doubleSpinBox_qda_reg_param.value())
            text_model = f'**QDA**: \nreg_param: {round(ui_cls.doubleSpinBox_qda_reg_param.value(), 2)}, '

        elif model == 'SVC':
            model_class = SVC(kernel=ui_cls.comboBox_svr_kernel.currentText(), probability=True,
                              C=ui_cls.doubleSpinBox_svr_c.value(), random_state=0, class_weight='balanced')
            text_model = (f'**SVC**: \nkernel: {ui_cls.comboBox_svr_kernel.currentText()}, '
                          f'\nC: {round(ui_cls.doubleSpinBox_svr_c.value(), 2)}, ')
        # elif model == 'XGB':
        #     model_class = XGBClassifier(objective='binary:logistic', min_child_weight=1, n_estimators=ui_cls.spinBox_xgb_estim.value(),
        #                                 learning_rate=0.01, max_depth=10, random_state=0)
        #     print('XGB CHOSEN')
        #     text_model = f'**XGB**: \nn estimators: {ui_cls.spinBox_xgb_estim.value()}, '

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

    def build_stacking_voting_model():
        """ Построить модель стекинга """
        estimators, list_model = [], []

        if ui_cls.checkBox_stv_mlpc.isChecked():
            mlpc = MLPClassifier(
                hidden_layer_sizes=tuple(map(int, ui_cls.lineEdit_layer_mlp.text().split())),
                activation=ui_cls.comboBox_activation_mlp.currentText(),
                solver=ui_cls.comboBox_solvar_mlp.currentText(),
                alpha=ui_cls.doubleSpinBox_alpha_mlp.value(),
                max_iter=5000,
                early_stopping=ui_cls.checkBox_e_stop_mlp.isChecked(),
                validation_fraction=ui_cls.doubleSpinBox_valid_mlp.value(),
                random_state=0
            )
            estimators.append(('mlpc', mlpc))
            list_model.append('mlpc')

        if ui_cls.checkBox_stv_knnc.isChecked():
            n_knn = ui_cls.spinBox_neighbors.value()
            weights_knn = 'distance' if ui_cls.checkBox_knn_weights.isChecked() else 'uniform'
            knnc = KNeighborsClassifier(n_neighbors=n_knn, weights=weights_knn, algorithm='auto')
            estimators.append(('knnc', knnc))
            list_model.append('knnc')

        if ui_cls.checkBox_stv_gbc.isChecked():
            est = ui_cls.spinBox_n_estimators.value()
            l_rate = ui_cls.doubleSpinBox_learning_rate.value()
            gbc = GradientBoostingClassifier(n_estimators=est, learning_rate=l_rate, random_state=0)
            estimators.append(('gbc', gbc))
            list_model.append('gbc')

        if ui_cls.checkBox_stv_gnb.isChecked():
            gnb = GaussianNB(var_smoothing=10 ** (-ui_cls.spinBox_gnb_var_smooth.value()))
            estimators.append(('gnb', gnb))
            list_model.append('gnb')

        if ui_cls.checkBox_stv_dtc.isChecked():
            spl = 'random' if ui_cls.checkBox_splitter_rnd.isChecked() else 'best'
            dtc = DecisionTreeClassifier(splitter=spl, random_state=0)
            estimators.append(('dtc', dtc))
            list_model.append('dtc')

        if ui_cls.checkBox_stv_rfc.isChecked():
            rfc = RandomForestClassifier(n_estimators=ui_cls.spinBox_rfc_n.value(), class_weight='balanced',
                                         bootstrap=True, oob_score=True, random_state=0, n_jobs=-1)
            estimators.append(('rfc', rfc))
            list_model.append('rfc')
        if ui_cls.checkBox_stv_abc.isChecked():
            abc = AdaBoostClassifier(n_estimators=ui_cls.spinBox_rfc_n.value(), random_state=0)
            estimators.append(('abc', abc))
            list_model.append('abc')
        if ui_cls.checkBox_stv_etc.isChecked():
            etc = ExtraTreesClassifier(n_estimators=ui_cls.spinBox_rfc_n.value(), class_weight='balanced', bootstrap=True, oob_score=True, random_state=0, n_jobs=-1)
            estimators.append(('etc', etc))
            list_model.append('etc')

        if ui_cls.checkBox_stv_gpc.isChecked():
            gpc_kernel_width = ui_cls.doubleSpinBox_gpc_wigth.value()
            gpc_kernel_scale = ui_cls.doubleSpinBox_gpc_scale.value()
            n_restart_optimization = ui_cls.spinBox_gpc_n_restart.value()
            multi_class = ui_cls.comboBox_gpc_multi.currentText()
            kernel = gpc_kernel_scale * RBF(gpc_kernel_width)
            gpc = GaussianProcessClassifier(
                kernel=kernel,
                n_restarts_optimizer=n_restart_optimization,
                random_state=0,
                multi_class=multi_class,
                n_jobs=-1
            )
            estimators.append(('gpc', gpc))
            list_model.append('gpc')

        if ui_cls.checkBox_stv_qda.isChecked():
            qda = QuadraticDiscriminantAnalysis(reg_param=ui_cls.doubleSpinBox_qda_reg_param.value())
            estimators.append(('qda', qda))
            list_model.append('qda')
        if ui_cls.checkBox_stv_svc.isChecked():
            svc = SVC(kernel=ui_cls.comboBox_svr_kernel.currentText(),
                      probability=True, C=ui_cls.doubleSpinBox_svr_c.value(), random_state=0, class_weight='balanced')
            estimators.append(('svc', svc))
            list_model.append('svc')
        final_model, final_text_model = choice_model_classifier(ui_cls.buttonGroup.checkedButton().text())
        list_model_text = ', '.join(list_model)
        if ui_cls.buttonGroup_stack_vote.checkedButton().text() == 'Voting':
            # hard_voting = 'hard' if ui_cls.checkBox_voting_hard.isChecked() else 'soft'
            model_class = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
            text_model = f'**Voting**: -soft-\n({list_model_text})\n'
            model_name = 'VOT'
        else:
            model_class = StackingClassifier(estimators=estimators, final_estimator=final_model, n_jobs=-1)
            text_model = f'**Stacking**:\nFinal estimator: {final_text_model}\n({list_model_text})\n'
            model_name = 'STACK'
        return model_class, text_model, model_name


    def add_model_class_to_lineup():
        """ Добавление модели в лайнап """

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

        over_sampling, text_over_sample = 'none', ''
        if ui_cls.checkBox_smote.isChecked():
            over_sampling, text_over_sample = 'smote', '\nSMOTE'
        if ui_cls.checkBox_adasyn.isChecked():
            over_sampling, text_over_sample = 'adasyn', '\nADASYN'

        if ui_cls.checkBox_pca.isChecked():
            n_comp = 'mle' if ui_cls.checkBox_pca_mle.isChecked() else ui_cls.spinBox_pca.value()
            pca = PCA(n_components=n_comp, random_state=0, svd_solver='auto')
            pipe_steps.append(('pca', pca))
        text_pca = f'\nPCA: n_components={n_comp}' if ui_cls.checkBox_pca.isChecked() else ''

        if ui_cls.checkBox_stack_vote.isChecked():
            model_class, text_model, model_name = build_stacking_voting_model()
        else:
            model_name = ui_cls.buttonGroup.checkedButton().text()
            model_class, text_model = choice_model_classifier(model_name)

        if ui_cls.checkBox_baggig.isChecked():
            model_class = BaggingClassifier(base_estimator=model_class, n_estimators=ui_cls.spinBox_bagging.value(),
                                            random_state=0, n_jobs=-1)
        bagging_text = f'\nBagging: n_estimators={ui_cls.spinBox_bagging.value()}' if ui_cls.checkBox_baggig.isChecked() else ''

        text_model += text_scaler
        text_model += text_pca
        text_model += text_over_sample
        text_model += bagging_text
        text_model += f'\nrandom_seed={ui.spinBox_seed.value()}'
        if ui_cls.checkBox_cvw.isChecked():
            text_model += '\nCVW (separate by well)'

        pipe_steps.append(('model', model_class))
        pipe = Pipeline(pipe_steps)

        if ui_cls.checkBox_calibr.isChecked():
            kf = StratifiedKFold(n_splits=ui_cls.spinBox_n_cross_val.value(), shuffle=True, random_state=0)

            pipe = CalibratedClassifierCV(
                estimator=pipe,
                cv=kf,
                method=ui_cls.comboBox_calibr_method.currentText(),
                n_jobs=-1
            )
            text_model += f'\ncalibration: method={ui_cls.comboBox_calibr_method.currentText()}'

        except_mlp = session.query(ExceptionMLP).filter_by(analysis_id=get_MLP_id()).first()

        new_lineup = LineupTrain(
            type_ml = 'cls',
            analysis_id = get_MLP_id(),
            list_param = json.dumps(list_param),
            list_param_short = json.dumps(list_param_save),
            except_signal = except_mlp.except_signal,
            except_crl = except_mlp.except_crl,
            text_model=text_model,
            model_name = model_name,
            over_sampling = over_sampling,
            pipe = pickle.dumps(pipe),
            random_seed = ui.spinBox_seed.value(),
            cvw = ui_cls.checkBox_cvw.isChecked()
        )
        session.add(new_lineup)
        session.commit()

        set_info(f'Модель {model_name} добавлена в очередь\n{text_model}', 'green')


    def draw_yellow_brick():
        # if ui_cls.checkBox_cvw.isChecked():
        #     training_sample_train, training_sample_test, markup_train, markup_test = train_test_split_cvw(data_train,
        #         list_marker, mark, list_param, random_seed=ui.spinBox_seed.value(), test_size=0.2)
        # else:
        #     training_sample_train, training_sample_test, markup_train, markup_test = train_test_split(
        #         training_sample, markup, test_size=0.20, random_state=ui.spinBox_seed.value(), stratify=markup)

        training_sample = np.array(data_train[list_param].values.tolist())
        markup = np.array(sum(data_train[[mark]].values.tolist(), []))
        list_marker = get_list_marker_mlp(type_case)

        (markup_train, model_class, model_name, pipe,
             text_model, training_sample_train) = build_pipeline(markup, training_sample)
        markup = np.array([0 if i == list_marker[0] else 1 for i in markup])
        visualizer = DiscriminationThreshold(pipe)
        visualizer.fit(training_sample, markup)
        visualizer.show()

    def calc_model_class_by_cvw():
        """ Кросс-валидация по скважинам """

        start_time = datetime.datetime.now()

        list_well = [i.split('_')[1] for i in data_train['prof_well_index'].values.tolist()]
        data_train_by_well = data_train.copy()
        data_train_by_well['well_id'] = list_well

        list_marker = get_list_marker_mlp(type_case)
        data_train_m1 = data_train_by_well[data_train_by_well[mark] == list_marker[0]]
        data_train_m2 = data_train_by_well[data_train_by_well[mark] == list_marker[1]]

        list_well_m1 = list(set(data_train_m1['well_id'].values.tolist()))
        list_well_m2 = list(set(data_train_m2['well_id'].values.tolist()))


        if '0' in list_well_m1:
            list_well_m1.remove('0')
        if '0' in list_well_m2:
            list_well_m2.remove('0')

        list_cvw_m1 = split_list_cvw(list_well_m1, 5)
        list_cvw_m2 = split_list_cvw(list_well_m2, 5)

        list_cvw = [list_cvw_m1[i] + list_cvw_m2[i] for i in range(5)]

        (fig_cvw, axes_cvw) = plt.subplots(nrows=2, ncols=3)
        fig_cvw.set_size_inches(25, 15)
        list_accuracy = []

        ui.progressBar.setMaximum(len(list_cvw))
        for n_cv, lcvw in enumerate(list_cvw):
            ui.progressBar.setValue(n_cv + 1)

            cvw_row, cvw_col = n_cv // 3, n_cv % 3

            data_test_well = data_train_by_well[data_train_by_well['well_id'].isin(lcvw)]
            data_train_well = data_train_by_well[~data_train_by_well['well_id'].isin(lcvw)]

            training_sample_train = np.array(data_train_well[list_param].values.tolist())
            training_sample_test = np.array(data_test_well[list_param].values.tolist())
            markup_train = np.array(sum(data_train_well[[mark]].values.tolist(), []))
            markup_test = np.array(sum(data_test_well[[mark]].values.tolist(), []))

            (markup_train, model_class, model_name, pipe,
             text_model, training_sample_train) = build_pipeline(markup_train, training_sample_train)

            pipe.fit(training_sample_train, markup_train)

            # Оценка точности на всей обучающей выборке
            train_accuracy = pipe.score(training_sample, markup)
            test_accuracy = pipe.score(training_sample_test, markup_test)

            list_accuracy.append(test_accuracy)

            preds_test = pipe.predict_proba(training_sample_test)[:, 0]
            fpr, tpr, thresholds = roc_curve(markup_test, preds_test, pos_label=list_marker[0])
            roc_auc = auc(fpr, tpr)

            # Строим ROC-кривую
            axes_cvw[cvw_row, cvw_col].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            axes_cvw[cvw_row, cvw_col].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes_cvw[cvw_row, cvw_col].set_xlim([0.0, 1.0])
            axes_cvw[cvw_row, cvw_col].set_ylim([0.0, 1.05])
            axes_cvw[cvw_row, cvw_col].set_xlabel('False Positive Rate')
            axes_cvw[cvw_row, cvw_col].set_ylabel('True Positive Rate')
            axes_cvw[cvw_row, cvw_col].set_title('ROC-кривая')
            axes_cvw[cvw_row, cvw_col].legend(loc="lower right")

        train_time = datetime.datetime.now() - start_time

        axes_cvw[1, 2].bar(range(5), list_accuracy)

        fig_cvw.tight_layout()
        fig_cvw.show()


    def calc_model_class():
        """ Создание и тренировка модели """

        # global training_sample, markup
        start_time = datetime.datetime.now()

        # Разделение данных на обучающую и тестовую выборки
        if ui_cls.checkBox_cvw.isChecked():
            training_sample_train, training_sample_test, markup_train, markup_test = train_test_split_cvw(data_train,
                list_marker, mark, list_param, random_seed=ui.spinBox_seed.value(), test_size=0.2)
        else:
            training_sample_train, training_sample_test, markup_train, markup_test = train_test_split(
                training_sample, markup, test_size=0.20, random_state=ui.spinBox_seed.value(), stratify=markup)

        (markup_train, model_class, model_name, pipe,
         text_model, training_sample_train) = build_pipeline(markup_train, training_sample_train)

        pipe.fit(training_sample_train, markup_train)

        # Оценка точности на всей обучающей выборке
        train_accuracy = pipe.score(training_sample, markup)
        test_accuracy = pipe.score(training_sample_test, markup_test)

        train_time = datetime.datetime.now() - start_time

        text_model += f'\ntrain_accuracy: {round(train_accuracy, 4)}, test_accuracy: {round(test_accuracy, 4)},'
        set_info(text_model, 'blue')
        preds_train = pipe.predict(training_sample)
        preds_test = pipe.predict_proba(training_sample_test)[:, 0]
        fpr, tpr, thresholds = roc_curve(markup_test, preds_test, pos_label=list_marker[0])
        roc_auc = auc(fpr, tpr)
        preds_t = pipe.predict(training_sample_test)
        precision = precision_score(markup_test, preds_t, average='binary', pos_label='bitum')
        recall = recall_score(markup_test, preds_t, average='binary', pos_label='bitum')
        f1 = f1_score(markup_test, preds_t, average='binary', pos_label='bitum')
        # fpr, tpr, thresholds = roc_curve(markup_test, probab)
        # roc_auc = auc(fpr, tpr)

        text_model += f'test_precision: {round(precision, 4)},\ntest_recall: {round(recall, 4)},' \
                      f'\ntest_f1: {round(f1, 4)},\nroc_auc: {round(roc_auc, 4)},\nвремя обучения: {train_time}'
        set_info(text_model, 'blue')

        # if (ui_cls.checkBox_stack_vote.isChecked() and ui_cls.buttonGroup_stack_vote.checkedButton().text() == 'Voting'
        #         and ui_cls.checkBox_voting_hard.isChecked()):
        #     hard_flag = True
        # else:
        #     hard_flag = False

        # if not hard_flag:
        preds_proba_train = pipe.predict_proba(training_sample)
        try:
            tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
            train_tsne = tsne.fit_transform(preds_proba_train)
            data_tsne = pd.DataFrame(train_tsne)
            data_tsne[mark] = preds_train
        except ValueError:
            tsne = TSNE(n_components=2, perplexity=len(data_train.index) - 1, learning_rate=200, random_state=42)
            train_tsne = tsne.fit_transform(preds_proba_train)
            data_tsne = pd.DataFrame(train_tsne)
            data_tsne[mark] = preds_train

        if ui_cls.checkBox_cross_val.isChecked():
            kf = StratifiedKFold(n_splits=ui_cls.spinBox_n_cross_val.value(), shuffle=True, random_state=0)

            # if ui_cls.checkBox_smote.isChecked():
            #     smote = SMOTE(random_state=0)
            #     training_sample, markup = smote.fit_resample(training_sample, markup)
            #
            # if ui_cls.checkBox_adasyn.isChecked():
            #     adasyn = ADASYN(random_state=0)
            #     training_sample, markup = adasyn.fit_resample(training_sample, markup)

            scores_cv = cross_val_score(pipe, training_sample, markup, cv=kf, n_jobs=-1)

        if model_name == 'RFC' or model_name == 'GBC' or model_name == 'DTC' or model_name == 'ETC' or model_name == 'ABC':
            if not ui_cls.checkBox_baggig.isChecked():
                imp_name_params, imp_params = [], []
                if not ui_cls.checkBox_calibr.isChecked():
                    for n, i in enumerate(model_class.feature_importances_):
                        if ui_cls.checkBox_all_imp.isChecked():
                            imp_name_params.append(list_param[n])
                            imp_params.append(i)
                        else:
                            if i >= np.mean(model_class.feature_importances_):
                                imp_name_params.append(list_param[n])
                                imp_params.append(i)

        # def model_predict_proba(X):
        #     return pipe['model'].predict_proba(X)
        #
        # print(training_sample_test.shape)
        # print(type(training_sample_test))
        # print('DATA ', training_sample_test[:100, :10])
        # list_names = data_train.iloc[:, 2:].columns
        # print('LIST_NAMES ', list_names)
        # explainer = shap.KernelExplainer(model_predict_proba, training_sample_train[:100])
        # shap_values = explainer(training_sample_test[:30])
        # print('shap values ', shap_values)
        # print('shap values shape ', shap_values.shape)
        # print(f"SHAP values shape for class 0: {shap_values[0].shape}")
        # print(f"SHAP values shape for class 1: {shap_values[0, 0].shape}")
        # print(f"SHAP values shape for class 2: {shap_values[0][0].shape}")
        # shap.plots.bar(shap_values[:, :, 0])
        # shap.summary_plot(shap_values, training_sample_test[:30], plot_type="bar", feature_names=list_names)
        # shap.summary_plot(shap_values, training_sample_test[:30], feature_names=list_names)



        # shap.plots.heatmap(shap_values[0])
        #
        # shap.plots.beeswarm(shap_values[0, 0], training_sample_test[:10])
        # shap.plots.waterfall(shap_values[0][0], max_display=20)


        (fig_train, axes) = plt.subplots(nrows=1, ncols=3)
        fig_train.set_size_inches(25, 10)

        # if not hard_flag:
        sns.scatterplot(data=data_tsne, x=0, y=1, hue=mark, s=200, palette=colors, ax=axes[0])
        axes[0].grid()
        axes[0].xaxis.grid(True, "minor", linewidth=.25)
        axes[0].yaxis.grid(True, "minor", linewidth=.25)
        axes[0].set_title(f'Диаграмма рассеяния для канонических значений {model_name}\nдля обучающей выборки и тестовой выборки')
        if len(list_marker) == 2:
            # Вычисляем ROC-кривую и AUC
            preds_test = pipe.predict_proba(training_sample_test)[:, 0]
            fpr, tpr, thresholds = roc_curve(markup_test, preds_test, pos_label=list_marker[0])
            roc_auc = auc(fpr, tpr)

            # Строим ROC-кривую
            axes[1].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[1].set_xlim([0.0, 1.0])
            axes[1].set_ylim([0.0, 1.05])
            axes[1].set_xlabel('False Positive Rate')
            axes[1].set_ylabel('True Positive Rate')
            axes[1].set_title('ROC-кривая')
            axes[1].legend(loc="lower right")

        title_graph = text_model
        if model_name == 'RFC' or model_name == 'ETC':
            if not ui_cls.checkBox_calibr.isChecked():
                if not ui_cls.checkBox_baggig.isChecked():
                    title_graph += f'\nOOB score: {round(model_class.oob_score_, 7)}, \n' \
                                   f'roc_auc: {round(roc_auc, 7)}'

        if (model_name == 'RFC' or model_name == 'GBC' or model_name == 'DTC' or model_name == 'ETC' or model_name == 'ABC') and not ui_cls.checkBox_cross_val.isChecked():
            if not ui_cls.checkBox_calibr.isChecked():
                if not ui_cls.checkBox_baggig.isChecked():
                    axes[2].bar(imp_name_params, imp_params)
                    axes[2].set_xticklabels(imp_name_params, rotation=90)
                    axes[2].set_title('Важность признаков')

        if ui_cls.checkBox_cross_val.isChecked():
            axes[2].bar(range(len(scores_cv)), scores_cv)
            axes[2].set_title('Кросс-валидация')

        elif ui_cls.checkBox_calibr.isChecked():
            preds_test = pipe.predict_proba(training_sample_test)[:, 0]
            true_probs, pred_probs = calibration_curve(markup_test, preds_test, n_bins=5, pos_label=list_marker[0])

            # Построение кривой надежности
            axes[2].plot(pred_probs, true_probs, marker='o')
            axes[2].plot([0, 1], [0, 1], linestyle='--')
            axes[2].set_xlabel('Predicted probabilities')
            axes[2].set_ylabel('True probabilities')
            axes[2].set_title('Reliability curve')
        fig_train.suptitle(title_graph)
        fig_train.tight_layout()
        fig_train.show()

        if ui_cls.checkBox_save_table.isChecked():
            preds_train = pipe.predict(training_sample)
            probability = pipe.predict_proba(training_sample)
            list_cat = list(pipe.classes_)

            # pd.concat работает не корректно, поэтому преобразуем в словари, складываем и создаем датафрейм
            data_train_dict = data_train.to_dict(orient='series')
            probability_dict = pd.DataFrame(probability, columns=list_cat).to_dict(orient='list')
            result_dict = {**data_train_dict, **probability_dict}
            data_result = pd.DataFrame(result_dict)
            # data_result = pd.concat([data_train, pd.DataFrame(probability, columns=list_cat)], axis=1)

            data_result['mark_result'] = preds_train
            table_name = QFileDialog.getSaveFileName(
                caption=f'Сохранить расчеты модели {model_name} в таблицу', directory=f'model_table_{model_name}.xlsx', filter="Excel Files (*.xlsx)")
            data_result.to_excel(table_name[0])
            set_info(f'Таблица сохранена в файл: {table_name[0]}', 'green')

        if not ui_cls.checkBox_save_model.isChecked():
            return
        if type_case == 'georadar':
            save_model_georadar_class(model_name, pipe, test_accuracy, text_model, list_param_save)
        if type_case == 'geochem':
            save_model_geochem_class(model_name, pipe, test_accuracy, text_model, list_param_save, data_train)

    def build_pipeline(markup_train, training_sample_train):
        pipe_steps = []
        text_scaler = ''

        # Нормализация данных
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
        text_over_sample = ''
        if ui_cls.checkBox_smote.isChecked():
            smote = SMOTE(random_state=0)
            training_sample_train, markup_train = smote.fit_resample(training_sample_train, markup_train)
            text_over_sample = '\nSMOTE'
        if ui_cls.checkBox_adasyn.isChecked():
            try:
                adasyn = ADASYN(random_state=0)
                training_sample_train, markup_train = adasyn.fit_resample(training_sample_train, markup_train)
            except ValueError:
                set_info('Невозможно применить ADASYN c n_neighbors=5, значение уменьшено до n_neighbors=3', 'red')
                adasyn = ADASYN(random_state=0, n_neighbors=3)
                training_sample_train, markup_train = adasyn.fit_resample(training_sample_train, markup_train)
            text_over_sample = '\nADASYN'
        if ui_cls.checkBox_pca.isChecked():
            n_comp = 'mle' if ui_cls.checkBox_pca_mle.isChecked() else ui_cls.spinBox_pca.value()
            pca = PCA(n_components=n_comp, random_state=0, svd_solver='auto')
            pipe_steps.append(('pca', pca))
        text_pca = f'\nPCA: n_components={n_comp}' if ui_cls.checkBox_pca.isChecked() else ''
        if ui_cls.checkBox_stack_vote.isChecked():
            model_class, text_model, model_name = build_stacking_voting_model()
        else:
            model_name = ui_cls.buttonGroup.checkedButton().text()
            model_class, text_model = choice_model_classifier(model_name)
        if ui_cls.checkBox_baggig.isChecked():
            model_class = BaggingClassifier(base_estimator=model_class, n_estimators=ui_cls.spinBox_bagging.value(),
                                            random_state=0, n_jobs=-1)
        bagging_text = f'\nBagging: n_estimators={ui_cls.spinBox_bagging.value()}' if ui_cls.checkBox_baggig.isChecked() else ''
        text_model += text_scaler
        text_model += text_pca
        text_model += text_over_sample
        text_model += bagging_text
        pipe_steps.append(('model', model_class))
        pipe = Pipeline(pipe_steps)
        if ui_cls.checkBox_calibr.isChecked():
            kf = StratifiedKFold(n_splits=ui_cls.spinBox_n_cross_val.value(), shuffle=True, random_state=0)

            pipe = CalibratedClassifierCV(
                estimator=pipe,
                cv=kf,
                method=ui_cls.comboBox_calibr_method.currentText(),
                n_jobs=-1
            )
            text_model += f'\ncalibration: method={ui_cls.comboBox_calibr_method.currentText()}'
        return markup_train, model_class, model_name, pipe, text_model, training_sample_train

    def calc_lof():
        """ Расчет выбросов методом LOF """
        global data_pca, data_tsne, colors_lof, factor_lof

        data_lof = data_train[list_param]

        scaler = StandardScaler()
        training_sample_lof = scaler.fit_transform(data_lof)
        n_LOF = ui_cls.spinBox_lof_neighbor.value()

        try:
            tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
            data_tsne = tsne.fit_transform(training_sample_lof)
        except ValueError:
            tsne = TSNE(n_components=2, perplexity=len(data_train.index) - 1, learning_rate=200, random_state=42)
            data_tsne = tsne.fit_transform(training_sample_lof)

        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(training_sample_lof)

        colors_lof, data_pca, data_tsne, factor_lof, label_lof = calc_lof_model(n_LOF, training_sample_lof)

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
        draw_lof_bar(colors_lof, factor_lof, label_lof, ui_lof)
        insert_list_samples(data_train, ui_lof.listWidget_samples, label_lof)
        insert_list_features(data_train, ui_lof.listWidget_features)


        def calc_lof_in_window():
            global data_pca, data_tsne, colors_lof, factor_lof
            colors_lof, data_pca, data_tsne, factor_lof, label_lof = calc_lof_model(ui_lof.spinBox_lof_n.value(), training_sample_lof)
            ui_lof.checkBox_samples.setChecked(False)
            draw_lof_tsne(data_tsne, ui_lof)
            draw_lof_pca(data_pca, ui_lof)
            draw_lof_bar(colors_lof, factor_lof, label_lof, ui_lof)

            set_title_lof_form(label_lof)
            insert_list_samples(data_train, ui_lof.listWidget_samples, label_lof)
            insert_list_features(data_train, ui_lof.listWidget_features)


        def calc_clean_model():
            _, _, _, _, label_lof = calc_lof_model(ui_lof.spinBox_lof_n.value(), training_sample_lof)
            lof_index = [i for i, x in enumerate(label_lof) if x == -1]
            if type_case == 'georadar':
                for ix in lof_index:
                    prof_well = data_train[point_name][ix]
                    prof_id, well_id, fake_id = int(prof_well.split('_')[0]), int(prof_well.split('_')[1]), int(prof_well.split('_')[2])
                    old_list_fake = session.query(MarkupMLP.list_fake).filter(
                        MarkupMLP.analysis_id == get_MLP_id(),
                        MarkupMLP.profile_id == prof_id,
                        MarkupMLP.well_id == well_id
                    ).first()[0]
                    if old_list_fake:
                        new_list_fake = json.loads(old_list_fake)
                        new_list_fake.append(fake_id)
                    else:
                        new_list_fake = [fake_id]
                    session.query(MarkupMLP).filter(
                        MarkupMLP.analysis_id == get_MLP_id(),
                        MarkupMLP.profile_id == prof_id,
                        MarkupMLP.well_id == well_id
                    ).update({'list_fake': json.dumps(new_list_fake)}, synchronize_session='fetch')
                    session.commit()

                Classifier.close()
                Form_LOF.close()
                session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
                session.commit()
                build_table_train(False, 'mlp')
                update_list_well_markup_mlp()
            if type_case == 'geochem':
                for ix in lof_index:
                    train_point = session.query(GeochemTrainPoint).join(GeochemCategory).join(GeochemMaket).filter(
                        GeochemMaket.id == get_maket_id(),
                        GeochemTrainPoint.title == data_train[point_name][ix]
                    ).first()
                    session.query(GeochemTrainPoint).filter_by(id=train_point.id).update(
                        {'fake': True}, synchronize_session='fetch'
                    )
                session.commit()
                Classifier.close()
                Form_LOF.close()
                update_g_train_point_list()

            # show_regression_form(data_train_clean, list_param)

        def draw_checkbox_samples():
            global data_pca, data_tsne, colors_lof, factor_lof
            if ui_lof.checkBox_samples.isChecked():
                col = ui_lof.listWidget_features.currentItem().text()
                draw_hist_sample_feature(data_train, col, data_train[col][int(ui_lof.listWidget_samples.currentItem().text().split(') ')[0])], ui_lof)
                draw_lof_bar(colors_lof, factor_lof, label_lof, ui_lof)
                draw_lof_pca(data_pca, ui_lof)
            else:
                draw_lof_tsne(data_tsne, ui_lof)
                draw_lof_bar(colors_lof, factor_lof, label_lof, ui_lof)
                draw_lof_pca(data_pca, ui_lof)


        # ui_lof.spinBox_lof_n.valueChanged.connect(calc_lof_in_window)
        ui_lof.pushButton_clean_lof.clicked.connect(calc_clean_model)
        ui_lof.checkBox_samples.clicked.connect(draw_checkbox_samples)
        ui_lof.listWidget_samples.currentItemChanged.connect(draw_checkbox_samples)

        ui_lof.listWidget_features.currentItemChanged.connect(draw_checkbox_samples)
        ui_lof.pushButton_lof.clicked.connect(calc_lof_in_window)

        Form_LOF.exec_()


    def insert_list_samples(data, list_widget, label_lof):
        list_widget.clear()
        for i in data.index:
            list_widget.addItem(f'{i}) {data[point_name][i]}')
            if label_lof[int(i)] == -1:
                list_widget.item(int(i)).setBackground(QBrush(QColor('red')))
        list_widget.setCurrentRow(0)


    def insert_list_features(data, list_widget):
        list_widget.clear()
        for param in list_param:
            list_widget.addItem(param)
        list_widget.setCurrentRow(0)


    def draw_hist_sample_feature(data, feature, value_sample, ui_widget):
        clear_horizontalLayout(ui_widget.horizontalLayout_tsne)
        figure_tsne = plt.figure()
        canvas_tsne = FigureCanvas(figure_tsne)
        figure_tsne.clear()
        ui_widget.horizontalLayout_tsne.addWidget(canvas_tsne)
        sns.histplot(data, x=feature, bins=50)
        plt.axvline(value_sample, color='r', linestyle='dashed', linewidth=2)
        plt.grid()
        figure_tsne.suptitle(f't-SNE')
        figure_tsne.tight_layout()
        canvas_tsne.draw()


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

        colors_lof = ['red' if label == -1 else 'blue' for label in label_lof]

        return colors_lof, data_pca_pd, data_tsne_pd, factor_lof, label_lof


    def class_exit():
        Classifier.close()

    ui_cls.pushButton_random_search.clicked.connect(class_exit)
    ui_cls.pushButton_random_search.clicked.connect(push_random_search)
    ui_cls.pushButton_random_param.clicked.connect(class_exit)
    ui_cls.pushButton_random_param.clicked.connect(push_random_param)
    ui_cls.pushButton_lof.clicked.connect(calc_lof)
    ui_cls.pushButton_calc.clicked.connect(calc_model_class)
    ui_cls.checkBox_smote.clicked.connect(push_checkbutton_smote)
    ui_cls.checkBox_adasyn.clicked.connect(push_checkbutton_adasyn)
    ui_cls.pushButton_add_to_lineup.clicked.connect(add_model_class_to_lineup)
    ui_cls.pushButton_cvw.clicked.connect(calc_model_class_by_cvw)
    ui_cls.pushButton_yellow_brick.clicked.connect(draw_yellow_brick)
    Classifier.exec_()


def save_model_georadar_class(model_name, pipe, test_accuracy, text_model, list_param):
    print(list_param)
    result = QtWidgets.QMessageBox.question(
        MainWindow,
        'Сохранение модели',
        f'Сохранить модель {model_name}?',
        QtWidgets.QMessageBox.Yes,
        QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        # Сохранение модели в файл с помощью pickle
        path_model = f'models/classifier/{model_name}_{round(test_accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y")}.pkl'
        if os.path.exists(path_model):
            path_model = f'models/classifier/{model_name}_{round(test_accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y_%H%M%S")}.pkl'
        with open(path_model, 'wb') as f:
            pickle.dump(pipe, f)

        new_trained_model = TrainedModelClass(
            analysis_id=get_MLP_id(),
            title=f'{model_name}_{round(test_accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y")}',
            path_model=path_model,
            list_params=json.dumps(list_param),
            except_signal = ui.lineEdit_signal_except.text(),
            except_crl = ui.lineEdit_crl_except.text(),
            comment=text_model
        )
        session.add(new_trained_model)
        session.commit()
        update_list_trained_models_class()
    else:
        pass

def save_model_geochem_class(model_name, pipe, test_accuracy, text_model, list_param, data_train):
    result = QtWidgets.QMessageBox.question(
        MainWindow,
        'Сохранение модели',
        f'Сохранить модель {model_name}?',
        QtWidgets.QMessageBox.Yes,
        QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        # Сохранение модели в файл с помощью pickle
        path_model = f'models/g_classifier/{model_name}_{round(test_accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y")}.pkl'
        if os.path.exists(path_model):
            path_model = f'models/g_classifier/{model_name}_{round(test_accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y_%H%M%S")}.pkl'
        with open(path_model, 'wb') as f:
            pickle.dump(pipe, f)

        text_model += get_text_train_point_geochem(data_train)
        text_model += get_text_train_param_geochem(list_param)

        new_trained_model = GeochemTrainedModel(
            maket_id = get_maket_id(),
            title=f'{model_name}_{round(test_accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y")}',
            path_model=path_model,
            list_params=json.dumps(list_param),
            comment=text_model
        )
        session.add(new_trained_model)
        session.commit()
        update_g_model_list()
    else:
        pass


def get_text_train_point_geochem(data):
    text = '\nМодель обучалась на точках:\n'
    list_cat = data['category'].unique().tolist()
    for cat in list_cat:
        text += f'Категория "{cat}":\n'
        for t in data.loc[data['category'] == cat, 'title']:
            if t != data.loc[data['category'] == cat, 'title'].iloc[-1]:
                text += f'{t}, '
            else:
                text += f'{t}\n'
    return text


def get_text_train_param_geochem(list_param):
    text = '\nПараметры модели:\n'
    text += ", ".join(list_param)
    return text























