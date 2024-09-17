from calc_additional_features import calc_wavelet_features, calc_fractal_features, calc_entropy_features, \
    calc_nonlinear_features, calc_morphology_features, calc_frequency_features, calc_envelope_feature, \
    calc_autocorr_feature, calc_emd_feature, calc_hht_features
from sklearn.metrics import mean_absolute_error

from func import *

class RegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, dropout_rate, activation_fn):
        super(RegressionModel, self).__init__()

        layers = []
        current_input_dim = input_dim

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


def push_random_param_reg():
    RandomParam = QtWidgets.QDialog()
    ui_rp = Ui_RandomParam()
    ui_rp.setupUi(RandomParam)
    RandomParam.show()
    RandomParam.setAttribute(Qt.WA_DeleteOnClose)

    m_width, m_height = get_width_height_monitor()
    RandomParam.resize(int(m_width / 1.5), int(m_height / 1.5))

    Regressor = QtWidgets.QDialog()
    ui_r = Ui_RegressorForm()
    ui_r.setupUi(Regressor)
    Regressor.show()
    Regressor.setAttribute(Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия


    def get_regmod_test_id():
        return ui_rp.comboBox_test_analysis.currentText().split(' id')[-1]


    def check_spinbox(spinbox1, spinbox2):
        spinbox1.setMaximum(spinbox2.value())
        spinbox2.setMinimum(spinbox1.value())

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

    def get_test_reg_id():
        return ui_rp.comboBox_test_analysis.currentText().split(' id')[-1]

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

    def build_table_random_param_reg(analisis_id: int, list_param: list) -> (pd.DataFrame, list):

        locals_dict = inspect.currentframe().f_back.f_locals #
        data_train = pd.DataFrame(columns=['prof_well_index', 'target_value'])

        # Получаем размеченные участки
        markups = session.query(MarkupReg).filter_by(analysis_id=analisis_id).all()
        ui.progressBar.setMaximum(len(markups))

        for nm, markup in enumerate(tqdm(markups)):
            # Получение списка фиктивных меток и границ слоев из разметки
            list_fake = json.loads(markup.list_fake) if markup.list_fake else []
            list_up = json.loads(markup.formation.layer_up.layer_line)
            list_down = json.loads(markup.formation.layer_down.layer_line)

            for measure in json.loads(markup.list_measure):
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
                    if i in list_fractal_features:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_fractal_features(markup.formation_id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                     json.loads(session.query(literal_column(f'fractal_feature.{i}')).filter(
                                         FractalFeature.formation_id == markup.formation_id
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
                    if i in list_nonlinear_features:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_nonlinear_features(markup.formation_id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                     json.loads(session.query(literal_column(f'nonlinear_feature.{i}')).filter(
                                         NonlinearFeature.formation_id == markup.formation_id
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
                    if i in list_frequency_feature:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_frequency_features(markup.formation_id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                     json.loads(session.query(literal_column(f'frequency_feature.{i}')).filter(
                                         FrequencyFeature.formation_id == markup.formation_id
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
                    if i in list_autocorr_feature:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_autocorr_feature(markup.formation_id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                     json.loads(session.query(literal_column(f'autocorr_feature.{i}')).filter(
                                         AutocorrFeature.formation_id == markup.formation_id
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
                    if i in list_hht_feature:
                        if not str(markup.profile.id) + '_' + i in locals_dict:
                            calc_hht_features(markup.formation_id)
                            locals_dict.update(
                                {str(markup.profile.id) + '_' + i:
                                     json.loads(session.query(literal_column(f'hht_feature.{i}')).filter(
                                         HHTFeature.formation_id == markup.formation_id
                                     ).first()[0])}
                            )

            # Обработка каждого измерения в разметке
            for measure in json.loads(markup.list_measure):
                # Пропустить измерение, если оно является фиктивным
                if measure in list_fake:
                    continue

                dict_value = {}
                dict_value['prof_well_index'] = f'{markup.profile_id}_{markup.well_id}_{measure}'
                dict_value['target_value'] = markup.target_value

                # Обработка каждого параметра в списке параметров
                for param in list_param:
                    if param.startswith('sig') or param.startswith('distr') or param.startswith('sep') or param.startswith('mfcc'):
                        if param.startswith('sig'):
                            p, atr, up, down = param.split('_')[0], param.split('_')[1], int(param.split('_')[2]), 512 - int(param.split('_')[3])
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

                # Добавление данных в обучающую выборку
                data_train = pd.concat([data_train, pd.DataFrame([dict_value])], ignore_index=True)

            ui.progressBar.setValue(nm + 1)

        new_list_param = []
        for param in list_param:
            if param.startswith('sig') or param.startswith('distr') or param.startswith('sep') or param.startswith(
                    'mfcc'):
                if param.startswith('sig'):
                    p, atr, up, down = param.split('_')[0], param.split('_')[1], int(param.split('_')[2]), 512 - int(
                        param.split('_')[3])
                    for i_sig in range(up, down):
                        new_list_param.append(f'sig_{atr}_{i_sig + 1}')
                else:
                    p, atr, n = param.split('_')[0], param.split('_')[1], int(param.split('_')[2])
                    for i_sig in range(n):
                        new_list_param.append(f'{p}_{atr}_{i_sig + 1}')
            else:
                new_list_param.append(param)

        return data_train, new_list_param



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
                n_sig_bot = 512 - (n_sig_top + n_sig_bot)
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
            list_param_attr += ['width', 'top', 'land', 'speed', 'speed_cover']
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

        n_param = random.randint(1, len(list_param_attr))
        list_param_all = random_combination(list_param_attr, n_param)

        print(len(list_param_all), list_param_all)
        return list_param_all


    def update_list_test_well():
        ui_rp.listWidget_test_point.clear()
        count_markup, count_measure, count_fake = 0, 0, 0
        for i in session.query(MarkupReg).filter(MarkupReg.analysis_id == get_test_reg_id()).all():
            fake = len(json.loads(i.list_fake)) if i.list_fake else 0
            measure = len(json.loads(i.list_measure))
            if i.type_markup == 'intersection':
                try:
                    inter_name = session.query(Intersection.name).filter(Intersection.id == i.well_id).first()[0]
                except TypeError:
                    session.query(MarkupReg).filter(MarkupReg.id == i.id).delete()
                    session.commit()
                    continue
                item = f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | {inter_name} | {measure - fake} из {measure} | id{i.id}'
            elif i.type_markup == 'profile':
                item = f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | | {measure - fake} из {measure} | id{i.id}'
            else:
                item = f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | {i.well.name} | {measure - fake} из {measure} | id{i.id}'
            ui_rp.listWidget_test_point.addItem(item)

            count_markup += 1
            count_measure += measure - fake
            count_fake += fake



    def update_test_analysis_combobox():
        ui_rp.comboBox_test_analysis.clear()
        for i in session.query(AnalysisReg).order_by(AnalysisReg.title).all():
            ui_rp.comboBox_test_analysis.addItem(f'{i.title} id{i.id}')
            update_list_test_well()

    def build_torch_model(pipe_steps, x_train):
        output_dim = 1

        epochs = ui_r.spinBox_epochs_torch.value()
        learning_rate = ui_r.doubleSpinBox_lr_torch.value()
        hidden_units = list(map(int, ui_r.lineEdit_layers_torch.text().split()))
        dropout_rate = ui_r.doubleSpinBox_dropout_torch.value()
        weight_decay = ui_r.doubleSpinBox_decay_torch.value()

        if ui_r.comboBox_activation_torch.currentText() == 'ReLU':
            activation_function = 'relu'
        elif ui_r.comboBox_activation_torch.currentText() == 'Sigmoid':
            activation_function = 'sigmoid'
        elif ui_r.comboBox_activation_torch.currentText() == 'Tanh':
            activation_function = 'tanh'

        if ui_r.comboBox_optimizer_torch.currentText() == 'Adam':
            optimizer = torch.optim.Adam
        elif ui_r.comboBox_optimizer_torch.currentText() == 'SGD':
            optimizer = torch.optim.SGD

        if ui_r.comboBox_loss_torch.currentText() == 'MSE':
            loss_function = nn.MSELoss
        elif ui_r.comboBox_loss_torch.currentText() == 'MAE':
            loss_function = nn.L1Loss
        elif ui_r.comboBox_loss_torch.currentText() == 'HuberLoss':
            loss_function = nn.HuberLoss
        elif ui_r.comboBox_loss_torch.currentText() == 'SmoothL1Loss':
            loss_function = nn.SmoothL1Loss

        patience = 0
        early_stopping_flag = False
        if ui_r.checkBox_estop_torch.isChecked():
            early_stopping_flag = True
            patience = ui_r.spinBox_stop_patience.value()

        early_stopping = EarlyStopping(
            monitor='valid_loss',
            patience=patience,
            threshold=1e-4,
            threshold_mode='rel',
            lower_is_better=True,
        )

        model = RegressionModel(x_train.shape[1], output_dim, hidden_units, dropout_rate,
                                activation_function)

        net = NeuralNetRegressor(
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
                            ui_r.comboBox_optimizer_torch.currentText() + '\ncriterion: ' + \
                            ui_r.comboBox_loss_torch.currentText() + '\nepochs: ' + str(epochs)

        return pipeline, text_model


    def choice_model_regressor(model):
        """ Выбор модели регрессии """
        if model == 'MLPR':
            model_reg = MLPRegressor(
                hidden_layer_sizes=tuple(map(int, ui_r.lineEdit_layer_mlp.text().split())),
                activation=ui_r.comboBox_activation_mlp.currentText(),
                solver=ui_r.comboBox_solvar_mlp.currentText(),
                alpha=ui_r.doubleSpinBox_alpha_mlp.value(),
                learning_rate_init=ui_r.doubleSpinBox_lr_mlp.value(),
                max_iter=5000,
                early_stopping=ui_r.checkBox_e_stop_mlp.isChecked(),
                validation_fraction=ui_r.doubleSpinBox_valid_mlp.value(),
                random_state=0
            )
            text_model = (
                f'**MLPR**: \n'
                f'hidden_layer_sizes: ({",".join(map(str, tuple(map(int, ui_r.lineEdit_layer_mlp.text().split()))))}), '
                f'\nactivation: {ui_r.comboBox_activation_mlp.currentText()}, '
                f'\nsolver: {ui_r.comboBox_solvar_mlp.currentText()}, '
                f'\nalpha: {round(ui_r.doubleSpinBox_alpha_mlp.value(), 2)}, '
                f'\nlearning_rate: {round(ui_r.doubleSpinBox_lr_mlp.value(), 2)}'
                f'\n{"early stopping, " if ui_r.checkBox_e_stop_mlp.isChecked() else ""}'
                f'\nvalidation_fraction: {round(ui_r.doubleSpinBox_valid_mlp.value(), 2)}'
            )
        elif model == 'KNNR':
            n_knn = ui_r.spinBox_neighbors.value()
            weights_knn = 'distance' if ui_r.checkBox_knn_weights.isChecked() else 'uniform'
            model_reg = KNeighborsRegressor(n_neighbors=n_knn, weights=weights_knn, algorithm='auto')
            text_model = f'**KNNR**: \nn_neighbors: {n_knn}, \nweights: {weights_knn}, '

        elif model == 'GBR':
            est = ui_r.spinBox_n_estimators_gbr.value()
            l_rate = ui_r.doubleSpinBox_learning_rate_gbr.value()
            model_reg = GradientBoostingRegressor(n_estimators=est,
                                                  learning_rate=l_rate,
                                                  max_depth=ui_r.spinBox_depth_gbr.value(),
                                                  min_samples_split=ui_r.spinBox_min_sample_split_gbr.value(),
                                                  min_samples_leaf=ui_r.spinBox_min_sample_leaf_gbr.value(),
                                                  subsample=ui_r.doubleSpinBox_subsample_gbr.value(),
                                                  random_state=0)
            text_model = f'**GBR**: \nn estimators: {round(est, 2)}, \nlearning rate: {round(l_rate, 2)}, ' \
                         f'max_depth: {ui_r.spinBox_depth_gbr.value()}, \nmin_samples_split: ' \
                         f'{ui_r.spinBox_min_sample_split_gbr.value()}, \nmin_samples_leaf: ' \
                         f'{ui_r.spinBox_min_sample_leaf_gbr.value()} \nsubsample: ' \
                         f'{round(ui_r.doubleSpinBox_subsample_gbr.value(), 2)}, '


        elif model == 'LR':
            model_reg = LinearRegression(fit_intercept=ui_r.checkBox_fit_intercept.isChecked())
            text_model = f'**LR**: \nfit_intercept: {"on" if ui_r.checkBox_fit_intercept.isChecked() else "off"}, '

        elif model == 'DTR':
            spl = 'random' if ui_r.checkBox_splitter_rnd.isChecked() else 'best'
            model_reg = DecisionTreeRegressor(splitter=spl, random_state=0)
            text_model = f'**DTR**: \nsplitter: {spl}, '

        elif model == 'RFR':
            model_reg = RandomForestRegressor(n_estimators=ui_r.spinBox_rfr_n.value(),
                                              max_depth=ui_r.spinBox_depth_rfr.value(),
                                              min_samples_split=ui_r.spinBox_min_sample_split.value(),
                                              min_samples_leaf=ui_r.spinBox_min_sample_leaf.value(),
                                              max_features=ui_r.comboBox_max_features_rfr.currentText(),
                                              oob_score=True, random_state=0, n_jobs=-1)
            text_model = f'**RFR**: \nn estimators: {ui_r.spinBox_rfr_n.value()}, ' \
                         f'\nmax_depth: {ui_r.spinBox_depth_rfr.value()},' \
                         f'\nmin_samples_split: {ui_r.spinBox_min_sample_split.value()}, ' \
                         f'\nmin_samples_leaf: {ui_r.spinBox_min_sample_leaf.value()}, ' \
                         f'\nmax_features: {ui_r.comboBox_max_features_rfr.currentText()}, '

        elif model == 'ABR':
            model_reg = AdaBoostRegressor(n_estimators=ui_r.spinBox_rfr_n.value(), random_state=0)
            text_model = f'**ABR**: \nn estimators: {ui_r.spinBox_rfr_n.value()}, '

        elif model == 'ETR':
            model_reg = ExtraTreesRegressor(n_estimators=ui_r.spinBox_rfr_n.value(), bootstrap=True, oob_score=True,
                                            random_state=0, n_jobs=-1)
            text_model = f'**ETR**: \nn estimators: {ui_r.spinBox_rfr_n.value()}, '

        elif model == 'GPR':
            constant = ui_r.doubleSpinBox_gpr_const.value()
            scale = ui_r.doubleSpinBox_gpr_scale.value()
            n_restart_optimization = ui_r.spinBox_gpr_n_restart.value()
            kernel = ConstantKernel(constant) * RBF(scale)
            model_reg = GaussianProcessRegressor(
                kernel=kernel,
                alpha=ui_r.doubleSpinBox_gpr_alpha.value(),
                n_restarts_optimizer=n_restart_optimization,
                random_state=0
            )
            text_model = (f'**GPR**: '
                          f'\nkernal: {kernel}, '
                          f'\nscale: {round(scale, 3)}, '
                          f'\nconstant: {round(constant, 3)}, '
                          f'\nn restart: {n_restart_optimization} ,')

        elif model == 'SVR':
            model_reg = SVR(kernel=ui_r.comboBox_svr_kernel.currentText(), C=ui_r.doubleSpinBox_svr_c.value(),
                            epsilon=ui_r.doubleSpinBox_epsilon_svr.value())
            text_model = (f'**SVR**: \nkernel: {ui_r.comboBox_svr_kernel.currentText()}, '
                          f'\nC: {round(ui_r.doubleSpinBox_svr_c.value(), 2)}, '
                          f'\nepsilon: {round(ui_r.doubleSpinBox_epsilon_svr.value(), 2)}')

        elif model == 'EN':
            model_reg = ElasticNet(
                alpha=ui_r.doubleSpinBox_alpha.value(),
                l1_ratio=ui_r.doubleSpinBox_l1_ratio.value(),
                random_state=0
            )
            text_model = (f'**EN**: \nalpha: {round(ui_r.doubleSpinBox_alpha.value(), 2)}, '
                          f'\nl1_ratio: {round(ui_r.doubleSpinBox_l1_ratio.value(), 2)}, ')

        elif model == 'LSS':
            model_reg = Lasso(alpha=ui_r.doubleSpinBox_alpha.value(), random_state=0)
            text_model = f'**LSS**: \nalpha: {round(ui_r.doubleSpinBox_alpha.value(), 2)}, '

        elif model == 'XGB':
            model_reg = XGBRegressor(n_estimators=ui_r.spinBox_n_estimators_xgb.value(),
                                     learning_rate=ui_r.doubleSpinBox_learning_rate_xgb.value(),
                                     max_depth=ui_r.spinBox_depth_xgb.value(),
                                     alpha=ui_r.doubleSpinBox_alpha_xgb.value(), booster='gbtree', random_state=0)
            text_model = f'**XGB**: \nn estimators: {ui_r.spinBox_n_estimators_xgb.value()}, ' \
                         f'\nlearning_rate: {ui_r.doubleSpinBox_learning_rate_xgb.value()}, ' \
                         f'\nmax_depth: {ui_r.spinBox_depth_xgb.value()} \nalpha: {ui_r.doubleSpinBox_alpha_xgb.value()}'


        elif model == 'LGBM':
            model_reg = lgb.LGBMRegressor(
                objective='binary',
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

        else:
            model_reg = QuadraticDiscriminantAnalysis()
            text_model = ''

        return model_reg, text_model

    def result_analysis(results, filename, reverse=True):
        if reverse is True:
            sorted_result = sorted(results, key=lambda x: x[0], reverse=True)
        else:
            sorted_result = sorted(results, key=lambda x: x[0], reverse=False)
        for item in sorted_result[:20]:
            print(item)

        twenty_percent = int(len(sorted_result) * 0.2)
        sorted_result = sorted_result[:twenty_percent]
        result_param = [item[3] for item in sorted_result]
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



    def test_classif_regressor(estimators, data_val, list_param, filename):
        list_estimator_accuracy = []
        list_estimator_mse = []
        list_estimator_r2 = []
        for model in estimators:
            start_time = datetime.datetime.now()
            data_test = data_val.copy()
            working_sample = data_test.iloc[:, 2:].values.tolist()
            try:
                y_pred = model.predict(working_sample)
            except ValueError:
                working_sample = imputer.fit_transform(working_sample)
                y_pred = model.predict(working_sample)

                for i in working_sample.index:
                    p_nan = [working_sample.columns[ic + 3] for ic, v in enumerate(working_sample.iloc[i, 3:].tolist()) if
                             np.isnan(v)]
                    if len(p_nan) > 0:
                        set_info(
                            f'Внимание для измерения "{i}" отсутствуют параметры "{", ".join(p_nan)}", поэтому расчёт для'
                            f' этого измерения может быть не корректен', 'red')
            data_test['y_pred'] = y_pred
            data_test['diff'] = data_test['target_value'] - data_test['y_pred']
            accuracy = model.score(working_sample, data_test['target_value'].values.tolist())
            mae = round(mean_absolute_error(data_test['target_value'].values.tolist(),
                                            data_test['y_pred'].values.tolist()), 5)
            mse = round(mean_squared_error(data_test['target_value'].values.tolist(),
                                           data_test['y_pred'].values.tolist()), 5)
            r2 = r2_score(data_test['target_value'].values.tolist(), data_test['y_pred'].values.tolist())

            list_estimator_accuracy.append(mae)
            list_estimator_mse.append(mse)
            list_estimator_r2.append(r2)

            ui_rp.textEdit_test_result.setTextColor(Qt.darkGreen)
            ui_rp.textEdit_test_result.append(f"{datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n"
                      f"Тестирование модели {model}\n"
                      f"Количество параметров: {len(list_param)}\n"
                      f"MAE: {mae:.3f}"
                      f'\nMSE: {mse:.3f}'
                        f'\nR2: {r2:.3f}')

            with open(filename, 'a') as f:
                print(f"{datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n"
                      f"Тестирование модели {model}\n"
                      f"Количество параметров: {len(list_param)}\n"
                      f"MAE: {mae:.3f}"
                      f'\nMSE: {mse:.3f}\nR2: {r2:.3f}', file=f)

            index = 0
            while index + 1 < len(data_test):
                comp, total = 0, 0
                diff_list, list_y = [], []
                while index + 1 < len(data_test) and \
                        data_test.loc[index, 'prof_well_index'].split('_')[0] == \
                        data_test.loc[index + 1, 'prof_well_index'].split('_')[0] and \
                        data_test.loc[index, 'prof_well_index'].split('_')[1] == \
                        data_test.loc[index + 1, 'prof_well_index'].split('_')[1]:
                    list_y.append(data_test.loc[index, 'y_pred'])
                    diff_list.append(data_test.loc[index, 'diff'])
                    total += 1
                    index += 1
                if data_test.loc[index, 'prof_well_index'].split('_')[1] == \
                        data_test.loc[index - 1, 'prof_well_index'].split('_')[1]:
                    total += 1
                if total == 0: total = 1
                profile = session.query(Profile).filter(
                    Profile.id == data_test.loc[index, 'prof_well_index'].split('_')[0]).first()
                well = session.query(Well).filter(
                    Well.id == data_test.loc[index, 'prof_well_index'].split('_')[1]).first()

                min_target, max_target = data_test['target_value'].min(), data_test['target_value'].max()
                lin_target = np.linspace(0, max_target - min_target, data_test['target_value'].size)
                percentile_20 = np.percentile(lin_target, 20)
                percentile_50 = np.percentile(lin_target, 50)
                mistake = math.fabs(round(data_test.loc[index, "target_value"] - sum(list_y) / total, 2))
                mean_pred = round(sum(list_y) / total, 2)

                color_text = Qt.black
                if percentile_20 <= mistake < percentile_50:
                    color_text = Qt.darkYellow
                if mistake >= percentile_50:
                    color_text = Qt.red
                ui_rp.textEdit_test_result.setTextColor(color_text)
                ui_rp.textEdit_test_result.insertPlainText(
                    f'{profile.research.object.title} - {profile.title} | Скв. {well.name} |'
                    f' predict {mean_pred} | target {round(data_test.loc[index, "target_value"], 2)} '
                    f'| погрешность: {mistake}\n')
                with open(filename, 'a') as f:
                    print(f'{profile.research.object.title} - {profile.title} | Скв. {well.name} |'
                          f' predict {mean_pred} | target {round(data_test.loc[index, "target_value"], 2)} '
                          f'| погрешность: {mistake}\n', file=f)
                index += 1

            ui_rp.textEdit_test_result.setTextColor(Qt.black)
            ui_rp.textEdit_test_result.append(f"Время работы: {datetime.datetime.now() - start_time}")
            with open(filename, 'a') as f:
                print(f"Время работы: {datetime.datetime.now() - start_time}", file=f)

        return list_estimator_accuracy, list_estimator_mse, list_estimator_r2, filename



    def start_random_param():
        start_time = datetime.datetime.now()
        filename, _ = QFileDialog.getSaveFileName(caption="Сохранить результаты подбора параметров?",
                                                  filter="TXT (*.txt)")
        with open(filename, 'w') as f:
            print(f"START SEARCH PARAMS\n{datetime.datetime.now()}\n\n", file=f)
        results = []

        for i in range(ui_rp.spinBox_n_iter.value()):
            pipe_steps = []
            text_scaler = ''
            if ui_r.checkBox_stdscaler_reg.isChecked():
                std_scaler = StandardScaler()
                pipe_steps.append(('std_scaler', std_scaler))
                text_scaler += '\nStandardScaler'
            if ui_r.checkBox_robscaler_reg.isChecked():
                robust_scaler = RobustScaler()
                pipe_steps.append(('robust_scaler', robust_scaler))
                text_scaler += '\nRobustScaler'
            if ui_r.checkBox_mnmxscaler_reg.isChecked():
                minmax_scaler = MinMaxScaler()
                pipe_steps.append(('minmax_scaler', minmax_scaler))
                text_scaler += '\nMinMaxScaler'
            if ui_r.checkBox_mxabsscaler_reg.isChecked():
                maxabs_scaler = MaxAbsScaler()
                pipe_steps.append(('maxabs_scaler', maxabs_scaler))
                text_scaler += '\nMaxAbsScaler'

            list_param = build_list_param()
            data_train, new_list_param = build_table_random_param_reg(get_regmod_id(), list_param)

            # Замена inf на 0
            data_train[new_list_param] = data_train[new_list_param].replace([np.inf, -np.inf], 0)

            list_col = data_train.columns.tolist()
            data_train = pd.DataFrame(imputer.fit_transform(data_train), columns=list_col)

            y_train = data_train['target_value'].values
            if not np.issubdtype(y_train.dtype, np.number):
                y_train = pd.to_numeric(y_train, errors='coerce')
            data_test, new_list_param = build_table_random_param_reg(get_regmod_test_id(), list_param)

            # Замена inf на 0
            data_test[new_list_param] = data_test[new_list_param].replace([np.inf, -np.inf], 0)

            list_col = data_test.columns.tolist()
            data_test = pd.DataFrame(imputer.fit_transform(data_test), columns=list_col)

            model_name = ui_r.buttonGroup.checkedButton().text()
            if model_name == 'TORCH':
                pipe, text_model = build_torch_model(pipe_steps, data_train.iloc[:, 2:])
                text_model += text_scaler
            else:
                model_reg, text_model = choice_model_regressor(model_name)
                text_model += text_scaler
                pipe_steps.append(('model', model_reg))
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

            kv = KFold(n_splits=5, shuffle=True, random_state=0)
            cv_results = cross_validate(pipe, data_train.iloc[:, 2:], np.array(y_train, dtype=np.float32), cv=kv, scoring='r2',
                                        return_estimator=True)
            estimators = cv_results['estimator']

            list_accuracy, list_mse, list_r2, filename = test_classif_regressor(estimators,data_test,list_param, filename)
            mae = np.array(list_accuracy).mean()
            mse = np.array(list_mse).mean()
            r2 = np.array(list_r2).mean()
            results_list = [mae, mse, r2, list_param]
            results.append(results_list)

            with open(filename, 'a') as f:
                print(f'\n!!!RESULT!!!\nMAE mean: {mae}\nMSE mean: {mse}\nR2 mean: {r2}', file=f)
            print(f'\n!!!RESULT!!!\nMAE mean: {mae}\nMSE mean: {mse}\nR2 mean: {r2}')

        result_analysis(results, filename, reverse=True)
        result_analysis(results, filename, reverse=False)
        # draw_result_rnd_prm(results)

        end_time = datetime.datetime.now() - start_time
        with open(filename, 'a') as f:
            print(f'\nВремя выполнения: {end_time}', file=f)
        ui_rp.textEdit_test_result.setTextColor(Qt.red)
        ui_rp.textEdit_test_result.insertPlainText(f'Время выполнения: {end_time}\n')
        print(f'Время выполнения: {end_time}')

    update_test_analysis_combobox()
    update_list_test_well()

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

    Regressor.exec_()
