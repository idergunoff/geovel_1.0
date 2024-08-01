from sklearn.metrics import mean_absolute_error

from func import *


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

    ui_rp.groupBox_minmax_attr.hide()

    def get_regmod_test_id():
        return ui_rp.comboBox_test_analysis.currentText().split(' id')[-1]

    def check_checkbox_flp_dstr_ts():
        push = True if ui_rp.checkBox_flp_dstr_ts_all.isChecked() else False
        ui_rp.checkBox_flp_dstr_ts_a.setChecked(push)
        ui_rp.checkBox_flp_dstr_ts_at.setChecked(push)
        ui_rp.checkBox_flp_dstr_ts_vt.setChecked(push)
        ui_rp.checkBox_flp_dstr_ts_pht.setChecked(push)
        ui_rp.checkBox_flp_dstr_ts_wt.setChecked(push)
        ui_rp.checkBox_flp_dstr_ts_diff.setChecked(push)
        ui_rp.checkBox_flp_dstr_ts_crl.setChecked(push)


    def check_checkbox_flp_sep_ts():
        push = True if ui_rp.checkBox_flp_sep_ts_all.isChecked() else False
        ui_rp.checkBox_flp_sep_ts_a.setChecked(push)
        ui_rp.checkBox_flp_sep_ts_at.setChecked(push)
        ui_rp.checkBox_flp_sep_ts_vt.setChecked(push)
        ui_rp.checkBox_flp_sep_ts_pht.setChecked(push)
        ui_rp.checkBox_flp_sep_ts_wt.setChecked(push)
        ui_rp.checkBox_flp_sep_ts_diff.setChecked(push)
        ui_rp.checkBox_flp_sep_ts_crl.setChecked(push)


    def check_checkbox_flp_mfcc_ts():
        push = True if ui_rp.checkBox_flp_mfcc_ts_all.isChecked() else False
        ui_rp.checkBox_flp_mfcc_ts_a.setChecked(push)
        ui_rp.checkBox_flp_mfcc_ts_at.setChecked(push)
        ui_rp.checkBox_flp_mfcc_ts_vt.setChecked(push)
        ui_rp.checkBox_flp_mfcc_ts_pht.setChecked(push)
        ui_rp.checkBox_flp_mfcc_ts_wt.setChecked(push)
        ui_rp.checkBox_flp_mfcc_ts_diff.setChecked(push)
        ui_rp.checkBox_flp_mfcc_ts_crl.setChecked(push)


    def check_checkbox_flp_sig_ts():
        push = True if ui_rp.checkBox_flp_sig_ts_all.isChecked() else False
        ui_rp.checkBox_flp_sig_ts_a.setChecked(push)
        ui_rp.checkBox_flp_sig_ts_at.setChecked(push)
        ui_rp.checkBox_flp_sig_ts_vt.setChecked(push)
        ui_rp.checkBox_flp_sig_ts_pht.setChecked(push)
        ui_rp.checkBox_flp_sig_ts_wt.setChecked(push)
        ui_rp.checkBox_flp_sig_ts_diff.setChecked(push)
        ui_rp.checkBox_flp_sig_ts_crlnf.setChecked(push)
        ui_rp.checkBox_flp_sig_ts_crl.setChecked(push)


    def check_group_attr():
        if ui_rp.checkBox_group.isChecked():
            ui_rp.groupBox_minmax_attr.hide()
        else:
            ui_rp.groupBox_minmax_attr.show()

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
        ui_rp.checkBox_prior.setChecked(push)
        ui_rp.checkBox_stat.setChecked(push)
        ui_rp.checkBox_crl_stat.setChecked(push)

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
        return data_train, list_param



    def build_list_param():
        list_param_all = []

        if ui_rp.checkBox_flp.isChecked():

            # signal
            n_sig_top = random.randint(ui_rp.spinBox_flp_top_skip_up.value(), ui_rp.spinBox_flp_top_skip_down.value())
            n_sig_bot = random.randint(ui_rp.spinBox_flp_bot_skip_up.value(), ui_rp.spinBox_flp_bot_skip_down.value())

            if ui_rp.checkBox_flp_sig_width.isChecked():
                if n_sig_top + n_sig_bot > 505:
                    n_sig_bot = 0
                else:
                    n_sig_bot = 512 - (n_sig_top + n_sig_bot)
            else:
                if n_sig_bot + n_sig_top > 505:
                    n_sig_bot = random.randint(0, 505 - n_sig_top)

            if ui_rp.checkBox_flp_sig_ts_a.isChecked():
                list_param_all.append(f'sig_Abase_{n_sig_top}_{n_sig_bot}')
            if ui_rp.checkBox_flp_sig_ts_at.isChecked():
                list_param_all.append(f'sig_At_{n_sig_top}_{n_sig_bot}')
            if ui_rp.checkBox_flp_sig_ts_vt.isChecked():
                list_param_all.append(f'sig_Vt_{n_sig_top}_{n_sig_bot}')
            if ui_rp.checkBox_flp_sig_ts_pht.isChecked():
                list_param_all.append(f'sig_Pht_{n_sig_top}_{n_sig_bot}')
            if ui_rp.checkBox_flp_sig_ts_wt.isChecked():
                list_param_all.append(f'sig_Wt_{n_sig_top}_{n_sig_bot}')
            if ui_rp.checkBox_flp_sig_ts_diff.isChecked():
                list_param_all.append(f'sig_diff_{n_sig_top}_{n_sig_bot}')
            if ui_rp.checkBox_flp_sig_ts_crlnf.isChecked():
                list_param_all.append(f'sig_CRLNF_{n_sig_top}_{n_sig_bot}')
            if ui_rp.checkBox_flp_sig_ts_crl.isChecked():
                list_param_all.append(f'sig_CRL_{n_sig_top}_{n_sig_bot}')

            # distribution
            n_distr = random.randint(ui_rp.spinBox_flp_distr_up.value(), ui_rp.spinBox_flp_distr_down.value())

            if ui_rp.checkBox_flp_dstr_ts_a.isChecked():
                list_param_all.append(f'distr_Abase_{n_distr}')
            if ui_rp.checkBox_flp_dstr_ts_at.isChecked():
                list_param_all.append(f'distr_At_{n_distr}')
            if ui_rp.checkBox_flp_dstr_ts_vt.isChecked():
                list_param_all.append(f'distr_Vt_{n_distr}')
            if ui_rp.checkBox_flp_dstr_ts_pht.isChecked():
                list_param_all.append(f'distr_Pht_{n_distr}')
            if ui_rp.checkBox_flp_dstr_ts_wt.isChecked():
                list_param_all.append(f'distr_Wt_{n_distr}')
            if ui_rp.checkBox_flp_dstr_ts_diff.isChecked():
                list_param_all.append(f'distr_diff_{n_distr}')
            if ui_rp.checkBox_flp_dstr_ts_crl.isChecked():
                list_param_all.append(f'distr_CRL_{n_distr}')

            # Interpolate (sep)
            n_sep = random.randint(ui_rp.spinBox_flp_sep_up.value(), ui_rp.spinBox_flp_sep_down.value())
            if ui_rp.checkBox_flp_sep_ts_a.isChecked():
                list_param_all.append(f'sep_Abase_{n_sep}')
            if ui_rp.checkBox_flp_sep_ts_at.isChecked():
                list_param_all.append(f'sep_At_{n_sep}')
            if ui_rp.checkBox_flp_sep_ts_vt.isChecked():
                list_param_all.append(f'sep_Vt_{n_sep}')
            if ui_rp.checkBox_flp_sep_ts_pht.isChecked():
                list_param_all.append(f'sep_Pht_{n_sep}')
            if ui_rp.checkBox_flp_sep_ts_wt.isChecked():
                list_param_all.append(f'sep_Wt_{n_sep}')
            if ui_rp.checkBox_flp_sep_ts_diff.isChecked():
                list_param_all.append(f'sep_diff_{n_sep}')
            if ui_rp.checkBox_flp_sep_ts_crl.isChecked():
                list_param_all.append(f'sep_CRL_{n_sep}')

            n_mfcc = random.randint(ui_rp.spinBox_flp_mfcc_up.value(), ui_rp.spinBox_flp_mfcc_down.value())
            if ui_rp.checkBox_flp_mfcc_ts_a.isChecked():
                list_param_all.append(f'mfcc_Abase_{n_mfcc}')
            if ui_rp.checkBox_flp_mfcc_ts_at.isChecked():
                list_param_all.append(f'mfcc_At_{n_mfcc}')
            if ui_rp.checkBox_flp_mfcc_ts_vt.isChecked():
                list_param_all.append(f'mfcc_Vt_{n_mfcc}')
            if ui_rp.checkBox_flp_mfcc_ts_pht.isChecked():
                list_param_all.append(f'mfcc_Pht_{n_mfcc}')
            if ui_rp.checkBox_flp_mfcc_ts_wt.isChecked():
                list_param_all.append(f'mfcc_Wt_{n_mfcc}')
            if ui_rp.checkBox_flp_mfcc_ts_diff.isChecked():
                list_param_all.append(f'mfcc_diff_{n_mfcc}')
            if ui_rp.checkBox_flp_mfcc_ts_crl.isChecked():
                list_param_all.append(f'mfcc_CRL_{n_mfcc}')

            # Attributes
            if ui_rp.checkBox_a_top.isChecked():
                list_param_all.append('A_top')
            if ui_rp.checkBox_a_bot.isChecked():
                list_param_all.append('A_bottom')
            if ui_rp.checkBox_da.isChecked():
                list_param_all.append('dA')
            if ui_rp.checkBox_a_mean.isChecked():
                list_param_all.append('A_mean')
            if ui_rp.checkBox_a_sum.isChecked():
                list_param_all.append('A_sum')
            if ui_rp.checkBox_a_max.isChecked():
                list_param_all.append('A_max')
            if ui_rp.checkBox_a_t_max.isChecked():
                list_param_all.append('A_T_max')
            if ui_rp.checkBox_a_sn.isChecked():
                list_param_all.append('A_Sn')
            if ui_rp.checkBox_a_wmf.isChecked():
                list_param_all.append('A_wmf')
            if ui_rp.checkBox_a_qf.isChecked():
                list_param_all.append('A_Qf')
            if ui_rp.checkBox_a_sn_wmf.isChecked():
                list_param_all.append('A_Sn_wmf')

            if ui_rp.checkBox_at_top.isChecked():
                list_param_all.append('At_top')
            if ui_rp.checkBox_dat.isChecked():
                list_param_all.append('dAt')
            if ui_rp.checkBox_at_mean.isChecked():
                list_param_all.append('At_mean')
            if ui_rp.checkBox_at_sum.isChecked():
                list_param_all.append('At_sum')
            if ui_rp.checkBox_at_max.isChecked():
                list_param_all.append('At_max')
            if ui_rp.checkBox_at_t_max.isChecked():
                list_param_all.append('At_T_max')
            if ui_rp.checkBox_at_sn.isChecked():
                list_param_all.append('At_Sn')
            if ui_rp.checkBox_at_wmf.isChecked():
                list_param_all.append('At_wmf')
            if ui_rp.checkBox_at_qf.isChecked():
                list_param_all.append('At_Qf')
            if ui_rp.checkBox_at_sn_wmf.isChecked():
                list_param_all.append('At_Sn_wmf')

            if ui_rp.checkBox_vt_top.isChecked():
                list_param_all.append('Vt_top')
            if ui_rp.checkBox_dvt.isChecked():
                list_param_all.append('dVt')
            if ui_rp.checkBox_vt_mean.isChecked():
                list_param_all.append('Vt_mean')
            if ui_rp.checkBox_vt_sum.isChecked():
                list_param_all.append('Vt_sum')
            if ui_rp.checkBox_vt_max.isChecked():
                list_param_all.append('Vt_max')
            if ui_rp.checkBox_vt_t_max.isChecked():
                list_param_all.append('Vt_T_max')
            if ui_rp.checkBox_vt_sn.isChecked():
                list_param_all.append('Vt_Sn')
            if ui_rp.checkBox_vt_wmf.isChecked():
                list_param_all.append('Vt_wmf')
            if ui_rp.checkBox_vt_qf.isChecked():
                list_param_all.append('Vt_Qf')
            if ui_rp.checkBox_vt_sn_wmf.isChecked():
                list_param_all.append('Vt_Sn_wmf')

            if ui_rp.checkBox_pht_top.isChecked():
                list_param_all.append('Pht_top')
            if ui_rp.checkBox_dpht.isChecked():
                list_param_all.append('dPht')
            if ui_rp.checkBox_pht_mean.isChecked():
                list_param_all.append('Pht_mean')
            if ui_rp.checkBox_pht_sum.isChecked():
                list_param_all.append('Pht_sum')
            if ui_rp.checkBox_pht_max.isChecked():
                list_param_all.append('Pht_max')
            if ui_rp.checkBox_pht_t_max.isChecked():
                list_param_all.append('Pht_T_max')
            if ui_rp.checkBox_pht_sn.isChecked():
                list_param_all.append('Pht_Sn')
            if ui_rp.checkBox_pht_wmf.isChecked():
                list_param_all.append('Pht_wmf')
            if ui_rp.checkBox_pht_qf.isChecked():
                list_param_all.append('Pht_Qf')
            if ui_rp.checkBox_pht_sn_wmf.isChecked():
                list_param_all.append('Pht_Sn_wmf')

            if ui_rp.checkBox_wt_top.isChecked():
                list_param_all.append('Wt_top')
            if ui_rp.checkBox_wt_mean.isChecked():
                list_param_all.append('Wt_mean')
            if ui_rp.checkBox_wt_sum.isChecked():
                list_param_all.append('Wt_sum')
            if ui_rp.checkBox_wt_max.isChecked():
                list_param_all.append('Wt_max')
            if ui_rp.checkBox_wt_t_max.isChecked():
                list_param_all.append('Wt_T_max')
            if ui_rp.checkBox_wt_sn.isChecked():
                list_param_all.append('Wt_Sn')
            if ui_rp.checkBox_wt_wmf.isChecked():
                list_param_all.append('Wt_wmf')
            if ui_rp.checkBox_wt_qf.isChecked():
                list_param_all.append('Wt_Qf')
            if ui_rp.checkBox_wt_sn_wmf.isChecked():
                list_param_all.append('Wt_Sn_wmf')

            if ui_rp.checkBox_crl_top.isChecked():
                list_param_all.append('CRL_top')
            if ui_rp.checkBox_crl_mean.isChecked():
                list_param_all.append('CRL_mean')
            if ui_rp.checkBox_crl_sum.isChecked():
                list_param_all.append('CRL_sum')
            if ui_rp.checkBox_crl_max.isChecked():
                list_param_all.append('CRL_max')
            if ui_rp.checkBox_crl_t_max.isChecked():
                list_param_all.append('CRL_T_max')
            if ui_rp.checkBox_crl_sn.isChecked():
                list_param_all.append('CRL_Sn')
            if ui_rp.checkBox_crl_wmf.isChecked():
                list_param_all.append('CRL_wmf')
            if ui_rp.checkBox_crl_qf.isChecked():
                list_param_all.append('CRL_Qf')
            if ui_rp.checkBox_crl_sn_wmf.isChecked():
                list_param_all.append('CRL_Sn_wmf')

            if ui_rp.checkBox_t_top.isChecked():
                list_param_all.append('T_top')
            if ui_rp.checkBox_t_bot.isChecked():
                list_param_all.append('T_bottom')
            if ui_rp.checkBox_dt.isChecked():
                list_param_all.append('dT')

            if ui_rp.checkBox_skew.isChecked():
                list_param_all.append('skew')
            if ui_rp.checkBox_kurt.isChecked():
                list_param_all.append('kurt')
            if ui_rp.checkBox_std.isChecked():
                list_param_all.append('std')
            if ui_rp.checkBox_k_var.isChecked():
                list_param_all.append('k_var')

            if ui_rp.checkBox_crl_skew.isChecked():
                list_param_all.append('CRL_skew')
            if ui_rp.checkBox_crl_kurt.isChecked():
                list_param_all.append('CRL_kurt')
            if ui_rp.checkBox_crl_std.isChecked():
                list_param_all.append('CRL_std')
            if ui_rp.checkBox_crl_k_var.isChecked():
                list_param_all.append('CRL_k_var')

            if ui_rp.checkBox_speed.isChecked():
                list_param_all.append('speed')
            if ui_rp.checkBox_speed_cover.isChecked():
                list_param_all.append('speed_cover')
            if ui_rp.checkBox_attr_width.isChecked():
                list_param_all.append('width')
            if ui_rp.checkBox_top.isChecked():
                list_param_all.append('top')
            if ui_rp.checkBox_land.isChecked():
                list_param_all.append('land')


            n_param = random.randint(1, len(list_param_all))
            list_param_all = random_combination(list_param_all, n_param)

        else:

            n_distr = random.randint(ui_rp.spinBox_distr_up.value(), ui_rp.spinBox_distr_down.value())
            n_sep = random.randint(ui_rp.spinBox_sep_up.value(), ui_rp.spinBox_sep_down.value())
            n_mfcc = random.randint(ui_rp.spinBox_mfcc_up.value(), ui_rp.spinBox_mfcc_down.value())
            n_sig_top = random.randint(ui_rp.spinBox_top_skip_up.value(), ui_rp.spinBox_top_skip_down.value())
            n_sig_bot = random.randint(ui_rp.spinBox_bot_skip_up.value(), ui_rp.spinBox_bot_skip_down.value())

            def get_n(ts: str) -> str:
                if ts == 'distr':
                    return str(n_distr)
                elif ts == 'sep':
                    return str(n_sep)
                elif ts == 'mfcc':
                    return str(n_mfcc)
                elif ts == 'sig':
                    return f'{n_sig_top}_{n_sig_bot}'

            list_param_group, list_ts = [], []
            if ui_rp.checkBox_distr.isChecked():
                list_ts.append('distr')
            if ui_rp.checkBox_sep.isChecked():
                list_ts.append('sep')
            if ui_rp.checkBox_mfcc.isChecked():
                list_ts.append('mfcc')
            if ui_rp.checkBox_sig.isChecked():
                list_ts.append('sig')

            if ui_rp.checkBox_ts_a.isChecked():
                for i in list_ts:
                    list_param_group.append(f'{i}_A')
            if ui_rp.checkBox_ts_at.isChecked():
                for i in list_ts:
                    list_param_group.append(f'{i}_At')
            if ui_rp.checkBox_ts_vt.isChecked():
                for i in list_ts:
                    list_param_group.append(f'{i}_Vt')
            if ui_rp.checkBox_ts_pht.isChecked():
                for i in list_ts:
                    list_param_group.append(f'{i}_Pht')
            if ui_rp.checkBox_ts_wt.isChecked():
                for i in list_ts:
                    list_param_group.append(f'{i}_Wt')
            if ui_rp.checkBox_ts_diff.isChecked():
                for i in list_ts:
                    list_param_group.append(f'{i}_diff')
            if ui_rp.checkBox_ts_crlnf.isChecked():
                if ui_rp.checkBox_sig.isChecked():
                    list_param_group.append(f'sig_CRLNF')
            if ui_rp.checkBox_ts_crl.isChecked():
                for i in list_ts:
                    list_param_group.append(f'{i}_CRL')

            if ui_rp.checkBox_group.isChecked():

                if ui_rp.checkBox_attr_a.isChecked():
                    list_param_group.append('attr_A')
                if ui_rp.checkBox_attr_at.isChecked():
                    list_param_group.append('attr_At')
                if ui_rp.checkBox_attr_vt.isChecked():
                    list_param_group.append('attr_Vt')
                if ui_rp.checkBox_attr_pht.isChecked():
                    list_param_group.append('attr_Pht')
                if ui_rp.checkBox_attr_wt.isChecked():
                    list_param_group.append('attr_Wt')
                if ui_rp.checkBox_attr_crl.isChecked():
                    list_param_group.append('attr_CRL')

                if ui_rp.checkBox_stat.isChecked():
                    list_param_group.append('stat')
                if ui_rp.checkBox_crl_stat.isChecked():
                    list_param_group.append('CRL_stat')

                if ui_rp.checkBox_form_t.isChecked():
                    list_param_group.append('form_t')
                if ui_rp.checkBox_prior.isChecked():
                    list_param_group.append('prior')

            else:
                list_param_attr = []
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
                if ui_rp.checkBox_prior.isChecked():
                    list_param_attr += ['width', 'top', 'land', 'speed', 'speed_cover']

            if ui_rp.checkBox_width.isChecked():
                if n_sig_top + n_sig_bot > 505:
                    n_sig_bot = 0
                else:
                    n_sig_bot = 512 - (n_sig_top + n_sig_bot)
            else:
                if n_sig_bot + n_sig_top > 505:
                    n_sig_bot = random.randint(0, 505 - n_sig_top)

            if ui_rp.checkBox_group.isChecked():
                n_param = random.randint(1, len(list_param_group))
                list_param_choice = random_combination(list_param_group, n_param)
            else:
                n_param_attr  = random.randint(ui_rp.spinBox_min_attr.value(), ui_rp.spinBox_max_attr.value())
                list_param_choice_attr = random_combination(list_param_attr, n_param_attr)
                list_param_group += list_param_choice_attr
                if len(list_param_choice_attr) == len(list_param_group):
                    list_param_choice = list_param_group
                else:
                    n_param = random.randint(ui_rp.spinBox_min_attr.value(), len(list_param_group))
                    list_param_choice = random_combination(list_param_group, n_param)


            if ui_rp.checkBox_ts_a.isChecked():
                for i in list_ts:
                    if f'{i}_A' in list_param_choice:
                        list_param_all.append(f'{i}_Abase_{get_n(i)}')
            if ui_rp.checkBox_ts_at.isChecked():
                for i in list_ts:
                    if f'{i}_At' in list_param_choice:
                        list_param_all.append(f'{i}_At_{get_n(i)}')
            if ui_rp.checkBox_ts_vt.isChecked():
                for i in list_ts:
                    if f'{i}_Vt' in list_param_choice:
                        list_param_all.append(f'{i}_Vt_{get_n(i)}')
            if ui_rp.checkBox_ts_pht.isChecked():
                for i in list_ts:
                    if f'{i}_Pht' in list_param_choice:
                        list_param_all.append(f'{i}_Pht_{get_n(i)}')
            if ui_rp.checkBox_ts_wt.isChecked():
                for i in list_ts:
                    if f'{i}_Wt' in list_param_choice:
                        list_param_all.append(f'{i}_Wt_{get_n(i)}')
            if ui_rp.checkBox_ts_diff.isChecked():
                for i in list_ts:
                    if f'{i}_diff' in list_param_choice:
                        list_param_all.append(f'{i}_diff_{get_n(i)}')
            if ui_rp.checkBox_ts_crlnf.isChecked():
                if ui_rp.checkBox_sig.isChecked():
                    if 'sig_CRLNF' in list_param_choice:
                        list_param_all.append(f'sig_CRLNF_{n_sig_top}_{n_sig_bot}')
            if ui_rp.checkBox_ts_crl.isChecked():
                for i in list_ts:
                    if f'{i}_CRL' in list_param_choice:
                        list_param_all.append(f'{i}_CRL_{get_n(i)}')

            if ui_rp.checkBox_group.isChecked():

                if ui_rp.checkBox_attr_a.isChecked():
                    if 'attr_A' in list_param_choice:
                        list_param_all += ['A_top', 'A_bottom', 'dA', 'A_sum', 'A_mean', 'A_max', 'A_T_max', 'A_Sn', 'A_wmf',
                                           'A_Qf', 'A_Sn_wmf']
                if ui_rp.checkBox_attr_at.isChecked():
                    if 'attr_At' in list_param_choice:
                        list_param_all += ['At_top', 'dAt', 'At_sum', 'At_mean', 'At_max', 'At_T_max', 'At_Sn',
                                           'At_wmf', 'At_Qf', 'At_Sn_wmf']
                if ui_rp.checkBox_attr_vt.isChecked():
                    if 'attr_Vt' in list_param_choice:
                        list_param_all += ['Vt_top', 'dVt', 'Vt_sum', 'Vt_mean', 'Vt_max', 'Vt_T_max', 'Vt_Sn', 'Vt_wmf',
                                           'Vt_Qf', 'Vt_Sn_wmf']
                if ui_rp.checkBox_attr_pht.isChecked():
                    if 'attr_Pht' in list_param_choice:
                        list_param_all += ['Pht_top', 'dPht', 'Pht_sum', 'Pht_mean', 'Pht_max', 'Pht_T_max', 'Pht_Sn',
                                           'Pht_wmf', 'Pht_Qf', 'Pht_Sn_wmf']
                if ui_rp.checkBox_attr_wt.isChecked():
                    if 'attr_Wt' in list_param_choice:
                        list_param_all += ['Wt_top', 'Wt_sum', 'Wt_mean', 'Wt_max', 'Wt_T_max', 'Wt_Sn', 'Wt_wmf', 'Wt_Qf',
                                           'Wt_Sn_wmf']
                if ui_rp.checkBox_attr_crl.isChecked():
                    if 'attr_CRL' in list_param_choice:
                        list_param_all += ['CRL_top', 'CRL_sum', 'CRL_mean', 'CRL_max', 'CRL_T_max', 'CRL_Sn', 'CRL_wmf',
                                           'CRL_Qf', 'CRL_Sn_wmf']

                if ui_rp.checkBox_form_t.isChecked():
                    if 'form_t' in list_param_choice:
                        list_param_all += ['T_top', 'T_bottom', 'dT']
                if ui_rp.checkBox_prior.isChecked():
                    if 'prior' in list_param_choice:
                        list_param_all += ['width', 'top', 'land', 'speed', 'speed_cover']

                if ui_rp.checkBox_stat.isChecked():
                    if 'stat' in list_param_choice:
                        list_param_all += ['skew', 'kurt', 'std', 'k_var']
                if ui_rp.checkBox_crl_stat.isChecked():
                    if 'CRL_stat' in list_param_choice:
                        list_param_all += ['CRL_skew', 'CRL_kurt', 'CRL_std', 'CRL_k_var']

            else:

                if ui_rp.checkBox_attr_a.isChecked():
                    for i in ['A_top', 'A_bottom', 'dA', 'A_sum', 'A_mean', 'A_max', 'A_T_max', 'A_Sn', 'A_wmf', 'A_Qf', 'A_Sn_wmf']:
                        if i in list_param_choice:
                            list_param_all.append(i)

                if ui_rp.checkBox_attr_at.isChecked():
                    for i in  ['At_top', 'dAt', 'At_sum', 'At_mean', 'At_max', 'At_T_max', 'At_Sn', 'At_wmf', 'At_Qf', 'At_Sn_wmf']:
                        if i in list_param_choice:
                            list_param_all.append(i)

                if ui_rp.checkBox_attr_vt.isChecked():
                    for i in ['Vt_top', 'dVt', 'Vt_sum', 'Vt_mean', 'Vt_max', 'Vt_T_max', 'Vt_Sn', 'Vt_wmf', 'Vt_Qf', 'Vt_Sn_wmf']:
                        if i in list_param_choice:
                            list_param_all.append(i)

                if ui_rp.checkBox_attr_pht.isChecked():
                    for i in ['Pht_top', 'dPht', 'Pht_sum', 'Pht_mean', 'Pht_max', 'Pht_T_max', 'Pht_Sn', 'Pht_wmf', 'Pht_Qf', 'Pht_Sn_wmf']:
                        if i in list_param_choice:
                            list_param_all.append(i)

                if ui_rp.checkBox_attr_wt.isChecked():
                    for i in ['Wt_top', 'Wt_sum', 'Wt_mean', 'Wt_max', 'Wt_T_max', 'Wt_Sn', 'Wt_wmf', 'Wt_Qf', 'Wt_Sn_wmf']:
                        if i in list_param_choice:
                            list_param_all.append(i)

                if ui_rp.checkBox_attr_crl.isChecked():
                    for i in ['CRL_top', 'CRL_sum', 'CRL_mean', 'CRL_max', 'CRL_T_max', 'CRL_Sn', 'CRL_wmf', 'CRL_Qf', 'CRL_Sn_wmf']:
                        if i in list_param_choice:
                            list_param_all.append(i)

                if ui_rp.checkBox_form_t.isChecked():
                    for i in ['T_top', 'T_bottom', 'dT']:
                        if i in list_param_choice:
                            list_param_all.append(i)

                if ui_rp.checkBox_prior.isChecked():
                    for i in ['width', 'top', 'land', 'speed', 'speed_cover']:
                        if i in list_param_choice:
                            list_param_all.append(i)

                if ui_rp.checkBox_stat.isChecked():
                    for i in ['skew', 'kurt', 'std', 'k_var']:
                        if i in list_param_choice:
                            list_param_all.append(i)

                if ui_rp.checkBox_crl_stat.isChecked():
                    for i in ['CRL_skew', 'CRL_kurt', 'CRL_std', 'CRL_k_var']:
                        if i in list_param_choice:
                            list_param_all.append(i)

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

    def choice_model_regressor(model):
        """ Выбор модели регрессии """
        if model == 'MLPR':
            model_reg = MLPRegressor(
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
            model_reg = KNeighborsRegressor(n_neighbors=n_knn, weights=weights_knn, algorithm='auto')
            text_model = f'**KNNR**: \nn_neighbors: {n_knn}, \nweights: {weights_knn}, '

        elif model == 'GBR':
            est = ui_r.spinBox_n_estimators.value()
            l_rate = ui_r.doubleSpinBox_learning_rate.value()
            model_reg = GradientBoostingRegressor(n_estimators=est, learning_rate=l_rate, random_state=0)
            text_model = f'**GBR**: \nn estimators: {round(est, 2)}, \nlearning rate: {round(l_rate, 2)}, '

        elif model == 'LR':
            model_reg = LinearRegression(fit_intercept=ui_r.checkBox_fit_intercept.isChecked())
            text_model = f'**LR**: \nfit_intercept: {"on" if ui_r.checkBox_fit_intercept.isChecked() else "off"}, '

        elif model == 'DTR':
            spl = 'random' if ui_r.checkBox_splitter_rnd.isChecked() else 'best'
            model_reg = DecisionTreeRegressor(splitter=spl, random_state=0)
            text_model = f'**DTR**: \nsplitter: {spl}, '

        elif model == 'RFR':
            model_reg = RandomForestRegressor(n_estimators=ui_r.spinBox_rfr_n.value(), oob_score=True, random_state=0,
                                              n_jobs=-1)
            text_model = f'**RFR**: \nn estimators: {ui_r.spinBox_rfr_n.value()}, '

        elif model == 'ABR':
            model_reg = AdaBoostRegressor(n_estimators=ui_r.spinBox_rfr_n.value(), random_state=0)
            text_model = f'**ABR**: \nn estimators: {ui_r.spinBox_rfr_n.value()}, '

        elif model == 'ETR':
            model_reg = ExtraTreesRegressor(n_estimators=ui_r.spinBox_rfr_n.value(), bootstrap=True, oob_score=True,
                                            random_state=0, n_jobs=-1)
            text_model = f'**ETR**: \nn estimators: {ui_r.spinBox_rfr_n.value()}, '

        elif model == 'GPR':
            gpc_kernel_width = ui_r.doubleSpinBox_gpc_wigth.value()
            gpc_kernel_scale = ui_r.doubleSpinBox_gpc_scale.value()
            n_restart_optimization = ui_r.spinBox_gpc_n_restart.value()
            kernel = gpc_kernel_scale * RBF(gpc_kernel_width)
            model_reg = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=n_restart_optimization,
                random_state=0
            )
            text_model = (
                f'**GPR**: \nwidth kernal: {round(gpc_kernel_width, 2)}, \nscale kernal: {round(gpc_kernel_scale, 2)}, '
                f'\nn restart: {n_restart_optimization} ,')

        elif model == 'SVR':
            model_reg = SVR(kernel=ui_r.comboBox_svr_kernel.currentText(), C=ui_r.doubleSpinBox_svr_c.value())
            text_model = (f'**SVR**: \nkernel: {ui_r.comboBox_svr_kernel.currentText()}, '
                          f'\nC: {round(ui_r.doubleSpinBox_svr_c.value(), 2)}, ')

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
                                     learning_rate=ui_r.doubleSpinBox_learning_rate_xgb.value(), random_state=0)
            text_model = f'**XGB**: \nn estimators: {ui_r.spinBox_n_estimators_xgb.value()}, ' \
                         f'\nlearning_rate: {ui_r.doubleSpinBox_learning_rate_xgb.value()}, '
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

            model_name = ui_r.buttonGroup.checkedButton().text()
            model_class, text_model = choice_model_regressor(model_name)
            text_model += text_scaler
            pipe_steps.append(('model', model_class))
            pipe = Pipeline(pipe_steps)

            list_param = build_list_param()
            data_train, list_param = build_table_random_param_reg(get_regmod_id(), list_param)
            list_col = data_train.columns.tolist()
            data_train = pd.DataFrame(imputer.fit_transform(data_train), columns=list_col)

            y_train = data_train['target_value'].values
            if not np.issubdtype(y_train.dtype, np.number):
                y_train = pd.to_numeric(y_train, errors='coerce')
            data_test, list_param = build_table_random_param_reg(get_regmod_test_id(), list_param)
            list_col = data_test.columns.tolist()
            data_test = pd.DataFrame(imputer.fit_transform(data_test), columns=list_col)

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
            cv_results = cross_validate(pipe, data_train.iloc[:, 2:], y_train, cv=kv, scoring='r2',
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

    ui_rp.spinBox_flp_bot_skip_down.valueChanged.connect(lambda: check_spinbox(ui_rp.spinBox_flp_bot_skip_up, ui_rp.spinBox_flp_bot_skip_down))
    ui_rp.spinBox_flp_bot_skip_up.valueChanged.connect(lambda: check_spinbox(ui_rp.spinBox_flp_bot_skip_up, ui_rp.spinBox_flp_bot_skip_down))

    ui_rp.spinBox_flp_distr_up.valueChanged.connect(lambda: check_spinbox(ui_rp.spinBox_flp_distr_up, ui_rp.spinBox_flp_distr_down))
    ui_rp.spinBox_flp_distr_down.valueChanged.connect(lambda: check_spinbox(ui_rp.spinBox_flp_distr_up, ui_rp.spinBox_flp_distr_down))

    ui_rp.spinBox_flp_sep_up.valueChanged.connect(lambda: check_spinbox(ui_rp.spinBox_flp_sep_up, ui_rp.spinBox_flp_sep_down))
    ui_rp.spinBox_flp_sep_down.valueChanged.connect(lambda: check_spinbox(ui_rp.spinBox_flp_sep_up, ui_rp.spinBox_flp_sep_down))

    ui_rp.spinBox_flp_mfcc_up.valueChanged.connect(lambda: check_spinbox(ui_rp.spinBox_flp_mfcc_up, ui_rp.spinBox_flp_mfcc_down))
    ui_rp.spinBox_flp_mfcc_down.valueChanged.connect(lambda: check_spinbox(ui_rp.spinBox_flp_mfcc_up, ui_rp.spinBox_flp_mfcc_down))

    ui_rp.checkBox_group.clicked.connect(check_group_attr)
    ui_rp.checkBox_flp_sig_ts_all.clicked.connect(check_checkbox_flp_sig_ts)
    ui_rp.checkBox_flp_sep_ts_all.clicked.connect(check_checkbox_flp_sep_ts)
    ui_rp.checkBox_flp_dstr_ts_all.clicked.connect(check_checkbox_flp_dstr_ts)
    ui_rp.checkBox_flp_mfcc_ts_all.clicked.connect(check_checkbox_flp_mfcc_ts)

    RandomParam.exec_()

    Regressor.exec_()
