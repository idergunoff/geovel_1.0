import inspect
import random

from scipy.stats import randint, uniform
from func import *


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

    # def test():
    #     print(inspect.currentframe().f_back.f_locals)
    #
    # ui_rp.pushButton.clicked.connect(test)

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



    def build_list_param():
        list_param_group, list_ts, list_param_all = [], [], []
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

        if ui_rp.checkBox_stat.isChecked():
            list_param_group.append('stat')
        if ui_rp.checkBox_crl_stat.isChecked():
            list_param_group.append('CRL_stat')

        if ui_rp.checkBox_form_t.isChecked():
            list_param_group.append('form_t')
        if ui_rp.checkBox_prior.isChecked():
            list_param_group.append('prior')

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

        n_param = random.randint(2, len(list_param_group))
        print(n_param, len(list_param_group))
        list_param_choice = random_combination(list_param_group, n_param)

        print(list_param_group)
        print(list_param_choice)


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

            # Загрузка сигналов из профилей, необходимых для параметров 'distr', 'sep' и 'mfcc'
            for param in list_param:
                # Если параметр является расчётным
                if param.startswith('sig') or param.startswith('distr') or param.startswith('sep') or param.startswith('mfcc'):
                    # Проверка, есть ли уже загруженный сигнал в локальных переменных
                    if not str(markup.profile.id) + '_signal' in locals_dict:
                        # Загрузка сигнала из профиля
                        locals_dict.update(
                            {str(markup.profile.id) + '_signal':
                                 json.loads(session.query(Profile.signal).filter(Profile.id == markup.profile_id).first()[0])}
                        )
                if ui_rp.checkBox_ts_crl.isChecked():
                    if not str(markup.profile.id) + '_CRL' in locals_dict:
                        locals_dict.update(
                            {str(markup.profile.id) + '_CRL':
                                calc_CRL_filter(json.loads(
                                    session.query(Profile.signal).filter(Profile.id == markup.profile_id).first()[0]))}
                        )
                if ui_rp.checkBox_ts_crlnf.isChecked():
                    if not str(markup.profile.id) + '_CRL_NF' in locals_dict:
                        locals_dict.update(
                            {str(markup.profile.id) + '_CRL_NF':
                             calc_CRL(json.loads(
                                 session.query(Profile.signal).filter(Profile.id == markup.profile_id).first()[0]))}
                        )
                # # Если параметр сохранён в базе
                # else:
                #     # Загрузка значений параметра из формации
                #     locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'Formation.{param}')).filter(
                #         Formation.id == markup.formation_id).first()[0])

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
                            p, atr, up, down = param.split('_')[0], param.split('_')[1], int(param.split('_')[2]), 512 - int(param.split('_')[3])
                        else:
                            p, atr, n = param.split('_')[0], param.split('_')[1], int(param.split('_')[2])

                        if atr == 'CRL':
                            sig_measure = locals_dict[str(markup.profile.id) + '_CRL'][measure]
                        elif atr == 'CRLNF':
                            sig_measure = locals_dict[str(markup.profile.id) + '_CRL_NF'][measure]
                        else:
                            sig_measure = calc_atrib_measure(locals_dict[str(markup.profile.id) + '_signal'][measure], atr)

                        if p == 'sig':
                            for i_sig in range(len(sig_measure[up:down])):
                                dict_value[f'{p}_{atr}_{up + i_sig + 1}'] = sig_measure[i_sig]
                        elif p == 'distr':
                            distr = get_distribution(sig_measure[list_up[measure]: list_down[measure]], n)
                            for num in range(n):
                                dict_value[f'{p}_{atr}_{num + 1}'] = distr[num]
                        elif p == 'sep':
                            sep = get_mean_values(sig_measure[list_up[measure]: list_down[measure]], n)
                            for num in range(n):
                                dict_value[f'{p}_{atr}_{num + 1}'] = sep[num]
                        elif p == 'mfcc':
                            mfcc = get_mfcc(sig_measure[list_up[measure]: list_down[measure]], n)
                            for num in range(n):
                                dict_value[f'{p}_{atr}_{num + 1}'] = mfcc[num]

                    else:
                        # Загрузка значения параметра из списка значений
                        dict_value[param] = json.loads(session.query(literal_column(f'Formation.{param}')).filter(
                        Formation.id == markup.formation_id).first()[0])[measure]

                # Добавление данных в обучающую выборку
                data_train = pd.concat([data_train, pd.DataFrame([dict_value])], ignore_index=True)

            ui.progressBar.setValue(nm + 1)

        return data_train, list_param


    def start_random_param():
        for _ in range(ui_rp.spinBox_n_iter.value()):
            list_param = build_list_param()
            data_train, list_param = build_table_random_param(get_MLP_id(), list_param)
            print(data_train)


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
        for i in session.query(AnalysisMLP).order_by(AnalysisMLP.title).all():
            ui_rp.comboBox_test_analysis.addItem(f'{i.title} id{i.id}')
            update_list_test_well()


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
