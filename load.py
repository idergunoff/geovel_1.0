import numpy as np
from matplotlib.pyplot import title

from func import *
from wgs_to_pulc import wgs84_to_pulc42
from qt.add_obj_dialog import *


def add_object():
    """Добавить ноывый объект в БД"""
    Add_Object = QtWidgets.QDialog()
    ui_ob = Ui_add_obj()
    ui_ob.setupUi(Add_Object)
    Add_Object.show()
    Add_Object.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия
    ui_ob.dateEdit.setDate(datetime.date.today())

    def object_to_db():
        name_object = ui_ob.lineEdit.text()
        date_research = ui_ob.dateEdit.date().toPyDate()
        if name_object != '':
            obj = session.query(GeoradarObject).filter(GeoradarObject.title == name_object).first()
            if not obj:
                new_object = GeoradarObject(title=name_object)
                session.add(new_object)
                session.commit()
                obj_id = new_object.id
                set_info(f'Объект "{name_object}" добавлен в базу данных.', 'green')
            else:
                obj_id = obj.id
            new_research = Research(object_id=obj_id, date_research=date_research)
            session.add(new_research)
            session.commit()
            update_object()
            Add_Object.close()
            set_info(f'Добавлено исследование /{date_research.strftime("%m.%Y")}/ для объекта "{name_object}".', 'green')

    def cancel_add_object():
        Add_Object.close()

    ui_ob.buttonBox.accepted.connect(object_to_db)
    ui_ob.buttonBox.rejected.connect(cancel_add_object)
    Add_Object.exec_()


def load_profile():
    """ загрузка профилей """
    try:
        file_name = QFileDialog.getOpenFileName(caption='Выберите файл профиля', filter='*.txt')[0]
        set_info(file_name, 'blue')
        pd_radar = pd.read_table(file_name, delimiter=';', header=0)
    except FileNotFoundError:
        return
    try:
        ui.progressBar.setMaximum(int(len(pd_radar.index)/512))
        signal = []
        for i in range(int(pd_radar['X'].max() + 1)):
            if len(set(pd_radar[pd_radar['X'] == i]['ALn'])) > 1:
                signal.append(pd_radar[pd_radar['X'] == i]['ALn'].tolist())
            ui.progressBar.setValue(i+1)

        new_profile = Profile(research_id=get_research_id(), title=file_name.split('/')[-1].split('.')[0], signal=json.dumps(signal))
        session.add(new_profile)
        session.commit()
        set_info(f'Профиль загружен ({get_object_name()})', 'green')
    except KeyError:
        set_info('Не правильный формат файла', 'red')
        return
    update_profile_combobox()


def load_param():
    """ загрузка файла интервала """
    try:
        file_name = QFileDialog.getOpenFileName(caption='Выберите файл интервала', filter='*.txt')[0]
        set_info(file_name, 'blue')
        pd_int = pd.read_table(file_name, delimiter=';', header=0)
        pd_int = pd_int.applymap(lambda x: float(str(x).replace(',', '.')))
    except FileNotFoundError:
        return
    signals = json.loads(session.query(Profile.signal).filter(Profile.id == get_profile_id()).first()[0])
    # проверяем соответствие количества измерений в загруженном файле и в БД
    if len(signals) != len(pd_int.index):
        set_info('ВНИМАНИЕ! ОШИБКА!!! Не совпадает количество измерений в файлах', 'red')
    else:
        x_wgs_l, y_wgs_l, x_pulc_l, y_pulc_l = [], [], [], []
        for i in pd_int.index:
            y, x = wgs84_to_pulc42(pd_int['Latd'][i], pd_int['Long'][i])
            x_wgs_l.append(pd_int['Long'][i])
            y_wgs_l.append(pd_int['Latd'][i])
            x_pulc_l.append(x)
            y_pulc_l.append(y)
        dict_signal = {'x_wgs': json.dumps(x_wgs_l), 'y_wgs': json.dumps(y_wgs_l),
                       'x_pulc': json.dumps(x_pulc_l), 'y_pulc': json.dumps(y_pulc_l)}
        session.query(Profile).filter(Profile.id == get_profile_id()).update(dict_signal, synchronize_session="fetch")
        session.commit()
        set_info(f'Загружены координаты ({get_object_name()}, {get_profile_name()})', 'green')
        check_coordinates_profile()
        check_coordinates_research()
        layer_top = list(map(lambda x: int(x / 40), pd_int['T01'].values.tolist()))
        layer_bottom = list(map(lambda x: int(x / 40), pd_int['D02'].values.tolist()))
        if all(i == 0 for i in layer_top) or all(i == 0 for i in layer_bottom):
            set_info('В выбранном файле границы пласта не отрисованы', 'red')
            return
        if session.query(Layers).filter(Layers.profile_id == get_profile_id(), Layers.layer_title == 'krot_top').count() == 0:
            new_layer_top = Layers(profile_id=get_profile_id(), layer_title='krot_top', layer_line=json.dumps(layer_top))
            session.add(new_layer_top)
            new_layer_bottom = Layers(profile_id=get_profile_id(), layer_title='krot_bottom', layer_line=json.dumps(layer_bottom))
            session.add(new_layer_bottom)
            session.commit()
            new_formation = Formation(title='KROT', profile_id=get_profile_id(), up=new_layer_top.id, down=new_layer_bottom.id)
            session.add(new_formation)
        else:
            session.query(Layers).filter(Layers.profile_id == get_profile_id(), Layers.layer_title == 'krot_top').update(
                {'layer_line': json.dumps(layer_top)}, synchronize_session="fetch")
            session.query(Layers).filter(Layers.profile_id == get_profile_id(), Layers.layer_title == 'krot_bottom').update(
                {'layer_line': json.dumps(layer_bottom)}, synchronize_session="fetch")
        session.commit()
        set_info(f'Загружены слои из программы KROT ({get_object_name()}, {get_profile_name()})', 'green')
        update_layers()

        ui.progressBar.setMaximum(len(layer_top))
        grid_db = session.query(Grid).filter(Grid.object_id == get_object_id()).first()
        if grid_db:
            # считываем сетку грида из БД
            pd_grid_uf = pd.DataFrame(json.loads(grid_db.grid_table_uf))
            pd_grid_m = pd.DataFrame(json.loads(grid_db.grid_table_m))
            pd_grid_r = pd.DataFrame(json.loads(grid_db.grid_table_r))
        # задаем списки для хранения данных о скважинах
        T_top_l, T_bottom_l, dT_l, A_top_l, A_bottom_l, dA_l, A_sum_l, A_mean_l, dVt_l, Vt_top_l, Vt_sum_l, \
        Vt_mean_l, dAt_l, At_top_l, At_sum_l, At_mean_l, dPht_l, Pht_top_l, Pht_sum_l, Pht_mean_l, Wt_top_l, \
        Wt_mean_l, Wt_sum_l, std_l, k_var_l, skew_l, kurt_l, width_l, top_l, land_l, speed_l, speed_cover_l, \
        A_max_l, Vt_max_l, At_max_l, Pht_max_l, Wt_max_l, A_T_max_l, Vt_T_max_l, At_T_max_l, Pht_T_max_l, Wt_T_max_l, \
        A_Sn_l, Vt_Sn_l, At_Sn_l, Pht_Sn_l, Wt_Sn_l, A_wmf_l, Vt_wmf_l, At_wmf_l, Pht_wmf_l, Wt_wmf_l, A_Qf_l, \
        Vt_Qf_l, At_Qf_l, Pht_Qf_l, Wt_Qf_l, A_Sn_wmf_l, Vt_Sn_wmf_l, At_Sn_wmf_l, Pht_Sn_wmf_l, Wt_Sn_wmf_l, k_r_l= \
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], \
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], \
        [], [], [], [], [], [], []

        try:
            list_param_1 = []
            for i in range(len(layer_top)):
                list_param_2 = []
                signal = signals[i]
                analytic_signal = hilbert(signal)
                At = np.hypot(signal, np.imag(analytic_signal)).tolist()
                Vt = np.imag(analytic_signal).tolist()
                Pht = np.angle(analytic_signal).tolist()
                Wt = np.diff(np.angle(analytic_signal)).tolist()
                nt = layer_top[i]
                nb = layer_bottom[i]
                T_top_l.append(layer_top[i] * 8)
                list_param_2.append(T_top_l[-1])
                T_bottom_l.append(layer_bottom[i] * 8)
                list_param_2.append(T_bottom_l[-1])
                dT_l.append(layer_bottom[i] * 8 - layer_top[i] * 8)
                list_param_2.append(dT_l[-1])
                A_top_l.append(signal[nt])
                list_param_2.append(A_top_l[-1])
                A_bottom_l.append(signal[nb])
                list_param_2.append(A_bottom_l[-1])
                dA_l.append(signal[nb] - signal[nt])
                list_param_2.append(dA_l[-1])
                A_sum_l.append(float(np.sum(signal[nt:nb])))
                list_param_2.append(A_sum_l[-1])
                A_mean_l.append(float(np.mean(signal[nt:nb])))
                list_param_2.append(A_mean_l[-1])
                dVt_l.append(Vt[nb] - Vt[nt])
                list_param_2.append(dVt_l[-1])
                Vt_top_l.append(Vt[nt])
                list_param_2.append(Vt_top_l[-1])
                Vt_sum_l.append(float(np.sum(Vt[nt:nb])))
                list_param_2.append(Vt_sum_l[-1])
                Vt_mean_l.append(float(np.mean(Vt[nt:nb])))
                list_param_2.append(Vt_mean_l[-1])
                dAt_l.append(At[nb] - At[nt])
                list_param_2.append(dAt_l[-1])
                At_top_l.append(At[nt])
                list_param_2.append(At_top_l[-1])
                At_sum_l.append(float(np.sum(At[nt:nb])))
                list_param_2.append(At_sum_l[-1])
                At_mean_l.append(float(np.mean(At[nt:nb])))
                list_param_2.append(At_mean_l[-1])
                dPht_l.append(Pht[nb] - Pht[nt])
                list_param_2.append(dPht_l[-1])
                Pht_top_l.append(Pht[nt])
                list_param_2.append(Pht_top_l[-1])
                Pht_sum_l.append(float(np.sum(Pht[nt:nb])))
                list_param_2.append(Pht_sum_l[-1])
                Pht_mean_l.append(float(np.mean(Pht[nt:nb])))
                list_param_2.append(Pht_mean_l[-1])
                Wt_top_l.append(Wt[nt])
                list_param_2.append(Wt_top_l[-1])
                Wt_mean_l.append(float(np.mean(Wt[nt:nb])))
                list_param_2.append(Wt_mean_l[-1])
                Wt_sum_l.append(float(np.sum(Wt[nt:nb])))
                list_param_2.append(Wt_sum_l[-1])
                std_l.append(float(np.std(signal[nt:nb])))
                list_param_2.append(std_l[-1])
                k_var_l.append(float(np.var(signal[nt:nb])))
                list_param_2.append(k_var_l[-1])
                skew_l.append(skew(signal[nt:nb]))
                list_param_2.append(skew_l[-1])
                kurt_l.append(kurtosis(signal[nt:nb]))
                list_param_2.append(kurt_l[-1])
                A_max_l.append(max(signal[nt:nb]))
                list_param_2.append(A_max_l[-1])
                Vt_max_l.append(max(Vt[nt:nb]))
                list_param_2.append(Vt_max_l[-1])
                At_max_l.append(max(At[nt:nb]))
                list_param_2.append(At_max_l[-1])
                Pht_max_l.append(max(Pht[nt:nb]))
                list_param_2.append(Pht_max_l[-1])
                Wt_max_l.append(max(Wt[nt:nb]))
                list_param_2.append(Wt_max_l[-1])
                A_T_max_l.append((signal[nt:nb].index(max(signal[nt:nb])) + nt) * 8)
                list_param_2.append(A_T_max_l[-1])
                Vt_T_max_l.append((Vt[nt:nb].index(max(Vt[nt:nb])) + nt) * 8)
                list_param_2.append(Vt_T_max_l[-1])
                At_T_max_l.append((At[nt:nb].index(max(At[nt:nb])) + nt) * 8)
                list_param_2.append(At_T_max_l[-1])
                Pht_T_max_l.append((Pht[nt:nb].index(max(Pht[nt:nb])) + nt) * 8)
                list_param_2.append(Pht_T_max_l[-1])
                Wt_T_max_l.append((Wt[nt:nb].index(max(Wt[nt:nb])) + nt) * 8)
                list_param_2.append(Wt_T_max_l[-1])
                A_Sn, A_wmf, A_Qf, A_Sn_wmf = calc_fft_attributes(signal[nt:nb])
                Vt_Sn, Vt_wmf, Vt_Qf, Vt_Sn_wmf = calc_fft_attributes(Vt[nt:nb])
                At_Sn, At_wmf, At_Qf, At_Sn_wmf = calc_fft_attributes(At[nt:nb])
                Pht_Sn, Pht_wmf, Pht_Qf, Pht_Sn_wmf = calc_fft_attributes(Pht[nt:nb])
                Wt_Sn, Wt_wmf, Wt_Qf, Wt_Sn_wmf = calc_fft_attributes(Wt[nt:nb])
                A_Sn_l.append(A_Sn)
                list_param_2.append(A_Sn_l[-1])
                Vt_Sn_l.append(Vt_Sn)
                list_param_2.append(Vt_Sn_l[-1])
                At_Sn_l.append(At_Sn)
                list_param_2.append(At_Sn_l[-1])
                Pht_Sn_l.append(Pht_Sn)
                list_param_2.append(Pht_Sn_l[-1])
                Wt_Sn_l.append(Wt_Sn)
                list_param_2.append(Wt_Sn_l[-1])
                A_wmf_l.append(At_wmf)
                list_param_2.append(A_wmf_l[-1])
                Vt_wmf_l.append(Vt_wmf)
                list_param_2.append(Vt_wmf_l[-1])
                At_wmf_l.append(At_wmf)
                list_param_2.append(At_wmf_l[-1])
                Pht_wmf_l.append(Pht_wmf)
                list_param_2.append(Pht_wmf_l[-1])
                Wt_wmf_l.append(Wt_wmf)
                list_param_2.append(Wt_wmf_l[-1])
                A_Qf_l.append(A_Qf)
                list_param_2.append(A_Qf_l[-1])
                Vt_Qf_l.append(Vt_Qf)
                list_param_2.append(Vt_Qf_l[-1])
                At_Qf_l.append(At_Qf)
                list_param_2.append(At_Qf_l[-1])
                Pht_Qf_l.append(Pht_Qf)
                list_param_2.append(Pht_Qf_l[-1])
                Wt_Qf_l.append(Wt_Qf)
                list_param_2.append(Wt_Qf_l[-1])
                A_Sn_wmf_l.append(A_Sn_wmf)
                list_param_2.append(A_Sn_wmf_l[-1])
                Vt_Sn_wmf_l.append(Vt_Sn_wmf)
                list_param_2.append(Vt_Sn_wmf_l[-1])
                At_Sn_wmf_l.append(At_Sn_wmf)
                list_param_2.append(At_Sn_wmf_l[-1])
                Pht_Sn_wmf_l.append(Pht_Sn_wmf)
                list_param_2.append(Pht_Sn_wmf_l[-1])
                Wt_Sn_wmf_l.append(Wt_Sn_wmf)
                list_param_2.append(Wt_Sn_wmf_l[-1])

                if grid_db:
                    pd_grid_uf['dist_y'] = abs(pd_grid_uf[1] - y_pulc_l[i])
                    pd_grid_uf['dist_x'] = abs(pd_grid_uf[0] - x_pulc_l[i])
                    pd_grid_m['dist_y'] = abs(pd_grid_m[1] - y_pulc_l[i])
                    pd_grid_m['dist_x'] = abs(pd_grid_m[0] - x_pulc_l[i])
                    pd_grid_r['dist_y'] = abs(pd_grid_r[1] - y_pulc_l[i])
                    pd_grid_r['dist_x'] = abs(pd_grid_r[0] - x_pulc_l[i])
                    i_uf = pd_grid_uf.loc[pd_grid_uf['dist_y'] == pd_grid_uf['dist_y'].min()].loc[
                        pd_grid_uf['dist_x'] == pd_grid_uf['dist_x'].min()].iat[0, 2]
                    i_m = pd_grid_m.loc[pd_grid_m['dist_y'] == pd_grid_m['dist_y'].min()].loc[
                        pd_grid_m['dist_x'] == pd_grid_m['dist_x'].min()].iat[0, 2]
                    i_r = pd_grid_r.loc[pd_grid_r['dist_y'] == pd_grid_r['dist_y'].min()].loc[
                        pd_grid_r['dist_x'] == pd_grid_r['dist_x'].min()].iat[0, 2]
                    im = i_m if i_m > 0 else 0
                    width_l.append(im)
                    list_param_2.append(width_l[-1])
                    top_l.append(i_uf)
                    list_param_2.append(top_l[-1])
                    land_l.append(i_r)
                    list_param_2.append(land_l[-1])
                    speed_l.append(im * 100 / (layer_bottom[i] * 8 - layer_top[i] * 8))
                    list_param_2.append(speed_l[-1])
                    speed_cover_l.append((i_r - i_uf) * 100 / (layer_top[i] * 8))
                    list_param_2.append(speed_cover_l[-1])
                if len(list_param_1) == len(list_param_2):
                    k_r_l.append(np.corrcoef(list_param_1, list_param_2)[0, 1])
                list_param_1 = list_param_2.copy()
                ui.progressBar.setValue(i + 1)
            k_r_l.append(k_r_l[-1])
            dict_signal = {'T_top': json.dumps(T_top_l),
                           'T_bottom': json.dumps(T_bottom_l),
                           'dT': json.dumps(dT_l),
                           'A_top': json.dumps(A_top_l),
                           'A_bottom': json.dumps(A_bottom_l),
                           'dA': json.dumps(dA_l),
                           'A_sum': json.dumps(A_sum_l),
                           'A_mean': json.dumps(A_mean_l),
                           'dVt': json.dumps(dVt_l),
                           'Vt_top': json.dumps(Vt_top_l),
                           'Vt_sum': json.dumps(Vt_sum_l),
                           'Vt_mean': json.dumps(Vt_mean_l),
                           'dAt': json.dumps(dAt_l),
                           'At_top': json.dumps(At_top_l),
                           'At_sum': json.dumps(At_sum_l),
                           'At_mean': json.dumps(At_mean_l),
                           'dPht': json.dumps(dPht_l),
                           'Pht_top': json.dumps(Pht_top_l),
                           'Pht_sum': json.dumps(Pht_sum_l),
                           'Pht_mean': json.dumps(Pht_mean_l),
                           'Wt_top': json.dumps(Wt_top_l),
                           'Wt_mean': json.dumps(Wt_mean_l),
                           'Wt_sum': json.dumps(Wt_sum_l),
                           'std': json.dumps(std_l),
                           'k_var': json.dumps(k_var_l),
                           'skew': json.dumps(skew_l),
                           'kurt': json.dumps(kurt_l),
                           'A_max': json.dumps(A_max_l),
                           'Vt_max': json.dumps(Vt_max_l),
                           'At_max': json.dumps(At_max_l),
                           'Pht_max': json.dumps(Pht_max_l),
                           'Wt_max': json.dumps(Wt_max_l),
                           'A_T_max': json.dumps(A_T_max_l),
                           'Vt_T_max': json.dumps(Vt_T_max_l),
                           'At_T_max': json.dumps(At_T_max_l),
                           'Pht_T_max': json.dumps(Pht_T_max_l),
                           'Wt_T_max': json.dumps(Wt_T_max_l),
                           'A_Sn': json.dumps(A_Sn_l),
                           'Vt_Sn': json.dumps(Vt_Sn_l),
                           'At_Sn': json.dumps(At_Sn_l),
                           'Pht_Sn': json.dumps(Pht_Sn_l),
                           'Wt_Sn': json.dumps(Wt_Sn_l),
                           'A_wmf': json.dumps(A_wmf_l),
                           'Vt_wmf': json.dumps(Vt_wmf_l),
                           'At_wmf': json.dumps(At_wmf_l),
                           'Pht_wmf': json.dumps(Pht_wmf_l),
                           'Wt_wmf': json.dumps(Wt_wmf_l),
                           'A_Qf': json.dumps(A_Qf_l),
                           'Vt_Qf': json.dumps(Vt_Qf_l),
                           'At_Qf': json.dumps(At_Qf_l),
                           'Pht_Qf': json.dumps(Pht_Qf_l),
                           'Wt_Qf': json.dumps(Wt_Qf_l),
                           'A_Sn_wmf': json.dumps(A_Sn_wmf_l),
                           'Vt_Sn_wmf': json.dumps(Vt_Sn_wmf_l),
                           'At_Sn_wmf': json.dumps(At_Sn_wmf_l),
                           'Pht_Sn_wmf': json.dumps(Pht_Sn_wmf_l),
                           'Wt_Sn_wmf': json.dumps(Wt_Sn_wmf_l),
                           'k_r': json.dumps(k_r_l)}

            if grid_db:
                dict_signal['width'] = json.dumps(width_l)
                dict_signal['top'] = json.dumps(top_l)
                dict_signal['land'] = json.dumps(land_l)
                dict_signal['speed'] = json.dumps(speed_l)
                dict_signal['speed_cover'] = json.dumps(speed_cover_l)
            form_krot = session.query(Formation).filter(Formation.profile_id == get_profile_id(), Formation.title == 'KROT').first()
            session.query(Formation).filter(Formation.id == form_krot.id).update(dict_signal, synchronize_session="fetch")
            session.commit()
            update_formation_combobox()

            set_info(f'Добавлен новый пласт - "KROT" на профиле - "{get_profile_name()}".', 'green')
        except ZeroDivisionError:
            set_info('ZeroDivisionError, проверьте файл выделенного интервала пласта', 'red')
            set_info(f'Невозможно добавить пласт - "KROT" на профиле - "{get_profile_name()}".', 'red')


# def load_param():
#     """ Загрузка параметров """
#     try:
#         # открываем диалоговое окно для выбора файла и получаем имя файла
#         file_name = QFileDialog.getOpenFileName(caption='Выберите файл выделенного интервала пласта', filter='*.txt')[0]
#         set_info(file_name, 'blue')  # выводим имя файла в информационное окно приложения
#         # считываем данные из файла в pandas DataFrame и заменяем запятые на точки
#         tab_int = pd.read_table(file_name, delimiter=';', header=0)
#         tab_int = tab_int.applymap(lambda x: float(str(x).replace(',', '.')))
#     except FileNotFoundError:
#         return
#     signals = json.loads(session.query(Profile.signal).filter(Profile.id == get_profile_id()).first()[0])
#     # проверяем соответствие количества измерений в загруженном файле и в БД
#     if len(signals) != len(tab_int.index):
#         set_info('ВНИМАНИЕ! ОШИБКА!!! Не совпадает количество измерений в файлах', 'red')
#     else:
#         # задаем максимальное значение для прогресс-бара
#         ui.progressBar.setMaximum(len(tab_int.index))
#         grid_db = session.query(Grid).filter(Grid.object_id == get_object_id()).first()
#         if grid_db:
#             # считываем сетку грида из БД
#             pd_grid_uf = pd.DataFrame(json.loads(grid_db.grid_table_uf))
#             pd_grid_m = pd.DataFrame(json.loads(grid_db.grid_table_m))
#             pd_grid_r = pd.DataFrame(json.loads(grid_db.grid_table_r))
#         # задаем списки для хранения данных о скважинах
#         x_wgs_l, y_wgs_l, x_pulc_l, y_pulc_l, T_top_l, T_bottom_l, dT_l, A_top_l, A_bottom_l, dA_l, A_sum_l, A_mean_l, dVt_l, Vt_top_l, Vt_sum_l, Vt_mean_l, dAt_l, At_top_l, \
#         At_sum_l, At_mean_l, dPht_l, Pht_top_l, Pht_sum_l, Pht_mean_l, Wt_top_l, Wt_mean_l, Wt_sum_l, std_l, k_var_l, skew_l, kurt_l, width_l, top_l, land_l, speed_l, speed_cover_l = \
#             [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
#         for i in tab_int.index:
#             y, x = wgs84_to_pulc42(tab_int['Latd'][i], tab_int['Long'][i])
#             x_wgs_l.append(tab_int['Long'][i])
#             y_wgs_l.append(tab_int['Latd'][i])
#             x_pulc_l.append(x)
#             y_pulc_l.append(y)
#             if all(x == 0 for x in tab_int['D02']):
#                 continue
#             signal = signals[i]
#             analytic_signal = hilbert(signal)
#             At = np.hypot(signal, np.imag(analytic_signal)).tolist()
#             Vt = np.imag(analytic_signal).tolist()
#             Pht = np.angle(analytic_signal).tolist()
#             Wt = np.diff(Pht).tolist()
#             nt = int(tab_int['T01'][i] / 40)
#             nb = int(tab_int['D02'][i] / 40)
#             T_top_l.append(tab_int['T01'][i] / 5)
#             T_bottom_l.append(tab_int['D02'][i] / 5)
#             dT_l.append(tab_int['D02'][i] / 5 - tab_int['T01'][i] / 5)
#             A_top_l.append(signal[nt])
#             A_bottom_l.append(signal[nb])
#             dA_l.append(signal[nb] - signal[nt])
#             A_sum_l.append(float(np.sum(signal[nt:nb])))
#             A_mean_l.append(float(np.mean(signal[nt:nb])))
#             dVt_l.append(Vt[nb] - Vt[nt])
#             Vt_top_l.append(Vt[nt])
#             Vt_sum_l.append(float(np.sum(Vt[nt:nb])))
#             Vt_mean_l.append(float(np.mean(Vt[nt:nb])))
#             dAt_l.append(At[nb] - At[nt])
#             At_top_l.append(At[nt])
#             At_sum_l.append(float(np.sum(At[nt:nb])))
#             At_mean_l.append(float(np.mean(At[nt:nb])))
#             dPht_l.append(Pht[nb] - Pht[nt])
#             Pht_top_l.append(Pht[nt])
#             Pht_sum_l.append(float(np.sum(Pht[nt:nb])))
#             Pht_mean_l.append(float(np.mean(Pht[nt:nb])))
#             Wt_top_l.append(Wt[nt])
#             Wt_mean_l.append(float(np.mean(Wt[nt:nb])))
#             Wt_sum_l.append(float(np.sum(Wt[nt:nb])))
#             std_l.append(float(np.std(signal[nt:nb])))
#             k_var_l.append(float(np.var(signal[nt:nb])))
#             skew_l.append(skew(signal[nt:nb]))
#             kurt_l.append(kurtosis(signal[nt:nb]))
#             if grid_db:
#                 pd_grid_uf['dist_y'] = abs(pd_grid_uf[1] - y)
#                 pd_grid_uf['dist_x'] = abs(pd_grid_uf[0] - x)
#                 pd_grid_m['dist_y'] = abs(pd_grid_m[1] - y)
#                 pd_grid_m['dist_x'] = abs(pd_grid_m[0] - x)
#                 pd_grid_r['dist_y'] = abs(pd_grid_r[1] - y)
#                 pd_grid_r['dist_x'] = abs(pd_grid_r[0] - x)
#                 i_uf = pd_grid_uf.loc[pd_grid_uf['dist_y'] == pd_grid_uf['dist_y'].min()].loc[pd_grid_uf['dist_x'] == pd_grid_uf['dist_x'].min()].iat[0, 2]
#                 i_m = pd_grid_m.loc[pd_grid_m['dist_y'] == pd_grid_m['dist_y'].min()].loc[pd_grid_m['dist_x'] == pd_grid_m['dist_x'].min()].iat[0, 2]
#                 i_r = pd_grid_r.loc[pd_grid_r['dist_y'] == pd_grid_r['dist_y'].min()].loc[pd_grid_r['dist_x'] == pd_grid_r['dist_x'].min()].iat[0, 2]
#                 im = i_m if i_m > 0 else 0
#                 width_l.append(im)
#                 top_l.append(i_uf)
#                 land_l.append(i_r)
#                 speed_l.append(im * 100 / (tab_int['D02'][i] / 5 - tab_int['T01'][i] / 5))
#                 speed_cover_l.append((i_r - i_uf) * 100 / (tab_int['T01'][i] / 5))
#             ui.progressBar.setValue(i+1)
#         if all(x == 0 for x in tab_int['D02']):
#             dict_signal = {'x_wgs': json.dumps(x_wgs_l),
#                            'y_wgs': json.dumps(y_wgs_l),
#                            'x_pulc': json.dumps(x_pulc_l),
#                            'y_pulc': json.dumps(y_pulc_l)}
#         else:
#             dict_signal = {'x_wgs': json.dumps(x_wgs_l),
#                        'y_wgs': json.dumps(y_wgs_l),
#                         'x_pulc': json.dumps(x_pulc_l),
#                         'y_pulc': json.dumps(y_pulc_l),
#                         'T_top': json.dumps(T_top_l),
#                         'T_bottom': json.dumps(T_bottom_l),
#                         'dT': json.dumps(dT_l),
#                         'A_top': json.dumps(A_top_l),
#                         'A_bottom': json.dumps(A_bottom_l),
#                         'dA': json.dumps(dA_l),
#                         'A_sum': json.dumps(A_sum_l),
#                         'A_mean': json.dumps(A_mean_l),
#                         'dVt': json.dumps(dVt_l),
#                         'Vt_top': json.dumps(Vt_top_l),
#                         'Vt_sum': json.dumps(Vt_sum_l),
#                         'Vt_mean': json.dumps(Vt_mean_l),
#                         'dAt': json.dumps(dAt_l),
#                         'At_top': json.dumps(At_top_l),
#                         'At_sum': json.dumps(At_sum_l),
#                         'At_mean': json.dumps(At_mean_l),
#                         'dPht': json.dumps(dPht_l),
#                         'Pht_top': json.dumps(Pht_top_l),
#                         'Pht_sum': json.dumps(Pht_sum_l),
#                         'Pht_mean': json.dumps(Pht_mean_l),
#                         'Wt_top': json.dumps(Wt_top_l),
#                         'Wt_mean': json.dumps(Wt_mean_l),
#                         'Wt_sum': json.dumps(Wt_sum_l),
#                         'std': json.dumps(std_l),
#                         'k_var': json.dumps(k_var_l),
#                         'skew': json.dumps(skew_l),
#                         'kurt': json.dumps(kurt_l)}
#             if grid_db:
#                 dict_signal['width'] = json.dumps(width_l)
#                 dict_signal['top'] = json.dumps(top_l)
#                 dict_signal['land'] = json.dumps(land_l)
#                 dict_signal['speed'] = json.dumps(speed_l)
#                 dict_signal['speed_cover'] = json.dumps(speed_cover_l)
#         session.query(Profile).filter(Profile.id == get_profile_id()).update(dict_signal, synchronize_session="fetch")
#         session.commit()
#         set_info(f'Параметры загружены ({get_object_name()}, {get_profile_name()})', 'green')
#         update_param_combobox()


def delete_profile():
    title_prof = ui.comboBox_profile.currentText().split(' id')[0]
    result = QtWidgets.QMessageBox.question(ui.listWidget_well_lda, 'Remove profile',
                f'Вы уверены, что хотите удалить профиль "{title_prof}" вместе со слоями, пластами и обучающими скважинами?',
                                            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        session.query(MarkupLDA).filter(MarkupLDA.profile_id == get_profile_id()).delete()
        session.query(Formation).filter(Formation.profile_id == get_profile_id()).delete()
        session.query(Layers).filter(Layers.profile_id == get_profile_id()).delete()
        session.query(Profile).filter(Profile.id == get_profile_id()).delete()
        session.commit()
        # vacuum()
        set_info(f'Профиль {title_prof} удалён', 'green')
        update_profile_combobox()
    else:
        pass


def load_uf_grid():
    try:
        file_name = QFileDialog.getOpenFileName(
            caption=f'Выберите grid-файл структурной карты по кровле продуктивного пласта по объекту {get_object_name()}',
            filter='*.dat')[0]
        set_info(file_name, 'blue')
        fn = file_name.split('/')[-1].split('.')[0]
        tab_grid = pd.read_table(file_name, delimiter=' ', header=0).values.tolist()
        new_common_grid = CommonGrid(title = fn, type='uf', grid_table=json.dumps(tab_grid))
        session.add(new_common_grid)
        # if session.query(Grid).filter(Grid.object_id == get_object_id()).count() > 0:
        #     session.query(Grid).filter(Grid.object_id == get_object_id()).update(
        #         {'grid_table_uf': json.dumps(tab_grid)}, synchronize_session="fetch"
        #     )
        # else:
        #     new_grid = Grid(object_id=get_object_id(), grid_table_uf=json.dumps(tab_grid))
        #     session.add(new_grid)
        session.commit()
        set_info(f'Загружен grid-файл структурной карты по кровле продуктивного пласта', 'green')
    except FileNotFoundError:
        return


def load_m_grid():
    try:
        file_name = QFileDialog.getOpenFileName(
            caption=f'Выберите grid-файл карты мощности продуктивного пласта по объекту {get_object_name()}',
            filter='*.dat')[0]
        set_info(file_name, 'blue')
        fn = file_name.split('/')[-1].split('.')[0]
        tab_grid = pd.read_table(file_name, delimiter=' ', header=0).values.tolist()
        new_common_grid = CommonGrid(title = fn, type='m', grid_table=json.dumps(tab_grid))
        session.add(new_common_grid)
        # if session.query(Grid).filter(Grid.object_id == get_object_id()).count() > 0:
        #     session.query(Grid).filter(Grid.object_id == get_object_id()).update(
        #         {'grid_table_m': json.dumps(tab_grid)}, synchronize_session="fetch"
        #     )
        # else:
        #     new_grid = Grid(object_id=get_object_id(), grid_table_m=json.dumps(tab_grid))
        #     session.add(new_grid)
        session.commit()
        set_info(f'Загружен grid-файл карты мощности продуктивного пласта', 'green')
    except FileNotFoundError:
        return


def load_r_grid():
    try:
        file_name = QFileDialog.getOpenFileName(
            caption=f'Выберите grid-файл карты рельефа по объекту {get_object_name()}',
            filter='*.dat')[0]
        set_info(file_name, 'blue')
        fn = file_name.split('/')[-1].split('.')[0]
        tab_grid = pd.read_table(file_name, delimiter=' ', header=0).values.tolist()
        new_common_grid = CommonGrid(title = fn, type='r', grid_table=json.dumps(tab_grid))
        session.add(new_common_grid)
        # if session.query(Grid).filter(Grid.object_id == get_object_id()).count() > 0:
        #     session.query(Grid).filter(Grid.object_id == get_object_id()).update(
        #         {'grid_table_r': json.dumps(tab_grid)}, synchronize_session="fetch"
        #     )
        # else:
        #     new_grid = Grid(object_id=get_object_id(), grid_table_r=json.dumps(tab_grid))
        #     session.add(new_grid)
        session.commit()
        set_info(f'Загружен grid-файл карты рельефа', 'green')
    except FileNotFoundError:
        return


def save_signal():
    radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])
    pd_radar = pd.DataFrame(radar).transpose()
    fn = QFileDialog.getSaveFileName(caption="Сохранить сигнал", filter="TXT (*.txt)")
    pd_radar.to_csv(fn[0], sep=';')

