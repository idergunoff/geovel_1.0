from model import *
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
        date_object = ui_ob.dateEdit.date().toPyDate()
        if name_object != '':
            new_object = GeoradarObject(title=name_object, date_exam=date_object)
            session.add(new_object)
            session.commit()
            update_object()
            Add_Object.close()
            set_info(f'Объект "{name_object}" добавлен в базу данных.', 'green')

    def cancel_add_object():
        Add_Object.close()

    ui_ob.buttonBox.accepted.connect(object_to_db)
    ui_ob.buttonBox.rejected.connect(cancel_add_object)
    Add_Object.exec_()


def update_object():
    """ Обновление списка объектов в выпадающем списке """
    ui.comboBox_object.clear()
    for i in session.query(GeoradarObject).order_by(GeoradarObject.date_exam).all():
        ui.comboBox_object.addItem(f'{i.title} {i.date_exam.strftime("%m.%Y")} id{i.id}')
    update_profile_combobox()


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
        for i in range(int(len(pd_radar.index)/512)):
            list_signal = list(pd_radar['ALn'].loc[i * 512:(i + 1) * 512])
            ui.progressBar.setValue(i+1)
            if len(set(list_signal)) > 1:
                signal.append(list_signal)
        new_profile = Profile(object_id=get_object_id(), title=file_name.split('/')[-1].split('.')[0], signal=json.dumps(signal))
        session.add(new_profile)
        session.commit()
        set_info(f'Профиль загружен ({get_object_name()})', 'green')
    except KeyError:
        set_info('Не правильный формат файла', 'red')
        return
    update_profile_combobox()


def update_profile_combobox():
    """ Обновление списка профилей в выпадающем списке """
    ui.comboBox_profile.clear()
    try:
        for i in session.query(Profile).filter(Profile.object_id == get_object_id()).all():
            count_measure = len(json.loads(session.query(Profile.signal).filter(Profile.id == i.id).first()[0]))
            ui.comboBox_profile.addItem(f'{i.title} ({count_measure} измерений) id{i.id}')
        update_param_combobox()
    except ValueError:
        pass
    if session.query(Grid).filter(Grid.object_id == get_object_id()).count() > 0:
        ui.pushButton_uf.setStyleSheet('background: rgb(191, 255, 191)')
        ui.pushButton_m.setStyleSheet('background: rgb(191, 255, 191)')
        ui.pushButton_r.setStyleSheet('background: rgb(191, 255, 191)')
    else:
        ui.pushButton_uf.setStyleSheet('background: rgb(255, 185, 185)')
        ui.pushButton_m.setStyleSheet('background:  rgb(255, 185, 185)')
        ui.pushButton_r.setStyleSheet('background: rgb(255, 185, 185)')


def load_param():
    """ Загрузка параметров """
    try:
        file_name = QFileDialog.getOpenFileName(caption='Выберите файл выделенного интервала пласта', filter='*.txt')[0]
        set_info(file_name, 'blue')
        tab_int = pd.read_table(file_name, delimiter=';', header=0)
        tab_int = tab_int.applymap(lambda x: float(str(x).replace(',', '.')))
    except FileNotFoundError:
        return
    signals = json.loads(session.query(Profile.signal).filter(Profile.id == get_profile_id()).first()[0])
    if len(signals) != len(tab_int.index):
        set_info('ВНИМАНИЕ! ОШИБКА!!! Не совпадает количество измерений в файлах', 'red')
    else:
        ui.progressBar.setMaximum(len(tab_int.index))
        grid_db = session.query(Grid).filter(Grid.object_id == get_object_id()).first()
        if grid_db:
            pd_grid_uf = pd.DataFrame(json.loads(grid_db.grid_table_uf))
            pd_grid_m = pd.DataFrame(json.loads(grid_db.grid_table_m))
            pd_grid_r = pd.DataFrame(json.loads(grid_db.grid_table_r))
        x_wgs_l, y_wgs_l, x_pulc_l, y_pulc_l, T_top_l, T_bottom_l, dT_l, A_top_l, A_bottom_l, dA_l, A_sum_l, A_mean_l, dVt_l, Vt_top_l, Vt_sum_l, Vt_mean_l, dAt_l, At_top_l, \
        At_sum_l, At_mean_l, dPht_l, Pht_top_l, Pht_sum_l, Pht_mean_l, Wt_top_l, Wt_mean_l, Wt_sum_l, std_l, k_var_l, skew_l, kurt_l, width_l, top_l, land_l, speed_l, speed_cover_l = \
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        for i in tab_int.index:
            signal = signals[i]
            analytic_signal = hilbert(signal)
            At = np.hypot(signal, np.imag(analytic_signal)).tolist()
            Vt = np.imag(analytic_signal).tolist()
            Pht = np.angle(analytic_signal).tolist()
            Wt = np.diff(Pht).tolist()
            nt = int(tab_int['T01'][i] / 40)
            nb = int(tab_int['D02'][i] / 40)
            y, x = wgs84_to_pulc42(tab_int['Latd'][i], tab_int['Long'][i])
            x_wgs_l.append(tab_int['Long'][i])
            y_wgs_l.append(tab_int['Latd'][i])
            x_pulc_l.append(x)
            y_pulc_l.append(y)
            T_top_l.append(tab_int['T01'][i] / 5)
            T_bottom_l.append(tab_int['D02'][i] / 5)
            dT_l.append(tab_int['D02'][i] / 5 - tab_int['T01'][i] / 5)
            A_top_l.append(signal[nt])
            A_bottom_l.append(signal[nb])
            dA_l.append(signal[nb] - signal[nt])
            A_sum_l.append(float(np.sum(signal[nt:nb])))
            A_mean_l.append(float(np.mean(signal[nt:nb])))
            dVt_l.append(Vt[nb] - Vt[nt])
            Vt_top_l.append(Vt[nt])
            Vt_sum_l.append(float(np.sum(Vt[nt:nb])))
            Vt_mean_l.append(float(np.mean(Vt[nt:nb])))
            dAt_l.append(At[nb] - At[nt])
            At_top_l.append(At[nt])
            At_sum_l.append(float(np.sum(At[nt:nb])))
            At_mean_l.append(float(np.mean(At[nt:nb])))
            dPht_l.append(Pht[nb] - Pht[nt])
            Pht_top_l.append(Pht[nt])
            Pht_sum_l.append(float(np.sum(Pht[nt:nb])))
            Pht_mean_l.append(float(np.mean(Pht[nt:nb])))
            Wt_top_l.append(Wt[nt])
            Wt_mean_l.append(float(np.mean(Wt[nt:nb])))
            Wt_sum_l.append(float(np.sum(Wt[nt:nb])))
            std_l.append(float(np.std(signal[nt:nb])))
            k_var_l.append(float(np.var(signal[nt:nb])))
            skew_l.append(skew(signal[nt:nb]))
            kurt_l.append(kurtosis(signal[nt:nb]))
            if grid_db:
                pd_grid_uf['dist_y'] = abs(pd_grid_uf[1] - y)
                pd_grid_uf['dist_x'] = abs(pd_grid_uf[0] - x)
                pd_grid_m['dist_y'] = abs(pd_grid_m[1] - y)
                pd_grid_m['dist_x'] = abs(pd_grid_m[0] - x)
                pd_grid_r['dist_y'] = abs(pd_grid_r[1] - y)
                pd_grid_r['dist_x'] = abs(pd_grid_r[0] - x)
                i_uf = pd_grid_uf.loc[pd_grid_uf['dist_y'] == pd_grid_uf['dist_y'].min()].loc[pd_grid_uf['dist_x'] == pd_grid_uf['dist_x'].min()].iat[0, 2]
                i_m = pd_grid_m.loc[pd_grid_m['dist_y'] == pd_grid_m['dist_y'].min()].loc[pd_grid_m['dist_x'] == pd_grid_m['dist_x'].min()].iat[0, 2]
                i_r = pd_grid_r.loc[pd_grid_r['dist_y'] == pd_grid_r['dist_y'].min()].loc[pd_grid_r['dist_x'] == pd_grid_r['dist_x'].min()].iat[0, 2]
                im = i_m if i_m > 0 else 0
                width_l.append(im)
                top_l.append(i_uf)
                land_l.append(i_r)
                speed_l.append(im * 100 / (tab_int['D02'][i] / 5 - tab_int['T01'][i] / 5))
                speed_cover_l.append((i_r - i_uf) * 100 / (tab_int['T01'][i] / 5))
            ui.progressBar.setValue(i+1)
        dict_signal = {'x_wgs': json.dumps(x_wgs_l),
                       'y_wgs': json.dumps(y_wgs_l),
                        'x_pulc': json.dumps(x_pulc_l),
                        'y_pulc': json.dumps(y_pulc_l),
                        'T_top': json.dumps(T_top_l),
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
                        'kurt': json.dumps(kurt_l)}
        if grid_db:
            dict_signal['width'] = json.dumps(width_l)
            dict_signal['top'] = json.dumps(top_l)
            dict_signal['land'] = json.dumps(land_l)
            dict_signal['speed'] = json.dumps(speed_l)
            dict_signal['speed_cover'] = json.dumps(speed_cover_l)
        session.query(Profile).filter(Profile.id == get_profile_id()).update(dict_signal, synchronize_session="fetch")
        session.commit()
        set_info(f'Параметры загружены ({get_object_name()}, {get_profile_name()})', 'green')
        update_param_combobox()


def delete_profile():
    title_prof = ui.comboBox_profile.currentText().split(' id')[0]
    session.query(Profile).filter(Profile.id == get_profile_id()).delete()
    session.commit()
    vacuum()
    set_info(f'Профиль {title_prof} удалён', 'green')
    update_profile_combobox()


def update_param_combobox():
    ui.comboBox_param_plast.clear()
    list_columns = Profile.__table__.columns.keys()  # список параметров таблицы
    [list_columns.remove(i) for i in ['id', 'object_id', 'title', 'x_wgs', 'y_wgs', 'x_pulc', 'y_pulc', 'signal']]  # удаляем не нужные колонки
    for i in list_columns:
        if session.query(Profile).filter(text(f"profile_id=:p_id and {i} NOT NULL")).params(p_id=get_profile_id()).count() > 0:
            ui.comboBox_param_plast.addItem(i)
    update_layers()


def load_uf_grid():
    try:
        file_name = QFileDialog.getOpenFileName(
            caption=f'Выберите grid-файл структурной карты по кровле продуктивного пласта по объекту {get_object_name()}',
            filter='*.dat')[0]
        set_info(file_name, 'blue')
        tab_grid = pd.read_table(file_name, delimiter=' ', header=0).values.tolist()
        if session.query(Grid).filter(Grid.object_id == get_object_id()).count() > 0:
            session.query(Grid).filter(Grid.object_id == get_object_id()).update(
                {'grid_table_uf': json.dumps(tab_grid)}, synchronize_session="fetch"
            )
        else:
            new_grid = Grid(object_id=get_object_id(), grid_table_uf=json.dumps(tab_grid))
            session.add(new_grid)
        session.commit()
        set_info(f'Загружен grid-файл структурной карты по кровле продуктивного пласта по объекту {get_object_name()}', 'green')
    except FileNotFoundError:
        return


def load_m_grid():
    try:
        file_name = QFileDialog.getOpenFileName(
            caption=f'Выберите grid-файл карты мощности продуктивного пласта по объекту {get_object_name()}',
            filter='*.dat')[0]
        set_info(file_name, 'blue')
        tab_grid = pd.read_table(file_name, delimiter=' ', header=0).values.tolist()
        if session.query(Grid).filter(Grid.object_id == get_object_id()).count() > 0:
            session.query(Grid).filter(Grid.object_id == get_object_id()).update(
                {'grid_table_m': json.dumps(tab_grid)}, synchronize_session="fetch"
            )
        else:
            new_grid = Grid(object_id=get_object_id(), grid_table_m=json.dumps(tab_grid))
            session.add(new_grid)
        session.commit()
        set_info(f'Загружен grid-файл карты мощности продуктивного пласта по объекту {get_object_name()}', 'green')
    except FileNotFoundError:
        return


def load_r_grid():
    try:
        file_name = QFileDialog.getOpenFileName(
            caption=f'Выберите grid-файл карты рельефа по объекту {get_object_name()}',
            filter='*.dat')[0]
        set_info(file_name, 'blue')
        tab_grid = pd.read_table(file_name, delimiter=' ', header=0).values.tolist()
        if session.query(Grid).filter(Grid.object_id == get_object_id()).count() > 0:
            session.query(Grid).filter(Grid.object_id == get_object_id()).update(
                {'grid_table_r': json.dumps(tab_grid)}, synchronize_session="fetch"
            )
        else:
            new_grid = Grid(object_id=get_object_id(), grid_table_r=json.dumps(tab_grid))
            session.add(new_grid)
        session.commit()
        set_info(f'Загружен grid-файл карты рельефа по объекту {get_object_name()}', 'green')
    except FileNotFoundError:
        return


def save_signal():
    radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])
    pd_radar = pd.DataFrame(radar).transpose()
    fn = QFileDialog.getSaveFileName(caption="Сохранить сигнал", filter="TXT (*.txt)")
    pd_radar.to_csv(fn[0], sep=';')

