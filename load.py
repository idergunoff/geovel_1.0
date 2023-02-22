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
        new_profile = Profile(object_id=get_object_id(), title=file_name.split('/')[-1].split('.')[0])
        session.add(new_profile)
        session.commit()
        ui.progressBar.setMaximum(int(len(pd_radar.index)/512))
        for i in range(int(len(pd_radar.index)/512)):
            list_signal = list(pd_radar['ALn'].loc[i * 512:(i + 1) * 512])
            ui.progressBar.setValue(i+1)
            if len(set(list_signal)) > 1:
                # analytic_signal = hilbert(list_signal)
                # Vt = list(map(lambda x: round(x, 2), np.imag(analytic_signal)))
                # At = list(map(lambda x: round(x, 2), np.hypot(list_signal, np.imag(analytic_signal))))
                # Pht = list(map(lambda x: round(x, 2), np.angle(analytic_signal)))
                # Wt = list(map(lambda x: round(x, 2), np.diff(np.angle(analytic_signal))))
                new_measure = Measure(
                    profile_id=new_profile.id,
                    signal = json.dumps(list_signal),
                    number = i
                    # diff = json.dumps(np.diff(list_signal).tolist()),
                    # Vt = json.dumps(Vt),
                    # At = json.dumps(At),
                    # Pht = json.dumps(Pht),
                    # Wt = json.dumps(Wt)
                )
                session.add(new_measure)
        session.commit()
    except KeyError:
        set_info('Не правильный формат файла', 'red')
        return
    #
    # session.commit()
    # for i in range(pd_radar['measure_id'].max() + 1):
    #     new_measure = Measure(profile_id=new_profile.id, number=i)
    #     session.add(new_measure)
    # session.commit()
    # l = query_to_list(session.query(Measure.id).filter(Measure.profile_id == new_profile.id).all())
    # m = [i for i in l for _ in range(512)]
    # # print(m)
    # pd_radar['measure_id'] = m
    # # print(pd_radar)
    # pd_radar.to_sql('signal', con=engine, if_exists='append', index=False)
    #
    # list_zero = list(set(query_to_list(session.query(Measure.id).join(Signal).filter(Signal.A == 0).all())))
    # for i in list_zero:
    #     session.query(Measure).filter(Measure.id == i).delete()
    # session.query(Signal).filter(Signal.A == 0).delete()
    # session.commit()
    #
    # l = query_to_list(session.query(Measure.id).filter(Measure.profile_id == new_profile.id).all())
    # ui.progressBar.setMaximum(len(l))
    # ui.progressBar.setValue(0)
    # for n, i in enumerate(l):
    #
    #     pd_signal = pd.read_sql(f'SELECT * FROM signal WHERE measure_id={i};', engine, index_col='id')
    #     session.query(Signal).filter(Signal.measure_id == i).delete()
    #     session.commit()
    #     pd_signal['diff'] = pd_signal['A'].diff()
    #     analytic_signal = hilbert(pd_signal['A'])
    #     pd_signal['Vt'] = np.imag(analytic_signal)
    #     pd_signal['At'] = np.hypot(pd_signal['A'], np.imag(analytic_signal))
    #     pd_signal['Pht'] = np.angle(analytic_signal)
    #     pd_signal['Wt'] = pd_signal['Pht'].diff()
    #     # print(pd_signal)
    #     pd_signal.to_sql('signal', con=engine, if_exists='append', index=True)
    #     ui.progressBar.setValue(n + 1)
    # session.commit()
    update_profile_combobox()


def update_profile_combobox():
    """ Обновление списка профилей в выпадающем списке """
    ui.comboBox_profile.clear()
    try:
        for i in session.query(Profile).filter(Profile.object_id == get_object_id()).all():
            count_measure = session.query(Measure).filter(Measure.profile_id == i.id).count()
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

    measures = session.query(Measure).filter(Measure.profile_id == get_profile_id()).order_by(Measure.number).all()
    if len(measures) != len(tab_int.index):
        set_info('ВНИМАНИЕ! ОШИБКА!!! Не совпадает количество измерений в файлах', 'red')
    else:
        ui.progressBar.setMaximum(len(tab_int.index))
        grid_db = session.query(Grid).filter(Grid.object_id == get_object_id()).first()
        if grid_db:
            pd_grid_uf = pd.DataFrame(json.loads(grid_db.grid_table_uf))
            pd_grid_m = pd.DataFrame(json.loads(grid_db.grid_table_m))
            pd_grid_r = pd.DataFrame(json.loads(grid_db.grid_table_r))
        for i in tab_int.index:
            signal = json.loads(measures[i].signal)
            analytic_signal = hilbert(signal)
            At = np.hypot(signal, np.imag(analytic_signal)).tolist()
            Vt = np.imag(analytic_signal).tolist()
            Pht = np.angle(analytic_signal).tolist()
            Wt = np.diff(Pht).tolist()
            nt = int(tab_int['T01'][i] / 40)
            nb = int(tab_int['D02'][i] / 40)
            y, x = wgs84_to_pulc42(tab_int['Latd'][i], tab_int['Long'][i])
            dict_measure = {
                'x_wgs': tab_int['Long'][i],
                'y_wgs': tab_int['Latd'][i],
                'x_pulc': x,
                'y_pulc': y,
                'T_top': tab_int['T01'][i] / 5,
                'T_bottom': tab_int['D02'][i] / 5,
                'dT': tab_int['D02'][i] / 5 - tab_int['T01'][i] / 5,
                'A_top': signal[nt],
                'A_bottom': signal[nb],
                'dA': signal[nb] - signal[nt],
                'A_sum': np.sum(signal[nt:nb]),
                'A_mean': np.mean(signal[nt:nb]),
                'dVt': Vt[nb] - Vt[nt],
                'Vt_top': Vt[nt],
                'Vt_sum': np.sum(Vt[nt:nb]),
                'Vt_mean': np.mean(Vt[nt:nb]),
                'dAt': At[nb] - At[nt],
                'At_top': At[nt],
                'At_sum': np.sum(At[nt:nb]),
                'At_mean': np.mean(At[nt:nb]),
                'dPht': Pht[nb] - Pht[nt],
                'Pht_top' : Pht[nt],
                'Pht_sum': np.sum(Pht[nt:nb]),
                'Pht_mean': np.mean(Pht[nt:nb]),
                'Wt_top': Wt[nt],
                'Wt_mean': np.mean(Wt[nt:nb]),
                'Wt_sum': np.sum(Wt[nt:nb]),
                'std': np.std(signal),
                'k_var': np.var(signal),
                'skew': skew(signal),
                'kurt': kurtosis(signal)}
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
                dict_measure['width'] = i_m if i_m > 0 else 0
                dict_measure['top'] = i_uf
                dict_measure['land'] = i_r
                dict_measure['speed'] = dict_measure['width'] * 100 / (tab_int['D02'][i] / 5 - tab_int['T01'][i] / 5)
                dict_measure['speed_cover'] = (i_r - i_uf) * 100 / (tab_int['T01'][i] / 5)
            session.query(Measure).filter(Measure.id == measures[i].id).update(dict_measure, synchronize_session="fetch")
            ui.progressBar.setValue(i+1)
        session.commit()
        update_param_combobox()


def delete_profile():
    title_prof = ui.comboBox_profile.currentText().split(' id')[0]
    session.query(Measure).filter(Measure.profile_id == get_profile_id()).delete()
    session.query(Profile).filter(Profile.id == get_profile_id()).delete()
    session.commit()
    vacuum()
    set_info(f'Профиль {title_prof} удалён', 'green')
    update_profile_combobox()


def update_param_combobox():
    ui.comboBox_param_plast.clear()
    list_columns = Measure.__table__.columns.keys()  # список параметров таблицы
    [list_columns.remove(i) for i in ['id', 'profile_id', 'number', 'x_wgs', 'y_wgs', 'x_pulc', 'y_pulc', 'signal']]  # удаляем не нужные колонки
    for i in list_columns:
        if session.query(Measure).filter(text(f"profile_id=:p_id and {i} NOT NULL")).params(p_id=get_profile_id()).count() > 0:
            ui.comboBox_param_plast.addItem(i)


def draw_radarogram():
    clear_current_profile()
    rad = session.query(Measure.signal).filter(Measure.profile_id == get_profile_id()).all()
    ui.progressBar.setMaximum(len(rad))
    radar = []
    for n, i in enumerate(rad):
        ui.progressBar.setValue(n + 1)
        if ui.comboBox_atrib.currentText() == 'A':
            radar.append(json.loads(i[0]))
        elif ui.comboBox_atrib.currentText() == 'diff':
            radar.append(np.diff(json.loads(i[0])).tolist())
        elif ui.comboBox_atrib.currentText() == 'At':
            analytic_signal = hilbert(json.loads(i[0]))
            radar.append(list(map(lambda x: round(x, 2), np.hypot(json.loads(i[0]), np.imag(analytic_signal)))))
        elif ui.comboBox_atrib.currentText() == 'Vt':
            analytic_signal = hilbert(json.loads(i[0]))
            radar.append(list(map(lambda x: round(x, 2), np.imag(analytic_signal))))
        elif ui.comboBox_atrib.currentText() == 'Pht':
            analytic_signal = hilbert(json.loads(i[0]))
            radar.append(list(map(lambda x: round(x, 2), np.angle(analytic_signal))))
        elif ui.comboBox_atrib.currentText() == 'Wt':
            analytic_signal = hilbert(json.loads(i[0]))
            radar.append(list(map(lambda x: round(x, 2), np.diff(np.angle(analytic_signal)))))
        else:
            radar.append(json.loads(i[0]))
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(radar))
    session.add(new_current)
    session.commit()
    draw_image(radar)
    updatePlot()



def draw_current_radarogram():
    rad = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])
    ui.progressBar.setMaximum(len(rad))
    radar = []
    for n, i in enumerate(rad):
        ui.progressBar.setValue(n + 1)
        if ui.comboBox_atrib.currentText() == 'A':
            radar.append(i)
        elif ui.comboBox_atrib.currentText() == 'diff':
            radar.append(np.diff(i).tolist())
        elif ui.comboBox_atrib.currentText() == 'At':
            analytic_signal = hilbert(i)
            radar.append(list(map(lambda x: round(x, 2), np.hypot(i, np.imag(analytic_signal)))))
        elif ui.comboBox_atrib.currentText() == 'Vt':
            analytic_signal = hilbert(i)
            radar.append(list(map(lambda x: round(x, 2), np.imag(analytic_signal))))
        elif ui.comboBox_atrib.currentText() == 'Pht':
            analytic_signal = hilbert(i)
            radar.append(list(map(lambda x: round(x, 2), np.angle(analytic_signal))))
        elif ui.comboBox_atrib.currentText() == 'Wt':
            analytic_signal = hilbert(i)
            radar.append(list(map(lambda x: round(x, 2), np.diff(np.angle(analytic_signal)))))
        else:
            radar.append(i)
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(radar))
    session.add(new_current)
    session.commit()
    draw_image(radar)
    updatePlot()





def draw_max_min():
    rad = session.query(CurrentProfile.signal).first()
    radar = json.loads(rad[0])
    radar_max_min = []
    print(len(radar))
    ui.progressBar.setMaximum(len(radar))
    for n, sig in enumerate(radar):
        diff_signal = np.diff(sig)

        max_points = argrelmax(diff_signal)
        min_points = argrelmin(diff_signal)
        signal_max_min = []
        for j in range(512):
            if j in max_points[0]:
                signal_max_min.append(1)
            elif j in min_points[0]:
                signal_max_min.append(-1)
            else:
                signal_max_min.append(0)
        radar_max_min.append(signal_max_min)
        ui.progressBar.setValue(n + 1)
    print(len(radar_max_min))
    draw_image(radar_max_min)
    updatePlot()


def draw_param():
    param = ui.comboBox_param_plast.currentText()
    graph = query_to_list(session.query(literal_column(f'Measure.{param}')).filter(Measure.profile_id == get_profile_id()).order_by(Measure.number).all())
    number = query_to_list(session.query(Measure.number).filter(Measure.profile_id == get_profile_id()).order_by(Measure.number).all())
    ui.graph.clear()
    curve = pg.PlotCurveItem(x=number, y=graph)
    curve_filter = pg.PlotCurveItem(x=number, y=savgol_filter(graph, 31, 3), pen=pg.mkPen(color='red', width=2.4))
    ui.graph.addItem(curve)
    ui.graph.addItem(curve_filter)
    ui.graph.showGrid(x=True, y=True)


def updatePlot():
    rad = session.query(CurrentProfile.signal).first()
    radar = json.loads(rad[0])
    selected = roi.getArrayRegion(np.array(radar), img)
    n = ui.spinBox_roi.value()//2
    ui.signal.plot(y=range(512, 0, -1), x=selected.mean(axis=0), clear=True, pen='r')
    ui.signal.plot(y=range(512, 0, -1), x=selected[n])
    ui.signal.showGrid(x=True, y=True)


def draw_rad_line():
    radarogramma.clear()
    radarogramma.addItem(img)
    radarogramma.addItem(roi)
    rad = session.query(CurrentProfile.signal).first()
    radar = json.loads(rad[0])
    draw_image(radar)
    updatePlot()
    line_up = ui.spinBox_rad_up.value()
    line_down = ui.spinBox_rad_down.value()
    l_up = pg.InfiniteLine(pos=line_up, angle=90, pen=pg.mkPen(color='darkred',width=1, dash=[8, 2]))
    l_down = pg.InfiniteLine(pos=line_down, angle=90, pen=pg.mkPen(color='darkgreen', width=1, dash=[8, 2]))
    radarogramma.addItem(l_up)
    radarogramma.addItem(l_down)


def clear_current_profile():
    session.query(CurrentProfile).delete()
    session.commit()


def vacuum():
    conn = connect(DATABASE_NAME)
    conn.execute("VACUUM")
    conn.close()


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
    except FileNotFoundError:
        return


def save_signal():
    radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])
    pd_radar = pd.DataFrame(radar).transpose()
    fn = QFileDialog.getSaveFileName(caption="Сохранить сигнал", filter="TXT (*.txt)")
    pd_radar.to_csv(fn[0], sep=';')


def changeSpinBox():
    roi.setSize([ui.spinBox_roi.value(), 512])
