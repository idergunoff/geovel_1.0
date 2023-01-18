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
        pd_radar = pd_radar.rename(columns={'X': 'measure_id', 'T': 'time_ns', 'ALn': 'A'})
        pd_radar = pd_radar.drop(columns=['Unnamed: 3'])
    except KeyError:
        set_info('Не правильный формат файла', 'red')
        return
    new_profile = Profile(object_id=get_object_id(), title=file_name.split('/')[-1].split('.')[0])
    session.add(new_profile)
    session.commit()
    for i in range(pd_radar['measure_id'].max() + 1):
        new_measure = Measure(profile_id=new_profile.id, number=i)
        session.add(new_measure)
    session.commit()
    l = query_to_list(session.query(Measure.id).filter(Measure.profile_id == new_profile.id).all())
    m = [i for i in l for _ in range(512)]
    print(m)
    pd_radar['measure_id'] = m
    print(pd_radar)
    pd_radar.to_sql('signal', con=engine, if_exists='append', index=False)

    list_zero = list(set(query_to_list(session.query(Measure.id).join(Signal).filter(Signal.A == 0).all())))
    for i in list_zero:
        session.query(Measure).filter(Measure.id == i).delete()
    session.query(Signal).filter(Signal.A == 0).delete()
    session.commit()

    l = query_to_list(session.query(Measure.id).filter(Measure.profile_id == new_profile.id).all())
    ui.progressBar.setMaximum(len(l))
    ui.progressBar.setValue(0)
    for n, i in enumerate(l):

        pd_signal = pd.read_sql(f'SELECT * FROM signal WHERE measure_id={i};', engine, index_col='id')
        session.query(Signal).filter(Signal.measure_id == i).delete()
        session.commit()
        pd_signal['diff'] = pd_signal['A'].diff()
        analytic_signal = hilbert(pd_signal['A'])
        pd_signal['Vt'] = np.imag(analytic_signal)
        pd_signal['At'] = np.hypot(pd_signal['A'], np.imag(analytic_signal))
        pd_signal['Pht'] = np.angle(analytic_signal)
        pd_signal['Wt'] = pd_signal['Pht'].diff()
        print(pd_signal)
        pd_signal.to_sql('signal', con=engine, if_exists='append', index=True)
        ui.progressBar.setValue(n + 1)
    session.commit()
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


def load_param():
    """ Загрузка параметров """
    try:
        file_name = QFileDialog.getOpenFileName(caption='Выберите файл выделенного интервала пласта', filter='*.txt')[0]
        set_info(file_name, 'blue')
        tab_int = pd.read_table(file_name, delimiter=';', header=0)
    except FileNotFoundError:
        return

    measures = session.query(Measure).filter(Measure.profile_id == get_profile_id()).order_by(Measure.number).all()
    if len(measures) != len(tab_int.index):
        set_info('ВНИМАНИЕ! ОШИБКА!!! Не совпадает количество измерений в файлах', 'red')
    else:
        ui.progressBar.setMaximum(len(tab_int.index))
        for i in tab_int.index:
            list_A = query_to_list(session.query(Signal.A).filter(
                Signal.measure_id == measures[i].id,
                Signal.time_ns >= tab_int['T01'][i] / 5,
                Signal.time_ns <= tab_int['D02'][i] / 5
            ).order_by(Signal.time_ns).all())
            list_At = query_to_list(session.query(Signal.At).filter(
                Signal.measure_id == measures[i].id,
                Signal.time_ns >= tab_int['T01'][i] / 5,
                Signal.time_ns <= tab_int['D02'][i] / 5
            ).order_by(Signal.time_ns).all())
            list_Vt = query_to_list(session.query(Signal.Vt).filter(
                Signal.measure_id == measures[i].id,
                Signal.time_ns >= tab_int['T01'][i] / 5,
                Signal.time_ns <= tab_int['D02'][i] / 5
            ).order_by(Signal.time_ns).all())
            list_Pht = query_to_list(session.query(Signal.Pht).filter(
                Signal.measure_id == measures[i].id,
                Signal.time_ns >= tab_int['T01'][i] / 5,
                Signal.time_ns <= tab_int['D02'][i] / 5
            ).order_by(Signal.time_ns).all())
            list_Wt = query_to_list(session.query(Signal.Wt).filter(
                Signal.measure_id == measures[i].id,
                Signal.time_ns >= tab_int['T01'][i] / 5,
                Signal.time_ns <= tab_int['D02'][i] / 5
            ).order_by(Signal.time_ns).all())
            y, x = wgs84_to_pulc42(tab_int['Latd'][i], tab_int['Long'][i])
            session.query(Measure).filter(Measure.id == measures[i].id).update({
                'x_wgs': tab_int['Long'][i],
                'y_wgs': tab_int['Latd'][i],
                'x_pulc': x,
                'y_pulc': y,
                'T_top': tab_int['T01'][i] / 5,
                'T_bottom': tab_int['D02'][i] / 5,
                'dT': tab_int['D02'][i] / 5 - tab_int['T01'][i] / 5,
                'A_top': list_A[0],
                'A_bottom': list_A[-1],
                'dA': list_A[-1] - list_A[0],
                'A_sum': np.sum(list_A),
                'A_mean': np.mean(list_A),
                'dVt': list_Vt[-1] - list_Vt[0],
                'Vt_top': list_Vt[0],
                'Vt_sum': np.sum(list_Vt),
                'Vt_mean': np.mean(list_Vt),
                'dAt': list_At[-1] - list_At[0],
                'At_top': list_At[0],
                'At_sum': np.sum(list_At),
                'At_mean': np.mean(list_At),
                'dPht': list_Pht[-1] - list_Pht[0],
                'Pht_top' : list_Pht[0],
                'Pht_sum': np.sum(list_Pht),
                'Pht_mean': np.mean(list_Pht),
                'Wt_top': list_Wt[0],
                'Wt_mean': np.mean(list_Wt),
                'Wt_sum': np.sum(list_Wt),
                'std': np.std(list_A),
                'k_var': np.var(list_A),
                'skew': skew(list_A),
                'kurt': kurtosis(list_A)},
            synchronize_session="fetch")
            ui.progressBar.setValue(i+1)
        session.commit()
        update_profile_combobox()


def delete_profile():
    title_prof = ui.comboBox_profile.currentText().split(' id')[0]
    l = query_to_list(session.query(Measure.id).filter(Measure.profile_id == get_profile_id()).all())
    for id_measure in l:
        session.query(Signal).filter(Signal.measure_id == id_measure).delete()
    session.query(Measure).filter(Measure.profile_id == get_profile_id()).delete()
    session.query(Profile).filter(Profile.id == get_profile_id()).delete()
    session.commit()
    set_info(f'Профиль {title_prof} удалён', 'green')
    update_profile_combobox()


def update_param_combobox():
    ui.comboBox_param_plast.clear()
    list_columns = Measure.__table__.columns.keys()  # список параметров таблицы
    [list_columns.remove(i) for i in ['id', 'profile_id', 'number', 'x_wgs', 'y_wgs', 'x_pulc', 'y_pulc']]  # удаляем не нужные колонки
    for i in list_columns:
        if session.query(Measure).filter(text(f"profile_id=:p_id and {i} NOT NULL")).params(p_id=get_profile_id()).count() > 0:
            ui.comboBox_param_plast.addItem(i)

    # with open(file_name) as f:
    #     count_line = sum(1 for _ in f)
    #
    # with open(file_name, 'rb') as f:
    #     ui.progressBar.setMaximum(count_line)
    #     ui.progressBar.setValue(0)
    #     n = 0
    #     n_izm = 1
    #     list_sig = []
    #     while True:
    #         try:
    #             line = f.readline()
    #             if not line:
    #                 break
    #             line_signal = line.strip().decode('utf-8').split(';')
    #             if not session.query(Measure).filter(Measure.profile_id == new_profile.id, Measure.number == int(line_signal[0])).first():
    #                 n_izm = int(line_signal[0])
    #                 new_measure = Measure(profile_id=new_profile.id, number=int(line_signal[0]))
    #                 session.add(new_measure)
    #
    #             new_signal_point = Signal(measure_id=n_izm, time_ns=int(line_signal[1]), A=int(line_signal[2]))
    #             list_sig.append(new_signal_point)
    #             # session.add(new_signal_point)
    #         except ValueError:
    #             pass
    #         n += 1
    #         ui.progressBar.setValue(n)
    #     session.add_all(list_sig)
    #     session.commit()


def draw_radarogram():
    rad = session.query(Signal.A).join(Measure).filter(Measure.profile_id == get_profile_id()).all()
    ui.progressBar.setMaximum(len(rad)/512)
    radar = []
    for i in range(int(len(rad)/512)):
        ui.progressBar.setValue(i+1)
        if i != len(rad)/512:
            radar.append(rad[i*512: (i+1)*512])
    ui.radarogram.setImage(np.array(radar))

    colors = [
        (255, 0, 0),
        (0, 0, 0),
        (0, 0, 255)
    ]
    cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 3), color=colors)
    ui.radarogram.setColorMap(cmap)