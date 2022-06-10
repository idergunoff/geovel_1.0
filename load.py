from model import *
from func import *
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


def load_profile():
    """ загрузка профилей """
    try:
        file_name = QFileDialog.getOpenFileName(filter='*.txt')[0]
        set_info(file_name, 'blue')
    except FileNotFoundError:
        return
    new_profile = Profile(object_id=get_object_id(), title=file_name.split('/')[-1].split('.')[0])
    session.add(new_profile)
    session.commit()

    pd_radar = pd.read_table(file_name, delimiter=';', header=0)

    pd_radar = pd_radar.rename(columns={'X': 'measure_id', 'T': 'time_ns', 'ALn': 'A'})
    pd_radar = pd_radar.drop(columns=['Unnamed: 3'])
    for i in range(pd_radar['measure_id'].max() + 1):
        new_measure = Measure(profile_id=new_profile.id, number=i)
        session.add(new_measure)
    session.commit()
    l = query_to_list(session.query(Measure.id).filter(Measure.profile_id == new_profile.id).all())
    m = [i for i in l for _ in range(512)]

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