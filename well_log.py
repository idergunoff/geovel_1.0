import json

from numba.scripts.generate_lower_listing import description

from func import *



def show_well_log():
    WellLogForm = QtWidgets.QDialog()
    ui_wl = Ui_Form_well_log()
    ui_wl.setupUi(WellLogForm)
    WellLogForm.show()
    WellLogForm.setAttribute(Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    m_width, m_height = get_width_height_monitor()
    WellLogForm.resize(int(m_width/4), m_height - 200)

    ui_wl.label.setText(f'Каротажные кривые скважины {get_well_name()}')

    def update_list_well_log():
        well_log = session.query(WellLog).filter(WellLog.well_id == get_well_id()).all()
        ui_wl.listWidget_well_log.clear()
        for log in well_log:
            item = QtWidgets.QListWidgetItem(f'{log.curve_name} ID{log.id}')
            item.setToolTip(f'{log.begin} - {log.end}; {log.step}\n{log.description}')
            ui_wl.listWidget_well_log.addItem(item)

    def draw_well_log():
        try:
            well_log_id = ui_wl.listWidget_well_log.currentItem().text().split(' ID')[-1]
            well_log = session.query(WellLog).filter_by(id=well_log_id).first()
        except AttributeError:
            return

        if not well_log:
            return

        try:
            Y = json.loads(well_log.curve_data)
            Y = np.array(Y)

            # Проверка на нулевой шаг
            if well_log.step == 0:
                print("Ошибка: шаг (step) равен нулю.")
                well_log.step = 0.1

            # Подсчёт длины массива X, чтобы совпадал с Y
            X = np.arange(well_log.begin, well_log.begin + len(Y) * well_log.step, well_log.step)

            # В случае если длины всё равно не совпадают — подрежем X
            if len(X) > len(Y):
                X = X[:len(Y)]
            elif len(Y) > len(X):
                Y = Y[:len(X)]

            curve = pg.PlotCurveItem(Y, X)

            ui_wl.widget_graph_well_log.clear()
            ui_wl.widget_graph_well_log.showGrid(x=True, y=True)
            ui_wl.widget_graph_well_log.invertY(True)
            ui_wl.widget_graph_well_log.addItem(curve)

        except Exception as e:
            print(f"Ошибка при построении графика: {e}")


    def load_well_log():
        filename = QtWidgets.QFileDialog.getOpenFileName(
            caption='Выберите файл LAS',
            filter='*.las'
        )[0]

        if filename:
            add_well_log_to_db(filename)
        update_list_well_log()


    def load_well_log_by_dir():
        filename = QtWidgets.QFileDialog.getExistingDirectory(
            caption='Выберите папку',
        )

        if filename:
            for file in os.listdir(filename):
                if file.endswith('.las'):
                    add_well_log_to_db(f'{filename}/{file}')
        update_list_well_log()


    def add_well_log_to_db(las_file):
        las = ls.read(las_file)
        list_curves = las.keys()

        for curve in list_curves:
            if curve in ['DEPT', 'DEPH', 'MD', 'DEPTH']:
                continue
            try:
                description = (f'Скв.: {las.well["WELL"].value}\n'
                               f'Площ.:' f'{las.well["AREA"].value}\n'
                               f'Дата: {las.well["DATE"].value}')
            except KeyError:
                description = (f'Скв.: {las.well["WELL"].value}\n'
                               f'Площ.:' f'{las.well["FLD"].value}\n'
                               f'Дата: {las.well["DATE"].value}')
            new_well_log = WellLog(
                well_id=get_well_id(),
                curve_name=curve,
                curve_data=json.dumps(list(las[curve])),
                begin=las.well["STRT"].value,
                end=las.well["STOP"].value,
                step=las.well["STEP"].value,
                description=description
            )
            session.add(new_well_log)
        session.commit()


    def remove_well_log():
        session.query(WellLog).filter_by(id=ui_wl.listWidget_well_log.currentItem().text().split(' ID')[-1]).delete()
        session.commit()
        update_list_well_log()


    def remove_all_well_log():
        session.query(WellLog).filter_by(well_id=get_well_id()).delete()
        session.commit()
        update_list_well_log()


    ui_wl.pushButton_add_well_log.clicked.connect(load_well_log)
    ui_wl.pushButton_add_dir_well_log.clicked.connect(load_well_log_by_dir)
    ui_wl.pushButton_rm_well_log.clicked.connect(remove_well_log)
    ui_wl.pushButton_rm_all_well_db.clicked.connect(remove_all_well_log)
    ui_wl.listWidget_well_log.currentItemChanged.connect(draw_well_log)

    update_list_well_log()

    WellLogForm.exec_()
