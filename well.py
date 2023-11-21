import numpy as np

from func import *
from monitoring import update_list_h_well
from qt.add_well_dialog import *
from qt.add_boundary_dialog import *
from qt.well_loader import *


def add_well():
    """Добавить новую скважину в БД"""
    Add_Well = QtWidgets.QDialog()
    ui_w = Ui_add_well()
    ui_w.setupUi(Add_Well)
    Add_Well.show()
    Add_Well.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    ui_w.lineEdit_well_alt.setText('0')
    ui_w.toolButton_del.hide()
    def well_to_db():
        name_well = ui_w.lineEdit_well_name.text()
        x_well = process_string(ui_w.lineEdit_well_x.text())
        y_well = process_string(ui_w.lineEdit_well_y.text())
        alt_well = process_string(ui_w.lineEdit_well_alt.text())
        if name_well != '' and x_well != '' and y_well != '' and alt_well != '':
            new_well = Well(name=name_well, x_coord=float(x_well), y_coord=float(y_well), alt=float(alt_well))
            session.add(new_well)
            session.commit()
            update_list_well()
            Add_Well.close()
            set_info(f'Добавлена новая скважина - "{name_well}".', 'green')

    def cancel_add_well():
        Add_Well.close()

    ui_w.buttonBox.accepted.connect(well_to_db)
    ui_w.buttonBox.rejected.connect(cancel_add_well)
    Add_Well.exec_()


def edit_well():
    """Изменить параметры скважины в БД"""
    Add_Well = QtWidgets.QDialog()
    ui_w = Ui_add_well()
    ui_w.setupUi(Add_Well)
    Add_Well.show()
    Add_Well.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    well = session.query(Well).filter(Well.id == get_well_id()).first()
    ui_w.lineEdit_well_x.setText(str(well.x_coord))
    ui_w.lineEdit_well_y.setText(str(well.y_coord))
    ui_w.lineEdit_well_alt.setText(str(well.alt))
    ui_w.lineEdit_well_name.setText(well.name)
    def well_update():
        name_well = ui_w.lineEdit_well_name.text()
        x_well = ui_w.lineEdit_well_x.text()
        y_well = ui_w.lineEdit_well_y.text()
        alt_well = ui_w.lineEdit_well_alt.text()
        if name_well != '' and x_well != '' and y_well != '' and alt_well != '':
            session.query(Well).filter(Well.id == get_well_id()).update(
                {'name': name_well, 'x_coord': float(x_well), 'y_coord': float(y_well), 'alt': float(alt_well)},
                synchronize_session="fetch")
            session.commit()
            update_list_well()
            Add_Well.close()
            set_info(f'Изменены параметры скважины - "{name_well}".', 'rgb(188, 160, 3)')

    def cancel_add_well():
        Add_Well.close()

    def well_delete():
        name_well = ui.listWidget_well.currentItem().text()
        session.query(Well).filter(Well.id == get_well_id()).delete()
        session.commit()
        update_list_well()
        Add_Well.close()
        set_info(f'Удалена скважина - "{name_well}".', 'rgb(188, 160, 3)')

    ui_w.buttonBox.accepted.connect(well_update)
    ui_w.buttonBox.rejected.connect(cancel_add_well)
    ui_w.toolButton_del.clicked.connect(well_delete)
    Add_Well.exec_()


def add_wells():
    """Пакетное добавление новых скважин в БД из файла Excel"""
    try:
        file_name = QFileDialog.getOpenFileName(caption='Выберите файл Excel или TXT (разделитель ";")', filter='*.xls *.xlsx *.txt')[0]
        set_info(file_name, 'blue')
        if file_name.lower().endswith('.txt'):
            try:
                pd_wells = pd.read_table(file_name, header=0, sep=';')
            except UnicodeDecodeError:
                pd_wells = pd.read_table(file_name, header=0, encoding='cp1251', sep=';')
        else:
            pd_wells = pd.read_excel(file_name, header=0)
    except FileNotFoundError:
        return

    pd_wells = clean_dataframe(pd_wells)

    WellLoader = QtWidgets.QDialog()
    ui_wl = Ui_WellLoader()
    ui_wl.setupUi(WellLoader)
    WellLoader.show()
    WellLoader.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    list_combobox = [ui_wl.comboBox_name, ui_wl.comboBox_x, ui_wl.comboBox_y, ui_wl.comboBox_alt,
                     ui_wl.comboBox_layers, ui_wl.comboBox_opt]
    for cmbx in list_combobox:
        for i in pd_wells.columns:
            cmbx.addItem(i)
    ui_wl.lineEdit_empty.setText('-999')

    def add_well_layer():
        list_layers = [] if ui_wl.lineEdit_layers.text() == '' else ui_wl.lineEdit_layers.text().split('/')
        if not ui_wl.comboBox_layers.currentText() in list_layers:
            list_layers.append(str(ui_wl.comboBox_layers.currentText()))
            ui_wl.lineEdit_layers.setText('/'.join(list_layers))

    def add_well_option():
        list_opt = [] if ui_wl.lineEdit_opt.text() == '' else ui_wl.lineEdit_opt.text().split('/')
        if not ui_wl.comboBox_opt.currentText() in list_opt:
            list_opt.append(str(ui_wl.comboBox_opt.currentText()))
            ui_wl.lineEdit_opt.setText('/'.join(list_opt))

    def load_wells():
        ui.progressBar.setMaximum(len(pd_wells.index))
        name_cell = ui_wl.comboBox_name.currentText()
        x_cell = ui_wl.comboBox_x.currentText()
        y_cell = ui_wl.comboBox_y.currentText()
        alt_cell = ui_wl.comboBox_alt.currentText()
        empty_value = '' if ui_wl.lineEdit_empty.text() == '' else int(ui_wl.lineEdit_empty.text())
        list_layers = [] if ui_wl.lineEdit_layers.text() == '' else ui_wl.lineEdit_layers.text().split('/')
        list_opt = [] if ui_wl.lineEdit_opt.text() == '' else ui_wl.lineEdit_opt.text().split('/')
        n_new, n_update = 0, 0
        for i in pd_wells.index:
            curr_well = session.query(Well).filter(Well.name == str(pd_wells[name_cell][i]),
                                          Well.x_coord == float(process_string(pd_wells[x_cell][i])),
                                          Well.y_coord == float(process_string(pd_wells[y_cell][i]))).first()
            if curr_well:
                n_update += 1
                set_info(f'Скважина {curr_well.name} уже есть в БД', 'red')
                alt = 0 if pd_wells[alt_cell][i] == '' else float(process_string(pd_wells[alt_cell][i]))
                session.query(Well).filter_by(id=curr_well.id).update(
                    {'alt': alt}, synchronize_session="fetch")
                for lr in list_layers:
                    try:
                        if pd_wells[lr][i] != empty_value:
                            bound = session.query(Boundary).filter(
                                Boundary.well_id == curr_well.id,
                                Boundary.title == str(lr)
                            ).first()
                            if not bound:
                                if ui_wl.checkBox_deep.isChecked():
                                    depth = round(float(process_string(pd_wells[lr][i])), 2)
                                else:
                                    depth = round(curr_well.alt - float(process_string(pd_wells[lr][i])), 2)
                                session.add(Boundary(
                                    well_id=curr_well.id,
                                    depth=depth,
                                    title=str(lr)
                                ))
                    except ValueError:
                        continue
                for opt in list_opt:
                    try:
                        if pd_wells[opt][i] != empty_value:
                            well_opt = session.query(WellOptionally).filter(
                                WellOptionally.well_id == curr_well.id,
                                WellOptionally.option == opt,
                                WellOptionally.value == str(pd_wells[opt][i])
                            ).first()
                            if not well_opt:
                                session.add(WellOptionally(
                                    well_id=curr_well.id,
                                    option=opt,
                                    value=str(pd_wells[opt][i])
                                ))
                    except ValueError:
                        continue
            else:
                n_new += 1
                new_well = Well(
                    name=str(pd_wells[name_cell][i]),
                    x_coord=float(process_string(pd_wells[x_cell][i])),
                    y_coord=float(process_string(pd_wells[y_cell][i])),
                    alt=round(float(process_string(pd_wells[alt_cell][i])), 2)
                )
                session.add(new_well)
                session.commit()
                for lr in list_layers:
                    try:
                        if pd_wells[lr][i] != empty_value:
                            if ui_wl.checkBox_deep.isChecked():
                                depth = round(float(process_string(pd_wells[lr][i])), 2)
                            else:
                                depth = round(new_well.alt - float(process_string(pd_wells[lr][i])), 2)
                            session.add(Boundary(
                                well_id=new_well.id,
                                depth=depth,
                                title=str(lr)
                            ))
                    except ValueError:
                        continue
                for opt in list_opt:
                    try:
                        if pd_wells[opt][i] != empty_value:
                            session.add(WellOptionally(
                                well_id=new_well.id,
                                option=opt,
                                value=str(pd_wells[opt][i])
                            ))
                    except ValueError:
                        continue
            session.commit()
            ui.progressBar.setValue(i + 1)
        session.commit()
        update_list_well()
        set_info(f'Добавлено {n_new} скважин, обновлено {n_update} скважин', 'green')


    def cancel_load():
        WellLoader.close()

    ui_wl.pushButton_add_layer.clicked.connect(add_well_layer)
    ui_wl.pushButton_add_opt.clicked.connect(add_well_option)
    ui_wl.buttonBox.accepted.connect(load_wells)
    ui_wl.buttonBox.rejected.connect(cancel_load)
    WellLoader.exec_()


def add_data_well():
    """ Добавить данные по скважинам """
    category, value = ui.lineEdit_string.text().split('; ')[0], ui.lineEdit_string.text().split('; ')[1]
    new_data_well = WellOptionally(
        well_id = get_well_id(),
        option = category,
        value = value
    )
    session.add(new_data_well)
    session.commit()
    show_data_well()




def show_data_well():
    ui.textEdit_datawell.clear()
    if ui.checkBox_profile_intersec.isChecked():
        inter = session.query(Intersection).filter_by(id=get_well_id()).first()
        if inter:
            therm = session.query(Thermogram).filter_by(id=inter.therm_id).first()
            target_date = session.query(Research).filter_by(id=get_research_id()).first().date_research
            target_datetime = datetime.datetime.combine(target_date, datetime.datetime.min.time())
            ui.textEdit_datawell.append(f'<p><b>Пересечение:</b> {inter.name}</p>'
                                        f'<p><b>X:</b> {round(inter.x_coord, 2)}</p>'
                                        f'<p><b>Y:</b> {round(inter.y_coord, 2)}</p>'
                                        f'<p><b>Дата термограммы:</b> {therm.date_time.strftime("%d.%m.%Y")} '
                                        f'({(therm.date_time - target_datetime).days} дней)</p>'
                                        f'<p><b>Температура:</b> {round(inter.temperature, 2)} °C</p>')
            for i in range(ui.comboBox_object_monitor.count()):
                if ui.comboBox_object_monitor.itemData(i)['id'] == get_object_id():
                    ui.comboBox_object_monitor.setCurrentIndex(i)
                    break
            update_list_h_well()
            for i in range(ui.listWidget_h_well.count()):
                if ui.listWidget_h_well.item(i).data(Qt.UserRole) == inter.thermogram.h_well_id:
                    ui.listWidget_h_well.setCurrentRow(i)
                    break
            for i in range(ui.listWidget_thermogram.count()):
                if ui.listWidget_thermogram.item(i).data(Qt.UserRole) == inter.therm_id:
                    ui.listWidget_thermogram.setCurrentRow(i)
                    break
            data_therm = [i for i in json.loads(inter.thermogram.therm_data) if len(i) > 2]
            int_therm = pg.InfiniteLine(pos=data_therm[inter.i_therm][0], angle=90, pen=pg.mkPen(color='#ff7800',width=3, dash=[8, 2]))
            ui.graph.addItem(int_therm)
            int_temp = pg.InfiniteLine(pos=inter.temperature, angle=0, pen=pg.mkPen(color='#ff7800',width=1, dash=[2, 2]))
            ui.graph.addItem(int_temp)
            text_item = pg.TextItem(text=get_profile_name().split(' (')[0], color='#ff7800')
            text_item.setFont(QtGui.QFont("Arial", 8))
            text_item.setPos(data_therm[inter.i_therm][0] + 5, data_therm[inter.i_therm][1] - 5)
            ui.graph.addItem(text_item)

    else:
        well = session.query(Well).filter_by(id=get_well_id()).first()
        if well:
            ui.textEdit_datawell.append(f'<p><b>Скважина №</b> {well.name}</p>'
                                f'<p><b>X:</b> {well.x_coord}</p>'
                                f'<p><b>Y:</b> {well.y_coord}</p>'
                                f'<p><b>Альтитуда:</b> {well.alt} м.</p>')
            for opt in session.query(WellOptionally).filter_by(well_id=well.id):
                ui.textEdit_datawell.append(f'<p><b>{opt.option}:</b> {opt.value}</p>')




def add_boundary():
    """Добавить новую границу для скважины в БД"""
    Add_Boundary = QtWidgets.QDialog()
    ui_b = Ui_add_bondary()
    ui_b.setupUi(Add_Boundary)
    Add_Boundary.show()
    Add_Boundary.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    last_boundary = session.query(Boundary).order_by(Boundary.id.desc()).first()
    ui_b.lineEdit_title.setText(last_boundary.title)
    ui_b.lineEdit_depth.setText(str(last_boundary.depth))

    def boundary_to_db():
        title_boundary = ui_b.lineEdit_title.text()
        depth_boundary = ui_b.lineEdit_depth.text()
        if title_boundary != '' and depth_boundary != '':
            new_boundary = Boundary(well_id=get_well_id(), title=title_boundary, depth=float(depth_boundary))
            session.add(new_boundary)
            session.commit()
            update_boundaries()
            update_list_well()
            Add_Boundary.close()
            set_info(f'Добавлена новая граница для текущей скважины - "{title_boundary}".', 'green')

    def cancel_add_boundary():
        Add_Boundary.close()

    ui_b.buttonBox.accepted.connect(boundary_to_db)
    ui_b.buttonBox.rejected.connect(cancel_add_boundary)
    Add_Boundary.exec_()


def remove_boundary():
    session.query(Boundary).filter(Boundary.id == get_boundary_id()).delete()
    session.commit()
    update_boundaries()
    update_list_well()


def draw_bound_int():
    for key, value in globals().items():
        if key.startswith('int_bound_') or key.startswith('bound_'):
            radarogramma.removeItem(globals()[key])

    if ui.listWidget_bound.currentItem():
        bound = session.query(Boundary).filter(Boundary.id == get_boundary_id()).first()
        x_prof = json.loads(session.query(Profile.x_pulc).filter(Profile.id == get_profile_id()).first()[0])
        y_prof = json.loads(session.query(Profile.y_pulc).filter(Profile.id == get_profile_id()).first()[0])
        index, dist = closest_point(bound.well.x_coord, bound.well.y_coord, x_prof, y_prof)
        # Получение значения средней скорости в среде
        Vmean = ui.doubleSpinBox_vsr.value()
        # Расчёт значения глубины, которое будет использоваться для отображения точки и текста на графике
        d = ((bound.depth * 100) / Vmean) / 8
        # Создание графического объекта точки с переданными параметрами
        scatter = pg.ScatterPlotItem(x=[index], y=[d], symbol='o', pen=pg.mkPen(None),
                                     brush=pg.mkBrush(color='#FFF500'), size=10)
        radarogramma.addItem(scatter)  # Добавление графического объекта точки на график
        globals()[f'bound_scatter'] = scatter  # Сохранение ссылки на графический объект точки в globals()

        # Создание графического объекта текста с переданными параметрами
        text_item = pg.TextItem(text=f'{bound.title} ({bound.depth})', color='white')
        text_item.setPos(index + 10, d)  # Установка позиции текста на графике
        radarogramma.addItem(text_item)  # Добавление графического объекта текста на график
        globals()[f'bound_text'] = text_item  # Сохранение ссылки на графический объект текста в globals()
        dmin = ((bound.depth * 100) / ui.doubleSpinBox_vmin.value()) / 8
        dmax = ((bound.depth * 100) / ui.doubleSpinBox_vmax.value()) / 8
        lmin = pg.InfiniteLine(pos=dmin, angle=0, pen=pg.mkPen(color='white', width=1, dash=[2, 2]))
        lmax = pg.InfiniteLine(pos=dmax, angle=0, pen=pg.mkPen(color='white', width=1, dash=[2, 2]))
        radarogramma.addItem(lmin)
        radarogramma.addItem(lmax)
        globals()[f'int_bound_min'] = lmin
        globals()[f'int_bound_max'] = lmax

        text_min = pg.TextItem(text=f'{bound.title} Vmin', color='white')
        text_min.setPos(0, dmin - 30)
        radarogramma.addItem(text_min)
        globals()[f'int_bound_text_min'] = text_min
        text_max = pg.TextItem(text=f'{bound.title} Vmax', color='white')
        text_max.setPos(0, dmax - 30)
        radarogramma.addItem(text_max)
        globals()[f'int_bound_text_max'] = text_max

        ui.doubleSpinBox_target_val.setValue(bound.depth)

