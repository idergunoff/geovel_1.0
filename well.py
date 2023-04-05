from func import *
from qt.add_well_dialog import *
from qt.add_boundary_dialog import *


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
        file_name = QFileDialog.getOpenFileName(caption='Выберите файл Excel', filter='*.xls *.xlsx')[0]
        set_info(file_name, 'blue')
        pd_wells = pd.read_excel(file_name, header=0)
    except FileNotFoundError:
        return
    ui.progressBar.setMaximum(len(pd_wells.index))

    for i in pd_wells.index:
        if session.query(Well).filter(Well.name == str(pd_wells['name'][i]), Well.x_coord == float(process_string(pd_wells['x'][i])),
                                      Well.y_coord == float(process_string(pd_wells['y'][i]))).count() > 0:
            set_info(f'Скважина {pd_wells["name"][i]} уже есть в БД', 'red')
        else:
            new_well = Well(name=str(pd_wells['name'][i]), x_coord=float(process_string(pd_wells['x'][i])), y_coord=float(process_string(pd_wells['y'][i])), alt=float(process_string(pd_wells['alt'][i])))
            session.add(new_well)
        ui.progressBar.setValue(i + 1)
    session.commit()
    update_list_well()


def show_data_well():
    ui.textEdit_datawell.clear()
    well = session.query(Well).filter_by(id=get_well_id()).first()
    if well:
        ui.textEdit_datawell.append(f'<p><b>Скважина №</b> {well.name}</p>'
                                f'<p><b>X:</b> {well.x_coord}</p>'
                                f'<p><b>Y:</b> {well.y_coord}</p>'
                                f'<p><b>Альтитуда:</b> {well.alt} м.</p>')


def add_boundary():
    """Добавить новую границу для скважины в БД"""
    Add_Boundary = QtWidgets.QDialog()
    ui_b = Ui_add_bondary()
    ui_b.setupUi(Add_Boundary)
    Add_Boundary.show()
    Add_Boundary.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

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
        if key.startswith('int_bound_'):
            radarogramma.removeItem(globals()[key])
    if ui.listWidget_bound.currentItem():
        bound = session.query(Boundary).filter(Boundary.id == get_boundary_id()).first()
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

