from func import *


def update_list_binding():
    ui.listWidget_bind.clear()
    for i in session.query(Binding).filter_by(profile_id=get_profile_id()).all():
        ui.listWidget_bind.addItem(f'{i.layer.layer_title} - {i.boundary.title} ({i.boundary.depth} м.) id{i.id}')


def add_binding():
    profile = session.query(Profile).filter_by(id=get_profile_id()).first()
    layer_id = 0
    for i in ui.widget_layer.findChildren(QtWidgets.QCheckBox):
        if i.isChecked():
            layer_id = int(i.text().split(' id')[-1])
            break
    if layer_id == 0:
        set_info('Выберите слой к которому нужно привязать границу', 'red')
        QMessageBox.critical(MainWindow, 'Ошибка', 'Выберите слой к которому нужно привязать границу')
        return
    if not get_boundary_id():
        set_info('Выберите границу', 'red')
        QMessageBox.critical(MainWindow, 'Ошибка', 'Выберите границу')
        return
    w = session.query(Well).filter_by(id=get_well_id()).first()

    index, dist = closest_point(w.x_coord, w.y_coord, json.loads(profile.x_pulc), json.loads(profile.y_pulc))

    new_bind = Binding(profile_id=profile.id, layer_id=layer_id, boundary_id=get_boundary_id(), index_measure=index)
    session.add(new_bind)
    session.commit()
    update_list_binding()