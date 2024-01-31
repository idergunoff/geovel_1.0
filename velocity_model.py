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


def remove_bind_point():
    for key, value in globals().items():
        if key == 'bind_scatter_id{}'.format(ui.listWidget_bind.currentItem().text().split(' id')[-1]):
            radarogramma.removeItem(globals()[key])

def draw_bind_point():
    for key, value in globals().items():
        if key.startswith('bind_'):
            radarogramma.removeItem(globals()[key])

    bind = session.query(Binding).filter_by(id=ui.listWidget_bind.currentItem().text().split(' id')[-1]).first()
    layer = session.query(Layers).filter_by(id=bind.layer_id).first()

    layer_line = json.loads(layer.layer_line)
    x_coord = bind.index_measure
    y_coord = layer_line[bind.index_measure]

    scatter = pg.ScatterPlotItem(x=[x_coord], y=[y_coord], pen=pg.mkPen('r', width=3), brush=pg.mkBrush('r'),
                                 symbol='o', size=15, hoverable=True)
    radarogramma.addItem(scatter)
    globals()[f'bind_scatter_id{bind.id}'] = scatter


def remove_binding():
    session.query(Binding).filter(Binding.id == int(ui.listWidget_bind.currentItem().text().split(' id')[-1])).delete()
    session.commit()
    update_list_binding()


def calc_velocity_model():
    list_layer_id = []
    for i in ui.widget_layer.findChildren(QtWidgets.QCheckBox):
        if i.isChecked():
            list_layer_id.append(int(i.text().split(' id')[-1]))

    if not list_layer_id:
        set_info('Должен быть выбран хотя бы один слой', 'red')
        QMessageBox.critical(MainWindow, 'Ошибка', 'Должен быть выбран хотя бы один слой')
        return

    list_layer_line = []
    for i in list_layer_id:
        list_layer_line.append(json.loads(session.query(Layers).filter_by(id=i).first().layer_line))

    print(list_layer_line)

