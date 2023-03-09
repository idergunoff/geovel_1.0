from func import *
from qt.add_layer_dialog import *


def add_layer():
    """Добавить ноывый слой в БД"""
    Add_Layer = QtWidgets.QDialog()
    ui_lay = Ui_add_layer()
    ui_lay.setupUi(Add_Layer)
    Add_Layer.show()
    Add_Layer.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    def layer_to_db():
        title_layer = ui_lay.lineEdit.text()
        if title_layer != '':
            new_layer = Layers(layer_title=title_layer, profile_id=get_profile_id())
            session.add(new_layer)
            session.commit()
            update_layers()
            Add_Layer.close()
            set_info(f'Добавлен новый слой - "{title_layer}" на профиле - "{get_profile_name()}".', 'green')

    def cancel_add_layer():
        Add_Layer.close()

    ui_lay.buttonBox.accepted.connect(layer_to_db)
    ui_lay.buttonBox.rejected.connect(cancel_add_layer)
    Add_Layer.exec_()


def remove_layer():
    for i in ui.widget_layer.findChildren(QtWidgets.QCheckBox):
        if i.isChecked():
            session.query(PointsOfLayer).filter(PointsOfLayer.layer_id == i.text().split(' id')[-1]).delete()
            session.query(Layers).filter(Layers.id == i.text().split(' id')[-1]).delete()
    session.commit()
    update_layers()


def mouseClicked(evt):
    """ Добавляем новую точку в слой """
    if not ui.checkBox_draw_layer.isChecked():
        return
    if not get_layer_id():
        return
    pos = evt.pos()
    vb = radarogramma.vb
    scene_pos = vb.mapToScene(pos)
    if radarogramma.sceneBoundingRect().contains(scene_pos):
        mousePoint = vb.mapSceneToView(scene_pos)
        count_sig = session.query(Measure).filter(Measure.profile_id == get_profile_id()).count()
        if 0 <= mousePoint.x() <= count_sig and 0 <= mousePoint.y() <= 513:
            new_point = PointsOfLayer(layer_id=get_layer_id(), point_x=int(mousePoint.x()), point_y=int(mousePoint.y()))
            session.add(new_point)
            session.commit()
            draw_layer(new_point.layer_id)


def add_edges():
    l_id = get_layer_id()
    layer_x = query_to_list(session.query(PointsOfLayer.point_x).filter(PointsOfLayer.layer_id == l_id).order_by(PointsOfLayer.point_x).all())
    layer_y = query_to_list(session.query(PointsOfLayer.point_y).filter(PointsOfLayer.layer_id == l_id).order_by(PointsOfLayer.point_x).all())
    count_sig = session.query(Measure).filter(Measure.profile_id == get_profile_id()).count()
    new_point = PointsOfLayer(layer_id=get_layer_id(), point_x=0, point_y=int(layer_y[layer_x.index(min(layer_x))]))
    session.add(new_point)
    new_point = PointsOfLayer(layer_id=get_layer_id(), point_x=count_sig, point_y=int(layer_y[layer_x.index(max(layer_x))]))
    session.add(new_point)
    session.commit()
    draw_layer(new_point.layer_id)






