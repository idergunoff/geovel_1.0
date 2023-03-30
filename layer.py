from func import *
from qt.add_layer_dialog import *
from qt.add_formation_dialog import *


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
            i_id = i.text().split(' id')[-1]
            if session.query(Formation).filter(or_(Formation.up == i_id, Formation.down == i_id)).count() == 0:
                session.query(PointsOfLayer).filter(PointsOfLayer.layer_id == i.text().split(' id')[-1]).delete()
                session.query(Layers).filter(Layers.id == i.text().split(' id')[-1]).delete()
            else:
                set_info(f'Невозможно удалить слой "{i.text()}", сначала удалите все пласты в которых используется данный слой', 'red')
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
        count_sig = len(json.loads(session.query(Profile.signal).filter(Profile.id == get_profile_id()).first()[0]))
        if 0 <= mousePoint.x() <= count_sig and 0 <= mousePoint.y() <= 513:
            new_point = PointsOfLayer(layer_id=get_layer_id(), point_x=int(mousePoint.x()), point_y=int(mousePoint.y()))
            session.add(new_point)
            session.commit()
            draw_layer(new_point.layer_id)


def save_layer():
    l_id = get_layer_id()
    layer_x = query_to_list(session.query(PointsOfLayer.point_x).filter(PointsOfLayer.layer_id == l_id).order_by(PointsOfLayer.point_x).all())
    layer_y = query_to_list(session.query(PointsOfLayer.point_y).filter(PointsOfLayer.layer_id == l_id).order_by(PointsOfLayer.point_x).all())
    count_sig = len(json.loads(session.query(Profile.signal).filter(Profile.id == get_profile_id()).first()[0]))
    if session.query(PointsOfLayer).filter(PointsOfLayer.layer_id == get_layer_id(), PointsOfLayer.point_x == 0).count() == 0:
        new_point = PointsOfLayer(layer_id=get_layer_id(), point_x=0, point_y=int(layer_y[layer_x.index(min(layer_x))]))
        session.add(new_point)
    if session.query(PointsOfLayer).filter(PointsOfLayer.layer_id == get_layer_id(), PointsOfLayer.point_x == count_sig).count() == 0:
        new_point = PointsOfLayer(layer_id=get_layer_id(), point_x=count_sig, point_y=int(layer_y[layer_x.index(max(layer_x))]))
        session.add(new_point)
    session.commit()
    draw_layer(l_id, True)


def crop_up():
    l_id, crop_line, y_new = crop()
    for i in range(len(y_new)):
        if y_new[i] <= crop_line[i]:
            y_new[i] = crop_line[i] + 1
    session.query(Layers).filter(Layers.id == l_id).update({'layer_line': json.dumps(y_new)}, synchronize_session="fetch")
    session.commit()
    draw_layers()


def crop_down():
    l_id, crop_line, y_new = crop()
    for i in range(len(y_new)):
        if y_new[i] >= crop_line[i]:
            y_new[i] = crop_line[i] - 1
    session.query(Layers).filter(Layers.id == l_id).update({'layer_line': json.dumps(y_new)}, synchronize_session="fetch")
    session.commit()
    draw_layers()


def add_formation():
    """Добавить ноывый пласт в БД"""
    Add_Form = QtWidgets.QDialog()
    ui_f = Ui_add_formation()
    ui_f.setupUi(Add_Form)
    Add_Form.show()
    Add_Form.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    def formation_to_db():
        title_form = ui_f.lineEdit.text()
        if title_form != '':
            list_id = get_layers_for_formation()
            if not list_id:
                set_info('Не выбраны границы пласта', 'red')
                return
            layer_0 = session.query(Layers.layer_line).filter(Layers.id == list_id[0]).first()[0]
            layer_1 = session.query(Layers.layer_line).filter(Layers.id == list_id[1]).first()[0]
            if not layer_0 or not layer_1:
                set_info('Одна из границ не сохранена', 'red')
                return
            if sum(json.loads(layer_0)) > sum(json.loads(layer_1)):
                up, down = list_id[1], list_id[0]
            else:
                up, down = list_id[0], list_id[1]
            new_formation = Formation(title=title_form, profile_id=get_profile_id(), up=up, down=down)
            session.add(new_formation)
            session.commit()

            signals = json.loads(session.query(Profile.signal).filter(Profile.id == get_profile_id()).first()[0])
            layer_up, layer_down = json.loads(layer_0), json.loads(layer_1)
            if len(signals) != len(layer_up) != len(layer_down):
                set_info('ВНИМАНИЕ! ОШИБКА!!! Не совпадает количество измерений в радарпограмме и в границах кровли/подошвы', 'red')
            else:
                ui.progressBar.setMaximum(len(layer_up))
                width_json = session.query(Formation.width).filter(Formation.profile_id == get_profile_id()).first()[0]
                top_json = session.query(Formation.top).filter(Formation.profile_id == get_profile_id()).first()[0]
                land_json = session.query(Formation.land).filter(Formation.profile_id == get_profile_id()).first()[0]
                if width_json and top_json and land_json:
                    width, top, land = json.loads(width_json), json.loads(top_json), json.loads(land_json)
                T_top_l, T_bottom_l, dT_l, A_top_l, A_bottom_l, dA_l, A_sum_l, A_mean_l, dVt_l, Vt_top_l, Vt_sum_l, Vt_mean_l, dAt_l, At_top_l, \
                At_sum_l, At_mean_l, dPht_l, Pht_top_l, Pht_sum_l, Pht_mean_l, Wt_top_l, Wt_mean_l, Wt_sum_l, std_l, k_var_l, skew_l, kurt_l, speed_l, speed_cover_l = \
                    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
                for i in range(len(layer_up)):
                    signal = signals[i]
                    analytic_signal = hilbert(signal)
                    At = np.hypot(signal, np.imag(analytic_signal)).tolist()
                    Vt = np.imag(analytic_signal).tolist()
                    Pht = np.angle(analytic_signal).tolist()
                    Wt = np.diff(Pht).tolist()
                    nt = layer_up[i]
                    nb = layer_down[i]
                    T_top_l.append(layer_up[i] * 8)
                    T_bottom_l.append(layer_down[i] * 8)
                    dT_l.append(layer_down[i] * 8 - layer_up[i] * 8)
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
                    if width_json and top_json and land_json:
                        speed_l.append(width[i] * 100 / (layer_down[i] * 8 - layer_up[i] * 8))
                        speed_cover_l.append((land[i] - top[i]) * 100 / (layer_up[i] * 8))
                    ui.progressBar.setValue(i + 1)
                dict_signal = {'T_top': json.dumps(T_top_l),
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
                if width_json and top_json and land_json:
                    dict_signal['speed'] = json.dumps(speed_l)
                    dict_signal['speed_cover'] = json.dumps(speed_cover_l)
                session.query(Formation).filter(Formation.id == new_formation.id).update(dict_signal,
                                                                                     synchronize_session="fetch")
                session.commit()
            update_formation_combobox()

            Add_Form.close()
            set_info(f'Добавлен новый пласт - "{title_form}" на профиле - "{get_profile_name()}".', 'green')

    def cancel_add_formation():
        Add_Form.close()

    ui_f.buttonBox.accepted.connect(formation_to_db)
    ui_f.buttonBox.rejected.connect(cancel_add_formation)
    Add_Form.exec_()


def remove_formation():
    session.query(Formation).filter(Formation.id == get_formation_id()).delete()
    session.commit()
    set_info(f'Пласт {ui.comboBox_plast.currentText()} удалён из БД', 'green')
    update_formation_combobox()