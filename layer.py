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
            if evt.button() == Qt.LeftButton:
                new_point = PointsOfLayer(layer_id=get_layer_id(), point_x=int(mousePoint.x()), point_y=int(mousePoint.y()))
                session.add(new_point)
            elif evt.button() == Qt.RightButton:
                session.query(PointsOfLayer).filter(
                    PointsOfLayer.layer_id == get_layer_id(),
                    PointsOfLayer.point_x <= int(mousePoint.x()) + 3,
                    PointsOfLayer.point_x >= int(mousePoint.x()) - 3,
                    PointsOfLayer.point_y <= int(mousePoint.y()) + 3,
                    PointsOfLayer.point_y >= int(mousePoint.y()) - 3
                ).delete()
            session.commit()
            draw_layer(get_layer_id())


def mouseLine(evt):
    if not ui.checkBox_draw_line.isChecked():
        return

    # Проверка на нажатие клавиши "Shift"
    if QApplication.keyboardModifiers() != Qt.ShiftModifier:
        return

    count_check = 0
    for i in ui.widget_layer.findChildren(QtWidgets.QCheckBox):
        if i.isChecked():
            count_check += 1
    if count_check > 1 or count_check == 0:
        QMessageBox.information(MainWindow, 'Внимание', 'Выберите один слой')


    pos = evt
    vb = radarogramma.vb

    # Проверка, находится ли курсор в пределах области графика
    if radarogramma.sceneBoundingRect().contains(pos):
        mousePoint = vb.mapSceneToView(pos)
        y = json.loads(session.query(Layers.layer_line).filter(Layers.id == get_layer_first_checkbox_id()).first()[0])
        try:
            y[int(mousePoint.x())] = int(mousePoint.y())
        except IndexError:
            return
        session.query(Layers).filter(Layers.id == get_layer_first_checkbox_id()).update({Layers.layer_line: json.dumps(y)}, synchronize_session="fetch")
        session.commit()
        draw_layer(get_layer_first_checkbox_id())



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
    last_form = session.query(Formation).order_by(Formation.id.desc()).first()
    ui_f.lineEdit.setText(last_form.title)

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
                x_pulc = json.loads(session.query(Profile.x_pulc).filter(Profile.id == get_profile_id()).first()[0])
                y_pulc = json.loads(session.query(Profile.y_pulc).filter(Profile.id == get_profile_id()).first()[0])
                # width_json = session.query(Formation.width).filter(Formation.profile_id == get_profile_id()).first()[0]
                # top_json = session.query(Formation.top).filter(Formation.profile_id == get_profile_id()).first()[0]
                # land_json = session.query(Formation.land).filter(Formation.profile_id == get_profile_id()).first()[0]
                # if width_json and top_json and land_json:
                #     width, top, land = json.loads(width_json), json.loads(top_json), json.loads(land_json)
                grid_db = session.query(Grid).filter(Grid.object_id == get_object_id()).first()
                if grid_db:
                    # считываем сетку грида из БД
                    pd_grid_uf = pd.DataFrame(json.loads(grid_db.grid_table_uf))
                    pd_grid_m = pd.DataFrame(json.loads(grid_db.grid_table_m))
                    pd_grid_r = pd.DataFrame(json.loads(grid_db.grid_table_r))
                T_top_l, T_bottom_l, dT_l, A_top_l, A_bottom_l, dA_l, A_sum_l, A_mean_l, dVt_l, Vt_top_l, Vt_sum_l, \
                Vt_mean_l, dAt_l, At_top_l, At_sum_l, At_mean_l, dPht_l, Pht_top_l, Pht_sum_l, Pht_mean_l, Wt_top_l, \
                Wt_mean_l, Wt_sum_l, std_l, k_var_l, skew_l, kurt_l, width_l, top_l, land_l, speed_l, speed_cover_l, \
                A_max_l, Vt_max_l, At_max_l, Pht_max_l, Wt_max_l, A_T_max_l, Vt_T_max_l, At_T_max_l, Pht_T_max_l, Wt_T_max_l, \
                A_Sn_l, Vt_Sn_l, At_Sn_l, Pht_Sn_l, Wt_Sn_l, A_wmf_l, Vt_wmf_l, At_wmf_l, Pht_wmf_l, Wt_wmf_l, A_Qf_l, \
                Vt_Qf_l, At_Qf_l, Pht_Qf_l, Wt_Qf_l, A_Sn_wmf_l, Vt_Sn_wmf_l, At_Sn_wmf_l, Pht_Sn_wmf_l, Wt_Sn_wmf_l, k_r_l= \
                [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], \
                [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], \
                [], [], [], [], [], [], []

                list_param_1 = []

                for i in range(len(layer_up)):
                    list_param_2 = []
                    signal = signals[i]
                    analytic_signal = hilbert(signal)
                    At = np.hypot(signal, np.imag(analytic_signal)).tolist()
                    Vt = np.imag(analytic_signal).tolist()
                    Pht = np.angle(analytic_signal).tolist()
                    Wt = np.diff(np.angle(analytic_signal)).tolist()
                    nt = layer_up[i]
                    nb = layer_down[i]
                    T_top_l.append(layer_up[i] * 8)
                    list_param_2.append(T_top_l[-1])
                    T_bottom_l.append(layer_down[i] * 8)
                    list_param_2.append(T_bottom_l[-1])
                    dT_l.append(layer_down[i] * 8 - layer_up[i] * 8)
                    list_param_2.append(dT_l[-1])
                    A_top_l.append(signal[nt])
                    list_param_2.append(A_top_l[-1])
                    A_bottom_l.append(signal[nb])
                    list_param_2.append(A_bottom_l[-1])
                    dA_l.append(signal[nb] - signal[nt])
                    list_param_2.append(dA_l[-1])
                    A_sum_l.append(float(np.sum(signal[nt:nb])))
                    list_param_2.append(A_sum_l[-1])
                    A_mean_l.append(float(np.mean(signal[nt:nb])))
                    list_param_2.append(A_mean_l[-1])
                    dVt_l.append(Vt[nb] - Vt[nt])
                    list_param_2.append(dVt_l[-1])
                    Vt_top_l.append(Vt[nt])
                    list_param_2.append(Vt_top_l[-1])
                    Vt_sum_l.append(float(np.sum(Vt[nt:nb])))
                    list_param_2.append(Vt_sum_l[-1])
                    Vt_mean_l.append(float(np.mean(Vt[nt:nb])))
                    list_param_2.append(Vt_mean_l[-1])
                    dAt_l.append(At[nb] - At[nt])
                    list_param_2.append(dAt_l[-1])
                    At_top_l.append(At[nt])
                    list_param_2.append(At_top_l[-1])
                    At_sum_l.append(float(np.sum(At[nt:nb])))
                    list_param_2.append(At_sum_l[-1])
                    At_mean_l.append(float(np.mean(At[nt:nb])))
                    list_param_2.append(At_mean_l[-1])
                    dPht_l.append(Pht[nb] - Pht[nt])
                    list_param_2.append(dPht_l[-1])
                    Pht_top_l.append(Pht[nt])
                    list_param_2.append(Pht_top_l[-1])
                    Pht_sum_l.append(float(np.sum(Pht[nt:nb])))
                    list_param_2.append(Pht_sum_l[-1])
                    Pht_mean_l.append(float(np.mean(Pht[nt:nb])))
                    list_param_2.append(Pht_mean_l[-1])
                    Wt_top_l.append(Wt[nt])
                    list_param_2.append(Wt_top_l[-1])
                    Wt_mean_l.append(float(np.mean(Wt[nt:nb])))
                    list_param_2.append(Wt_mean_l[-1])
                    Wt_sum_l.append(float(np.sum(Wt[nt:nb])))
                    list_param_2.append(Wt_sum_l[-1])
                    std_l.append(float(np.std(signal[nt:nb])))
                    list_param_2.append(std_l[-1])
                    k_var_l.append(float(np.var(signal[nt:nb])))
                    list_param_2.append(k_var_l[-1])
                    skew_l.append(skew(signal[nt:nb]))
                    list_param_2.append(skew_l[-1])
                    kurt_l.append(kurtosis(signal[nt:nb]))
                    list_param_2.append(kurt_l[-1])
                    A_max_l.append(max(signal[nt:nb]))
                    list_param_2.append(A_max_l[-1])
                    Vt_max_l.append(max(Vt[nt:nb]))
                    list_param_2.append(Vt_max_l[-1])
                    At_max_l.append(max(At[nt:nb]))
                    list_param_2.append(At_max_l[-1])
                    Pht_max_l.append(max(Pht[nt:nb]))
                    list_param_2.append(Pht_max_l[-1])
                    Wt_max_l.append(max(Wt[nt:nb]))
                    list_param_2.append(Wt_max_l[-1])
                    A_T_max_l.append((signal[nt:nb].index(max(signal[nt:nb])) + nt) * 8)
                    list_param_2.append(A_T_max_l[-1])
                    Vt_T_max_l.append((Vt[nt:nb].index(max(Vt[nt:nb])) + nt) * 8)
                    list_param_2.append(Vt_T_max_l[-1])
                    At_T_max_l.append((At[nt:nb].index(max(At[nt:nb])) + nt) * 8)
                    list_param_2.append(At_T_max_l[-1])
                    Pht_T_max_l.append((Pht[nt:nb].index(max(Pht[nt:nb])) + nt) * 8)
                    list_param_2.append(Pht_T_max_l[-1])
                    Wt_T_max_l.append((Wt[nt:nb].index(max(Wt[nt:nb])) + nt) * 8)
                    list_param_2.append(Wt_T_max_l[-1])
                    A_Sn, A_wmf, A_Qf, A_Sn_wmf = calc_fft_attributes(signal[nt:nb])
                    Vt_Sn, Vt_wmf, Vt_Qf, Vt_Sn_wmf = calc_fft_attributes(Vt[nt:nb])
                    At_Sn, At_wmf, At_Qf, At_Sn_wmf = calc_fft_attributes(At[nt:nb])
                    Pht_Sn, Pht_wmf, Pht_Qf, Pht_Sn_wmf = calc_fft_attributes(Pht[nt:nb])
                    Wt_Sn, Wt_wmf, Wt_Qf, Wt_Sn_wmf = calc_fft_attributes(Wt[nt:nb])
                    A_Sn_l.append(A_Sn)
                    list_param_2.append(A_Sn_l[-1])
                    Vt_Sn_l.append(Vt_Sn)
                    list_param_2.append(Vt_Sn_l[-1])
                    At_Sn_l.append(At_Sn)
                    list_param_2.append(At_Sn_l[-1])
                    Pht_Sn_l.append(Pht_Sn)
                    list_param_2.append(Pht_Sn_l[-1])
                    Wt_Sn_l.append(Wt_Sn)
                    list_param_2.append(Wt_Sn_l[-1])
                    A_wmf_l.append(At_wmf)
                    list_param_2.append(A_wmf_l[-1])
                    Vt_wmf_l.append(Vt_wmf)
                    list_param_2.append(Vt_wmf_l[-1])
                    At_wmf_l.append(At_wmf)
                    list_param_2.append(At_wmf_l[-1])
                    Pht_wmf_l.append(Pht_wmf)
                    list_param_2.append(Pht_wmf_l[-1])
                    Wt_wmf_l.append(Wt_wmf)
                    list_param_2.append(Wt_wmf_l[-1])
                    A_Qf_l.append(A_Qf)
                    list_param_2.append(A_Qf_l[-1])
                    Vt_Qf_l.append(Vt_Qf)
                    list_param_2.append(Vt_Qf_l[-1])
                    At_Qf_l.append(At_Qf)
                    list_param_2.append(At_Qf_l[-1])
                    Pht_Qf_l.append(Pht_Qf)
                    list_param_2.append(Pht_Qf_l[-1])
                    Wt_Qf_l.append(Wt_Qf)
                    list_param_2.append(Wt_Qf_l[-1])
                    A_Sn_wmf_l.append(A_Sn_wmf)
                    list_param_2.append(A_Sn_wmf_l[-1])
                    Vt_Sn_wmf_l.append(Vt_Sn_wmf)
                    list_param_2.append(Vt_Sn_wmf_l[-1])
                    At_Sn_wmf_l.append(At_Sn_wmf)
                    list_param_2.append(At_Sn_wmf_l[-1])
                    Pht_Sn_wmf_l.append(Pht_Sn_wmf)
                    list_param_2.append(Pht_Sn_wmf_l[-1])
                    Wt_Sn_wmf_l.append(Wt_Sn_wmf)
                    list_param_2.append(Wt_Sn_wmf_l[-1])

                    if grid_db:
                        pd_grid_uf['dist_y'] = abs(pd_grid_uf[1] - y_pulc[i])
                        pd_grid_uf['dist_x'] = abs(pd_grid_uf[0] - x_pulc[i])
                        pd_grid_m['dist_y'] = abs(pd_grid_m[1] - y_pulc[i])
                        pd_grid_m['dist_x'] = abs(pd_grid_m[0] - x_pulc[i])
                        pd_grid_r['dist_y'] = abs(pd_grid_r[1] - y_pulc[i])
                        pd_grid_r['dist_x'] = abs(pd_grid_r[0] - x_pulc[i])
                        i_uf = pd_grid_uf.loc[pd_grid_uf['dist_y'] == pd_grid_uf['dist_y'].min()].loc[
                            pd_grid_uf['dist_x'] == pd_grid_uf['dist_x'].min()].iat[0, 2]
                        i_m = pd_grid_m.loc[pd_grid_m['dist_y'] == pd_grid_m['dist_y'].min()].loc[
                            pd_grid_m['dist_x'] == pd_grid_m['dist_x'].min()].iat[0, 2]
                        i_r = pd_grid_r.loc[pd_grid_r['dist_y'] == pd_grid_r['dist_y'].min()].loc[
                            pd_grid_r['dist_x'] == pd_grid_r['dist_x'].min()].iat[0, 2]
                        im = i_m if i_m > 0 else 0
                        width_l.append(im)
                        list_param_2.append(width_l[-1])
                        top_l.append(i_uf)
                        list_param_2.append(top_l[-1])
                        land_l.append(i_r)
                        list_param_2.append(land_l[-1])
                        speed_l.append(im * 100 / (layer_down[i] * 8 - layer_up[i] * 8))
                        list_param_2.append(speed_l[-1])
                        speed_cover_l.append((i_r - i_uf) * 100 / (layer_up[i] * 8))
                        list_param_2.append(speed_cover_l[-1])
                    if len(list_param_1) == len(list_param_2):
                        k_r_l.append(np.corrcoef(list_param_1, list_param_2)[0, 1])
                    list_param_1 = list_param_2.copy()
                    ui.progressBar.setValue(i + 1)
                k_r_l.append(k_r_l[-1])
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
                               'kurt': json.dumps(kurt_l),
                               'A_max': json.dumps(A_max_l),
                               'Vt_max': json.dumps(Vt_max_l),
                               'At_max': json.dumps(At_max_l),
                               'Pht_max': json.dumps(Pht_max_l),
                               'Wt_max': json.dumps(Wt_max_l),
                               'A_T_max': json.dumps(A_T_max_l),
                               'Vt_T_max': json.dumps(Vt_T_max_l),
                               'At_T_max': json.dumps(At_T_max_l),
                               'Pht_T_max': json.dumps(Pht_T_max_l),
                               'Wt_T_max': json.dumps(Wt_T_max_l),
                               'A_Sn': json.dumps(A_Sn_l),
                               'Vt_Sn': json.dumps(Vt_Sn_l),
                               'At_Sn': json.dumps(At_Sn_l),
                               'Pht_Sn': json.dumps(Pht_Sn_l),
                               'Wt_Sn': json.dumps(Wt_Sn_l),
                               'A_wmf': json.dumps(A_wmf_l),
                               'Vt_wmf': json.dumps(Vt_wmf_l),
                               'At_wmf': json.dumps(At_wmf_l),
                               'Pht_wmf': json.dumps(Pht_wmf_l),
                               'Wt_wmf': json.dumps(Wt_wmf_l),
                               'A_Qf': json.dumps(A_Qf_l),
                               'Vt_Qf': json.dumps(Vt_Qf_l),
                               'At_Qf': json.dumps(At_Qf_l),
                               'Pht_Qf': json.dumps(Pht_Qf_l),
                               'Wt_Qf': json.dumps(Wt_Qf_l),
                               'A_Sn_wmf': json.dumps(A_Sn_wmf_l),
                               'Vt_Sn_wmf': json.dumps(Vt_Sn_wmf_l),
                               'At_Sn_wmf': json.dumps(At_Sn_wmf_l),
                               'Pht_Sn_wmf': json.dumps(Pht_Sn_wmf_l),
                               'Wt_Sn_wmf': json.dumps(Wt_Sn_wmf_l),
                               'k_r': json.dumps(k_r_l)}

                if grid_db:
                    dict_signal['width'] = json.dumps(width_l)
                    dict_signal['top'] = json.dumps(top_l)
                    dict_signal['land'] = json.dumps(land_l)
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
    session.query(FormationAI).filter_by(formation_id=get_formation_id()).delete()
    session.query(Formation).filter(Formation.id == get_formation_id()).delete()
    session.commit()
    set_info(f'Пласт {ui.comboBox_plast.currentText()} удалён из БД', 'green')
    update_formation_combobox()


def filter_layer():
    try:
        l_id = get_layer_id()
        layer = session.query(Layers).filter(Layers.id == l_id).first()
        filter_layer = list(map(int, savgol_filter(json.loads(layer.layer_line), 31, 3)))
        session.query(Layers).filter(Layers.id == l_id).update({'layer_line': json.dumps(filter_layer)}, synchronize_session="fetch")
        session.commit()
        draw_layer(l_id)
    except AttributeError:
        QMessageBox.critical(MainWindow, 'Ошибка', 'Выберите слой для фильтрации.')


def remove_well():
    w_id = get_well_id()
    if ui.checkBox_profile_intersec.isChecked():
        session.query(Intersection).filter_by(id=w_id).delete()
        set_info('Пересечение удалено', 'green')
    else:
        session.query(Boundary).filter_by(well_id=w_id).delete()
        session.query(MarkupLDA).filter_by(well_id=w_id).delete()
        session.query(MarkupMLP).filter_by(well_id=w_id).delete()
        session.query(Well).filter_by(id=w_id).delete()
        set_info('Скважина удалена', 'green')
    session.commit()
    update_list_well()
