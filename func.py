from object import *


def set_info(text, color):
    ui.info.append(f'{datetime.datetime.now().strftime("%H:%M:%S")}:\t<span style =\"color:{color};\" >{text}</span>')


def get_object_id():
    try:
        return int(ui.comboBox_object.currentText().split('id')[-1])
    except ValueError:
        pass


def get_object_name():
    return ui.comboBox_object.currentText().split(' id')[0]


def get_profile_id():
    return int(ui.comboBox_profile.currentText().split('id')[-1])


def get_profile_name():
    return ui.comboBox_profile.currentText().split(' id')[0]


def query_to_list(query):
    """ результаты запроса в список """
    return sum(list(map(list, query)), [])


def draw_image(radar):
    hist.setImageItem(img)
    hist.setLevels(np.array(radar).min(), np.array(radar).max())
    colors = [
        (255, 0, 0),
        (0, 0, 0),
        (0, 0, 255)
    ]
    cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 3), color=colors)
    img.setColorMap(cmap)
    hist.gradient.setColorMap(cmap)
    img.setImage(np.array(radar))


def clear_current_profile():
    session.query(CurrentProfile).delete()
    session.commit()


def clear_current_profile_min_max():
    session.query(CurrentProfileMinMax).delete()
    session.commit()


def clear_window_profile():
    session.query(WindowProfile).delete()
    session.commit()


def vacuum():
    conn = connect(DATABASE_NAME)
    conn.execute("VACUUM")
    conn.close()


def changeSpinBox():
    roi.setSize([ui.spinBox_roi.value(), 512])


def clear_spectr():
    session.query(FFTSpectr).delete()
    session.commit()


def reset_spinbox_fft():
    ui.spinBox_ftt_up.setValue(0)
    ui.spinBox_fft_down.setValue(0)


def build_list(indexes_1, indexes_minus_1):
    # Создаем пустой список длины 512
    result = [0] * 512
    # Устанавливаем 1 в соответствующих индексах первого списка
    for i in indexes_1:
        result[i] = 1
    # Устанавливаем -1 в соответствующих индексах второго списка
    for i in indexes_minus_1:
        result[i] = -1
    return result


def updatePlot():
    rad = session.query(CurrentProfile.signal).first()
    radar = json.loads(rad[0])
    selected = roi.getArrayRegion(np.array(radar), img)
    n = ui.spinBox_roi.value()//2
    ui.signal.plot(y=range(0, 512), x=selected.mean(axis=0), clear=True, pen='r')
    ui.signal.plot(y=range(0, 512), x=selected[n])
    ui.signal.showGrid(x=True, y=True)
    ui.signal.invertY(True)


def save_max_min(radar):
    radar_max_min = []
    ui.progressBar.setMaximum(len(radar))
    for n, sig in enumerate(radar):
        max_points, _ = find_peaks(np.array(sig))
        min_points, _ = find_peaks(-np.array(sig))
        # diff_signal = np.diff(sig) возможно 2min/max
        # max_points = argrelmax(diff_signal)
        # min_points = argrelmin(diff_signal)
        signal_max_min = build_list(max_points, min_points)

        radar_max_min.append(signal_max_min)
        ui.progressBar.setValue(n + 1)
    clear_current_profile_min_max()
    new_current_min_max = CurrentProfileMinMax(profile_id=get_profile_id(), signal=json.dumps(radar_max_min))
    session.add(new_current_min_max)
    session.commit()


########################################
##############   layer   ###############
########################################


def clear_layers():
    for i in ui.widget_layer.findChildren(QtWidgets.QCheckBox):
        i.deleteLater()
        if f'scatter_id{i.text().split(" id")[-1]}' in globals():
            radarogramma.removeItem(globals()[f'scatter_id{i.text().split(" id")[-1]}'])
            del globals()[f'scatter_id{i.text().split(" id")[-1]}']

        if f'curve_id{i.text().split(" id")[-1]}' in globals():
            radarogramma.removeItem(globals()[f'curve_id{i.text().split(" id")[-1]}'])
            del globals()[f'curve_id{i.text().split(" id")[-1]}']
        if f'text_id{i.text().split(" id")[-1]}' in globals():
            radarogramma.removeItem(globals()[f'text_id{i.text().split(" id")[-1]}'])
            del globals()[f'text_id{i.text().split(" id")[-1]}']
    for i in ui.widget_layer_radio.findChildren(QtWidgets.QRadioButton):
        i.deleteLater()


def get_layer_id():
    for i in ui.widget_layer_radio.findChildren(QtWidgets.QRadioButton):
        if i.isChecked():
            return int(i.text())


def draw_layers():
    for i in ui.widget_layer.findChildren(QtWidgets.QCheckBox):
        if f'scatter_id{i.text().split(" id")[-1]}' in globals():
            globals()[f'scatter_id{i.text().split(" id")[-1]}'] = globals()[f'scatter_id{i.text().split(" id")[-1]}']
            radarogramma.removeItem(globals()[f'scatter_id{i.text().split(" id")[-1]}'])
        if f'curve_id{i.text().split(" id")[-1]}' in globals():
            globals()[f'curve_id{i.text().split(" id")[-1]}'] = globals()[f'curve_id{i.text().split(" id")[-1]}']
            radarogramma.removeItem(globals()[f'curve_id{i.text().split(" id")[-1]}'])
        if f'text_id{i.text().split(" id")[-1]}' in globals():
            globals()[f'text_id{i.text().split(" id")[-1]}'] = globals()[f'text_id{i.text().split(" id")[-1]}']
            radarogramma.removeItem(globals()[f'text_id{i.text().split(" id")[-1]}'])
        if i.isChecked():
            draw_layer(i.text().split(' id')[-1])



def draw_layer(layer_id):
    if f'curve_id{layer_id}' in globals():
        radarogramma.removeItem(globals()[f'curve_id{layer_id}'])
    if f'scatter_id{layer_id}' in globals():
        radarogramma.removeItem(globals()[f'scatter_id{layer_id}'])
    if f'text_id{layer_id}' in globals():
        radarogramma.removeItem(globals()[f'text_id{layer_id}'])
    layer_x = np.array(query_to_list(session.query(PointsOfLayer.point_x).filter(PointsOfLayer.layer_id == layer_id).order_by(PointsOfLayer.point_x).all()))
    layer_y = np.array(query_to_list(session.query(PointsOfLayer.point_y).filter(PointsOfLayer.layer_id == layer_id).order_by(PointsOfLayer.point_x).all()))

    scatter = pg.ScatterPlotItem(x=layer_x, y=layer_y, symbol='o', pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120), size=10)
    radarogramma.addItem(scatter)
    globals()[f'scatter_id{layer_id}'] = scatter
    if len(layer_x) > 2:
        # вычисляем кубический сплайн
        tck = splrep(layer_x, layer_y, k=2)
        # определяем новые точки для интерполяции
        x_new = np.linspace(layer_x.min(), layer_x.max(), int(layer_x.max() - layer_x.min()) + 2)
        # вычисляем значения интерполированной кривой
        y_new = splev(x_new, tck)
        curve = pg.PlotCurveItem(x=x_new, y=y_new, pen=pg.mkPen(color='white', width=2))
        radarogramma.addItem(curve)
        text_item = pg.TextItem(text=f'{layer_id}', color='white')
        text_item.setPos(layer_x.min() - int((layer_x.max() - layer_x.min())/50), int(layer_y.tolist()[layer_x.tolist().index(min(layer_x.tolist()))]))
        radarogramma.addItem(text_item)
        globals()[f'curve_id{layer_id}'] = curve
        globals()[f'text_id{layer_id}'] =text_item


def update_layers():
    clear_layers()
    layers = session.query(Layers).filter(Layers.profile_id == get_profile_id()).all()
    for n, lay in enumerate(layers):
        ui.checkBox_new = QtWidgets.QCheckBox()
        ui.radio_new = QtWidgets.QRadioButton()
        ui.checkBox_new.setText(f'{lay.layer_title} id{lay.id}')
        ui.radio_new.setText(f'{lay.id}')
        ui.verticalLayout_layer.addWidget(ui.checkBox_new)
        ui.verticalLayout_layer_radio.addWidget(ui.radio_new)
        ui.checkBox_new.clicked.connect(draw_layers)






