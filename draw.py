from func import *


def draw_radarogram():
    global l_up, l_down
    if 'curve_up' in globals():
        radarogramma.removeItem(globals()['curve_up'])
    if 'curve_down' in globals():
        radarogramma.removeItem(globals()['curve_down'])
    if 'text_item' in globals():
        radarogramma.removeItem(globals()['text_item'])
    if 'poly_item' in globals():
        radarogramma.removeItem(globals()['poly_item'])
    remove_poly_item()
    remove_curve_fake()
    ui.info.clear()
    clear_current_profile()
    rad = json.loads(session.query(Profile.signal).filter(Profile.id == get_profile_id()).first()[0])
    ui.progressBar.setMaximum(len(rad))
    radar = []
    for n, i in enumerate(rad):
        ui.progressBar.setValue(n + 1)
        if ui.comboBox_atrib.currentText() == 'A':
            radar.append(i)
        elif ui.comboBox_atrib.currentText() == 'diff':
            radar.append(np.diff(i).tolist())
        elif ui.comboBox_atrib.currentText() == 'At':
            analytic_signal = hilbert(i)
            radar.append(list(map(lambda x: round(x, 2), np.hypot(i, np.imag(analytic_signal)))))
        elif ui.comboBox_atrib.currentText() == 'Vt':
            analytic_signal = hilbert(i)
            radar.append(list(map(lambda x: round(x, 2), np.imag(analytic_signal))))
        elif ui.comboBox_atrib.currentText() == 'Pht':
            analytic_signal = hilbert(i)
            radar.append(list(map(lambda x: round(x, 2), np.angle(analytic_signal))))
        elif ui.comboBox_atrib.currentText() == 'Wt':
            analytic_signal = hilbert(i)
            radar.append(list(map(lambda x: round(x, 2), np.diff(np.angle(analytic_signal)))))
        else:
            radar.append(i)
    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(radar))
    session.add(new_current)
    session.commit()
    save_max_min(radar)
    ui.checkBox_minmax.setCheckState(0)
    draw_image(radar)
    set_info(f'Отрисовка "{ui.comboBox_atrib.currentText()}" профиля ({get_object_name()}, {get_profile_name()})', 'blue')
    updatePlot()
    line_up = ui.spinBox_rad_up.value()
    line_down = ui.spinBox_rad_down.value()
    l_up = pg.InfiniteLine(pos=line_up, angle=90, pen=pg.mkPen(color='darkred',width=1, dash=[8, 2]))
    l_down = pg.InfiniteLine(pos=line_down, angle=90, pen=pg.mkPen(color='darkgreen', width=1, dash=[8, 2]))
    radarogramma.addItem(l_up)
    radarogramma.addItem(l_down)
    update_layers()
    draw_layers()
    update_formation_combobox()


def draw_current_radarogram():
    global l_up, l_down
    if 'curve_up' in globals():
        radarogramma.removeItem(globals()['curve_up'])
    if 'curve_down' in globals():
        radarogramma.removeItem(globals()['curve_down'])
    if 'text_item' in globals():
        radarogramma.removeItem(globals()['text_item'])
    if 'poly_item' in globals():
        radarogramma.removeItem(globals()['poly_item'])
    remove_poly_item()
    remove_curve_fake()
    rad = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])
    ui.progressBar.setMaximum(len(rad))
    radar = []
    for n, i in enumerate(rad):
        ui.progressBar.setValue(n + 1)
        if ui.comboBox_atrib.currentText() == 'A':
            radar.append(i)
        elif ui.comboBox_atrib.currentText() == 'diff':
            radar.append(np.diff(i).tolist())
        elif ui.comboBox_atrib.currentText() == 'At':
            analytic_signal = hilbert(i)
            radar.append(list(map(lambda x: round(x, 2), np.hypot(i, np.imag(analytic_signal)))))
        elif ui.comboBox_atrib.currentText() == 'Vt':
            analytic_signal = hilbert(i)
            radar.append(list(map(lambda x: round(x, 2), np.imag(analytic_signal))))
        elif ui.comboBox_atrib.currentText() == 'Pht':
            analytic_signal = hilbert(i)
            radar.append(list(map(lambda x: round(x, 2), np.angle(analytic_signal))))
        elif ui.comboBox_atrib.currentText() == 'Wt':
            analytic_signal = hilbert(i)
            radar.append(list(map(lambda x: round(x, 2), np.diff(np.angle(analytic_signal)))))
        else:
            radar.append(i)
    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(radar))
    session.add(new_current)
    session.commit()
    save_max_min(radar)
    if ui.checkBox_minmax.isChecked():
        radar = json.loads(session.query(CurrentProfileMinMax.signal).filter(CurrentProfileMinMax.id == 1).first()[0])
    draw_image(radar)
    set_info(f'Отрисовка "{ui.comboBox_atrib.currentText()}" для текущего профиля', 'blue')
    updatePlot()
    line_up = ui.spinBox_rad_up.value()
    line_down = ui.spinBox_rad_down.value()
    l_up = pg.InfiniteLine(pos=line_up, angle=90, pen=pg.mkPen(color='darkred',width=1, dash=[8, 2]))
    l_down = pg.InfiniteLine(pos=line_down, angle=90, pen=pg.mkPen(color='darkgreen', width=1, dash=[8, 2]))
    radarogramma.addItem(l_up)
    radarogramma.addItem(l_down)
    update_formation_combobox()


def draw_max_min():
    rad = session.query(CurrentProfile.signal).first()
    radar = json.loads(rad[0])
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
    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(radar_max_min))
    session.add(new_current)
    session.commit()

    draw_image(radar_max_min)
    updatePlot()
    set_info(f'Отрисовка "max/min" для текущего профиля', 'blue')


def draw_rad_line():
    global l_up, l_down
    radarogramma.removeItem(l_up)
    radarogramma.removeItem(l_down)
    line_up = ui.spinBox_rad_up.value()
    line_down = ui.spinBox_rad_down.value()
    l_up = pg.InfiniteLine(pos=line_up, angle=90, pen=pg.mkPen(color='darkred',width=1, dash=[8, 2]))
    l_down = pg.InfiniteLine(pos=line_down, angle=90, pen=pg.mkPen(color='darkgreen', width=1, dash=[8, 2]))
    radarogramma.addItem(l_up)
    radarogramma.addItem(l_down)


def choose_minmax():
    remove_poly_item()
    remove_curve_fake()
    if ui.checkBox_minmax.isChecked():
        radar = json.loads(session.query(CurrentProfileMinMax.signal).filter(CurrentProfileMinMax.id == 1).first()[0])
    else:
        radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])
    draw_image(radar)
    updatePlot()


def draw_formation():
    remove_poly_item()
    remove_curve_fake()
    if 'curve_up' in globals():
        radarogramma.removeItem(globals()['curve_up'])
    if 'curve_down' in globals():
        radarogramma.removeItem(globals()['curve_down'])
    if 'text_item' in globals():
        radarogramma.removeItem(globals()['text_item'])
    if ui.comboBox_plast.currentText() == '-----':
        return
    # elif ui.comboBox_plast.currentText() == 'KROT':
    #     t_top = json.loads(session.query(Profile.T_top).filter(Profile.id == get_profile_id()).first()[0])
    #     t_bot = json.loads(session.query(Profile.T_bottom).filter(Profile.id == get_profile_id()).first()[0])
    #     layer_up = [x / 8 for x in t_top]
    #     layer_down = [x / 8 for x in t_bot]
    #     title_text = 'KROT'
    else:
        formation = session.query(Formation).filter(Formation.id == get_formation_id()).first()
        layer_up = json.loads(session.query(Layers.layer_line).filter(Layers.id == formation.up).first()[0])
        layer_down = json.loads(session.query(Layers.layer_line).filter(Layers.id == formation.down).first()[0])
        title_text = formation.title
    x = list(range(len(layer_up)))
    # Создаем объект линии и добавляем его на радарограмму
    curve_up = pg.PlotCurveItem(x=x, y=layer_up, pen=pg.mkPen(color='white', width=2))
    curve_down = pg.PlotCurveItem(x=x, y=layer_down, pen=pg.mkPen(color='white', width=2))
    radarogramma.addItem(curve_up)
    radarogramma.addItem(curve_down)
    # Создаем объект текста для отображения id слоя и добавляем его на радарограмму
    text_item = pg.TextItem(text=f'{title_text}', color='white')
    text_item.setPos(min(x) - int(max(x) - min(x)) / 50, int(layer_down[x.index(min(x))]))
    radarogramma.addItem(text_item)
    # Добавляем созданные объекты в глобальные переменные для возможности последующего удаления
    globals()['curve_up'] = curve_up
    globals()['curve_down'] = curve_down
    globals()['text_item'] = text_item


def draw_fill(x, y1, y2, color):
    remove_poly_item()
    remove_curve_fake()
    curve_up = pg.PlotCurveItem(x=x, y=y1)
    curve_down = pg.PlotCurveItem(x=x, y=y2)
    poly_item = pg.FillBetweenItem(curve1=curve_down, curve2=curve_up, brush=color)
    radarogramma.addItem(poly_item)
    poly_item.setOpacity(0.5)
    poly_item.setZValue(1)
    globals()['poly_item'] = poly_item


def draw_fake(list_fake, list_up, list_down):
    remove_curve_fake()
    for f in list_fake:
        curve_fake = pg.PlotCurveItem(x=[f, f], y=[list_up[f], list_down[f]], pen=pg.mkPen(color='white', width=1))
        radarogramma.addItem(curve_fake)
        curve_fake.setZValue(1)
        globals()[f'curve_fake_{f}'] = curve_fake


def remove_poly_item():
    for key, value in globals().items():
        if key.startswith('poly_item'):
            radarogramma.removeItem(globals()[key])


def remove_curve_fake():
    for key, value in globals().items():
        if key.startswith('curve_fake_'):
            radarogramma.removeItem(globals()[key])


def draw_fill_result(x, y1, y2, color):
    curve_up = pg.PlotCurveItem(x=x, y=y1)
    curve_down = pg.PlotCurveItem(x=x, y=y2)
    poly_item = pg.FillBetweenItem(curve1=curve_down, curve2=curve_up, brush=color)
    radarogramma.addItem(poly_item)
    poly_item.setOpacity(0.5)
    poly_item.setZValue(1)
    globals()[f'poly_item{x[0]}'] = poly_item
