from func import *


def draw_radarogram():
    global l_up, l_down
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


def draw_current_radarogram():
    global l_up, l_down
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


def draw_param():
    param = ui.comboBox_param_plast.currentText()
    graph = json.loads(session.query(literal_column(f'Profile.{param}')).filter(Profile.id == get_profile_id()).first()[0])
    number = list(range(1, len(graph) + 1))
    ui.graph.clear()
    curve = pg.PlotCurveItem(x=number, y=graph)
    curve_filter = pg.PlotCurveItem(x=number, y=savgol_filter(graph, 31, 3), pen=pg.mkPen(color='red', width=2.4))
    ui.graph.addItem(curve)
    ui.graph.addItem(curve_filter)
    ui.graph.showGrid(x=True, y=True)
    set_info(f'Отрисовка параметра "{param}" для текущего профиля', 'blue')


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
    if ui.checkBox_minmax.isChecked():
        radar = json.loads(session.query(CurrentProfileMinMax.signal).filter(CurrentProfileMinMax.id == 1).first()[0])
    else:
        radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])
    draw_image(radar)
    updatePlot()
