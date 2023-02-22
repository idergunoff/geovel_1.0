from model import *
from func import *
from load import updatePlot, clear_current_profile, draw_radarogram


def clear_spectr():
    session.query(FFTSpectr).delete()
    session.commit()


def calc_add_fft():
    radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])

    axes_fft = (0, 1) if ui.checkBox_fft_2axes.isChecked() else 1
    if ui.checkBox_fft_int.isChecked():
        line_up = ui.spinBox_rad_up.value()
        line_down = ui.spinBox_rad_down.value()
        radar_fft = rfft2(radar[line_up:line_down], axes=axes_fft)
    else:
        radar_fft = rfft2(radar, axes=axes_fft)

    sum_radar_fft = [0]*len(radar_fft[0])
    for i in range(len(radar_fft)):
        sum_radar_fft += radar_fft[i]

    ui.spinBox_fft_down.setMaximum(len(radar_fft[0]))
    ui.spinBox_ftt_up.setMaximum(len(radar_fft[0]))
    up = ui.spinBox_ftt_up.value()
    down = ui.spinBox_fft_down.value()

    window = cosine(down - up + 1)
    for i in range(len(window)):
        window[i] = 1 - window[i]
    for i in range(len(radar_fft)):
        radar_fft[i][up:down+1] = radar_fft[i][up:down+1] * window

    radar_filter = irfft2(radar_fft, axes=axes_fft)

    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(radar_filter.tolist()))
    session.add(new_current)
    session.commit()

    radar_fft = rfft2(radar_filter, axes=axes_fft)
    sum_radar_fft = [0]*len(radar_fft[0])
    for i in range(len(radar_fft)):
        sum_radar_fft += radar_fft[i]

    clear_spectr()
    new_spectr = FFTSpectr(spectr=json.dumps(np.abs(sum_radar_fft[1:]).tolist()))
    session.add(new_spectr)
    session.commit()
    draw_fft_spectr()


def calc_fft():
    draw_radarogram()

    radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])

    axes_fft = (0, 1) if ui.checkBox_fft_2axes.isChecked() else 1
    if ui.checkBox_fft_int.isChecked():
        line_up = ui.spinBox_rad_up.value()
        line_down = ui.spinBox_rad_down.value()
        radar_fft = rfft2(radar[line_up:line_down], axes=axes_fft)
    else:
        radar_fft = rfft2(radar, axes=axes_fft)

    sum_radar_fft = [0]*len(radar_fft[0])
    for i in range(len(radar_fft)):
        sum_radar_fft += radar_fft[i]

    ui.spinBox_fft_down.setMaximum(len(radar_fft[0]))
    ui.spinBox_ftt_up.setMaximum(len(radar_fft[0]))
    up = ui.spinBox_ftt_up.value()
    down = ui.spinBox_fft_down.value()

    window = cosine(down - up + 1)
    for i in range(len(window)):
        window[i] = 1 - window[i]
    for i in range(len(radar_fft)):
        radar_fft[i][up:down+1] = radar_fft[i][up:down+1] * window


    radar_filter = irfft2(radar_fft, axes=axes_fft)


    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(radar_filter.tolist()))
    session.add(new_current)
    session.commit()

    radar_fft = rfft2(radar_filter, axes=axes_fft)
    sum_radar_fft = [0]*len(radar_fft[0])
    for i in range(len(radar_fft)):
        sum_radar_fft += radar_fft[i]

    clear_spectr()
    new_spectr = FFTSpectr(spectr=json.dumps(np.abs(sum_radar_fft[1:]).tolist()))
    session.add(new_spectr)
    session.commit()
    draw_fft_spectr()


def calc_ifft():
    radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])

    axes_fft = (0, 1) if ui.checkBox_fft_2axes.isChecked() else 1
    if ui.checkBox_fft_int.isChecked():
        line_up = ui.spinBox_rad_up.value()
        line_down = ui.spinBox_rad_down.value()
        radar_fft = rfft2(radar[line_up:line_down], axes=axes_fft)
    else:
        radar_fft = rfft2(radar, axes=axes_fft)

    sum_radar_fft = [0]*len(radar_fft[0])
    for i in range(len(radar_fft)):
        sum_radar_fft += radar_fft[i]

    ui.spinBox_fft_down.setMaximum(len(radar_fft[0]))
    ui.spinBox_ftt_up.setMaximum(len(radar_fft[0]))
    up = ui.spinBox_ftt_up.value()
    down = ui.spinBox_fft_down.value()

    window = cosine(down - up + 1)
    for i in range(len(window)):
        window[i] = 1 - window[i]
    for i in range(len(radar_fft)):
        radar_fft[i][up:down+1] = radar_fft[i][up:down+1] * window

    radar_filter = irfft2(radar_fft, axes=axes_fft)

    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(radar_filter.tolist()))
    session.add(new_current)
    session.commit()
    draw_image(radar_filter)
    updatePlot()

    radar_fft = rfft2(radar_filter, axes=axes_fft)
    sum_radar_fft = [0]*len(radar_fft[0])
    for i in range(len(radar_fft)):
        sum_radar_fft += radar_fft[i]

    clear_spectr()
    new_spectr = FFTSpectr(spectr=json.dumps(np.abs(sum_radar_fft[1:]).tolist()))
    session.add(new_spectr)
    session.commit()
    draw_fft_spectr()



def calc_medfilt():
    a = ui.spinBox_a_medfilt.value()
    b = ui.spinBox_b_medfilt.value()
    radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])
    radar_filter = medfilt2d(radar, [a, b])
    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(radar_filter.tolist()))
    session.add(new_current)
    session.commit()
    draw_image(radar_filter)
    updatePlot()


def calc_wiener():
    a = ui.spinBox_a_wiener.value()
    b = ui.spinBox_b_wiener.value()
    radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])
    radar_filter = wiener(radar, (a, b))
    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(radar_filter.tolist()))
    session.add(new_current)
    session.commit()
    draw_image(radar_filter)
    updatePlot()


def calc_savgol():
    a = ui.spinBox_a_savgol.value()
    b = ui.spinBox_b_savgol.value()
    radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])
    radar_filter = savgol_filter(radar, a, b)
    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(radar_filter.tolist()))
    session.add(new_current)
    session.commit()
    draw_image(radar_filter)
    updatePlot()


def calc_filtfilt():
    a = ui.spinBox_a_filtfilt.value()
    b = ui.doubleSpinBox_b_filtfilt.value()
    k_b, k_a = butter(a, b)
    radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])
    radar_filter = filtfilt(k_b, k_a, radar)
    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(radar_filter.tolist()))
    session.add(new_current)
    session.commit()
    draw_image(radar_filter)
    updatePlot()


def draw_fft_spectr():
    ui.graph.clear()
    spectr = json.loads(session.query(FFTSpectr.spectr).filter(FFTSpectr.id == 1).first()[0])
    curve = pg.PlotCurveItem(spectr, pen=pg.mkPen(color='red', width=2.4))
    ui.graph.addItem(curve)
    line_up = ui.spinBox_ftt_up.value() - 1
    line_down = ui.spinBox_fft_down.value() - 1
    l_up = pg.InfiniteLine(pos=line_up, angle=90, pen=pg.mkPen(color='darkred',width=1, dash=[8, 2]))
    l_down = pg.InfiniteLine(pos=line_down, angle=90, pen=pg.mkPen(color='darkgreen', width=1, dash=[8, 2]))
    ui.graph.addItem(l_up)
    ui.graph.addItem(l_down)
    ui.graph.showGrid(x=True, y=True)


def reset_spinbox_fft():
    ui.spinBox_ftt_up.setValue(0)
    ui.spinBox_fft_down.setValue(0)
