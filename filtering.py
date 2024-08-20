
from func import *
from qt.new_window_dialog import *
from load import clear_current_profile
from draw import draw_radarogram



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
    int_fft = '' if up == 0 and down == 0 else f' int({down}:{up})'
    axes_info = 'по 2 осям' if ui.checkBox_fft_2axes.isChecked() else 'по 1 оси'
    set_info(f'add_fft {axes_info}{int_fft}', 'blue')


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
    int_fft = '' if up == 0 and down == 0 else f' int({down}:{up})'
    axes_info = 'по 2 осям' if ui.checkBox_fft_2axes.isChecked() else 'по 1 оси'
    set_info(f'fft {axes_info}{int_fft}', 'blue')


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
    int_fft = '' if up == 0 and down == 0 else f' int({down}:{up})'
    axes_info = 'по 2 осям' if ui.checkBox_fft_2axes.isChecked() else 'по 1 оси'
    set_info(f'ifft {axes_info}{int_fft}', 'blue')



def calc_medfilt():
    a = ui.spinBox_a_medfilt.value()
    b = ui.spinBox_b_medfilt.value()
    radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])
    radar_filter = medfilt2d(radar, [a, b])
    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(radar_filter.tolist()))
    session.add(new_current)
    session.commit()
    save_max_min(radar_filter)
    if ui.checkBox_minmax.isChecked():
        radar_filter = json.loads(session.query(CurrentProfileMinMax.signal).filter(CurrentProfileMinMax.id == 1).first()[0])
    draw_image(radar_filter)
    updatePlot()
    set_info(f'medfilt({a}, {b})', 'blue')


def calc_wiener():
    a = ui.spinBox_a_wiener.value()
    b = ui.spinBox_b_wiener.value()
    radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])
    radar_filter = wiener(radar, (a, b))
    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(radar_filter.tolist()))
    session.add(new_current)
    session.commit()
    save_max_min(radar_filter)
    if ui.checkBox_minmax.isChecked():
        radar_filter = json.loads(session.query(CurrentProfileMinMax.signal).filter(CurrentProfileMinMax.id == 1).first()[0])
    draw_image(radar_filter)
    updatePlot()
    set_info(f'wiener({a}, {b})', 'blue')


def calc_savgol():
    a = ui.spinBox_a_savgol.value()
    b = ui.spinBox_b_savgol.value()
    radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])
    radar_filter = savgol_filter(radar, a, b)
    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(radar_filter.tolist()))
    session.add(new_current)
    session.commit()
    save_max_min(radar_filter)
    if ui.checkBox_minmax.isChecked():
        radar_filter = json.loads(session.query(CurrentProfileMinMax.signal).filter(CurrentProfileMinMax.id == 1).first()[0])
    draw_image(radar_filter)
    updatePlot()
    set_info(f'savgol({a}, {b})', 'blue')


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
    save_max_min(radar_filter)
    if ui.checkBox_minmax.isChecked():
        radar_filter = json.loads(session.query(CurrentProfileMinMax.signal).filter(CurrentProfileMinMax.id == 1).first()[0])
    draw_image(radar_filter)
    updatePlot()
    set_info(f'filtfilt({a}, {b})', 'blue')


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


def calc_rfft2():
    radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])

    axes_fft = (0, 1) if ui.checkBox_fft_2axes.isChecked() else 1

    radar_fft = np.repeat(rfft2(radar, axes=axes_fft), 2, axis=1)

    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(abs(radar_fft).tolist()))
    session.add(new_current)
    session.commit()

    save_max_min(abs(radar_fft))
    if ui.checkBox_minmax.isChecked():
        radar_fft = json.loads(session.query(CurrentProfileMinMax.signal).filter(CurrentProfileMinMax.id == 1).first()[0])
        draw_image(radar_fft)
    else:
        draw_image(abs(radar_fft))
    updatePlot()
    axes_info = ' по 2 осям' if ui.checkBox_fft_2axes.isChecked() else ' по 1 оси'
    set_info(f'calc FFT{axes_info}', 'blue')


def calc_dctn():
    radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])

    axes_dct = (0, 1) if ui.checkBox_fft_2axes.isChecked() else 1

    radar_fft = dctn(radar, type=ui.spinBox_type_dct.value(), norm='ortho', axes=axes_dct)

    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(radar_fft.tolist()))
    session.add(new_current)
    session.commit()
    save_max_min(radar_fft)
    if ui.checkBox_minmax.isChecked():
        radar_fft = json.loads(session.query(CurrentProfileMinMax.signal).filter(CurrentProfileMinMax.id == 1).first()[0])

    draw_image(radar_fft)
    updatePlot()
    axes_info = ' по 2 осям' if ui.checkBox_fft_2axes.isChecked() else ' по 1 оси'
    set_info(f'calc DCT{axes_info}', 'blue')


def calc_idctn():
    radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])

    axes_dct = (0, 1) if ui.checkBox_fft_2axes.isChecked() else 1

    radar_fft = idctn(radar, type=ui.spinBox_type_dct.value(), norm='ortho', axes=axes_dct)

    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(radar_fft.tolist()))
    session.add(new_current)
    session.commit()
    save_max_min(radar_fft)
    if ui.checkBox_minmax.isChecked():
        radar_fft = json.loads(session.query(CurrentProfileMinMax.signal).filter(CurrentProfileMinMax.id == 1).first()[0])

    draw_image(radar_fft)
    updatePlot()
    axes_info = ' по 2 осям' if ui.checkBox_fft_2axes.isChecked() else ' по 1 оси'
    set_info(f'calc IDCT{axes_info}', 'blue')



def calc_irfft2():
    radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])

    axes_fft = (0, 1) if ui.checkBox_fft_2axes.isChecked() else 1

    if ui.checkBox_2degree.isChecked():
        radar_fft = irfft2((np.log(rfft2(radar, axes=axes_fft)))**2, axes=axes_fft)
    else:
        radar_fft = irfft2(np.log(rfft2(radar, axes=axes_fft)), axes=axes_fft)

    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(radar_fft.tolist()))
    session.add(new_current)
    session.commit()
    save_max_min(radar_fft)
    if ui.checkBox_minmax.isChecked():
        radar_fft = json.loads(session.query(CurrentProfileMinMax.signal).filter(CurrentProfileMinMax.id == 1).first()[0])

    draw_image(radar_fft)
    updatePlot()
    axes_info = ' по 2 осям' if ui.checkBox_fft_2axes.isChecked() else ' по 1 оси'
    set_info(f'calc IFFT{axes_info}', 'blue')


def calc_log():
    radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])

    radar_log = np.log(radar)

    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(radar_log.tolist()))
    session.add(new_current)
    session.commit()
    save_max_min(radar_log)
    if ui.checkBox_minmax.isChecked():
        radar_log = json.loads(session.query(CurrentProfileMinMax.signal).filter(CurrentProfileMinMax.id == 1).first()[0])

    draw_image(radar_log)
    updatePlot()
    set_info('calc_LOG', 'blue')


def calc_rang():
    radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])

    radar_rang = rankdata(radar, axis=1)

    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(radar_rang.tolist()))
    session.add(new_current)
    session.commit()
    save_max_min(radar_rang)
    if ui.checkBox_minmax.isChecked():
        radar_rang = json.loads(session.query(CurrentProfileMinMax.signal).filter(CurrentProfileMinMax.id == 1).first()[0])

    draw_image(radar_rang)
    updatePlot()
    set_info('calc_RANG', 'blue')


def add_window():
    """Добавить ноывый объект в БД"""
    New_Window = QtWidgets.QDialog()
    ui_nw = Ui_new_window()
    ui_nw.setupUi(New_Window)
    New_Window.show()
    New_Window.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    radarogramma = ui_nw.radarogram.addPlot()
    img = pg.ImageItem()
    radarogramma.invertY(True)
    radarogramma.addItem(img)

    hist = pg.HistogramLUTItem(gradientPosition='left')

    ui_nw.radarogram.addItem(hist)

    roi = pg.ROI(pos=[0, 0], size=[ui.spinBox_roi.value(), 512], maxBounds=QRect(0, 0, 100000000, 512))
    radarogramma.addItem(roi)

    radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])
    if ui.checkBox_minmax.isChecked():
        radar = json.loads(session.query(CurrentProfileMinMax.signal).filter(CurrentProfileMinMax.id == 1).first()[0])

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

    if ui.checkBox_minmax.isChecked():
        radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])
    new_window = WindowProfile(profile_id=get_profile_id(), signal=json.dumps(radar))
    session.add(new_window)
    session.commit()

    ui_nw.label_id_nw_rad.setText(str(new_window.id))
    ui_nw.info.setText(ui.info.toHtml())

    def updatePlot_nw():
        rad = session.query(WindowProfile.signal).filter(WindowProfile.id == ui_nw.label_id_nw_rad.text()).first()
        radar = json.loads(rad[0])
        selected = roi.getArrayRegion(np.array(radar), img)
        n = ui.spinBox_roi.value() // 2
        ui_nw.signal.plot(y=range(512, 0, -1), x=selected.mean(axis=0), clear=True, pen='r')
        ui_nw.signal.plot(y=range(512, 0, -1), x=selected[n])
        ui_nw.signal.showGrid(x=True, y=True)


    def rollback_radar():
        radar = json.loads(session.query(WindowProfile.signal).filter(WindowProfile.id == ui_nw.label_id_nw_rad.text()).first()[0])
        clear_current_profile()
        new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(radar))
        session.add(new_current)
        session.commit()
        save_max_min(radar)
        if ui.checkBox_minmax.isChecked():
            radar =json.loads(session.query(CurrentProfileMinMax.signal).filter(CurrentProfileMinMax.id == 1).first()[0])
        draw_image(radar)
        updatePlot()
        ui.info.setText(ui_nw.info.toHtml())


    roi.sigRegionChanged.connect(updatePlot_nw)
    ui_nw.pushButton_rollback.clicked.connect(rollback_radar)
    updatePlot_nw()

    New_Window.exec_()


def filter19():
    ui.spinBox_a_medfilt.setValue(19)
    ui.spinBox_b_medfilt.setValue(1)
    ui.pushButton_medfilt.click()
    ui.spinBox_b_medfilt.setValue(19)
    ui.pushButton_medfilt.click()
    ui.spinBox_a_medfilt.setValue(1)
    ui.pushButton_medfilt.click()


def secret_filter():
    ui.pushButton_dct.click()
    ui.pushButton_rang.click()
    ui.pushButton_log.click()
    ui.pushButton_idct.click()
    ui.pushButton_rang.click()
    ui.pushButton_filtfilt.click()
    filter19()


def universal_threshold(data):
    n = len(data)
    sigma = np.median(np.abs(data)) / 0.6745
    return sigma * np.sqrt(2 * np.log(n))


def sure_shrink(data):
    n = len(data)
    sigma = np.median(np.abs(data)) / 0.6745
    squared_data = data**2
    t = np.sqrt(2 * np.log(n))
    risk = ((n - 2 * np.sum(norm.cdf(-data / sigma))) + 
            (squared_data / sigma**2) * (1 - norm.cdf(-data / sigma)) + 
            t**2 * norm.cdf(-t))
    best_t = t[np.argmin(risk)]
    return sigma * best_t


def denoise_profile(profile, wavelet='db4', level=None, mode='soft', threshold_func=universal_threshold):
    # Если уровень не задан, используем максимально возможный
    if level is None:
        level = pywt.dwt_max_level(len(profile[0]), wavelet)

    denoised_profile = np.zeros_like(profile)

    for i, signal in enumerate(profile):
        # Применяем вейвлет-преобразование
        coeffs = pywt.wavedec(signal, wavelet, level=level)

        # Применяем пороговую обработку к каждому уровню детализации
        for j in range(1, len(coeffs)):
            threshold = threshold_func(coeffs[j])
            coeffs[j] = pywt.threshold(coeffs[j], threshold, mode=mode)

        # Выполняем обратное вейвлет-преобразование
        denoised_profile[i] = pywt.waverec(coeffs, wavelet)

    return denoised_profile


# Пример использования


def calc_wavelet_filter():
    type_wavelet = ui.comboBox_type_wavelet.currentText()
    level_wavelet = ui.spinBox_level_wavelet.value()
    mode_wavelet = 'soft' if ui.checkBox_mode_threshold.isChecked() else 'hard'
    threshold_func = universal_threshold

    radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])

    denoised = denoise_profile(radar, wavelet=type_wavelet, level=level_wavelet, mode=mode_wavelet, threshold_func=threshold_func)
    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(denoised.tolist()))
    session.add(new_current)
    session.commit()
    save_max_min(denoised)
    if ui.checkBox_minmax.isChecked():
        denoised = json.loads(session.query(CurrentProfileMinMax.signal).filter(CurrentProfileMinMax.id == 1).first()[0])
    draw_image(denoised)
    updatePlot()
    set_info(f'wavelet filter {type_wavelet}, level {level_wavelet}, mode {mode_wavelet}', 'blue')
