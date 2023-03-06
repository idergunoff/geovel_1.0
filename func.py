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
    ui.signal.plot(y=range(512, 0, -1), x=selected.mean(axis=0), clear=True, pen='r')
    ui.signal.plot(y=range(512, 0, -1), x=selected[n])
    ui.signal.showGrid(x=True, y=True)


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



