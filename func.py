import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from object import *

list_param_geovel = [
    'T_top', 'T_bottom', 'dT', 'A_top', 'A_bottom', 'dA', 'A_sum', 'A_mean', 'dVt', 'Vt_top', 'Vt_sum', 'Vt_mean',
    'dAt', 'At_top', 'At_sum', 'At_mean', 'dPht', 'Pht_top', 'Pht_sum', 'Pht_mean', 'Wt_top', 'Wt_mean', 'Wt_sum',
    'width', 'top', 'land', 'speed', 'speed_cover', 'skew', 'kurt', 'std', 'k_var', 'A_max', 'Vt_max', 'At_max',
    'Pht_max', 'Wt_max', 'A_T_max', 'Vt_T_max', 'At_T_max', 'Pht_T_max', 'Wt_T_max', 'A_Sn', 'Vt_Sn', 'At_Sn',
    'Pht_Sn', 'Wt_Sn', 'A_wmf', 'Vt_wmf', 'At_wmf', 'Pht_wmf', 'Wt_wmf', 'A_Qf', 'Vt_Qf', 'At_Qf', 'Pht_Qf', 'Wt_Qf',
    'A_Sn_wmf', 'Vt_Sn_wmf', 'At_Sn_wmf', 'Pht_Sn_wmf', 'Wt_Sn_wmf', 'k_r'
    ]

# Функция добавления информации в окно информации с указанием времени и цвета текста
def set_info(text, color):
    ui.info.append(f'{datetime.datetime.now().strftime("%H:%M:%S")}:\t<span style ="color:{color};" >{text}</span>')


# Функция получения id выбранного объекта
def get_object_id():
    try:
        return int(ui.comboBox_object.currentText().split('id')[-1])
    except ValueError:
        pass


def get_research_id():
    return int(ui.comboBox_research.currentText().split(' id')[-1])


def get_research_name():
    return ui.comboBox_research.currentText().split(' id')[0]


# Функция получения имени выбранного объекта
def get_object_name():
    return ui.comboBox_object.currentText().split(' id')[0]


# Функция получения имени выбранного объекта мониторинга
def get_obj_monitor_name():
    return ui.comboBox_object_monitor.currentText()


# Функция получения id выбранного профиля
def get_profile_id():
    try:
        return int(ui.comboBox_profile.currentText().split(' id')[-1])
    except ValueError:
        return False


# Функция получения имени выбранного профиля
def get_profile_name():
    return ui.comboBox_profile.currentText().split(' id')[0]


# Функция преобразования результатов запроса в список
def query_to_list(query):
    """ результаты запроса в список """
    return sum(list(map(list, query)), [])


# Функция отображения радарограммы
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


# Функция очистки текущего профиля
def clear_current_profile():
    session.query(CurrentProfile).delete()
    session.commit()


def clear_current_profile_min_max():
    """Очистить таблицу текущего профиля минимальных и максимальных значений"""
    session.query(CurrentProfileMinMax).delete()
    session.commit()


def clear_window_profile():
    """Очистить таблицу оконного профиля"""
    session.query(WindowProfile).delete()
    session.commit()


def vacuum():
    """Выполнить VACUUM базы данных"""
    conn = connect(DATABASE_NAME)
    conn.execute("VACUUM")
    conn.close()


def set_random_color():
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)
    color = f'#{red:02x}{green:02x}{blue:02x}'
    ui.pushButton_color.setStyleSheet(f"background-color: {color};")
    ui.pushButton_color.setText(color)


def changeSpinBox():
    """Изменить размер выделенной области"""
    roi.setSize([ui.spinBox_roi.value(), 512])


def clear_spectr():
    """Очистить таблицу спектра"""
    session.query(FFTSpectr).delete()
    session.commit()


def reset_spinbox_fft():
    """Сбросить значения спинбоксов FFT"""
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
    # Получаем данные с радара из базы данных
    rad = session.query(CurrentProfile.signal).first()
    # Преобразуем данные из строки json в словарь python
    radar = json.loads(rad[0])
    # Выбираем область интереса (ROI) изображения
    selected = roi.getArrayRegion(np.array(radar), img)
    # Вычисляем половину размера области интереса
    n = ui.spinBox_roi.value()//2
    # Очищаем график перед отрисовкой новых данных
    ui.signal.plot(y=range(0, 512), x=selected.mean(axis=0), clear=True, pen='r')
    # Добавляем график визуализации одного из профилей данных
    ui.signal.plot(y=range(0, 512), x=selected[n])
    # Включаем отображение координатной сетки на графике
    ui.signal.showGrid(x=True, y=True)
    # Инвертируем направление оси Y на графике
    ui.signal.invertY(True)


def update_profile_combobox():
    """ Обновление списка профилей в выпадающем списке """
    ui.label_4.setText(f'Объект ({calc_object_measures()} изм)')
    # Очистка выпадающего списка
    ui.comboBox_profile.clear()
    try:
        # Запрос на получение всех профилей, относящихся к объекту, и их добавление в выпадающий список
        for i in session.query(Profile).filter(Profile.research_id == get_research_id()).all():
            count_measure = len(json.loads(session.query(Profile.signal).filter(Profile.id == i.id).first()[0]))
            ui.comboBox_profile.addItem(f'{i.title} ({count_measure} измерений) id{i.id}')
        # Обновление списка формирований
        update_formation_combobox()
        update_layers()
    except ValueError:
        # Если возникла ошибка при обновлении списка профилей, просто проигнорировать ее
        pass
    # Если в объекте есть график, изменить цвет кнопок на зеленый
    if session.query(Grid).filter(Grid.object_id == get_object_id()).count() > 0:
        ui.pushButton_uf.setStyleSheet('background: rgb(191, 255, 191)')
        ui.pushButton_m.setStyleSheet('background: rgb(191, 255, 191)')
        ui.pushButton_r.setStyleSheet('background: rgb(191, 255, 191)')
    else:
        # Если в объекте нет графика, изменить цвет кнопок на красный
        ui.pushButton_uf.setStyleSheet('background: rgb(255, 185, 185)')
        ui.pushButton_m.setStyleSheet('background:  rgb(255, 185, 185)')
        ui.pushButton_r.setStyleSheet('background: rgb(255, 185, 185)')

    check_coordinates_profile()


def update_object():
    """ Функция для обновления списка объектов в выпадающем списке """
    # Очистка выпадающего списка объектов
    ui.comboBox_object.clear()
    # Получение всех объектов из базы данных, отсортированных по дате исследования
    for i in session.query(GeoradarObject).order_by(GeoradarObject.title).all():
        # Добавление названия объекта, даты исследования и идентификатора объекта в выпадающий список
        ui.comboBox_object.addItem(f'{i.title} id{i.id}')
    # Обновление выпадающего списка профилей
    update_research_combobox()
    check_coordinates_research()
    check_coordinates_profile()


def update_research_combobox():
    ui.comboBox_research.clear()
    for i in session.query(Research).filter(Research.object_id == get_object_id()).order_by(Research.date_research).all():
        ui.comboBox_research.addItem(f'{i.date_research.strftime("%m.%Y")} id{i.id}')
    update_profile_combobox()
    check_coordinates_profile()
    check_coordinates_research()


def update_param_combobox():
    """Обновление выпадающего списка параметров формаций/пластов"""
    current_text = ui.comboBox_param_plast.currentText()  # сохраняем текущий текст в комбобоксе
    ui.comboBox_param_plast.clear()  # очищаем комбобокс
    if ui.comboBox_plast.currentText() == '-----':  # если выбрана пустая строка, то ничего не делаем
        pass
    # elif ui.comboBox_plast.currentText() == 'KROT':  # если выбран КРОТ, то добавляем в комбобокс параметры профиля
    #     list_columns = Profile.__table__.columns.keys()  # список параметров таблицы профиля
    #     # удаляем не нужные колонки
    #     [list_columns.remove(i) for i in ['id', 'object_id', 'title', 'x_wgs', 'y_wgs', 'x_pulc', 'y_pulc', 'signal']]
    #     for i in list_columns:
    #         # если в таблице профиля есть хотя бы одна запись, где значение параметра не NULL, то добавляем параметр в комбобокс
    #         if session.query(Profile).filter(text(f"profile_id=:p_id and {i} NOT NULL")).params(p_id=get_profile_id()).count() > 0:
    #             ui.comboBox_param_plast.addItem(i)
    else:  # если выбрана какая-то формация, то добавляем в комбобокс параметры формации
        list_columns = Formation.__table__.columns.keys()  # список параметров таблицы формаций
        [list_columns.remove(i) for i in  ['id', 'profile_id', 'title', 'up', 'down']]  # удаляем не нужные колонки
        for i in list_columns:
            # если в таблице формаций есть хотя бы одна запись, где значение параметра не NULL, то добавляем параметр в комбобокс
            if session.query(Formation).filter(text(f"profile_id=:p_id and {i} NOT NULL")).params(p_id=get_profile_id()).count() > 0:
                ui.comboBox_param_plast.addItem(i)
    index = ui.comboBox_param_plast.findText(current_text)  # находим индекс сохраненного текста в комбобоксе
    if index != -1:  # если сохраненный текст есть в комбобоксе, то выбираем его
        ui.comboBox_param_plast.setCurrentIndex(index)
    draw_param()  # отрисовываем параметр
    update_layers()  # обновляем список слоев в соответствии с выбранным параметром


def draw_param():
    # Очищаем график
    ui.graph.clear()
    # Получаем параметр из выпадающего списка
    param = ui.comboBox_param_plast.currentText()
    # Если выбрана опция "все пласты"
    if ui.checkBox_all_formation.isChecked():
        # Если не выбран конкретный пласт
        if ui.comboBox_plast.currentText() == '-----':
            return
        # # Получаем данные для текущего пласта
        # graph = json.loads(session.query(literal_column(f'Formation.{param}')).filter(
        #     Formation.profile_id == get_profile_id()).first()[0])
        # # Создаем список значений по порядку
        # number = list(range(1, len(graph) + 1))
        # # Создаем кривую и кривую, отфильтрованную с помощью savgol_filter
        # curve = pg.PlotCurveItem(x=number, y=graph)
        # wl = 2.4 if ui.comboBox_plast.currentText() == 'KROT' else 1
        # cl = 'red' if ui.comboBox_plast.currentText() == 'KROT' else 'green'
        # curve_filter = pg.PlotCurveItem(x=number, y=savgol_filter(graph, 31, 3),
        #                                 pen=pg.mkPen(color=cl, width=wl))
        # # Если выбран пласт КРОТ, то добавляем только кривую на график
        # if ui.comboBox_plast.currentText() == 'KROT':
        #     ui.graph.addItem(curve)
        # # Добавляем кривую и отфильтрованную кривую на график для всех пластов
        # ui.graph.addItem(curve_filter)
        # Для каждого пласта
        for f in session.query(Formation).filter(Formation.profile_id == get_profile_id()).all():
            # Получаем данные для текущего пласта
            graph = json.loads(session.query(literal_column(f'Formation.{param}')).filter(Formation.id == f.id).first()[0])
            # Создаем список значений по порядку
            number = list(range(1, len(graph) + 1))
            # Создаем кривую и кривую, отфильтрованную с помощью savgol_filter
            curve = pg.PlotCurveItem(x=number, y=graph)
            wl = 2.4 if f.id == get_formation_id() else 1
            cl = 'red' if f.id == get_formation_id() else 'green'
            curve_filter = pg.PlotCurveItem(x=number, y=savgol_filter(graph, 31, 3),
                                            pen=pg.mkPen(color=cl, width=wl))
            # Если выбран текущий пласт, то добавляем только кривую на график
            if f.id == get_formation_id():
                ui.graph.addItem(curve)
            # Добавляем кривую и отфильтрованную кривую на график для всех пластов
            ui.graph.addItem(curve_filter)
    # Если выбран конкретный пласт
    else:
        # если текущий выбранный элемент равен '-----', то ничего не делаем и выходим из функции
        if ui.comboBox_plast.currentText() == '-----':
            return
        else:  # в остальных случаях получаем данные для формации
            # получаем данные для выбранного параметра из таблицы Formation и преобразуем их из строки в список с помощью json.loads()
            graph = json.loads(session.query(literal_column(f'Formation.{param}')).filter(
                Formation.id == get_formation_id()).first()[0])
        number = list(range(1, len(graph) + 1))  # создаем список номеров элементов данных
        curve = pg.PlotCurveItem(x=number, y=graph)  # создаем объект класса PlotCurveItem для отображения графика данных
        # создаем объект класса PlotCurveItem для отображения фильтрованных данных с помощью savgol_filter()
        curve_filter = pg.PlotCurveItem(x=number, y=savgol_filter(graph, 31, 3), pen=pg.mkPen(color='red', width=2.4))
        ui.graph.addItem(curve)  # добавляем график данных на график
        ui.graph.addItem(curve_filter)  # добавляем фильтрованный график данных на график
        ui.graph.showGrid(x=True, y=True)  # отображаем сетку на графике
        set_info(f'Отрисовка параметра "{param}" для текущего профиля', 'blue')  # выводим информационное сообщение в лог синим цветом


def save_max_min(radar):
    radar_max_min = []
    ui.progressBar.setMaximum(len(radar))
    for n, sig in enumerate(radar):
        max_points, _ = find_peaks(np.array(sig))
        min_points, _ = find_peaks(-np.array(sig))
        # diff_signal = np.diff(sig) #возможно 2min/max
        # max_points = argrelmax(diff_signal)[0]
        # min_points = argrelmin(diff_signal)[0]
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
    # удаляем все CheckBox виджеты, находящиеся в виджете ui.widget_layer
    # и удаляем соответствующие объекты графика radarogramma из глобального пространства имен
    for i in ui.widget_layer.findChildren(QtWidgets.QCheckBox):
        i.deleteLater()  # удаляем CheckBox виджет
        # если объект scatter_id или curve_id существует, удаляем его из графика и из глобального пространства имен
        if f'scatter_id{i.text().split(" id")[-1]}' in globals():
            radarogramma.removeItem(globals()[f'scatter_id{i.text().split(" id")[-1]}'])
            del globals()[f'scatter_id{i.text().split(" id")[-1]}']
        if f'curve_id{i.text().split(" id")[-1]}' in globals():
            radarogramma.removeItem(globals()[f'curve_id{i.text().split(" id")[-1]}'])
            del globals()[f'curve_id{i.text().split(" id")[-1]}']
        if f'text_id{i.text().split(" id")[-1]}' in globals():
            radarogramma.removeItem(globals()[f'text_id{i.text().split(" id")[-1]}'])
            del globals()[f'text_id{i.text().split(" id")[-1]}']
    # удаляем все RadioButton виджеты, находящиеся в виджете ui.widget_layer_radio
    for i in ui.widget_layer.findChildren(QtWidgets.QRadioButton):
        i.deleteLater()


def get_layer_id():
    # Найти id выбранного слоя в списке радиокнопок
    for i in ui.widget_layer.findChildren(QtWidgets.QRadioButton):
        if i.isChecked():
            return int(i.text())


def get_layer_first_checkbox_id():
    # Найти id выбранного слоя в списке радиокнопок
    for i in ui.widget_layer.findChildren(QtWidgets.QCheckBox):
        if i.isChecked():
            return int(i.text().split(' id')[-1])


def get_layer_id_to_crop():
    # Найти id слоя, который нужно обрезать, из списка чекбоксов слоев
    l_id = get_layer_id()
    for i in ui.widget_layer.findChildren(QtWidgets.QCheckBox):
        if i.isChecked() and str(l_id) != i.text().split(' id')[-1]:
            return int(i.text().split(' id')[-1])


def get_layers_for_formation():
    # Получить список выбранных для формирования слоев
    list_id = []
    for i in ui.widget_layer.findChildren(QtWidgets.QCheckBox):
        if i.isChecked():
            list_id.append(int(i.text().split(' id')[-1]))
            if len(list_id) == 2:
                return list_id
    return False


def get_formation_id():
    # Найти id выбранного пласта
    if ui.comboBox_plast.currentText() == '-----' or ui.comboBox_plast.currentText() == 'KROT':
        return False

    return int(ui.comboBox_plast.currentText().split(' id')[-1])


def draw_layers():
    # Проходим по всем CheckBox объектам на widget_layer
    for i in ui.widget_layer.findChildren(QtWidgets.QCheckBox):
        # Если объект уже отображен на радарограмме, удаляем его
        if f'scatter_id{i.text().split(" id")[-1]}' in globals():
            # globals()[f'scatter_id{i.text().split(" id")[-1]}'] = globals()[f'scatter_id{i.text().split(" id")[-1]}']
            radarogramma.removeItem(globals()[f'scatter_id{i.text().split(" id")[-1]}'])
        if f'curve_id{i.text().split(" id")[-1]}' in globals():
            # globals()[f'curve_id{i.text().split(" id")[-1]}'] = globals()[f'curve_id{i.text().split(" id")[-1]}']
            radarogramma.removeItem(globals()[f'curve_id{i.text().split(" id")[-1]}'])
        if f'text_id{i.text().split(" id")[-1]}' in globals():
            # globals()[f'text_id{i.text().split(" id")[-1]}'] = globals()[f'text_id{i.text().split(" id")[-1]}']
            radarogramma.removeItem(globals()[f'text_id{i.text().split(" id")[-1]}'])
        # Если CheckBox отмечен, отображаем соответствующий слой на радарограмме
        if i.isChecked():
            draw_layer(i.text().split(' id')[-1])


def draw_layer(layer_id, save=False):
    # если существуют глобальные объекты линий и точек, удаляем их с радарограммы
    if f'curve_id{layer_id}' in globals():
        radarogramma.removeItem(globals()[f'curve_id{layer_id}'])
    if f'scatter_id{layer_id}' in globals():
        radarogramma.removeItem(globals()[f'scatter_id{layer_id}'])
    if f'text_id{layer_id}' in globals():
        radarogramma.removeItem(globals()[f'text_id{layer_id}'])
    # Если включен режим рисования, рисуем интерполяцию линии по точкам
    if ui.checkBox_draw_layer.isChecked() or not session.query(Layers.layer_line).filter(Layers.id == layer_id).first()[0]:
        # Получаем из базы данных координаты точек для данного слоя
        layer_x = np.array(query_to_list(session.query(PointsOfLayer.point_x).filter(PointsOfLayer.layer_id == layer_id).order_by(PointsOfLayer.point_x).all()))
        layer_y = np.array(query_to_list(session.query(PointsOfLayer.point_y).filter(PointsOfLayer.layer_id == layer_id).order_by(PointsOfLayer.point_x).all()))
        # Создаем объект точек и добавляем его на радарограмму
        scatter = pg.ScatterPlotItem(x=layer_x, y=layer_y, symbol='o', pen=pg.mkPen(None),
                                     brush=pg.mkBrush('#FFE900'), size=12)
        radarogramma.addItem(scatter)
        globals()[f'scatter_id{layer_id}'] = scatter
        # Если точек для интерполяции больше двух, вычисляем кубический сплайн
        if len(layer_x) > 2:
            tck = splrep(layer_x, layer_y, k=2)
            # Определяем новые точки для интерполяции
            x_new = list(range(int(layer_x.min()), int(layer_x.max())))
            # Вычисляем значения интерполированной кривой
            y_new = list(map(int, splev(x_new, tck)))
            # Создаем объект линии и добавляем его на радарограмму
            curve = pg.PlotCurveItem(x=x_new, y=y_new, pen=pg.mkPen(color='white', width=2))
            radarogramma.addItem(curve)
            # Создаем объект текста для отображения id слоя и добавляем его на радарограмму
            text_item = pg.TextItem(text=f'{layer_id}', color='white')
            text_item.setPos(layer_x.min() - int((layer_x.max() - layer_x.min()) / 50),
                             int(layer_y.tolist()[layer_x.tolist().index(min(layer_x.tolist()))]))
            radarogramma.addItem(text_item)
            # Добавляем созданные объекты в глобальные переменные для возможности последующего удаления
            globals()[f'curve_id{layer_id}'] = curve
            globals()[f'text_id{layer_id}'] = text_item
            # сохранение кривой
            if save:
                session.query(Layers).filter(Layers.id == layer_id).update({'layer_line': json.dumps(y_new)}, synchronize_session="fetch")
                session.commit()
    else:
        y = json.loads(session.query(Layers.layer_line).filter(Layers.id == layer_id).first()[0])
        x = list(range(len(y)))
        # Создаем объект линии и добавляем его на радарограмму
        curve = pg.PlotCurveItem(x=x, y=y, pen=pg.mkPen(color='white', width=2))
        radarogramma.addItem(curve)
        # Создаем объект текста для отображения id слоя и добавляем его на радарограмму
        text_item = pg.TextItem(text=f'{layer_id}', color='white')
        text_item.setPos(min(x) - int(max(x) - min(x)) / 50, int(y[x.index(min(x))]))
        radarogramma.addItem(text_item)
        # Добавляем созданные объекты в глобальные переменные для возможности последующего удаления
        globals()[f'curve_id{layer_id}'] = curve
        globals()[f'text_id{layer_id}'] = text_item


def update_layers():
    clear_layers()
    # Если в базе данных нет профиля, то выйти из функции
    if session.query(Profile).count() == 0:
        return
    # Получить все слои текущего профиля
    layers = session.query(Layers).filter(Layers.profile_id == get_profile_id()).all()
    # Для каждого слоя создать элементы управления
    for n, lay in enumerate(layers):
        ui.new_horizontalLayout = QtWidgets.QHBoxLayout()
        ui.verticalLayout_layer.addLayout(ui.new_horizontalLayout)
        ui.checkBox_new = QtWidgets.QCheckBox()
        # Задать текст для флажка и радиокнопки
        ui.checkBox_new.setText(f'{lay.layer_title} id{lay.id}')
        # Добавить флажок и радиокнопку на соответствующие макеты
        ui.new_horizontalLayout.addWidget(ui.checkBox_new)
        if not lay.layer_title.startswith('krot_'):
            ui.radio_new = QtWidgets.QRadioButton()
            ui.radio_new.setText(f'{lay.id}')
            ui.new_horizontalLayout.addWidget(ui.radio_new)
        # Связать событие нажатия на флажок с функцией draw_layers()
        ui.checkBox_new.clicked.connect(draw_layers)


def crop():
    # Получаем ID слоя, который необходимо обрезать
    l_crop_id = get_layer_id_to_crop()
    # Получаем линию обрезки в формате JSON из БД
    json_crop_line = session.query(Layers.layer_line).filter(Layers.id == l_crop_id).first()[0]
    # Если линия обрезки не найдена, выводим сообщение об ошибке и выходим из функции
    if not json_crop_line:
        set_info('Выбранный для обрезки слой не сохранён', 'red')
        return
    # Преобразуем линию обрезки из формата JSON в Python-объект
    crop_line = json.loads(json_crop_line)
    # Получаем ID слоя, который нужно обрезать до линии обрезки
    l_id = get_layer_id()
    # Получаем координаты точек слоя, который нужно обрезать, по оси X и Y
    layer_x = query_to_list(session.query(PointsOfLayer.point_x).filter(PointsOfLayer.layer_id == l_id).order_by(
        PointsOfLayer.point_x).all())
    layer_y = query_to_list(session.query(PointsOfLayer.point_y).filter(PointsOfLayer.layer_id == l_id).order_by(
        PointsOfLayer.point_x).all())
    # Получаем количество сигналов в профиле
    count_sig = len(json.loads(session.query(Profile.signal).filter(Profile.id == get_profile_id()).first()[0]))
    # Добавляем точки, если их нет на границах слоя
    if session.query(PointsOfLayer).filter(PointsOfLayer.layer_id == get_layer_id(),
                                           PointsOfLayer.point_x == 0).count() == 0:
        new_point = PointsOfLayer(layer_id=get_layer_id(), point_x=0, point_y=int(layer_y[layer_x.index(min(layer_x))]))
        session.add(new_point)
    if session.query(PointsOfLayer).filter(PointsOfLayer.layer_id == get_layer_id(),
                                           PointsOfLayer.point_x == count_sig).count() == 0:
        new_point = PointsOfLayer(layer_id=get_layer_id(), point_x=count_sig,
                                  point_y=int(layer_y[layer_x.index(max(layer_x))]))
        session.add(new_point)
    # Сохраняем изменения в БД
    session.commit()
    # Обновляем координаты точек слоя, который нужно обрезать, по оси X и Y
    layer_x = query_to_list(session.query(PointsOfLayer.point_x).filter(PointsOfLayer.layer_id == l_id).order_by(
        PointsOfLayer.point_x).all())
    layer_y = query_to_list(session.query(PointsOfLayer.point_y).filter(PointsOfLayer.layer_id == l_id).order_by(
        PointsOfLayer.point_x).all())
    # Используем кубическую сплайн-интерполяцию для получения новых координат точек слоя по оси Y
    tck = splrep(layer_x, layer_y, k=2)
    # Определяем новые точки для интерполяции по оси X
    x_new = list(range(int(min(layer_x)), int(max(layer_x))))
    # Вычисляем значения интерполированной кривой по оси Y
    y_new = list(map(int, splev(x_new, tck)))
    return l_id, crop_line, y_new


def update_formation_combobox():
    # Очищаем список выбора пластов
    ui.comboBox_plast.clear()
    # Получаем все формации текущего профиля
    formations = session.query(Formation).filter(Formation.profile_id == get_profile_id()).all()
    # Добавляем первые два пункта в список
    ui.comboBox_plast.addItem('-----')
    # if session.query(Profile.T_top).filter(Profile.id == get_profile_id()).first()[0]:
    #     ui.comboBox_plast.addItem('KROT')
    # Добавляем все формации в список выбора пластов
    for form in formations:
        ui.comboBox_plast.addItem(f'{form.title} id{form.id}')
    # Обновляем список выбора параметров для выбранного пласта
    update_param_combobox()
    update_list_well()


def check_coordinates_profile():
    profile_id = get_profile_id()
    if profile_id:
        x_pulc = session.query(Profile.x_pulc).filter(Profile.id == profile_id).scalar()
        ui.label_5.setStyleSheet('background: #D6FCE5' if x_pulc else 'background: #F7B9B9')
    else:
        ui.label_5.setStyleSheet('background: #F7B9B9')



def check_coordinates_research():
    profiles = session.query(Profile).filter(Profile.research_id == get_research_id()).all()
    if len(profiles) == 0:
        ui.label_4.setStyleSheet('background: #F7B9B9')
        return
    for prof in profiles:
        if not prof.x_pulc:
            ui.label_4.setStyleSheet('background: #F7B9B9')
            return
    else:
        ui.label_4.setStyleSheet('background: #D6FCE5')


def calc_atrib(rad, atr):
    radar = []
    for n, i in enumerate(rad):
        ui.progressBar.setValue(n + 1)
        radar.append(calc_atrib_measure(i, atr))
    return radar


def calc_atrib_measure(rad, atr):
    if atr == 'Abase':
        return rad
    elif atr == 'diff':
        return np.diff(rad).tolist()
    elif atr == 'At':
        analytic_signal = hilbert(rad)
        return list(map(lambda x: round(x, 2), np.hypot(rad, np.imag(analytic_signal))))
    elif atr == 'Vt':
        analytic_signal = hilbert(rad)
        return list(map(lambda x: round(x, 2), np.imag(analytic_signal)))
    elif atr == 'Pht':
        analytic_signal = hilbert(rad)
        return list(map(lambda x: round(x, 2), np.angle(analytic_signal)))
    elif atr == 'Wt':
        analytic_signal = hilbert(rad)
        return list(map(lambda x: round(x, 2), np.diff(np.angle(analytic_signal))))
    else:
        return rad


###################################################################
############################   Well   #############################
###################################################################


# Функция для нахождения ближайшей точки на профиле к заданной скважине
def closest_point(well_x, well_y, profile_x, profile_y):
    # используем lambda-функцию для нахождения расстояния между скважиной и каждой точкой на профиле
    closest = min(range(len(profile_x)), key=lambda i: calc_distance(well_x, well_y, profile_x[i], profile_y[i]))
    # возвращает индекс ближайшей точки и расстояние до нее
    return (closest, calc_distance(well_x, well_y, profile_x[closest], profile_y[closest]))


def update_list_well():
    """Обновить виджет списка скважин"""
    for key, value in globals().items():
        if key.startswith('bound_') or key.startswith('well_'):
            radarogramma.removeItem(globals()[key])
    ui.listWidget_well.clear()
    if ui.checkBox_profile_intersec.isChecked():
        intersections = session.query(Intersection).order_by(Intersection.name).all()
        if ui.checkBox_profile_well.isChecked():
            for intersec in intersections:
                if intersec.profile_id == get_profile_id():
                    ui.listWidget_well.addItem(f'{intersec.name} id{intersec.id}')
            draw_wells()
        else:
            for intersec in intersections:
                ui.listWidget_well.addItem(f'{intersec.name} id{intersec.id}')
    else:
        if ui.checkBox_profile_well.isChecked():
            wells = get_list_nearest_well(get_profile_id())
            if wells:
                for w in wells:
                    ui.listWidget_well.addItem(f'скв.№ {w[0].name} - ({w[1]}) {round(w[2], 2)} м. id{w[0].id}')
                draw_wells()
        else:
            wells = session.query(Well).order_by(Well.name).all()
            for w in wells:
                ui.listWidget_well.addItem(f'скв.№ {w.name} id{w.id}')


def get_list_nearest_well(profile_id):
    profile = session.query(Profile).filter_by(id=profile_id).first()
    if session.query(Profile.x_pulc).filter_by(id=profile_id).first()[0]:
        x_prof = json.loads(profile.x_pulc)
        y_prof = json.loads(profile.y_pulc)
        profile_plus_dist = calc_distance(x_prof[0], y_prof[0], x_prof[-1], y_prof[-1]) + ui.spinBox_well_distance.value()
        wells = session.query(Well).order_by(Well.name).all()
        list_nearest_well = []
        for w in wells:
            start_profile_dist = calc_distance(x_prof[0], y_prof[0], w.x_coord, w.y_coord)
            end_profile_dist = calc_distance(x_prof[-1], y_prof[-1], w.x_coord, w.y_coord)
            if start_profile_dist <= profile_plus_dist or end_profile_dist <= profile_plus_dist:
                index, dist = closest_point(w.x_coord, w.y_coord, x_prof, y_prof)
                if dist <= ui.spinBox_well_distance.value():
                    list_nearest_well.append([w, index, dist])
        return list_nearest_well
    else:
        ui.listWidget_well.addItem(f'Координаты профиля {profile.title} не загружены')


def set_title_list_widget_wells():
    if ui.checkBox_profile_intersec.isChecked():
        ui.label_11.setText('Intersections:')
    else:
        ui.label_11.setText('Wells:')


def get_well_id():
    if ui.listWidget_well.currentItem():
        return ui.listWidget_well.currentItem().text().split(' id')[-1]


def get_well_name():
    if ui.listWidget_well.currentItem():
        return ui.listWidget_well.currentItem().text().split(' id')[0]


def process_string(s):
    """Удалить пробелы из строки и заменить запятую на точку"""
    if type(s) == str:
        s = s.replace(" ", "") # удалить пробелы
        s = s.replace("\xa0", "")  # удалить пробелы
        s = s.replace(",", ".") # заменить запятые на точки
    return s


def update_boundaries():
    ui.listWidget_bound.clear()
    boundaries = session.query(Boundary).filter(Boundary.well_id == get_well_id()).order_by(Boundary.depth).all()
    for b in boundaries:
        ui.listWidget_bound.addItem(f'{b.title} - {b.depth}m. id{b.id}')


def get_boundary_id():
    return ui.listWidget_bound.currentItem().text().split(' id')[-1]


def draw_wells():
    # for key, value in globals().items():
    #     if key.startswith('well_'):
    #         radarogramma.removeItem(globals()[key])
    for i in range(ui.listWidget_well.count()):
        well_id = ui.listWidget_well.item(i).text().split(' id')[-1]
        draw_intersection(well_id) if ui.checkBox_profile_intersec.isChecked() else draw_well(well_id)


def draw_well(well_id):
    # если существуют глобальные объекты линий и точек, удаляем их с радарограммы
    if f'well_curve_id{well_id}' in globals():
        radarogramma.removeItem(globals()[f'well_curve_id{well_id}'])
    if f'well_text_id{well_id}' in globals():
        radarogramma.removeItem(globals()[f'well_text_id{well_id}'])

    well = session.query(Well).filter(Well.id == well_id).first()
    x_prof = json.loads(session.query(Profile.x_pulc).filter(Profile.id == get_profile_id()).first()[0])
    y_prof = json.loads(session.query(Profile.y_pulc).filter(Profile.id == get_profile_id()).first()[0])
    index, dist = closest_point(well.x_coord, well.y_coord, x_prof, y_prof)
    curve = pg.PlotCurveItem(x=[index, index], y=[0, 512], pen=pg.mkPen(color='white', width=2))
    radarogramma.addItem(curve)
    globals()[f'well_curve_id{well_id}'] = curve


    well_text = f'int ({well.name.split("/")[2]})' if well.name.startswith('int') else well.name
    text_item = pg.TextItem(text=f'{well_text} ({round(dist, 2)})', color='white')
    text_item.setPos(index, -30)
    radarogramma.addItem(text_item)
    globals()[f'well_text_id{well_id}'] = text_item

    boundaries = session.query(Boundary).filter(Boundary.well_id == well_id).all()
    # for key, value in globals().items():
    #     if key.startswith('bound_'):
    #         radarogramma.removeItem(globals()[key])
    for b in boundaries:
        draw_boundary(b.id, index)



def draw_intersection(int_id):
    # если существуют глобальные объекты линий и точек, удаляем их с радарограммы
    if f'well_curve_id{int_id}' in globals():
        radarogramma.removeItem(globals()[f'well_curve_id{int_id}'])
    if f'well_text_id{int_id}' in globals():
        radarogramma.removeItem(globals()[f'well_text_id{int_id}'])

    intersection = session.query(Intersection).filter_by(id=int_id).first()

    curve = pg.PlotCurveItem(x=[intersection.i_profile, intersection.i_profile], y=[0, 512], pen=pg.mkPen(color='white', width=2))
    radarogramma.addItem(curve)
    globals()[f'well_curve_id{int_id}'] = curve


    well_text = intersection.name.split("_")[0]
    text_item = pg.TextItem(text=f'{well_text}', color='white')
    font = QtGui.QFont()
    font.setPointSize(7)  # Установите размер шрифта, который вам нужен
    text_item.setFont(font)

    text_item.setPos(intersection.i_profile, -10)
    radarogramma.addItem(text_item)
    globals()[f'well_text_id{int_id}'] = text_item


def draw_boundary(bound_id, index):
    # Проверка наличия ранее добавленных графических элементов (точки) с данным bound_id
    if f'bound_scatter_id{bound_id}' in globals():
        # Удаление точки с графика, если она есть в globals()
        radarogramma.removeItem(globals()[f'bound_scatter_id{bound_id}'])
    # Проверка наличия ранее добавленных графических элементов (текста) с данным bound_id
    if f'bound_text_id{bound_id}' in globals():
        # Удаление текста с графика, если он есть в globals()
        radarogramma.removeItem(globals()[f'bound_text_id{bound_id}'])
    # Запрос на получение объекта границы из базы данных
    bound = session.query(Boundary).filter(Boundary.id == bound_id).first()
    # Получение значения средней скорости в среде
    Vmean = ui.doubleSpinBox_vsr.value()
    # Расчёт значения глубины, которое будет использоваться для отображения точки и текста на графике
    d = ((bound.depth * 100) / Vmean) / 8
    # Создание графического объекта точки с переданными параметрами
    scatter = pg.ScatterPlotItem(x=[index], y=[d], symbol='o', pen=pg.mkPen(None),
                                 brush=pg.mkBrush(255, 255, 255, 120), size=10)
    radarogramma.addItem(scatter)  # Добавление графического объекта точки на график
    globals()[f'bound_scatter_id{bound_id}'] = scatter  # Сохранение ссылки на графический объект точки в globals()

    # Создание графического объекта текста с переданными параметрами
    text_item = pg.TextItem(text=f'{bound.title} ({bound.depth})', color='white')
    text_item.setPos(index + 10, d)  # Установка позиции текста на графике
    radarogramma.addItem(text_item)  # Добавление графического объекта текста на график
    globals()[f'bound_text_id{bound_id}'] = text_item  # Сохранение ссылки на графический объект текста в globals()


#################################################################
###################### LDA MLP KNN GPC ##########################
#################################################################


def get_LDA_id():
    return ui.comboBox_lda_analysis.currentText().split(' id')[-1]


def get_MLP_id():
    return ui.comboBox_mlp_analysis.currentText().split(' id')[-1]


def get_regmod_id():
    return ui.comboBox_regmod.currentText().split(' id')[-1]


def get_lda_title():
    return ui.comboBox_lda_analysis.currentText().split(' id')[0]


def get_mlp_title():
    return ui.comboBox_mlp_analysis.currentText().split(' id')[0]


def get_regmod_title():
    return ui.comboBox_regmod.currentText().split(' id')[0]


def get_marker_id():
    return ui.comboBox_mark_lda.currentText().split(' id')[-1]


def get_marker_mlp_id():
    return ui.comboBox_mark_mlp.currentText().split(' id')[-1]


def get_marker_title():
    return ui.comboBox_mark_lda.currentText().split(' id')[0]


def get_marker_mlp_title():
    return ui.comboBox_mark_mlp.currentText().split(' id')[0]


def get_markup_id():
    if ui.listWidget_well_lda.currentItem():
        return ui.listWidget_well_lda.currentItem().text().split(' id')[-1]


def get_markup_mlp_id():
    if ui.listWidget_well_mlp.currentItem():
        return ui.listWidget_well_mlp.currentItem().text().split(' id')[-1]


def get_markup_regmod_id():
    if ui.listWidget_well_regmod.currentItem():
        return ui.listWidget_well_regmod.currentItem().text().split(' id')[-1]


def set_param_lda_to_combobox():
    for param in list_param_geovel:
        ui.comboBox_geovel_param_lda.addItem(param)


def set_param_mlp_to_combobox():
    for param in list_param_geovel:
        ui.comboBox_geovel_param_mlp.addItem(param)


def set_param_regmod_to_combobox():
    for param in list_param_geovel:
        ui.comboBox_geovel_param_reg.addItem(param)


def add_param_lda(param):
    if param.startswith('distr') or param.startswith('sep'):
        atr, count = ui.comboBox_atrib_distr_lda.currentText(), ui.spinBox_count_distr_lda.value()
        param = f'{param}_{atr}_{count}'
    elif param.startswith('mfcc'):
        atr, count = ui.comboBox_atrib_mfcc_lda.currentText(), ui.spinBox_count_mfcc.value()
        param = f'{param}_{atr}_{count}'
    new_param_lda = ParameterLDA(analysis_id=get_LDA_id(), parameter=param)
    session.add(new_param_lda)
    session.commit()


def add_param_mlp(param):
    if param.startswith('distr') or param.startswith('sep'):
        atr, count = ui.comboBox_atrib_distr_mlp.currentText(), ui.spinBox_count_distr_mlp.value()
        param = f'{param}_{atr}_{count}'
    elif param.startswith('mfcc'):
        atr, count = ui.comboBox_atrib_mfcc_mlp.currentText(), ui.spinBox_count_mfcc_mlp.value()
        param = f'{param}_{atr}_{count}'
    new_param_mlp = ParameterMLP(analysis_id=get_MLP_id(), parameter=param)
    session.add(new_param_mlp)
    session.commit()


def add_param_regmod(param):
    if param.startswith('distr') or param.startswith('sep'):
        atr, count = ui.comboBox_atrib_distr_reg.currentText(), ui.spinBox_count_distr_reg.value()
        param = f'{param}_{atr}_{count}'
    elif param.startswith('mfcc'):
        atr, count = ui.comboBox_atrib_mfcc_reg.currentText(), ui.spinBox_count_mfcc_reg.value()
        param = f'{param}_{atr}_{count}'
    new_param_regmod = ParameterReg(analysis_id=get_regmod_id(), parameter=param)
    session.add(new_param_regmod)
    session.commit()



def build_table_train(db=False, analisis='lda'):
    # Получение списка параметров
    if analisis == 'lda':
        list_param = get_list_param_lda()
        analisis_id = get_LDA_id()
        analis = session.query(AnalysisLDA).filter_by(id=get_LDA_id()).first()
    elif analisis == 'mlp':
        list_param = get_list_param_mlp()
        analisis_id = get_MLP_id()
        analis = session.query(AnalysisMLP).filter_by(id=get_MLP_id()).first()
    elif analisis == 'regmod':
        list_param = get_list_param_regmod()
        analisis_id = get_regmod_id()
        analis = session.query(AnalysisReg).filter_by(id=get_regmod_id()).first()
    # Если в базе есть сохранённая обучающая выборка, забираем ее оттуда
    if db or analis.up_data:
        if analisis == 'lda':
            data = session.query(AnalysisLDA.data).filter_by(id=get_LDA_id()).first()
        elif analisis == 'mlp':
            data = session.query(AnalysisMLP.data).filter_by(id=get_MLP_id()).first()
        elif analisis == 'regmod':
            data = session.query(AnalysisReg.data).filter_by(id=get_regmod_id()).first()

        if data[0]:
            data_train = pd.DataFrame(json.loads(data[0]))
            return data_train, list_param

    data_train, _ = build_table_test_no_db(analisis, analisis_id, list_param)
    return data_train, list_param


def build_table_test_no_db(analisis, analisis_id, list_param):

    # Если в базе нет сохранённой обучающей выборки. Создание таблицы
    if analisis == 'regmod':
        data_train = pd.DataFrame(columns=['prof_well_index', 'target_value'])
    else:
        data_train = pd.DataFrame(columns=['prof_well_index', 'mark'])

    # Получаем размеченные участки
    if analisis == 'lda':
        markups = session.query(MarkupLDA).filter_by(analysis_id=analisis_id).all()
    elif analisis == 'mlp':
        markups = session.query(MarkupMLP).filter_by(analysis_id=analisis_id).all()
    elif analisis == 'regmod':
        markups = session.query(MarkupReg).filter_by(analysis_id=analisis_id).all()

    ui.progressBar.setMaximum(len(markups))

    for nm, markup in enumerate(markups):
        # Получение списка фиктивных меток и границ слоев из разметки
        list_fake = json.loads(markup.list_fake) if markup.list_fake else []
        list_up = json.loads(markup.formation.layer_up.layer_line)
        list_down = json.loads(markup.formation.layer_down.layer_line)

        # Загрузка сигналов из профилей, необходимых для параметров 'distr', 'sep' и 'mfcc'
        for param in list_param:
            # Если параметр является расчётным
            if param.startswith('distr') or param.startswith('sep') or param.startswith('mfcc'):
                # Проверка, есть ли уже загруженный сигнал в локальных переменных
                if not str(markup.profile.id) + '_signal' in locals():
                    # Загрузка сигнала из профиля
                    locals()[str(markup.profile.id) + '_signal'] = json.loads(
                        session.query(Profile.signal).filter(Profile.id == markup.profile_id).first()[0])
            # Если параметр сохранён в базе
            else:
                # Загрузка значений параметра из формации
                locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'Formation.{param}')).filter(
                    Formation.id == markup.formation_id).first()[0])


        # Обработка каждого измерения в разметке
        for measure in json.loads(markup.list_measure):
            # Пропустить измерение, если оно является фиктивным
            if measure in list_fake:
                continue

            dict_value = {}
            dict_value['prof_well_index'] = f'{markup.profile_id}_{markup.well_id}_{measure}'
            if analisis == 'regmod':
                dict_value['target_value'] = markup.target_value
            else:
                dict_value['mark'] = markup.marker.title

            # Обработка каждого параметра в списке параметров
            for param in list_param:

                if param.startswith('distr'):
                    # Обработка параметра 'distr'
                    p, atr, n = param.split('_')[0], param.split('_')[1], int(param.split('_')[2])
                    sig_measure = calc_atrib_measure(locals()[str(markup.profile.id) + '_signal'][measure], atr)
                    distr = get_distribution(sig_measure[list_up[measure]: list_down[measure]], n)
                    for num in range(n):
                        dict_value[f'{p}_{atr}_{num + 1}'] = distr[num]
                elif param.startswith('sep'):
                    # Обработка параметра 'sep'
                    p, atr, n = param.split('_')[0], param.split('_')[1], int(param.split('_')[2])
                    sig_measure = calc_atrib_measure(locals()[str(markup.profile.id) + '_signal'][measure], atr)
                    sep = get_mean_values(sig_measure[list_up[measure]: list_down[measure]], n)
                    for num in range(n):
                        dict_value[f'{p}_{atr}_{num + 1}'] = sep[num]
                elif param.startswith('mfcc'):
                    # Обработка параметра 'mfcc'
                    p, atr, n = param.split('_')[0], param.split('_')[1], int(param.split('_')[2])
                    sig_measure = calc_atrib_measure(locals()[str(markup.profile.id) + '_signal'][measure], atr)
                    mfcc = get_mfcc(sig_measure[list_up[measure]: list_down[measure]], n)
                    for num in range(n):
                        dict_value[f'{p}_{atr}_{num + 1}'] = mfcc[num]
                else:
                    # Загрузка значения параметра из списка значений
                    dict_value[param] = locals()[f'list_{param}'][measure]

            # Добавление данных в обучающую выборку
            data_train = pd.concat([data_train, pd.DataFrame([dict_value])], ignore_index=True)

        ui.progressBar.setValue(nm + 1)
    data_train_to_db = json.dumps(data_train.to_dict())
    if analisis == 'lda':
        session.query(AnalysisLDA).filter_by(id=analisis_id).update({'data': data_train_to_db, 'up_data': True}, synchronize_session='fetch')
    elif analisis == 'mlp':
        session.query(AnalysisMLP).filter_by(id=analisis_id).update({'data': data_train_to_db, 'up_data': True}, synchronize_session='fetch')
    elif analisis == 'regmod':
        session.query(AnalysisReg).filter_by(id=analisis_id).update({'data': data_train_to_db, 'up_data': True}, synchronize_session='fetch')
    session.commit()
    return data_train, list_param


def build_table_test(analisis='lda'):
    if analisis == 'lda':
        list_param, analisis_title = get_list_param_lda(), ui.comboBox_lda_analysis.currentText()
    elif analisis == 'mlp':
        list_param, analisis_title = get_list_param_mlp(), ui.comboBox_mlp_analysis.currentText()
    elif analisis == 'regmod':
        model = session.query(TrainedModelReg).filter_by(id=ui.listWidget_trained_model_reg.currentItem().data(Qt.UserRole)).first()
        list_param, analisis_title = json.loads(model.list_params), model.title
    test_data = pd.DataFrame(columns=['prof_index', 'x_pulc', 'y_pulc'])
    curr_form = session.query(Formation).filter(Formation.id == get_formation_id()).first()
    list_up = json.loads(curr_form.layer_up.layer_line)
    list_down = json.loads(curr_form.layer_down.layer_line)
    x_pulc = json.loads(curr_form.profile.x_pulc)
    y_pulc = json.loads(curr_form.profile.y_pulc)
    for param in list_param:
        if param.startswith('distr') or param.startswith('sep') or param.startswith('mfcc'):
            if not str(curr_form.profile.id) + '_signal' in locals():
                locals()[str(curr_form.profile.id) + '_signal'] = json.loads(
                    session.query(Profile.signal).filter(Profile.id == curr_form.profile_id).first()[0])
        else:
            locals()[f'list_{param}'] = json.loads(getattr(curr_form, param))
    ui.progressBar.setMaximum(len(list_up))
    set_info(f'Процесс сбора параметров {analisis_title} по профилю {curr_form.profile.title}',
             'blue')
    for i in range(len(list_up)):
        dict_value = {}
        for param in list_param:
            if param.startswith('distr'):
                p, atr, n = param.split('_')[0], param.split('_')[1], int(param.split('_')[2])
                sig_measure = calc_atrib_measure(locals()[str(curr_form.profile.id) + '_signal'][i], atr)
                distr = get_distribution(sig_measure[list_up[i]: list_down[i]], n)
                for num in range(n):
                    dict_value[f'{p}_{atr}_{num + 1}'] = distr[num]
            elif param.startswith('sep'):
                p, atr, n = param.split('_')[0], param.split('_')[1], int(param.split('_')[2])
                sig_measure = calc_atrib_measure(locals()[str(curr_form.profile.id) + '_signal'][i], atr)
                sep = get_mean_values(sig_measure[list_up[i]: list_down[i]], n)
                for num in range(n):
                    dict_value[f'{p}_{atr}_{num + 1}'] = sep[num]
            elif param.startswith('mfcc'):
                p, atr, n = param.split('_')[0], param.split('_')[1], int(param.split('_')[2])
                sig_measure = calc_atrib_measure(locals()[str(curr_form.profile.id) + '_signal'][i], atr)
                mfcc = get_mfcc(sig_measure[list_up[i]: list_down[i]], n)
                for num in range(n):
                    dict_value[f'{p}_{atr}_{num + 1}'] = mfcc[num]
            else:
                dict_value[param] = locals()[f'list_{param}'][i]
        dict_value['prof_index'] = f'{curr_form.profile_id}_{i}'
        test_data = pd.concat([test_data, pd.DataFrame([dict_value])], ignore_index=True)
        ui.progressBar.setValue(i + 1)
    test_data['x_pulc'] = x_pulc
    test_data['y_pulc'] = y_pulc
    return test_data, curr_form


def get_list_marker():
    markers = session.query(MarkerLDA).filter_by(analysis_id=get_LDA_id()).all()
    return [m.title for m in markers]


def get_list_marker_mlp():
    markers = session.query(MarkerMLP).filter_by(analysis_id=get_MLP_id()).all()
    return [m.title for m in markers]


def get_list_param_lda():
    parameters = session.query(ParameterLDA).filter_by(analysis_id=get_LDA_id()).all()
    return [p.parameter for p in parameters]


def get_list_param_mlp():
    parameters = session.query(ParameterMLP).filter_by(analysis_id=get_MLP_id()).all()
    return [p.parameter for p in parameters]


def get_list_param_regmod():
    parameters = session.query(ParameterReg).filter_by(analysis_id=get_regmod_id()).all()
    return [p.parameter for p in parameters]


def get_working_data_lda():
    data_train, list_param = build_table_train(True)
    list_param_lda = data_train.columns.tolist()[2:]
    training_sample = data_train[list_param_lda].values.tolist()
    markup = sum(data_train[['mark']].values.tolist(), [])
    clf = LinearDiscriminantAnalysis()
    try:
        trans_coef = clf.fit(training_sample, markup).transform(training_sample)
    except ValueError:
        set_info(f'Ошибка в расчетах LDA! Возможно значения одного из параметров отсутствуют в интервале обучающей '
                 f'выборки.', 'red')
        return
    data_trans_coef = pd.DataFrame(trans_coef)
    data_trans_coef['mark'] = data_train['mark'].values.tolist()
    data_trans_coef['shape'] = ['train']*len(data_trans_coef)
    list_cat = list(clf.classes_)
    working_data, curr_form = build_table_test()
    profile_title = session.query(Profile.title).filter_by(id=working_data['prof_index'][0].split('_')[0]).first()[0][0]
    set_info(f'Процесс расчёта LDA. {ui.comboBox_lda_analysis.currentText()} по профилю {profile_title}', 'blue')
    try:
        new_trans_coef = clf.transform(working_data.iloc[:, 3:])
        new_mark = clf.predict(working_data.iloc[:, 3:])
        probability = clf.predict_proba(working_data.iloc[:, 3:])
    except ValueError:
        data = imputer.fit_transform(working_data.iloc[:, 3:])
        new_trans_coef = clf.transform(data)
        new_mark = clf.predict(data)
        probability = clf.predict_proba(data)
        for i in working_data.index:
            p_nan = [working_data.columns[ic + 3] for ic, v in enumerate(working_data.iloc[i, 3:].tolist()) if
                     np.isnan(v)]
            if len(p_nan) > 0:
                set_info(f'Внимание для измерения "{i}" отсутствуют параметры "{", ".join(p_nan)}", поэтому расчёт для'
                         f' этого измерения может быть не корректен', 'red')
    working_data = pd.concat([working_data, pd.DataFrame(probability, columns=list_cat)], axis=1)
    working_data['mark'] = new_mark
    test_data_trans_coef = pd.DataFrame(new_trans_coef)
    test_data_trans_coef['mark'] = new_mark
    test_data_trans_coef['shape'] = ['test'] * len(new_trans_coef)
    data_trans_coef = pd.concat([data_trans_coef, test_data_trans_coef], ignore_index=True)
    return working_data, data_trans_coef, curr_form


def get_distribution(values: list, n: int) -> list:
    # Находим минимальное и максимальное значения в наборе данных
    min_val = min(values)
    max_val = max(values)
    # Вычисляем размер интервала
    interval = (max_val - min_val) / n
    if interval == 0:
        return [len(values)] + [0] * (n - 1)
    # Создаем список, который будет содержать количество значений в каждом интервале
    distribution = [0] * n
    # Итерируем по значениям и распределяем их по соответствующему интервалу
    for value in values:
        index = int((value - min_val) / interval)
        # Если значение попадает в последний интервал, то оно приравнивается к последнему интервалу
        if index == n:
            index -= 1
        distribution[index] += 1
    # Возвращаем количество значений в каждом интервале
    return distribution


def get_mean_values(values: list, n: int) -> list:
    mean_values = []
    start = 0
    intd = int(round(len(values) / n, 0))
    for i in range(n):
        end = start + intd  # вычисление конца интервала
        mean = np.mean(values[start:end]) # вычисление среднего значения в интервале
        mean_values.append(mean)
        start = end  # начало следующего интервала
    return mean_values


def get_mfcc(values: list, n: int):
    return list(mfcc(signal=np.array(values), samplerate=125000000, winlen=0.000004104, nfilt=513, nfft=513, numcep=n)[0])


def check_list_lengths(list_of_lists):
    """ Проверяет, что все остальные элементы имеют равную длину """
    # Получаем длину первого элемента в списке
    first_length = len(list_of_lists[0])

    # Проверяем, что все остальные элементы имеют такую же длину
    all_same_length = all(len(lst) == first_length for lst in list_of_lists[1:])

    return all_same_length


def string_to_unique_number(strings, type_analysis):
    unique_strings = {}  # Словарь для хранения уникальных строк и их численного представления
    result = []  # Список для хранения результата
    if type_analysis == 'lda':
        markers = session.query(MarkerLDA).filter(MarkerLDA.analysis_id == get_LDA_id()).all()
    elif type_analysis == 'mlp':
        markers = session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).all()

    for marker in markers:
        num = len(unique_strings) + 1
        unique_strings[marker.title] = num
    for s in strings:
        result.append(unique_strings[s])

    return result


def calc_distance(x1, y1, x2, y2):
    """ Функция для вычисления расстояния между двумя точками """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def find_intersection_points(list_x1, list_y1, list_x2, list_y2):
    """ Функция для поиска точек пересечения и выбора ближайшей """
    intersection_points = []

    # Перебираем все пары соседних точек в профилях
    for i in range(len(list_x1) - 1):
        for j in range(len(list_x2) - 1):
            x1, y1 = list_x1[i], list_y1[i]
            x2, y2 = list_x1[i + 1], list_y1[i + 1]
            x3, y3 = list_x2[j], list_y2[j]
            x4, y4 = list_x2[j + 1], list_y2[j + 1]

            # Вычисляем знаменатель для проверки пересечения отрезков
            den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

            # Если отрезки пересекаются, вычисляем координаты точки пересечения
            if den != 0:
                ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
                ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den

                # Проверяем, что точка пересечения находится на отрезках
                if 0 <= ua <= 1 and 0 <= ub <= 1:
                    intersect_x = x1 + ua * (x2 - x1)
                    intersect_y = y1 + ua * (y2 - y1)
                    intersection_points.append((intersect_x, intersect_y, i, j))
    return intersection_points

    # # Если нет точек пересечения, возвращаем пустой список
    # if not intersection_points:
    #     return []
    #
    # # Находим ближайшую точку пересечения к началу координат
    # closest_point = min(intersection_points, key=lambda p: calc_distance(0, 0, p[0], p[1]))
    # return closest_point[2], closest_point[3]


def get_attributes():
    pass
    # attr = session.query()


def calc_object_measures():
    """ Расчет количества измерений на объекте """
    count_measure = 0
    for i in session.query(Profile).filter_by(research_id=get_research_id()).all():
        count_measure += len(json.loads(i.signal))
    return count_measure


def calc_fft_attributes(signal):
    """ Расчет аттрибутов спектра Фурье """

    if len(signal) < 3:
        return np.NaN, np.NaN, np.NaN, np.NaN

    # Вычислите преобразование Фурье сигнала
    # result = dct(signal)
    result = rfft(signal)
    # result_real = result.real
    # print(result)


    # Найдите модуль комплексных чисел преобразования Фурье
    magnitude = np.abs(result)
    # magnitude_real = np.abs(result_real)
    # plt.plot(magnitude[1:])
    # plt.plot(magnitude_real[1:])
    # plt.plot(result_real[1:])
    # plt.show()

    # Нормализуйте спектр, разделив на длину сигнала
    normalized_magnitude = magnitude / len(signal)

    # Рассчитайте площадь нормированного спектра
    area_under_spectrum = np.sum(normalized_magnitude)

    # Получите частоты, соответствующие спектру
    n = len(signal)
    frequencies = rfftfreq(n, 8E-9)

    # Рассчитайте средневзвешенную частоту
    weighted_average_frequency = np.sum(frequencies * magnitude) / np.sum(magnitude)

    # Найдите максимальную амплитуду и соответствующую центральную частоту
    max_amplitude = np.max(magnitude[1:])
    index_max_amplitude = list(magnitude[1:]).index(max_amplitude) + 1
    central_frequency = frequencies[index_max_amplitude]

    # Определите пороговое значение
    threshold = 0.7 * max_amplitude

    left_index, right_index = 0, len(frequencies) - 1
    for i in range(index_max_amplitude, -1, -1):
        if magnitude[i] < threshold:
            left_index = i
            break
    for i in range(index_max_amplitude, len(frequencies)):
        if magnitude[i] < threshold:
            right_index = i
            break

    # Найдите ширину спектра
    spectrum_width = frequencies[right_index] - frequencies[left_index]

    # Вычислите Q-фактор
    q_factor = central_frequency / spectrum_width

    return area_under_spectrum, weighted_average_frequency, q_factor, area_under_spectrum / weighted_average_frequency


def calc_fft_attributes_profile(signals, top, bottom):

    list_Sn, list_fcb, list_Q_f, list_Sn_fcb = [], [], [], []

    for i in range(len(top)):
        signal = signals[i]
        form_signal = signal[top[i]:bottom[i]]
        try:
            Sn, fcb, Q_f, Sn_fcb = calc_fft_attributes(form_signal)
            list_Sn.append(Sn)
            list_fcb.append(fcb)
            list_Q_f.append(Q_f)
            list_Sn_fcb.append(Sn_fcb)
        except ValueError:
            list_Sn.append(np.NaN)
            list_fcb.append(np.NaN)
            list_Q_f.append(np.NaN)
            list_Sn_fcb.append(np.NaN)
            print('ValueError')
        except TypeError:
            list_Sn.append(np.NaN)
            list_fcb.append(np.NaN)
            list_Q_f.append(np.NaN)
            list_Sn_fcb.append(np.NaN)
            print('TypeError')
        except IndexError:
            list_Sn.append(np.NaN)
            list_fcb.append(np.NaN)
            list_Q_f.append(np.NaN)
            list_Sn_fcb.append(np.NaN)
            print('IndexError')

    return list_Sn, list_fcb, list_Q_f, list_Sn_fcb


def calc_correlation_profile():
    pass
