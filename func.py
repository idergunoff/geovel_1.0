from object import *


# Функция добавления информации в окно информации с указанием времени и цвета текста
def set_info(text, color):
    ui.info.append(f'{datetime.datetime.now().strftime("%H:%M:%S")}:\t<span style =\"color:{color};\" >{text}</span>')


# Функция получения id выбранного объекта
def get_object_id():
    try:
        return int(ui.comboBox_object.currentText().split('id')[-1])
    except ValueError:
        pass


# Функция получения имени выбранного объекта
def get_object_name():
    return ui.comboBox_object.currentText().split(' id')[0]


# Функция получения id выбранного профиля
def get_profile_id():
    return int(ui.comboBox_profile.currentText().split('id')[-1])


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
    for i in ui.widget_layer_radio.findChildren(QtWidgets.QRadioButton):
        i.deleteLater()


def get_layer_id():
    # Найти id выбранного слоя в списке радиокнопок
    for i in ui.widget_layer_radio.findChildren(QtWidgets.QRadioButton):
        if i.isChecked():
            return int(i.text())


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
            globals()[f'scatter_id{i.text().split(" id")[-1]}'] = globals()[f'scatter_id{i.text().split(" id")[-1]}']
            radarogramma.removeItem(globals()[f'scatter_id{i.text().split(" id")[-1]}'])
        if f'curve_id{i.text().split(" id")[-1]}' in globals():
            globals()[f'curve_id{i.text().split(" id")[-1]}'] = globals()[f'curve_id{i.text().split(" id")[-1]}']
            radarogramma.removeItem(globals()[f'curve_id{i.text().split(" id")[-1]}'])
        if f'text_id{i.text().split(" id")[-1]}' in globals():
            globals()[f'text_id{i.text().split(" id")[-1]}'] = globals()[f'text_id{i.text().split(" id")[-1]}']
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
                                     brush=pg.mkBrush(255, 255, 255, 120), size=10)
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
        ui.checkBox_new = QtWidgets.QCheckBox()
        ui.radio_new = QtWidgets.QRadioButton()
        # Задать текст для флажка и радиокнопки
        ui.checkBox_new.setText(f'{lay.layer_title} id{lay.id}')
        ui.radio_new.setText(f'{lay.id}')
        # Добавить флажок и радиокнопку на соответствующие макеты
        ui.verticalLayout_layer.addWidget(ui.checkBox_new)
        ui.verticalLayout_layer_radio.addWidget(ui.radio_new)
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
    if session.query(Profile.T_top).filter(Profile.id == get_profile_id()).first()[0]:
        ui.comboBox_plast.addItem('KROT')
    # Добавляем все формации в список выбора пластов
    for form in formations:
        ui.comboBox_plast.addItem(f'{form.title} id{form.id}')
    # Обновляем список выбора параметров для выбранного пласта
    update_param_combobox()


def update_profile_combobox():
    """ Обновление списка профилей в выпадающем списке """
    # Очистка выпадающего списка
    ui.comboBox_profile.clear()
    try:
        # Запрос на получение всех профилей, относящихся к объекту, и их добавление в выпадающий список
        for i in session.query(Profile).filter(Profile.object_id == get_object_id()).all():
            count_measure = len(json.loads(session.query(Profile.signal).filter(Profile.id == i.id).first()[0]))
            ui.comboBox_profile.addItem(f'{i.title} ({count_measure} измерений) id{i.id}')
        # Обновление списка формирований
        update_formation_combobox()
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


def update_object():
    """ Функция для обновления списка объектов в выпадающем списке """
    # Очистка выпадающего списка объектов
    ui.comboBox_object.clear()
    # Получение всех объектов из базы данных, отсортированных по дате исследования
    for i in session.query(GeoradarObject).order_by(GeoradarObject.date_exam).all():
        # Добавление названия объекта, даты исследования и идентификатора объекта в выпадающий список
        ui.comboBox_object.addItem(f'{i.title} {i.date_exam.strftime("%m.%Y")} id{i.id}')
    # Обновление выпадающего списка профилей
    update_profile_combobox()


def update_param_combobox():
    """Обновление выпадающего списка параметров формаций/пластов"""
    current_text = ui.comboBox_param_plast.currentText()  # сохраняем текущий текст в комбобоксе
    ui.comboBox_param_plast.clear()  # очищаем комбобокс
    if ui.comboBox_plast.currentText() == '-----':  # если выбрана пустая строка, то ничего не делаем
        pass
    elif ui.comboBox_plast.currentText() == 'KROT':  # если выбран КРОТ, то добавляем в комбобокс параметры профиля
        list_columns = Profile.__table__.columns.keys()  # список параметров таблицы профиля
        # удаляем не нужные колонки
        [list_columns.remove(i) for i in ['id', 'object_id', 'title', 'x_wgs', 'y_wgs', 'x_pulc', 'y_pulc', 'signal']]
        for i in list_columns:
            # если в таблице профиля есть хотя бы одна запись, где значение параметра не NULL, то добавляем параметр в комбобокс
            if session.query(Profile).filter(text(f"profile_id=:p_id and {i} NOT NULL")).params(p_id=get_profile_id()).count() > 0:
                ui.comboBox_param_plast.addItem(i)
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
        # Получаем данные для текущего пласта
        graph = json.loads(session.query(literal_column(f'Profile.{param}')).filter(Profile.id == get_profile_id()).first()[0])
        # Создаем список значений по порядку
        number = list(range(1, len(graph) + 1))
        # Создаем кривую и кривую, отфильтрованную с помощью savgol_filter
        curve = pg.PlotCurveItem(x=number, y=graph)
        wl = 2.4 if ui.comboBox_plast.currentText() == 'KROT' else 1
        cl = 'red' if ui.comboBox_plast.currentText() == 'KROT' else 'green'
        curve_filter = pg.PlotCurveItem(x=number, y=savgol_filter(graph, 31, 3),
                                        pen=pg.mkPen(color=cl, width=wl))
        # Если выбран пласт КРОТ, то добавляем только кривую на график
        if ui.comboBox_plast.currentText() == 'KROT':
            ui.graph.addItem(curve)
        # Добавляем кривую и отфильтрованную кривую на график для всех пластов
        ui.graph.addItem(curve_filter)
        # Для каждого пласта
        for f in session.query(Formation).filter(Formation.profile_id == get_profile_id()).all():
            # Если выбран параметр ширина, верхняя или нижняя граница пласта, то не рисуем кривые
            if param in ['width', 'top', 'land']:
                return
            # Получаем данные для текущего пласта
            graph = json.loads(session.query(literal_column(f'Formation.{param}')).filter(Formation.id == f.id).first()[0])
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
        # если текущий выбранный элемент равен 'KROT', то получаем данные для профиля
        elif ui.comboBox_plast.currentText() == 'KROT':
            # получаем данные для выбранного параметра из таблицы Profile и преобразуем их из строки в список с помощью json.loads()
            graph = json.loads(session.query(literal_column(f'Profile.{param}')).filter(
                Profile.id == get_profile_id()).first()[0])
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

