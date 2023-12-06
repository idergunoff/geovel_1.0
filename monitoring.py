from func import *
from krige import draw_map


def update_list_object_monitor():
    """Обновить список объектов"""
    ui.comboBox_object_monitor.clear()
    for i in session.query(GeoradarObject).order_by(GeoradarObject.title).all():
        if len(i.researches) > 1:
            ui.comboBox_object_monitor.addItem(f'{i.title}')
            ui.comboBox_object_monitor.setItemData(ui.comboBox_object_monitor.count() - 1, {'id': i.id})
    update_list_h_well()


def update_list_h_well():
    """Обновить список горизонтальных скважин"""
    ui.listWidget_h_well.clear()
    for h_well in session.query(GeoradarObject).filter_by(id=get_obj_monitor_id()).first().h_wells:
        count_therm = session.query(Thermogram).filter_by(h_well_id=h_well.id).count()
        item_text = f'{h_well.title}\t+{count_therm} термограмм' if count_therm > 0 else h_well.title
        item = QListWidgetItem(item_text)
        if count_therm > 0:
            therm = session.query(Thermogram).filter_by(h_well_id=h_well.id).first()
            incl = False
            for i in json.loads(therm.therm_data):
                if len(i) > 2:
                    item.setBackground(QBrush(QColor('#ADFCDF')))
                    incl = True
                    break
            if not incl:
                item.setBackground(QBrush(QColor('#FBD59E')))
        item.setData(Qt.UserRole, h_well.id)
        ui.listWidget_h_well.addItem(item)
    ui.listWidget_h_well.sortItems()


def update_list_param_h_well():
    """Обновить список параметров горизонтальных скважин"""
    ui.listWidget_param_h_well.clear()
    h_well = session.query(HorizontalWell).filter_by(id=get_h_well_id()).first()
    if h_well and h_well.x_coord:
        item = QListWidgetItem(f'X: {round(h_well.x_coord, 2)}')
        item.setData(Qt.UserRole, None)
        item.setBackground(QBrush(QColor('#FBD59E')))
        ui.listWidget_param_h_well.addItem(item)
        item = QListWidgetItem(f'Y: {round(h_well.y_coord, 2)}')
        item.setData(Qt.UserRole, None)
        item.setBackground(QBrush(QColor('#FBD59E')))
        ui.listWidget_param_h_well.addItem(item)
        item = QListWidgetItem(f'Alt: {round(h_well.alt, 2)}')
        item.setData(Qt.UserRole, None)
        item.setBackground(QBrush(QColor('#FBD59E')))
        ui.listWidget_param_h_well.addItem(item)
    incl = session.query(ParameterHWell).filter_by(h_well_id=get_h_well_id(), parameter='Инклинометрия').first()
    if incl:
        length = max([p[3] for p in json.loads(incl.data)])
        item = QListWidgetItem(f'{incl.parameter} - {round(length, 1)} m.')
        item.setData(Qt.UserRole, incl.id)
        item.setBackground(QBrush(QColor('#ADFCDF')))
        ui.listWidget_param_h_well.addItem(item)
    for p in session.query(ParameterHWell).filter_by(h_well_id=get_h_well_id()).all():
        if p.parameter == 'Инклинометрия':
            continue
        item = QListWidgetItem(p.parameter)
        item.setData(Qt.UserRole, p.id)
        ui.listWidget_param_h_well.addItem(item)


def remove_parameter():
    """Удалить параметр"""
    result = QtWidgets.QMessageBox.question(
        MainWindow,
        'Удаление параметра',
        f'Вы уверены, что хотите удалить параметр {ui.listWidget_param_h_well.currentItem().text()}?',
        QtWidgets.QMessageBox.Yes,
        QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        id_param = ui.listWidget_param_h_well.currentItem().data(Qt.UserRole)
        # print(id_param)
        session.query(ParameterHWell).filter_by(id=id_param).delete()
        session.commit()
        update_list_param_h_well()
    else:
        pass


def add_h_well():
    """Добавить горизонтальную скважину"""
    if ui.lineEdit_string.text() == '':
        set_info('Внимание! Введите название горизонтальной скважины в поле ввода в верхней части окна', 'red')
        return
    result = QtWidgets.QMessageBox.question(
        MainWindow,
        'Добавление горизонтальной скважины',
        f'Вы уверены, что хотите добавить горизонтальную скважину {ui.lineEdit_string.text()} для объекта'
        f' {ui.comboBox_object_monitor.currentText()}?',
        QtWidgets.QMessageBox.Yes,
        QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        h_well = HorizontalWell(title=ui.lineEdit_string.text(), object_id=get_obj_monitor_id())
        session.add(h_well)
        session.commit()
        set_info(f'Горизонтальная скважина "{h_well.title}" добавлена', 'green')
        update_list_h_well()
    else:
        set_info('Горизонтальная скважина не добавлена', 'blue')


def remove_h_well():
    """Удалить горизонтальную скважину"""
    hw_id = get_h_well_id()
    if hw_id is None:
        set_info('Внимание! Не выбрана горизонтальная скважина', 'red')
        return
    result = QtWidgets.QMessageBox.question(
        MainWindow,
        'Удаление горизонтальной скважины',
        f'Вы уверены, что хотите удалить горизонтальную скважину {get_h_well_title()} вместе со всеми параметрами и данными термометрии?',
        QtWidgets.QMessageBox.Yes,
        QtWidgets.QMessageBox.No
    )
    if result == QtWidgets.QMessageBox.Yes:
        result_new = QtWidgets.QMessageBox.question(
            MainWindow,
            'Последнее предупреждение об удаледении',
            'Вы абсолютно уверены, что хотите удалить горизонтальную скважину? Для загрузки данных было затрачено огромное количество времени и сил.',
            QtWidgets.QMessageBox.Yes,
            QtWidgets.QMessageBox.No
        )
        if result_new == QtWidgets.QMessageBox.Yes:
            session.query(HorizontalWell).filter_by(id=hw_id).delete()
            session.commit()
            set_info(f'Горизонтальная скважина "{get_h_well_title()}" удалена', 'green')
            update_list_h_well()
        if result_new == QtWidgets.QMessageBox.No:
            set_info('Горизонтальная скважина не удалена', 'blue')
    if result == QtWidgets.QMessageBox.No:
        set_info('Горизонтальная скважина не удалена', 'blue')


def edit_h_well():
    """Редактировать горизонтальную скважину"""
    if ui.lineEdit_string.text() == '':
        set_info('Внимание! Введите новое название горизонтальной скважины в поле ввода в верхней части окна', 'red')
        return
    hw_id = get_h_well_id()
    if hw_id is None:
        set_info('Внимание! Не выбрана горизонтальная скважина', 'red')
        return
    result = QtWidgets.QMessageBox.question(
        MainWindow,
        'Редактирование горизонтальной скважины',
        f'Вы уверены, что хотите изменить название горизонтальной скважины '
        f'"{get_h_well_title()}" на "{ui.lineEdit_string.text()}"?',
        QtWidgets.QMessageBox.Yes,
        QtWidgets.QMessageBox.No
    )
    if result == QtWidgets.QMessageBox.Yes:
        session.query(HorizontalWell).filter_by(id=hw_id).update({'title': ui.lineEdit_string.text()}, synchronize_session='fetch')
        session.commit()
        set_info(f'Название горизонтальной скважины "{get_h_well_title()}" изменено на "{ui.lineEdit_string.text()}"', 'green')
        update_list_h_well()
    if result == QtWidgets.QMessageBox.No:
        pass


def  load_param_h_well():
    """Загрузить параметры горизонтальных скважин"""
    file_name = QFileDialog.getOpenFileName(MainWindow, 'Выбрать файл', '', 'Excel files (*.xls *.xlsx)')[0]
    if not file_name:
        return
    set_info(f'Загрузка параметров скважин из файла "{file_name.split("/")[-1]}"', 'green')
    pd_param = pd.read_excel(file_name, sheet_name=None, header=None, index_col=None)

    ui.progressBar.setMaximum(len(pd_param))
    sheet = 0

    for skv in pd_param:
        set_info(f'Загрузка параметров скважины "{skv}"', 'blue')
        list_skv, list_h_well_id = [skv_name for skv_name in skv.split('-')], []
        for h_well_name in list_skv:
            h_well = session.query(HorizontalWell).filter_by(title=h_well_name, object_id=get_obj_monitor_id()).first()
            if not h_well:
                new_h_well = HorizontalWell(title=h_well_name, object_id=get_obj_monitor_id())
                session.add(new_h_well)
                session.commit()
                list_h_well_id.append(new_h_well.id)
            else:
                list_h_well_id.append(h_well.id)

        pd_skv = pd_param[skv]

        # найти строку с заголовками столбцов, содержащую слово "Дата"
        header_row = pd_skv[pd_skv.eq('Дата').any(axis=1)].index[0]

        # удалить строки до строки с заголовками столбцов
        pd_skv = pd_skv.iloc[header_row:]

        # оставить только строки, содержащие хотя бы одно ненулевое значение
        pd_skv = pd_skv.dropna(how='all')

        # сброс индексов
        pd_skv = pd_skv.reset_index(drop=True)

        # получить индексы столбцов с датами
        list_date = [i for i, val in enumerate(pd_skv.iloc[0].tolist()) if val == 'Дата']

        for n_col, col_name in enumerate(pd_skv.iloc[0]):
            if isinstance(col_name, str):
                if col_name not in ['Дата', 'Время работы']:

                    if len(list_skv) > 1:
                        if n_col > list_date[1]:
                            h_well_id = list_h_well_id[1]
                        else:
                            h_well_id = list_h_well_id[0]
                    else:
                        h_well_id = list_h_well_id[0]

                    try:
                        dict_param = {str(pd_skv.iloc[i][list_date[0]].date()): pd_skv.iloc[i][n_col]
                                      for i in pd_skv.index
                                      if i > 0 and
                                      not np.isnan(pd_skv.iloc[i][n_col]) and
                                      isinstance(pd_skv.iloc[i][list_date[0]], datetime.datetime)}
                        if set(dict_param.values()) != {0} and len(dict_param) > 0:
                            param_h_well = session.query(ParameterHWell).filter_by(h_well_id=h_well_id,
                                                                                   parameter=col_name).first()
                            if param_h_well:
                                data_param = json.loads(param_h_well.data)
                                data_param.update(dict_param)
                                session.query(ParameterHWell).filter_by(
                                    h_well_id=h_well_id,
                                    parameter=col_name
                                ).update({'data': json.dumps(data_param)}, synchronize_session='fetch')
                                # session.commit()
                            else:
                                new_param_h_well = ParameterHWell(
                                    h_well_id=h_well_id,
                                    parameter=col_name,
                                    data=json.dumps(dict_param)
                                )
                                session.add(new_param_h_well)
                                # session.commit()
                    except TypeError:
                        set_info(f'Ошибка при загрузке параметра "{col_name}" скважины "{skv}"', 'red')
            session.commit()
        sheet += 1
        ui.progressBar.setValue(sheet)
    set_info('Параметры горизонтальных скважин загружены', 'green')
    update_list_h_well()

    # добавить все столбцы, кроме "Дата" и "Время работы", в выпадающий список параметров
        # list_param = []
        # for i, col_name in enumerate(pd_skv.iloc[0]):
        #     if isinstance(col_name, str):
        #         if col_name not in ['Дата', 'Время работы']:
        #             list_param.append(f"{i}. {col_name}")
        # # print(list_param)


def get_obj_monitor_id():
    """Получить id объекта мониторинга"""
    return ui.comboBox_object_monitor.itemData(ui.comboBox_object_monitor.currentIndex())['id']


def get_h_well_id():
    """Получить id горизонтальной скважины"""
    item = ui.listWidget_h_well.currentItem()
    if item:
        return item.data(Qt.UserRole)


def get_h_well_title():
    """Получить название горизонтальной скважины"""
    item = ui.listWidget_h_well.currentItem()
    if item:
        return item.text().split('\t')[0]


def get_therm_id():
    """Получить id термограммы"""
    item = ui.listWidget_thermogram.currentItem()
    if item:
        return item.data(Qt.UserRole)


def load_inclinometry_h_well():
    """Загрузить инклинометрические данные горизонтальных скважин"""
    file_dir = QFileDialog.getExistingDirectory(MainWindow, 'Выбрать папку с инклинометрическими данными')
    for file in os.listdir(file_dir):
        if file.endswith('.txt'):
            h_well = session.query(HorizontalWell).filter_by(object_id=get_obj_monitor_id(), title=file.split('.')[0]).first()
            if not h_well:
                set_info(f'Не найдена горизонтальная скважина "{file.split(".")[0]}" для {get_obj_monitor_name()}', 'red')
                continue
            file_name = os.path.join(file_dir, file)
            if not h_well.x_coord or not h_well.y_coord or not h_well.alt:
                set_info(f'Не найдены координаты устья горизонтальной скважины "{file.split(".")[0]}" для {get_obj_monitor_name()}', 'red')
                continue
            inclinometry = calc_inclinometry(file_name, [h_well.x_coord, h_well.y_coord, h_well.alt])
            new_param_h_well = ParameterHWell(
                h_well_id=h_well.id,
                parameter='Инклинометрия',
                data=json.dumps(inclinometry)
            )
            session.add(new_param_h_well)
    session.commit()
    update_list_param_h_well()


def load_thermogram_h_well():
    """Загрузить термограммы горизонтальных скважин"""
    h_well_id = get_h_well_id()
    if not h_well_id:
        set_info('Не выбрана горизонтальная скважина', 'red')
        return
    file_dir = QFileDialog.getExistingDirectory(
        MainWindow, f'Выбрать папку с термограммами скважины {ui.listWidget_h_well.currentItem().text()}')
    if not file_dir:
        return
    ui.progressBar.setMaximum(len(os.listdir(file_dir)))
    n_load = 0
    for n_file, file in enumerate(os.listdir(file_dir)):
        ui.progressBar.setValue(n_file)
        set_info(f'Загружается термограмма {file}', 'blue')
        if file.endswith('.las'):
            try:
                las = ls.read(os.path.join(file_dir, file))
            except KeyError:
                set_info(f'Ошибка при чтении файла {file}', 'red')
                continue
            # todo проверка совпадения названия
            if str(las.well['WELL'].value) != ui.listWidget_h_well.currentItem().text() and ui.checkBox_check_name.isChecked():
                set_info(f'Выбраная скважина ({ui.listWidget_h_well.currentItem().text()}) не совпадает с указанной '
                         f'в las-файле - {las.well["WELL"].value}', 'red')
                continue
            date_time = datetime.datetime.strptime(las.well['DATE'].value, '%d.%m.%Y %H-%M-%S')
            date_notime = date_time.date()
            if session.query(Thermogram).filter(
                    Thermogram.h_well_id == h_well_id,
                    func.DATE(Thermogram.date_time) == date_notime).first():
                set_info(f'Термограмма {file} уже существует', 'red')
                continue
            add_update_therm_to_db(h_well_id, date_time, list(las['DEPTH']), list(las['TEMP']))
            n_load += 1
        elif file.endswith('.xls') or file.endswith('.xlsx'):
            xl = pd.read_excel(os.path.join(file_dir, file))
            date_row = 0
            for i, row in xl.iterrows():
                try:
                    pd.to_datetime(row[0], format='%d.%m.%Y %H:%M')
                    date_row = i
                    break
                except ValueError:
                    pass
            pd_therm = pd.read_excel(os.path.join(file_dir, file), header=date_row)
            print(pd_therm)
            list_index = [0]
            date = pd_therm.iloc[i, 0]
            print(date)
            for i in pd_therm.index:
                if pd_therm.iloc[i, 0] != date:
                    list_index.append(i)
                    date = pd_therm.iloc[i, 0]
            list_index.append(i)
            for a, b in zip(list_index[:-1], list_index[1:]):
                set_info(f'Термограмма {pd_therm.iloc[a, 0]}', 'blue')
                date_time = datetime.datetime.strptime(pd_therm.iloc[a, 0], '%d.%m.%Y %H:%M')
                depth, therm = pd_therm.iloc[a:b, 1].tolist(), pd_therm.iloc[a:b, 2].tolist()
                add_update_therm_to_db(h_well_id, date_time, depth, therm)
                n_load += 1

    session.commit()
    set_info(f'Для скважины {ui.listWidget_h_well.currentItem().text()} загружено {n_load} термограмм', 'green')
    update_list_thermogram()
    update_list_h_well()


def add_update_therm_to_db(h_well_id: int, date_time: datetime.datetime, depth: list, temp: list) -> None:
    """Добавить или обновить термограмму в базу"""
    if len(depth) == 0 or len(temp) == 0:
        set_info('Пустая термограмма', 'red')
        return
    if len(set(temp)) == 1:
        set_info('Термограмма с одним значением температуры', 'red')
        return
    therm = session.query(Thermogram).filter_by(h_well_id=h_well_id, date_time=date_time).first()
    if therm:
        session.query(Thermogram).filter_by(h_well_id=h_well_id, date_time=date_time).update(
            {'therm_data': json.dumps([[float(depth[i]), float(temp[i])] for i in range(len(depth))])}, synchronize_session='fetch')
    else:
        new_therm = Thermogram(
            h_well_id=h_well_id,
            date_time=date_time,
            therm_data=json.dumps([[float(depth[i]), float(temp[i])] for i in range(len(depth))])
        )
        session.add(new_therm)


def update_list_thermogram():
    """Обновить список термограмм"""
    ui.listWidget_thermogram.clear()
    thermograms = session.query(Thermogram).filter_by(h_well_id=get_h_well_id()).order_by(Thermogram.date_time).all()
    for therm in thermograms:
        start_therm = min([l[0] for l in json.loads(therm.therm_data)])
        end_therm = max([l[0] for l in json.loads(therm.therm_data)])
        item = QListWidgetItem(f'{therm.date_time.strftime("%d.%m.%Y %H:%M:%S")} ({start_therm} - {end_therm})')
        # item.setToolTip(therm.date_time.strftime("%H:%M:%S"))
        item.setData(Qt.UserRole, therm.id)
        ui.listWidget_thermogram.addItem(item)
        if len(therm.intersections) > 0:
            item.setBackground(QBrush(QColor('#FBD59E')))
    ui.label_25.setText(f'Thermograms: {len(thermograms)}')


def show_thermogram():
    """Показать термограмму"""
    therm = session.query(Thermogram).filter_by(id=get_therm_id()).first()
    if not therm:
        return
    therm_data = json.loads(therm.therm_data)
    if ui.checkBox_3d_therm.isChecked():
        fig = plt.figure(figsize=(10, 8), dpi=120)
        ax = fig.add_subplot(111, projection='3d')
        xs = [t[2] for t in therm_data if len(t) == 5]
        ys = [t[3] for t in therm_data if len(t) == 5]
        zs = [t[4] for t in therm_data if len(t) == 5]
        ts = [t[1] for t in therm_data if len(t) == 5]

        ax.scatter(xs, ys, zs, c=ts, cmap='gist_rainbow_r')

        # Добавление цветной шкалы
        cbar = plt.colorbar(ax.scatter(xs, ys, zs, c=ts, cmap='gist_rainbow_r'))
        cbar.set_label('Температура')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Термограмма {therm.date_time.strftime("%d.%m.%Y %H:%M:%S")} скважины {ui.listWidget_h_well.currentItem().text()}')

        # задаем реальные масштабы для всех трех осей
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        z_range = max(zs) - min(zs)
        max_range = max(x_range, y_range, z_range)
        x_center = (max(xs) + min(xs)) / 2
        y_center = (max(ys) + min(ys)) / 2
        z_center = (max(zs) + min(zs)) / 2
        ax.set_xlim((x_center - max_range / 2, x_center + max_range / 2))
        ax.set_ylim((y_center - max_range / 2, y_center + max_range / 2))
        ax.set_zlim((z_center - max_range / 2, z_center + max_range / 2))
        plt.tight_layout()
        plt.show()
    else:
        x = [l[0] for l in therm_data]
        y = [t[1] for t in therm_data]
        ui.doubleSpinBox_end_therm.setMaximum(max(x))
        ui.doubleSpinBox_end_therm.setValue(max(x))
        ui.graph.clear()
        curve = pg.PlotCurveItem(x=x, y=y, pen='r', name='Температура')
        ui.graph.addItem(curve)
        ui.graph.showGrid(x=True, y=True)  # отображаем сетку на графике
        show_start_therm()
        show_end_therm()


def show_start_therm():
    """ Показать ноль термограммы """
    global start_therm
    if 'start_therm' in globals():
        ui.graph.removeItem(globals()['start_therm'])
    pos_start = ui.doubleSpinBox_start_therm.value()
    start_therm = pg.InfiniteLine(pos=pos_start, angle=90, pen=pg.mkPen(color='green',width=3, dash=[8, 2]))
    ui.graph.addItem(start_therm)


def show_end_therm():
    """ Показать конец термограммы """
    global end_therm
    if 'end_therm' in globals():
        ui.graph.removeItem(globals()['end_therm'])
    pos_end = ui.doubleSpinBox_end_therm.value()
    end_therm = pg.InfiniteLine(pos=pos_end, angle=90, pen=pg.mkPen(color='yellow',width=3, dash=[8, 2]))
    ui.graph.addItem(end_therm)


def set_start_therm():
    """ Установить ноль термограммы """
    therm = session.query(Thermogram).filter_by(id=get_therm_id()).first()
    try:
        temp_curr = [temp[1] for temp in json.loads(therm.therm_data)]
    except AttributeError:
        set_info('Выберите термограмму', 'red')
        return
    start_therm = min([l[0] for l in json.loads(therm.therm_data)])
    end_therm = max([l[0] for l in json.loads(therm.therm_data)])
    start_value, n_set = ui.doubleSpinBox_start_therm.value(), 0
    ui.progressBar.setMaximum(session.query(Thermogram).filter_by(h_well_id=get_h_well_id()).count())
    for n_t, t in enumerate(session.query(Thermogram).filter_by(h_well_id=get_h_well_id()).all()):
        ui.progressBar.setValue(n_t)
        start_t = min([l[0] for l in json.loads(t.therm_data)])
        end_t = max([l[0] for l in json.loads(t.therm_data)])
        temp_list = [temp[1] for temp in json.loads(t.therm_data)]
        if start_t != start_therm or end_t != end_therm:
            continue
        corr_spear, _ = spearmanr(temp_curr, temp_list)
        if corr_spear < ui.doubleSpinBox_corr_therm.value():
            continue
        new_therm = [[v[0] - start_value, v[1]] for v in json.loads(t.therm_data)]
        session.query(Thermogram).filter_by(id=t.id).update({'therm_data': json.dumps(new_therm)}, synchronize_session='fetch')
        n_set += 1
    session.commit()
    set_info(f'Обновлено {n_set} термограмм для скважины {ui.listWidget_h_well.currentItem().text()}', 'green')
    update_list_thermogram()
    ui.doubleSpinBox_start_therm.setValue(0)


def cut_end_therm():
    """ Убрать конец термограммы """
    therm = session.query(Thermogram).filter_by(id=get_therm_id()).first()
    try:
        temp_curr = [temp[1] for temp in json.loads(therm.therm_data)]
    except AttributeError:
        set_info('Выберите термограмму', 'red')
        return
    depth_curr = [d[0] for d in json.loads(therm.therm_data)]
    cut_i = len(depth_curr)
    for n, d in enumerate(depth_curr):
        if d > ui.doubleSpinBox_end_therm.value():
            cut_i = n
            break
    therms = session.query(Thermogram).filter_by(h_well_id=get_h_well_id()).order_by(Thermogram.date_time).all()
    ui.progressBar.setMaximum(len(therms))
    n_cut = 0
    for n_t, t in enumerate(therms):
        temp_list = [temp[1] for temp in json.loads(t.therm_data)]
        depth_list = [d[0] for d in json.loads(t.therm_data)]
        if len(temp_curr) != len(temp_list) or min(depth_curr) != min(depth_list) or max(depth_curr) != max(depth_list):
            continue
        corr_spear, _ = spearmanr(temp_curr, temp_list)
        if corr_spear > ui.doubleSpinBox_corr_therm.value():
            t_therm_data = json.loads(t.therm_data)
            session.query(Thermogram).filter_by(id=t.id).update({
                'therm_data': json.dumps(t_therm_data[:cut_i])}, synchronize_session='fetch')
            n_cut += 1
    session.commit()
    update_list_thermogram()
    set_info(f'Выполнена обрезка {n_cut} термограмм для скважины {ui.listWidget_h_well.currentItem().text()}', 'green')


def show_corr_therm():
    """Показать коррелируемые термограммы"""
    therm = session.query(Thermogram).filter_by(id=get_therm_id()).first()
    try:
        temp_curr = [temp[1] for temp in json.loads(therm.therm_data)]
    except AttributeError:
        set_info('Выберите термограмму', 'red')
        return
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(111)
    n_corr = 0
    for t in session.query(Thermogram).filter_by(h_well_id=get_h_well_id()).order_by(Thermogram.date_time).all():
        temp_list = [temp[1] for temp in json.loads(t.therm_data)]
        if len(temp_curr) != len(temp_list):
            continue
        corr_spear, _ = spearmanr(temp_curr, temp_list)
        if corr_spear > ui.doubleSpinBox_corr_therm.value():
            ax.plot(temp_list, label=t.date_time.strftime('%d.%m.%Y %H-%M-%S'))
            n_corr += 1
    ax.set_title(f'Коррелируют {n_corr} термограмм')
    plt.tight_layout()
    ax.legend()
    ax.grid(True)
    plt.show()


def remove_thermogram():
    """Удалить термограмму и подобные"""
    h_well_id = get_h_well_id()
    i_therm = ui.listWidget_thermogram.currentRow()
    if ui.checkBox_only_one.isChecked():
        session.query(Thermogram).filter_by(id=get_therm_id()).delete()
        set_info('Термограмма удалена', 'green')
    else:
        therm = session.query(Thermogram).filter_by(id=get_therm_id()).first()
        try:
            temp_curr = [temp[1] for temp in json.loads(therm.therm_data)]
        except AttributeError:
            set_info('Выберите термограмму', 'red')
            return
        n_rem = 0
        for t in session.query(Thermogram).filter_by(h_well_id=get_h_well_id()).all():
            temp_list = [temp[1] for temp in json.loads(t.therm_data)]
            if len(temp_curr) != len(temp_list):
                continue
            corr_spear, _ =spearmanr(temp_curr, temp_list)
            if corr_spear > ui.doubleSpinBox_corr_therm.value():
                session.delete(t)
                n_rem += 1
        set_info(f'Удалено {n_rem} термограмм', 'green')
    session.commit()
    update_list_h_well()
    for i in range(ui.listWidget_h_well.count()):
        if ui.listWidget_h_well.item(i).data(Qt.UserRole) == h_well_id:
            ui.listWidget_h_well.setCurrentRow(i)
            break
    update_list_thermogram()
    ui.listWidget_thermogram.setCurrentRow(i_therm)


def remove_therm_by_date():
    """Удалить термограммы за выбранный период"""
    h_well_id = get_h_well_id()

    FormDelTherm = QtWidgets.QDialog()
    ui_fdt = Ui_Form_delete_therm_by_date()
    ui_fdt.setupUi(FormDelTherm)
    FormDelTherm.show()
    FormDelTherm.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    date_start = session.query(Thermogram).filter_by(h_well_id=h_well_id).order_by(Thermogram.date_time).first().date_time
    date_stop = session.query(Thermogram).filter_by(h_well_id=h_well_id).order_by(Thermogram.date_time.desc()).first().date_time
    ui_fdt.dateEdit_start.setDate(date_start)
    ui_fdt.dateEdit_stop.setDate(date_stop)

    def remove_therm_by_start_stop():
        start = ui_fdt.dateEdit_start.date().toPyDate()
        stop = ui_fdt.dateEdit_stop.date().toPyDate()
        if ui_fdt.checkBox_all_hwell.isChecked():
            h_well_curr = session.query(HorizontalWell).filter_by(id=h_well_id).first()
            for h_well in session.query(HorizontalWell).filter_by(object_id=h_well_curr.object_id).all():
                session.query(Thermogram).filter(
                    Thermogram.date_time >= start,
                    Thermogram.date_time <= stop,
                    Thermogram.h_well_id == h_well.id).delete()
        else:
            session.query(Thermogram).filter(Thermogram.date_time.between(start, stop), Thermogram.h_well_id == h_well_id).delete()
        session.commit()
        update_list_h_well()
        for i in range(ui.listWidget_h_well.count()):
            if ui.listWidget_h_well.item(i).data(Qt.UserRole) == h_well_id:
                ui.listWidget_h_well.setCurrentRow(i)
                break
        update_list_thermogram()
        FormDelTherm.close()

    ui_fdt.buttonBox.accepted.connect(remove_therm_by_start_stop)
    ui_fdt.buttonBox.rejected.connect(FormDelTherm.close)
    FormDelTherm.exec_()



def draw_param_h_well():
    """Отрисовать параметр горизонтальных скважин"""
    item = ui.listWidget_param_h_well.currentItem()
    if not item:
        return
    id_param = item.data(Qt.UserRole)
    if not id_param:
        return
    param = session.query(ParameterHWell).filter_by(id=id_param).first()
    if param.parameter == 'Инклинометрия':
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        coord_inc = json.loads(param.data)
        xs = [coord[0] for coord in coord_inc]
        ys = [coord[1] for coord in coord_inc]
        zs = [coord[2] for coord in coord_inc]

        ax.plot(xs, ys, zs)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # задаем реальные масштабы для всех трех осей
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        z_range = max(zs) - min(zs)
        max_range = max(x_range, y_range, z_range)
        x_center = (max(xs) + min(xs)) / 2
        y_center = (max(ys) + min(ys)) / 2
        z_center = (max(zs) + min(zs)) / 2
        ax.set_xlim((x_center - max_range / 2, x_center + max_range / 2))
        ax.set_ylim((y_center - max_range / 2, y_center + max_range / 2))
        ax.set_zlim((z_center - max_range / 2, z_center + max_range / 2))
    else:
        param_data = json.loads(param.data)
        sorted_param_data = sorted(param_data)
        x, y = [], []
        for key in sorted_param_data:
            x.append(datetime.datetime.strptime(key, '%Y-%m-%d'))
            y.append(param_data[key])

        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(111)
        ax.plot(x, y, label=f'{param.parameter} скв. {param.h_well.title}', marker='.', linestyle='-', linewidth=1, color='blue')
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.show()

        # Вариант отображения параметров в pyqtgraph
        # x_num = [mdates.date2num(date) for date in x]
        # ui.graph.clear()
        # date_axis = DateAxisItem(orientation='bottom')
        # date_axis.setTicks([[(x_num[i], x[i].strftime('%m.%y')) for i in range(len(x_num)) if x[i].day == 1 and x[i].month % 2 == 0]])
        # ui.graph.setAxisItems({'bottom': date_axis})
        # curve_param = pg.PlotCurveItem(x=x_num, y=y)
        # ui.graph.addItem(curve_param)


def calc_inclinometry(input_file_path, initial_coordinates):
    output_coordinates = []

    x, y, z = initial_coordinates
    prev_length = 0

    with open(input_file_path, 'r') as input_file:
        for line in input_file:
            if not line.strip():
                continue
            try:
                length, dip_angle, azimuth = map(float, line.strip().split('\t'))
            except ValueError:
                continue

            delta_length = length - prev_length  # разница между текущим и предыдущим значением length
            delta_x = delta_length * math.sin(math.radians(180 - dip_angle)) * math.sin(math.radians(azimuth))
            delta_y = delta_length * math.sin(math.radians(180 - dip_angle)) * math.cos(math.radians(azimuth))
            delta_z = delta_length * math.cos(math.radians(180 - dip_angle))

            x += delta_x
            y += delta_y
            z += delta_z

            prev_length = length  # сохраняем текущее значение length для следующей итерации

            output_coordinates.append((x, y, z, length, dip_angle, azimuth))

    return output_coordinates


def load_wellhead():
    """Загрузить координаты устья горизонтальной скважины"""
    h_well_id = get_h_well_id()
    if h_well_id:
        try:
            x, y = float(ui.lineEdit_string.text().split(';')[0]), float(ui.lineEdit_string.text().split(';')[1])
            grid_db = session.query(Grid).filter(Grid.object_id == get_obj_monitor_id()).first()
            if grid_db:
                pd_grid_r = pd.DataFrame(json.loads(grid_db.grid_table_r))
                save_wellhead_to_db(x, y, pd_grid_r, h_well_id)
                session.commit()
                update_list_param_h_well()
            else:
                set_info(f'Не загружени grid-файл рельефа для {get_obj_monitor_name()} площади', 'red')
        except ValueError:
            set_info('Введите координаты устья горизонтальной скважины - "x;y"', 'red')
    else:
        set_info('Выберите горизонтальную скважину', 'red')


def load_wellhead_batch():
    """Загрузить координаты устья горизонтальной скважины"""
    grid_db = session.query(Grid).filter(Grid.object_id == get_obj_monitor_id()).first()
    if grid_db:
        pd_grid_r = pd.DataFrame(json.loads(grid_db.grid_table_r))
    else:
        set_info(f'Не загружени grid-файл рельефа для {get_obj_monitor_name()} площади', 'red')
        return
    file_name = QFileDialog.getOpenFileName(MainWindow, 'Выбрать файл с координатами устья скважин', '', 'Текстовые файлы (*.txt)')[0]
    if file_name:
        with open(file_name, 'r') as input_file:
            lines = input_file.readlines()
            ui.progressBar.setMaximum(len(lines))
            for n_line, line in enumerate(lines):
                ui.progressBar.setValue(n_line)
                if not line.strip():
                    continue
                well_name, x, y = line.strip().split('\t')
                h_well = session.query(HorizontalWell).filter_by(title=well_name, object_id=get_obj_monitor_id()).first()
                if not h_well:
                    set_info(f'Не загружена скважина {well_name} для {get_obj_monitor_name()}', 'red')
                else:
                    save_wellhead_to_db(float(x), float(y), pd_grid_r, h_well.id)
                    set_info(f'Загружены координаты устья скважины {well_name} для {get_obj_monitor_name()}', 'green')
            session.commit()
            update_list_param_h_well()
            set_info('Готово!', 'green')


def save_wellhead_to_db(x, y, land_grid, h_well_id):
    """Сохранить координаты устья горизонтальной скважины"""
    land_grid['dist_y'] = abs(land_grid[1] - y)
    land_grid['dist_x'] = abs(land_grid[0] - x)
    z = land_grid.loc[land_grid['dist_y'] == land_grid['dist_y'].min()].loc[
        land_grid['dist_x'] == land_grid['dist_x'].min()].iat[0, 2]
    session.query(HorizontalWell).filter(HorizontalWell.id == h_well_id).update({
        'x_coord': x,
        'y_coord': y,
        'alt': z
    }, synchronize_session='fetch')


def show_inclinometry():
    """Показать инклинометрию всех скважин объекта"""
    fig = plt.figure(figsize=(14, 14))
    if ui.checkBox_2d.isChecked():
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection='3d')

    data_incl = session.query(ParameterHWell).join(HorizontalWell).filter(
        HorizontalWell.object_id == get_obj_monitor_id(),
        ParameterHWell.parameter == 'Инклинометрия').all()

    obj = get_object_id()
    obj_m = get_obj_monitor_id()

    if obj_m == obj:
        intersections = session.query(Intersection).join(Profile).filter(Profile.research_id == get_research_id()).all()
    else:
        intersections = session.query(Intersection).join(Profile).join(Research).filter(Research.object_id == obj_m).all()

    all_x, all_y, all_z, all_x_t, all_y_t, all_z_t = [], [], [], [], [], []

    for incl in data_incl:
        coord_inc = json.loads(incl.data)
        xs = [coord[0] for coord in coord_inc]
        ys = [coord[1] for coord in coord_inc]
        zs = [coord[2] for coord in coord_inc]
        if len(incl.h_well.thermograms) > 0:
            all_x_t.extend(xs)
            all_y_t.extend(ys)
            all_z_t.extend(zs)
        else:
            all_x.extend(xs)
            all_y.extend(ys)
            all_z.extend(zs)

    if ui.checkBox_incl_int.isChecked():
        for int_db in intersections:
            if ui.checkBox_2d.isChecked():
                ax.scatter(int_db.x_coord, int_db.y_coord, marker='o', color='#2C145E', s=150)
            else:
                ax.plot([int_db.x_coord, int_db.x_coord], [int_db.y_coord, int_db.y_coord], [-1000, 1000], color='#ff7800', linewidth=2)
                ax.scatter(int_db.x_coord, int_db.y_coord, 1000, 'o', s=100, color='#ff7800')

    if ui.checkBox_2d.isChecked():
        ax.plot(all_x, all_y, '.')
        ax.plot(all_x_t, all_y_t, '.', color='red')
    else:
        ax.plot(all_x, all_y, all_z, '.')
        ax.plot(all_x_t, all_y_t, all_z_t, '.', color='red')

    if ui.checkBox_incl_prof.isChecked():

        if obj_m == obj:
            profiles = session.query(Profile).filter(Profile.research_id == get_research_id()).all()
        else:
            research_id = session.query(Research.id).filter(Research.object_id == obj_m).first()[0]
            profiles = session.query(Profile).filter(Profile.research_id == research_id).all()
        all_x_p, all_y_p, all_z_p = [], [], []
        for p in profiles:
            xs = json.loads(p.x_pulc)
            ys = json.loads(p.y_pulc)
            try:
                zs = json.loads(session.query(Formation.land).filter(Formation.profile_id == p.id).first()[0])
            except TypeError:
                set_info('Для профилей не рассчитан рельеф', 'red')
                return
            all_x_p.extend(xs)
            all_y_p.extend(ys)
            all_z_p.extend(zs)
            if ui.checkBox_2d.isChecked():
                ax.plot(xs, ys, '.', color='green')
            else:
                ax.plot(all_x_p, all_y_p, all_z_p, '.', color='green')

    if ui.checkBox_incl_therm.isChecked():
        for int_db in intersections:
            xs_tm = [i[2] for i in json.loads(int_db.thermogram.therm_data) if len(i)>2]
            ys_tm = [i[3] for i in json.loads(int_db.thermogram.therm_data) if len(i)>2]
            zs_tm = [i[4] for i in json.loads(int_db.thermogram.therm_data) if len(i)>2]
            if ui.checkBox_2d.isChecked():
                ax.plot(xs_tm, ys_tm, '.', color='darkred')
            else:
                ax.plot(xs_tm, ys_tm, zs_tm, '.', color='darkred')


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if not ui.checkBox_2d.isChecked():
        ax.set_zlabel('Z')

        # задаем реальные масштабы для всех трех осей
        x_range = max(all_x) - min(all_x)
        y_range = max(all_y) - min(all_y)
        z_range = max(all_z) - min(all_z)
        max_range = max(x_range, y_range, z_range)
        x_center = (max(all_x) + min(all_x)) / 2
        y_center = (max(all_y) + min(all_y)) / 2
        z_center = (max(all_z) + min(all_z)) / 2
        ax.set_xlim((x_center - max_range / 2, x_center + max_range / 2))
        ax.set_ylim((y_center - max_range / 2, y_center + max_range / 2))
        ax.set_zlim((z_center - max_range / 2, z_center + max_range / 2))


        if ui.checkBox_animation.isChecked():
            def rotate(angle):
                ax.view_init(azim=angle)

            ani = FuncAnimation(fig, rotate, frames=np.arange(0, 360, 1), interval=ui.spinBox_interval_animation.value())

    plt.tight_layout()
    plt.show()


def coordinate_binding_thermogram():
    """ Координатная привязка термограммы """
    incl_param = session.query(ParameterHWell).filter_by(parameter='Инклинометрия', h_well_id=get_h_well_id()).first()
    if not incl_param:
        hwell = session.query(HorizontalWell).filter_by(id=get_h_well_id()).first()
        set_info(f'Нет инклинометрии для скважины {hwell.title}', 'red')
        return
    data_incl = json.loads(incl_param.data)
    length_incl = [i[3] for i in data_incl]
    therms = session.query(Thermogram).filter_by(h_well_id=get_h_well_id()).all()
    list_no_start, list_too_long = [], []
    for therm in therms:
        start_t = min([l[0] for l in json.loads(therm.therm_data)])
        end_t = max([l[0] for l in json.loads(therm.therm_data)])
        if start_t == 0:
            list_no_start.append(therm.date_time.strftime('%Y-%m-%d %H:%M:%S'))
        if end_t > max(length_incl):
            list_too_long.append(therm.date_time.strftime('%Y-%m-%d %H:%M:%S'))
    if len(list_no_start) > 0:
        result = QtWidgets.QMessageBox.question(
            MainWindow,
            'Внимание!',
            f'Для термограмм {", ".join(list_no_start[:10])}{" и др." if len(list_no_start) > 10 else ""} '
            f'не установлена отметка устья скважины. Продолжить?',
            QtWidgets.QMessageBox.Yes,
            QtWidgets.QMessageBox.No
        )
        if result == QtWidgets.QMessageBox.No:
            return
    if len(list_too_long) > 0:
        set_info('Координатная привязка термограмм не выполнена', 'red')
        set_info(f'Длина термограмм больше длины инклинометрии: {", ".join(list_too_long[:10])}'
                 f'{" и др." if len(list_too_long) > 10 else ""}', 'red')
        return
    ui.progressBar.setMaximum(len(therms))
    for n_therm, therm in enumerate(therms):
        set_info(f'Привязка координат для термограммы {therm.date_time.strftime("%Y-%m-%d %H:%M:%S")}', 'blue')
        ui.progressBar.setValue(n_therm)
        t_data = json.loads(therm.therm_data)
        for i, point_therm in enumerate(t_data):
            if point_therm[0] < 0:
                continue
            l_therm = point_therm[0]
            left_i, right_i = find_interval(length_incl, l_therm)
            if left_i == right_i:
                t_data[i] = [l_therm, point_therm[1], data_incl[left_i][0], data_incl[left_i][1], data_incl[left_i][2]]
            else:
                xt, yt, zt = calc_coord_point_of_therm(data_incl[left_i][0], data_incl[left_i][1], data_incl[left_i][2],
                                                       data_incl[right_i][0], data_incl[right_i][1], data_incl[right_i][2],
                                                       l_therm - length_incl[left_i],)
                t_data[i] = [l_therm, point_therm[1], xt, yt, zt]
        session.query(Thermogram).filter_by(id=therm.id).update({'therm_data': json.dumps(t_data)}, synchronize_session='fetch')
    session.commit()
    set_info('Координатная привязка термограмм завершена', 'green')
    update_list_h_well()



def calc_coord_point_of_therm(x1, y1, z1, x2, y2, z2, l_therm):
    """Рассчитать координату точки термограммы """
    # Вычисление вектора между начальной и конечной точками
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    # Вычисление длины вектора
    l = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    # Вычисление значения t по пропорции
    t = l_therm / l
    # Вычисление координат искомой точки на линии между начальной и конечной точками
    x, y, z = x1 + t * dx, y1 + t * dy, z1 + t * dz
    return x, y, z


def find_interval(values, target):
    """
    Находит индексы ближайших элементов в списке values к числу target
    """
    left, right = 0, len(values) - 1

    while left <= right:
        middle = (left + right) // 2
        if target < values[middle]:
            right = middle - 1
        elif target > values[middle]:
            left = middle + 1
        else:
            return middle, middle

    if right < 0:
        return None, None

    if left >= len(values):
        return None, None

    return right, left


def show_therms_animation():
    """ Показать анимацию изменения термограмм """
    fig = plt.figure(figsize=(12, 15), dpi=120)
    ax1 = fig.add_subplot(211, projection='3d')

    therms = session.query(Thermogram).filter_by(h_well_id=get_h_well_id()).order_by(Thermogram.date_time).all()
    try:
        t_data = json.loads(therms[0].therm_data)
    except IndexError:
        set_info('Выберите горизонтальную скважину', 'red')
        return
    x_min = min([x[2] for x in t_data if len(x) > 4])
    x_max = max([x[2] for x in t_data if len(x) > 4])
    y_min = min([y[3] for y in t_data if len(y) > 4])
    y_max = max([y[3] for y in t_data if len(y) > 4])
    z_min = min([z[4] for z in t_data if len(z) > 4])
    z_max = max([z[4] for z in t_data if len(z) > 4])
    t_min = min([t[1] for t in t_data if len(t) > 4])
    t_max = max([t[1] for t in t_data if len(t) > 4])
    count_therm = len([t for t in t_data if len(t) > 4])
    for t in therms:
        t_data = json.loads(t.therm_data)
        count_t = len([t for t in t_data if len(t) > 4])
        count_therm = min(count_therm, count_t)
        x = [l[2] for l in t_data if len(l) > 4][:count_therm]
        y = [t[3] for t in t_data if len(t) > 4][:count_therm]
        z = [t[4] for t in t_data if len(t) > 4][:count_therm]
        t = [t[1] for t in t_data if len(t) > 4][:count_therm]
        x_min, x_max = min(x_min, min(x)), max(x_max, max(x))
        y_min, y_max = min(y_min, min(y)), max(y_max, max(y))
        z_min, z_max = min(z_min, min(z)), max(z_max, max(z))
        t_min, t_max = min(t_min, min(t)), max(t_max, max(t))
    t_data = json.loads(therms[0].therm_data)
    x = [l[2] for l in t_data if len(l) > 4][:count_therm]
    y = [t[3] for t in t_data if len(t) > 4][:count_therm]
    z = [t[4] for t in t_data if len(t) > 4][:count_therm]
    t = [t[1] for t in t_data if len(t) > 4][:count_therm]

    sc = ax1.scatter(x, y, z, c=t, cmap='gist_rainbow_r')
    cbar = plt.colorbar(ax1.scatter(x, y, z, c=t, cmap='gist_rainbow_r'))
    cbar.set_label('Температура')

    label_3d = ax1.text(x[0], y[0], z[0], '', fontsize=14)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlim(x_min, x_max)
    ax1.set_zlim(z_min, z_max)
    sc.set_clim(t_min, t_max)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Функция обновления графика на каждом кадре анимации
    def animate(frame):
        t_date = json.loads(therms[frame].therm_data)
        xi = [l[2] for l in t_date if len(l) > 4][:count_therm]
        yi = [l[3] for l in t_date if len(l) > 4][:count_therm]
        zi = [l[4] for l in t_date if len(l) > 4][:count_therm]
        ti = [l[1] for l in t_date if len(l) > 4][:count_therm]
        lbl = f'Термограмма {frame + 1} из {len(therms)}: {therms[frame].date_time.strftime("%Y-%m-%d %H:%M:%S")}'
        sc._offset3d = (xi, yi, zi)
        sc.set_array(ti)
        label_3d.set_text(lbl)
        return sc, label_3d

    animation_3d = FuncAnimation(fig, animate, frames=len(therms), blit=True,
                              interval=ui.spinBox_interval_animation.value())

    ax2 = fig.add_subplot(212)


    # Инициализация пустого графика
    line, = ax2.plot([], [], label='Термограмма', color='red', linewidth=2)

    x_min = min([l[0] for l in json.loads(therms[0].therm_data)])
    x_max = max([l[0] for l in json.loads(therms[0].therm_data)])
    y_min = min([t[1] for t in json.loads(therms[0].therm_data)])
    y_max = max([t[1] for t in json.loads(therms[0].therm_data)])
    for t in therms:
        x = [l[0] for l in json.loads(t.therm_data)]
        y = [t[1] for t in json.loads(t.therm_data)]
        x_min, x_max, y_min, y_max = min(x_min, min(x)), max(x_max, max(x)), min(y_min, min(y)), max(y_max, max(y))
    ax2.set_ylim(y_min, y_max)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylabel('Температура, С')
    ax2.set_xlabel('Глубина по стволу скважины, м')

    label = ax2.text(0.05, 0.95, '', transform=ax2.transAxes, fontsize=14, ha='left')
    ax2.grid(True)
    ax2.legend()


    # Функция инициализации анимации
    def init():
        line.set_data([], [])
        label.set_text('')
        return line, label


    # Функция обновления графика на каждом кадре анимации
    def update(frame):
        global a
        a = frame
        x = [l[0] for l in json.loads(therms[frame].therm_data)]
        y = [t[1] for t in json.loads(therms[frame].therm_data)]
        lbl = f'Термограмма {frame + 1} из {len(therms)}: {therms[frame].date_time.strftime("%Y-%m-%d %H:%M:%S")}'
        line.set_data(x, y)
        label.set_text(lbl)

        return line, label


    # Создание анимации
    animation = FuncAnimation(fig, update, frames=len(therms), init_func=init, blit=True,
                              interval=ui.spinBox_interval_animation.value())


    button_ax = fig.add_axes([0.8, 0.02, 0.1, 0.03])
    pause_button = Button(button_ax, "Pause")

    # Функция приостановки анимации при нажатии на кнопку
    def pause_animation(event):

        if pause_button.label.get_text() == "Pause":
            animation.event_source.stop()
            animation_3d.event_source.stop()
            pause_button.label.set_text("Resume")
        else:
            animation.event_source.start()
            animation_3d.event_source.start()
            pause_button.label.set_text("Pause")

    # Подключение функции обработки события к кнопке
    pause_button.on_clicked(pause_animation)

    app.processEvents()
    plt.tight_layout()
    plt.show()


def show_therms_animation_sep():
    """ Показать анимацию изменения термограмм отдельно """
    therms = session.query(Thermogram).filter_by(h_well_id=get_h_well_id()).order_by(Thermogram.date_time).all()

    # Создание фигуры и осей
    if ui.checkBox_3d_therm.isChecked():
        fig = plt.figure(figsize=(8, 6), dpi=120)
        ax = fig.add_subplot(111, projection='3d')
        t_data = json.loads(therms[0].therm_data)
        x_min = min([x[2] for x in t_data if len(x) > 4])
        x_max = max([x[2] for x in t_data if len(x) > 4])
        y_min = min([y[3] for y in t_data if len(y) > 4])
        y_max = max([y[3] for y in t_data if len(y) > 4])
        z_min = min([z[4] for z in t_data if len(z) > 4])
        z_max = max([z[4] for z in t_data if len(z) > 4])
        t_min = min([t[1] for t in t_data if len(t) > 4])
        t_max = max([t[1] for t in t_data if len(t) > 4])
        count_therm = len([t for t in t_data if len(t) > 4])
        for t in therms:
            t_data = json.loads(t.therm_data)
            count_t = len([t for t in t_data if len(t) > 4])
            count_therm = min(count_therm, count_t)
            x = [l[2] for l in t_data if len(l) > 4][:count_therm]
            y = [t[3] for t in t_data if len(t) > 4][:count_therm]
            z = [t[4] for t in t_data if len(t) > 4][:count_therm]
            t = [t[1] for t in t_data if len(t) > 4][:count_therm]
            x_min, x_max = min(x_min, min(x)), max(x_max, max(x))
            y_min, y_max = min(y_min, min(y)), max(y_max, max(y))
            z_min, z_max = min(z_min, min(z)), max(z_max, max(z))
            t_min, t_max = min(t_min, min(t)), max(t_max, max(t))
        t_data = json.loads(therms[0].therm_data)
        x = [l[2] for l in t_data if len(l) > 4][:count_therm]
        y = [t[3] for t in t_data if len(t) > 4][:count_therm]
        z = [t[4] for t in t_data if len(t) > 4][:count_therm]
        t = [t[1] for t in t_data if len(t) > 4][:count_therm]

        sc = ax.scatter(x, y, z, c=t, cmap='gist_rainbow_r')
        cbar = plt.colorbar(ax.scatter(x, y, z, c=t, cmap='gist_rainbow_r'))
        cbar.set_label('Температура')

        label = ax.text(x[0], y[0], z[0], '', fontsize=14)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(x_min, x_max)
        ax.set_zlim(z_min, z_max)
        sc.set_clim(t_min, t_max)



        # Функция обновления графика на каждом кадре анимации
        def animate(frame):
            t_date = json.loads(therms[frame].therm_data)
            xi = [l[2] for l in t_date if len(l) > 4][:count_therm]
            yi = [l[3] for l in t_date if len(l) > 4][:count_therm]
            zi = [l[4] for l in t_date if len(l) > 4][:count_therm]
            ti = [l[1] for l in t_date if len(l) > 4][:count_therm]
            # print(t)
            lbl = f'Термограмма {frame + 1} из {len(therms)}: {therms[frame].date_time.strftime("%Y-%m-%d %H:%M:%S")}'
            sc._offset3d = (xi, yi, zi)
            sc.set_array(ti)
            label.set_text(lbl)
            return sc, label

            # return line, label
        animation = FuncAnimation(fig, animate, frames=len(therms), blit=True, interval=ui.spinBox_interval_animation.value())

        button_ax = fig.add_axes([0.2, 0.9, 0.1, 0.1])
        pause_button = Button(button_ax, "Pause")

        # Функция приостановки анимации при нажатии на кнопку
        def pause_animation(event):

            if pause_button.label.get_text() == "Pause":
                animation.event_source.stop()
                pause_button.label.set_text("Resume")
            else:
                animation.event_source.start()
                pause_button.label.set_text("Pause")

        # Подключение функции обработки события к кнопке
        pause_button.on_clicked(pause_animation)
        plt.tight_layout()

    else:
        fig, ax = plt.subplots(figsize=(16, 8))

        # Инициализация пустого графика
        line, = ax.plot([], [], label='', color='red', linewidth=2)
        label = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=14, ha='left')

        x_min = min([l[0] for l in json.loads(therms[0].therm_data)])
        x_max = max([l[0] for l in json.loads(therms[0].therm_data)])
        y_min = min([t[1] for t in json.loads(therms[0].therm_data)])
        y_max = max([t[1] for t in json.loads(therms[0].therm_data)])
        for t in therms:
            x = [l[0] for l in json.loads(t.therm_data)]
            y = [t[1] for t in json.loads(t.therm_data)]
            x_min, x_max, y_min, y_max = min(x_min, min(x)),  max(x_max, max(x)), min(y_min, min(y)), max(y_max, max(y))
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(x_min, x_max)
        ax.grid(True)


        # Функция инициализации анимации
        def init():
            line.set_data([], [])
            label.set_text('')
            return line, label

        # Функция обновления графика на каждом кадре анимации
        def update(frame):
            global a
            a = frame
            x = [l[0] for l in json.loads(therms[frame].therm_data)]
            y = [t[1] for t in json.loads(therms[frame].therm_data)]
            lbl = f'Термограмма {frame + 1} из {len(therms)}: {therms[frame].date_time.strftime("%Y-%m-%d %H:%M:%S")}'
            line.set_data(x, y)
            label.set_text(lbl)

            return line, label

        # Создание анимации
        animation = FuncAnimation(fig, update, frames=len(therms), init_func=init, blit=True, interval=ui.spinBox_interval_animation.value())

        button_ax = fig.add_axes([0.2, 0.9, 0.1, 0.1])
        pause_button = Button(button_ax, "Pause")

        # Функция приостановки анимации при нажатии на кнопку
        def pause_animation(event):

            if pause_button.label.get_text() == "Pause":
                animation.event_source.stop()
                pause_button.label.set_text("Resume")
            else:
                animation.event_source.start()
                pause_button.label.set_text("Pause")

        # Подключение функции обработки события к кнопке
        pause_button.on_clicked(pause_animation)
    # Отображение

    app.processEvents()
    # plt.tight_layout()
    plt.show()


def average_lists(lists):
    # Проверка, что списки не пустые
    if not lists:
        return []
    # Использование zip для объединения элементов на одной позиции из всех списков
    combined = zip(*lists)
    # Вычисление среднего значения на каждой позиции
    average = [sum(values) / len(values) for values in combined]
    return average


def mean_day_thermogram():
    """ Вычисление средней термограммы за день """
    therms = session.query(Thermogram).filter_by(h_well_id=get_h_well_id()).all()
    uniq_dates = sorted(set(t.date_time.date() for t in therms))
    ui.progressBar.setMaximum(len(uniq_dates))
    for nd, d in enumerate(uniq_dates):
        ui.progressBar.setValue(nd + 1)
        day_therm_list, id_therm_list = [], []
        for t in therms:
            if t.date_time.date() == d:
                day_therm_list.append(json.loads(t.therm_data))
                id_therm_list.append(t.id)
        if len(day_therm_list) == 1:
            continue
        if not check_list_lengths(day_therm_list):
            set_info(f'Не совпадает длина термограммы {d.strftime("%d.%m.%Y")}', 'red')
            day_therm_list = trim_lists(day_therm_list, 30)
            if not day_therm_list:
                set_info(f'Не совпадает длина термограммы {d.strftime("%d.%m.%Y")}, обрезка не возможна', 'red')
                continue
        list_therm_temp = [[temp[1] for temp in day_therm] for day_therm in day_therm_list]
        set_info(f'Усреднение {len(list_therm_temp)} термограмм за {d.strftime("%d.%m.%Y")}', 'blue')
        averege_therm = average_lists(list_therm_temp)
        depth_list = [d[0] for d in day_therm_list[0]]
        for it in id_therm_list:
            session.query(Thermogram).filter_by(id=it).delete()
        add_update_therm_to_db(get_h_well_id(), d, depth_list, averege_therm)
    session.commit()
    update_list_thermogram()
    count_therm = session.query(Thermogram).filter_by(h_well_id=get_h_well_id()).count()
    item_text = f'{get_h_well_title()}\t+{count_therm} термограмм'
    ui.listWidget_h_well.currentItem().setText(item_text)


def add_intersection():
    """ Добавление пересечений профилей в базу данных """

    # Получение данных инклинометрии по скважинам выбранного объекта
    data_incl = session.query(ParameterHWell).join(HorizontalWell).filter(
        HorizontalWell.object_id == get_obj_monitor_id(),
        ParameterHWell.parameter == 'Инклинометрия'
    ).all()

    # Цикл по исследованиям на объекте
    count_add_int = 0
    for res in session.query(Research).filter(Research.object_id == get_obj_monitor_id()).all():
        ui.progressBar.setMaximum(session.query(Profile).filter(Profile.research_id == res.id).count())
        for n_prof, prof in enumerate(session.query(Profile).filter(Profile.research_id == res.id).all()):
            ui.progressBar.setValue(n_prof + 1)
            xs_prof = json.loads(prof.x_pulc)
            ys_prof = json.loads(prof.y_pulc)
            target_date = prof.research.date_research
            target_datetime = datetime.datetime.combine(target_date, datetime.datetime.min.time())

            for incl in data_incl:
                if len(incl.h_well.thermograms) > 0:
                    coord_inc = json.loads(incl.data)  # данные по инклинометрии
                    xs_hwell = [coord[0] for coord in coord_inc]
                    ys_hwell = [coord[1] for coord in coord_inc]

                    int_p_hw = find_intersection_points(xs_prof, ys_prof, xs_hwell, ys_hwell)
                    print(int_p_hw)
                    if len(int_p_hw) > 0:

                        therm1 = session.query(Thermogram).filter(
                            Thermogram.h_well_id == incl.h_well_id,
                            Thermogram.date_time > target_date
                        ).order_by(Thermogram.date_time.asc()).first()
                        therm2 = session.query(Thermogram).filter(
                            Thermogram.h_well_id == incl.h_well_id,
                            Thermogram.date_time < target_date
                        ).order_by(Thermogram.date_time.desc()).first()


                        therm1_diff = (therm1.date_time - target_datetime).total_seconds() if therm1 else math.inf
                        therm2_diff = (target_datetime - therm2.date_time).total_seconds() if therm2 else math.inf

                        print(therm1_diff/86400, therm2_diff/86400)

                        result_therm = therm1 if therm1_diff < therm2_diff else therm2
                        result_therm_diff = (abs(result_therm.date_time - target_datetime).total_seconds()) / 86400
                        if result_therm_diff > 100:
                            continue

                        xs_therm = [t[2] for t in json.loads(result_therm.therm_data) if len(t) > 2]
                        ys_therm = [t[3] for t in json.loads(result_therm.therm_data) if len(t) > 2]
                        ts_therm = [t[1] for t in json.loads(result_therm.therm_data) if len(t) > 2]

                        int_p_th = find_intersection_points(xs_prof, ys_prof, xs_therm, ys_therm)
                        print(int_p_th)

                        for intersect in int_p_th:
                            name_int = f'{incl.h_well.title}_{prof.id}_{result_therm.id}_{intersect[2]}_{intersect[3]}'
                            print(name_int)
                            if session.query(Intersection).filter_by(name=name_int).count() > 0:
                                set_info(f'Пересечение {name_int} уже есть в БД', 'red')
                            else:
                                new_int = Intersection(
                                    therm_id = result_therm.id, profile_id=prof.id, name=name_int,
                                    x_coord=intersect[0], y_coord=intersect[1], temperature=ts_therm[intersect[3]],
                                    i_therm=intersect[3], i_profile=intersect[2]
                                )
                                session.add(new_int)
                                session.commit()
                                count_add_int += 1

    set_info(f'Добавлено {count_add_int} пересечений', 'green')
    update_list_well()


def trim_lists(list_of_lists, max_difference):
    if not list_of_lists:
        return []

    min_length = min(len(lst) for lst in list_of_lists)
    max_length = max(len(lst) for lst in list_of_lists)
    if max_length - min_length > max_difference:
        return False
    trimmed_lists = [lst[:min_length] for lst in list_of_lists]

    return trimmed_lists


def draw_map_by_thermogram():
    research = session.query(Research).filter_by(id=get_research_id()).first()
    target_date = research.date_research
    target_datetime = datetime.datetime.combine(target_date, datetime.datetime.min.time())
    list_x, list_y, list_t = [], [], []
    for h_well in session.query(HorizontalWell).filter_by(object_id=get_object_id()).all():
        if len(h_well.thermograms) > 0:
            therm1 = session.query(Thermogram).filter(
                Thermogram.h_well_id == h_well.id,
                Thermogram.date_time > target_date
            ).order_by(Thermogram.date_time.asc()).first()
            therm2 = session.query(Thermogram).filter(
                Thermogram.h_well_id == h_well.id,
                Thermogram.date_time < target_date
            ).order_by(Thermogram.date_time.desc()).first()

            therm1_diff = (therm1.date_time - target_datetime).total_seconds() if therm1 else math.inf
            therm2_diff = (target_datetime - therm2.date_time).total_seconds() if therm2 else math.inf

            result_therm = therm1 if therm1_diff < therm2_diff else therm2
            result_therm_diff = (abs(result_therm.date_time - target_datetime).total_seconds()) / 86400
            if result_therm_diff < 100:
                xs_therm = [t[2] for t in json.loads(result_therm.therm_data) if len(t) > 2]
                ys_therm = [t[3] for t in json.loads(result_therm.therm_data) if len(t) > 2]
                ts_therm = [t[1] for t in json.loads(result_therm.therm_data) if len(t) > 2]
                list_x += xs_therm
                list_y += ys_therm
                list_t += ts_therm
    draw_map(list_x, list_y, list_t, f'{research.object.title} {target_date.strftime("%d.%m.%Y")}', False)
    result = QMessageBox.question(MainWindow, 'Сохранить рабочий набор?', 'Сохранить рабочий набор?', QMessageBox.Yes, QMessageBox.No)
    if result == QMessageBox.Yes:
        result_table = pd.DataFrame({'x': list_x, 'y': list_y, 'temp': list_t})
        file_name = f'{research.object.title}_{target_date.strftime("%d.%m.%Y")}.xlsx'
        fn = QFileDialog.getSaveFileName(caption='Сохранить рабочий набор', directory=file_name, filter="Excel Files (*.xlsx)")
        result_table.to_excel(fn[0], index=False)
        set_info(f'Рабочий набор сохранен в файл {fn[0]}', 'green')
    else:
        pass


def test():
    h = session.query(HorizontalWell).filter_by(object_id=get_obj_monitor_id()).all()
    x = [i.x_coord for i in h]
    y = [i.y_coord for i in h]
    title = [i.title for i in h]
    fig, ax = plt.subplots(figsize=(16, 8))

    ax.scatter(x, y)

    for i, label in enumerate(title):
        plt.annotate(label, (x[i], y[i]))

    plt.show()