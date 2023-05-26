import lasio
import numpy as np
from pyqtgraph import DateAxisItem
import matplotlib.dates as mdates

from func import *

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
        item = QListWidgetItem(h_well.title)
        item.setData(Qt.UserRole, h_well.id)
        ui.listWidget_h_well.addItem(item)
    ui.listWidget_h_well.sortItems()


def check_inclinometry_h_well():
    """Проверить инклинометрические данные горизонтальных скважин"""
    pass


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
        item = QListWidgetItem(incl.parameter)
        item.setData(Qt.UserRole, incl.id)
        item.setBackground(QBrush(QColor('#ADFCDF')))
        ui.listWidget_param_h_well.addItem(item)
    for p in session.query(ParameterHWell).filter_by(h_well_id=get_h_well_id()).all():
        if p.parameter == 'Инклинометрия':
            continue
        item = QListWidgetItem(p.parameter)
        item.setData(Qt.UserRole, p.id)
        ui.listWidget_param_h_well.addItem(item)


def add_h_well():
    """Добавить горизонтальную скважину"""
    pass


def remove_h_well():
    """Удалить горизонтальную скважину"""
    pass


def edit_h_well():
    """Редактировать горизонтальную скважину"""
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
            las = lasio.read(os.path.join(file_dir, file))
            if str(las.well['WELL'].value) != ui.listWidget_h_well.currentItem().text():
                set_info(f'Выбраная скважина ({ui.listWidget_h_well.currentItem().text()}) не совпадает с указанной '
                         f'в las-файле - {las.well["WELL"].value}', 'red')
                continue
            date_time = datetime.datetime.strptime(las.well['DATE'].value, '%d.%m.%Y %H-%M-%S')
            therm = session.query(Thermogram).filter_by(h_well_id=h_well_id, date_time=date_time).first()
            if therm:
                session.query(Thermogram).filter_by(h_well_id=h_well_id, date_time=date_time).update(
                    {'therm_data': json.dumps(dict(zip(las['DEPTH'], las['TEMP'])))}, synchronize_session='fetch')
            else:
                new_therm = Thermogram(
                    h_well_id=h_well_id,
                    date_time=date_time,
                    therm_data=json.dumps(dict(zip(las['DEPTH'], las['TEMP'])))
                )
                session.add(new_therm)
            n_load += 1
    session.commit()
    set_info(f'Для скважины {ui.listWidget_h_well.currentItem().text()} загружено {n_load} термограмм', 'green')
    update_list_thermogram()


def update_list_thermogram():
    """Обновить список термограмм"""
    ui.listWidget_thermogram.clear()
    thermograms = session.query(Thermogram).filter_by(h_well_id=get_h_well_id()).order_by(Thermogram.date_time).all()
    for therm in thermograms:
        item = QListWidgetItem(therm.date_time.strftime('%d.%m.%Y %H-%M-%S'))
        item.setData(Qt.UserRole, therm.id)
        ui.listWidget_thermogram.addItem(item)
    ui.label_25.setText(f'Thermograms: {len(thermograms)}')


def show_thermogram():
    """Показать термограмму"""
    therm = session.query(Thermogram).filter_by(id=get_therm_id()).first()
    if not therm:
        return
    therm_data = json.loads(therm.therm_data)
    x = [float(length) for length in therm_data.keys()]
    y = [float(temp) for temp in therm_data.values()]
    ui.graph.clear()
    curve = pg.PlotCurveItem(x=x, y=y, pen='r', name='Температура')
    ui.graph.addItem(curve)
    ui.graph.showGrid(x=True, y=True)  # отображаем сетку на графике


def show_corr_therm():
    """Показать коррелируемые термограммы"""
    therm = session.query(Thermogram).filter_by(id=get_therm_id()).first()
    temp_curr = [float(temp) for temp in json.loads(therm.therm_data).values()]
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(111)
    n_corr = 0
    for t in session.query(Thermogram).filter_by(h_well_id=get_h_well_id()).order_by(Thermogram.date_time).all():
        temp_list = [float(temp) for temp in json.loads(t.therm_data).values()]
        if len(temp_curr) != len(temp_list):
            continue
        corr_spear, _ =spearmanr(temp_curr, temp_list)
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
    therm = session.query(Thermogram).filter_by(id=get_therm_id()).first()
    temp_curr = [float(temp) for temp in json.loads(therm.therm_data).values()]
    n_rem = 0
    for t in session.query(Thermogram).filter_by(h_well_id=get_h_well_id()).all():
        temp_list = [float(temp) for temp in json.loads(t.therm_data).values()]
        if len(temp_curr) != len(temp_list):
            continue
        corr_spear, _ =spearmanr(temp_curr, temp_list)
        if corr_spear > ui.doubleSpinBox_corr_therm.value():
            session.delete(t)
            n_rem += 1
    session.commit()
    set_info(f'Удалено {n_rem} термограмм', 'green')
    update_list_thermogram()


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
    data_incl = session.query(ParameterHWell).join(HorizontalWell).filter(
        HorizontalWell.object_id == get_obj_monitor_id(),
        ParameterHWell.parameter == 'Инклинометрия').all()
    all_x, all_y, all_z = [], [], []
    for incl in data_incl:
        coord_inc = json.loads(incl.data)
        xs = [coord[0] for coord in coord_inc]
        ys = [coord[1] for coord in coord_inc]
        zs = [coord[2] for coord in coord_inc]
        all_x.extend(xs)
        all_y.extend(ys)
        all_z.extend(zs)

    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(all_x, all_y, all_z, '.')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
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

    plt.tight_layout()
    plt.show()
