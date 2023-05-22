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
    for p in session.query(ParameterHWell).filter_by(h_well_id=get_h_well_id()).all():
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


def load_inclinometry_h_well():
    """Загрузить инклинометрические данные горизонтальных скважин"""
    if ui.lineEdit_string.text() == '':
        well_coord = [0, 0, 0]
    else:
        well_coord = ui.lineEdit_string.text().split(';')
        well_coord = [float(i) for i in well_coord]
    if not len(well_coord) == 3:
        set_info('Неверные координаты горизонтальной скважины, введите "x;y;z"', 'red')
        return
    file_name = QFileDialog.getOpenFileName(MainWindow, 'Выбрать файл инклинометрическии', '', 'Текстовые файлы (*.txt)')[0]
    if file_name:
        coord_inc = calc_coord_inclinometry(file_name, well_coord)
        print(coord_inc)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xs = [coord[0] for coord in coord_inc]
        ys = [coord[1] for coord in coord_inc]
        zs = [coord[2] for coord in coord_inc]

        ax.plot(xs, ys, zs)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # задаем реальные масштабы для всех трех осей
        # x_range = max(xs) - min(xs)
        # y_range = max(ys) - min(ys)
        # z_range = max(zs) - min(zs)
        # max_range = max(x_range, y_range, z_range)
        # x_center = (max(xs) + min(xs)) / 2
        # y_center = (max(ys) + min(ys)) / 2
        # z_center = (max(zs) + min(zs)) / 2
        # ax.set_xlim((x_center - max_range / 2, x_center + max_range / 2))
        # ax.set_ylim((y_center - max_range / 2, y_center + max_range / 2))
        # ax.set_zlim((z_center - max_range / 2, z_center + max_range / 2))



        plt.show()



def load_thermogram_h_well():
    """Загрузить термограммы горизонтальных скважин"""
    pass


def draw_param_h_well():
    """Отрисовать параметр горизонтальных скважин"""
    item = ui.listWidget_param_h_well.currentItem()
    if not item:
        return
    id_param = item.data(Qt.UserRole)
    param = session.query(ParameterHWell).filter_by(id=id_param).first()

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


def calc_coord_inclinometry(input_file_path, initial_coordinates):
    output_coordinates = []

    x, y, z = initial_coordinates
    prev_length = 0

    with open(input_file_path, 'r') as input_file:
        for line in input_file:
            length, dip_angle, azimuth = map(float, line.strip().split('\t'))

            delta_length = length - prev_length  # разница между текущим и предыдущим значением length
            delta_x = delta_length * math.sin(math.radians(180 - dip_angle)) * math.sin(math.radians(azimuth))
            delta_y = delta_length * math.sin(math.radians(180 - dip_angle)) * math.cos(math.radians(azimuth))
            delta_z = delta_length * math.cos(math.radians(180 - dip_angle))

            x += delta_x
            y += delta_y
            z += delta_z

            prev_length = length  # сохраняем текущее значение length для следующей итерации

            output_coordinates.append((x, y, z))

    return output_coordinates

