import numpy as np

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
    pass


def load_thermogram_h_well():
    """Загрузить термограммы горизонтальных скважин"""
    pass