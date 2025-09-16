import numpy as np
from PyQt5.QtWidgets import QListWidget

from func import *
from monitoring import update_list_h_well
from qt.add_well_dialog import *
from qt.add_boundary_dialog import *
from qt.well_loader import *


def add_well():
    """Добавить новую скважину в БД"""
    Add_Well = QtWidgets.QDialog()
    ui_w = Ui_add_well()
    ui_w.setupUi(Add_Well)
    Add_Well.show()
    Add_Well.setAttribute(Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    ui_w.lineEdit_well_alt.setText('0')
    ui_w.toolButton_del.hide()
    def well_to_db():
        name_well = ui_w.lineEdit_well_name.text()
        x_well = process_string(ui_w.lineEdit_well_x.text())
        y_well = process_string(ui_w.lineEdit_well_y.text())
        alt_well = process_string(ui_w.lineEdit_well_alt.text())
        if name_well != '' and x_well != '' and y_well != '' and alt_well != '':
            new_well = Well(name=name_well, x_coord=float(x_well), y_coord=float(y_well), alt=float(alt_well))
            session.add(new_well)
            session.commit()
            update_list_well(select_well=True)
            Add_Well.close()
            set_info(f'Добавлена новая скважина - "{name_well}".', 'green')

    def cancel_add_well():
        Add_Well.close()

    ui_w.buttonBox.accepted.connect(well_to_db)
    ui_w.buttonBox.rejected.connect(cancel_add_well)
    Add_Well.exec_()


def search_well():
    """Фильтрация скважин с подсветкой и автопрокруткой к первому совпадению"""
    search_text = ui.lineEdit_well_search.text().lower().strip()
    first_match = None  # Для хранения первого совпадения

    ui.listWidget_well.setUpdatesEnabled(False)
    try:
        for i in range(ui.listWidget_well.count()):
            item = ui.listWidget_well.item(i)
            item_text = item.text().lower()
            matches = search_text in item_text if search_text else False

            # Запоминаем первое совпадение
            if matches and first_match is None:
                first_match = item

        # Прокручиваем к первому совпадению
        if first_match:
            ui.listWidget_well.scrollToItem(
                first_match,
                QListWidget.PositionAtTop  # Прокрутить чтобы элемент был сверху
            )
            ui.listWidget_well.setCurrentItem(first_match)  # Опционально: выделить элемент
    finally:
        ui.listWidget_well.setUpdatesEnabled(True)


def edit_well():
    if not get_well_id():
        set_info('Скважина не выбрана', 'red')
        QMessageBox.critical(MainWindow, 'Ошибка', 'Скважина не выбрана')
    else:
        """Изменить параметры скважины в БД"""
        Add_Well = QtWidgets.QDialog()
        ui_w = Ui_add_well()
        ui_w.setupUi(Add_Well)
        Add_Well.show()
        Add_Well.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

        well = session.query(Well).filter(Well.id == get_well_id()).first()
        ui_w.lineEdit_well_x.setText(str(well.x_coord))
        ui_w.lineEdit_well_y.setText(str(well.y_coord))
        ui_w.lineEdit_well_alt.setText(str(well.alt))
        ui_w.lineEdit_well_name.setText(well.name)
        def well_update():
            name_well = ui_w.lineEdit_well_name.text()
            x_well = ui_w.lineEdit_well_x.text()
            y_well = ui_w.lineEdit_well_y.text()
            alt_well = ui_w.lineEdit_well_alt.text()
            if name_well != '' and x_well != '' and y_well != '' and alt_well != '':
                session.query(Well).filter(Well.id == get_well_id()).update(
                    {'name': name_well, 'x_coord': float(x_well), 'y_coord': float(y_well), 'alt': float(alt_well)},
                    synchronize_session="fetch")
                session.commit()
                update_list_well(select_well=True, selected_well_id=get_well_id())
                Add_Well.close()
                set_info(f'Изменены параметры скважины - "{name_well}".', 'rgb(188, 160, 3)')

        def cancel_add_well():
            Add_Well.close()

        def well_delete():
            name_well = ui.listWidget_well.currentItem().text()
            session.query(Well).filter(Well.id == get_well_id()).delete()
            session.commit()
            update_list_well()
            Add_Well.close()
            set_info(f'Удалена скважина - "{name_well}".', 'rgb(188, 160, 3)')

        ui_w.buttonBox.accepted.connect(well_update)
        ui_w.buttonBox.rejected.connect(cancel_add_well)
        ui_w.toolButton_del.clicked.connect(well_delete)
        Add_Well.exec_()


def add_wells():
    """Пакетное добавление новых скважин в БД из файла Excel/TXT"""
    try:
        file_name = QFileDialog.getOpenFileName(
            caption='Выберите файл Excel или TXT (разделитель ";")',
            filter='*.xls *.xlsx *.txt'
        )[0]
        set_info(file_name, 'blue')
        if file_name.lower().endswith('.txt'):
            try:
                pd_wells = pd.read_table(file_name, header=0, sep=';')
            except UnicodeDecodeError:
                pd_wells = pd.read_table(file_name, header=0, encoding='cp1251', sep=';')
        else:
            pd_wells = pd.read_excel(file_name, header=0)
    except FileNotFoundError:
        return

    pd_wells = clean_dataframe(pd_wells)

    WellLoader = QtWidgets.QDialog()
    ui_wl = Ui_WellLoader()
    ui_wl.setupUi(WellLoader)
    WellLoader.show()
    WellLoader.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    list_combobox = [
        ui_wl.comboBox_name, ui_wl.comboBox_x, ui_wl.comboBox_y,
        ui_wl.comboBox_alt, ui_wl.comboBox_layers, ui_wl.comboBox_opt
    ]
    for cmbx in list_combobox:
        for i in pd_wells.columns:
            cmbx.addItem(i)
    ui_wl.lineEdit_empty.setText('-999')

    def add_well_layer():
        list_layers = [] if ui_wl.lineEdit_layers.text() == '' else ui_wl.lineEdit_layers.text().split('/')
        if ui_wl.comboBox_layers.currentText() not in list_layers:
            list_layers.append(str(ui_wl.comboBox_layers.currentText()))
            ui_wl.lineEdit_layers.setText('/'.join(list_layers))

    def add_well_option():
        list_opt = [] if ui_wl.lineEdit_opt.text() == '' else ui_wl.lineEdit_opt.text().split('/')
        if ui_wl.comboBox_opt.currentText() not in list_opt:
            list_opt.append(str(ui_wl.comboBox_opt.currentText()))
            ui_wl.lineEdit_opt.setText('/'.join(list_opt))

    def load_wells():
        ui.progressBar.setMaximum(len(pd_wells.index))
        name_cell = ui_wl.comboBox_name.currentText()
        x_cell = ui_wl.comboBox_x.currentText()
        y_cell = ui_wl.comboBox_y.currentText()
        alt_cell = ui_wl.comboBox_alt.currentText()
        empty_value = '' if ui_wl.lineEdit_empty.text() == '' else int(ui_wl.lineEdit_empty.text())
        list_layers = [] if ui_wl.lineEdit_layers.text() == '' else ui_wl.lineEdit_layers.text().split('/')
        list_opt = [] if ui_wl.lineEdit_opt.text() == '' else ui_wl.lineEdit_opt.text().split('/')

        distance_threshold = 5.0   # метры
        name_ratio = 0.5           # доля совпадения названий
        n_new, n_update = 0, 0

        for i in pd_wells.index:
            try:
                name = str(pd_wells[name_cell][i])
                x = float(process_string(pd_wells[x_cell][i]))
                y = float(process_string(pd_wells[y_cell][i]))
            except ValueError:
                continue

            # поиск кандидатов поблизости
            candidates = session.query(Well).filter(
                Well.x_coord.between(x - distance_threshold, x + distance_threshold),
                Well.y_coord.between(y - distance_threshold, y + distance_threshold)
            ).all()

            curr_well = None
            for cand in candidates:
                dist = math.hypot((cand.x_coord or 0) - x, (cand.y_coord or 0) - y)
                name_sim = SequenceMatcher(None, cand.name or "", name).ratio()
                if dist <= distance_threshold and name_sim >= name_ratio:
                    curr_well = cand
                    break

            if curr_well:
                n_update += 1
                set_info(f'Скважина {curr_well.name} уже есть в БД', 'red')
                alt = 0 if pd_wells[alt_cell][i] == '' else float(process_string(pd_wells[alt_cell][i]))
                session.query(Well).filter_by(id=curr_well.id).update(
                    {'alt': alt},
                    synchronize_session="fetch"
                )
                for lr in list_layers:
                    try:
                        if pd_wells[lr][i] != empty_value:
                            bound = session.query(Boundary).filter(
                                Boundary.well_id == curr_well.id,
                                Boundary.title == str(lr)
                            ).first()
                            if not bound:
                                if ui_wl.checkBox_deep.isChecked():
                                    depth = round(float(process_string(pd_wells[lr][i])), 2)
                                else:
                                    depth = round(curr_well.alt - float(process_string(pd_wells[lr][i])), 2)
                                session.add(Boundary(
                                    well_id=curr_well.id,
                                    depth=depth,
                                    title=str(lr)
                                ))
                                session.commit()
                    except ValueError:
                        continue
                for opt in list_opt:
                    try:
                        if pd_wells[opt][i] != empty_value:
                            well_opt = session.query(WellOptionally).filter(
                                WellOptionally.well_id == curr_well.id,
                                WellOptionally.option == opt,
                                WellOptionally.value == str(pd_wells[opt][i])
                            ).first()
                            if not well_opt:
                                session.add(WellOptionally(
                                    well_id=curr_well.id,
                                    option=opt,
                                    value=str(pd_wells[opt][i])
                                ))
                                session.commit()
                    except ValueError:
                        continue
            else:
                try:
                    n_new += 1
                    new_well = Well(
                        name=name,
                        x_coord=x,
                        y_coord=y,
                        alt=round(float(process_string(pd_wells[alt_cell][i])), 2),
                    )
                    session.add(new_well)
                    session.commit()
                    for lr in list_layers:
                        if pd_wells[lr][i] != empty_value:
                            if ui_wl.checkBox_deep.isChecked():
                                depth = round(float(process_string(pd_wells[lr][i])), 2)
                            else:
                                depth = round(new_well.alt - float(process_string(pd_wells[lr][i])), 2)
                            session.add(Boundary(
                                well_id=new_well.id,
                                depth=depth,
                                title=str(lr)
                            ))
                except ValueError:
                    continue
                for opt in list_opt:
                    try:
                        if pd_wells[opt][i] != empty_value:
                            session.add(WellOptionally(
                                well_id=new_well.id,
                                option=opt,
                                value=str(pd_wells[opt][i])
                            ))
                    except ValueError:
                        continue

            session.commit()
            ui.progressBar.setValue(i + 1)

        session.commit()
        update_list_well(select_well=True)
        set_info(f'Добавлено {n_new} скважин, обновлено {n_update} скважин', 'green')

    def cancel_load():
        WellLoader.close()

    ui_wl.pushButton_add_layer.clicked.connect(add_well_layer)
    ui_wl.pushButton_add_opt.clicked.connect(add_well_option)
    ui_wl.buttonBox.accepted.connect(load_wells)
    ui_wl.buttonBox.rejected.connect(cancel_load)
    WellLoader.exec_()



def add_data_well():
    """ Добавить данные по скважинам """
    try:
        category, value = ui.lineEdit_string.text().split(': ')[0], ui.lineEdit_string.text().split(': ')[1]

        existing_data = session.query(WellOptionally).filter(
            WellOptionally.well_id == get_well_id(),
            WellOptionally.option == category,
            WellOptionally.value == value
        ).first()

        if existing_data:
            set_info(f"Данные '{category}: {value}' уже существуют", 'red')
            return

        new_data_well = WellOptionally(
            well_id = get_well_id(),
            option = category,
            value = value
        )
        session.add(new_data_well)
        session.commit()
        set_info(f"Данные '{category}: {value}' успешно добавлены", 'green')
        show_data_well()

    except IndexError:
        set_info('Введите данные в формате "Категория: Значение"', 'red')
        return


def delete_data_well():
    """ Удалить данные по скважинам """
    try:
        category, value = ui.lineEdit_string.text().split(': ')[0], ui.lineEdit_string.text().split(': ')[1]

        # Находим запись для удаления
        data_to_delete = session.query(WellOptionally).filter(
            WellOptionally.well_id == get_well_id(),
            WellOptionally.option == category,
            WellOptionally.value == value
        ).first()

        if data_to_delete:
            session.delete(data_to_delete)
            session.commit()
            set_info(f"Данные '{category}: {value}' успешно удалены", 'green')
            show_data_well()
        else:
            set_info("Данные не найдены для удаления", 'red')

    except IndexError:
        set_info('Введите данные в формате "Категория: Значение"', 'red')
        return



def show_data_well():
    ui.textEdit_datawell.clear()
    if ui.checkBox_profile_intersec.isChecked():
        inter = session.query(Intersection).filter_by(id=get_well_id()).first()
        if inter:
            therm = session.query(Thermogram).filter_by(id=inter.therm_id).first()
            target_date = session.query(Research).filter_by(id=get_research_id()).first().date_research
            target_datetime = datetime.datetime.combine(target_date, datetime.datetime.min.time())
            ui.textEdit_datawell.append(f'<p><b>Пересечение:</b> {inter.name}</p>'
                                        f'<p><b>X:</b> {round(inter.x_coord, 2)}</p>'
                                        f'<p><b>Y:</b> {round(inter.y_coord, 2)}</p>'
                                        f'<p><b>Дата термограммы:</b> {therm.date_time.strftime("%d.%m.%Y")} '
                                        f'({(therm.date_time - target_datetime).days} дней)</p>'
                                        f'<p><b>Температура:</b> {round(inter.temperature, 2)} °C</p>')
            for i in range(ui.comboBox_object_monitor.count()):
                if ui.comboBox_object_monitor.itemData(i)['id'] == get_object_id():
                    ui.comboBox_object_monitor.setCurrentIndex(i)
                    break
            update_list_h_well()
            for i in range(ui.listWidget_h_well.count()):
                if ui.listWidget_h_well.item(i).data(Qt.UserRole) == inter.thermogram.h_well_id:
                    ui.listWidget_h_well.setCurrentRow(i)
                    break
            for i in range(ui.listWidget_thermogram.count()):
                if ui.listWidget_thermogram.item(i).data(Qt.UserRole) == inter.therm_id:
                    ui.listWidget_thermogram.setCurrentRow(i)
                    break
            data_therm = [i for i in json.loads(inter.thermogram.therm_data) if len(i) > 2]
            int_therm = pg.InfiniteLine(pos=data_therm[inter.i_therm][0], angle=90, pen=pg.mkPen(color='#ff7800',width=3, dash=[8, 2]))
            ui.graph.addItem(int_therm)
            int_temp = pg.InfiniteLine(pos=inter.temperature, angle=0, pen=pg.mkPen(color='#ff7800',width=1, dash=[2, 2]))
            ui.graph.addItem(int_temp)
            text_item = pg.TextItem(text=get_profile_name().split(' (')[0], color='#ff7800')
            text_item.setFont(QtGui.QFont("Arial", 8))
            text_item.setPos(data_therm[inter.i_therm][0] + 5, data_therm[inter.i_therm][1] - 5)
            ui.graph.addItem(text_item)

    else:
        well = session.query(Well).filter_by(id=get_well_id()).first()
        if not well:
            return

        count_well_log = session.query(WellLog).filter_by(well_id=well.id).count()

        text_content = []
        if count_well_log > 0:
            text_content.append(
                f'<p style="background-color:#ADFCDF">'
                f'<b>Количество каротажных кривых:</b> {count_well_log}</p>'
            )

        text_content.extend([
            f'<p><b>Скважина №</b> {well.name}</p>',
            f'<p><b>X:</b> {well.x_coord}</p>',
            f'<p><b>Y:</b> {well.y_coord}</p>',
            f'<p><b>Альтитуда:</b> {well.alt} м.</p>'
        ])

        for opt in session.query(WellOptionally).filter_by(well_id=well.id):
            text_content.append(f'<p><b>{opt.option}:</b> {opt.value}</p>')

        ui.textEdit_datawell.setHtml(''.join(text_content))






def add_boundary():
    """Добавить новую границу для скважины в БД"""
    Add_Boundary = QtWidgets.QDialog()
    ui_b = Ui_add_bondary()
    ui_b.setupUi(Add_Boundary)
    Add_Boundary.show()
    Add_Boundary.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    last_boundary = session.query(Boundary).order_by(Boundary.id.desc()).first()
    ui_b.lineEdit_title.setText(last_boundary.title)
    ui_b.lineEdit_depth.setText(str(last_boundary.depth))

    def boundary_to_db():
        title_boundary = ui_b.lineEdit_title.text()
        depth_boundary = ui_b.lineEdit_depth.text()
        if title_boundary != '' and depth_boundary != '':
            new_boundary = Boundary(well_id=get_well_id(), title=title_boundary, depth=float(depth_boundary))
            session.add(new_boundary)
            session.commit()
            update_boundaries()
            update_list_well(select_well=True, selected_well_id=get_well_id())
            Add_Boundary.close()
            set_info(f'Добавлена новая граница для текущей скважины - "{title_boundary}".', 'green')

    def cancel_add_boundary():
        Add_Boundary.close()

    ui_b.buttonBox.accepted.connect(boundary_to_db)
    ui_b.buttonBox.rejected.connect(cancel_add_boundary)
    Add_Boundary.exec_()


def remove_boundary():
    try:
        session.query(Boundary).filter(Boundary.id == get_boundary_id()).delete()
        session.commit()
        update_boundaries()
        update_list_well(select_well=True, selected_well_id=get_well_id())
    except AttributeError:
        set_info('Необходимо выбрать границу для удаления', 'red')
        return


def draw_bound_int():
    for key, value in globals().items():
        if key.startswith('int_bound_') or key.startswith('bound_'):
            radarogramma.removeItem(globals()[key])

    if ui.listWidget_bound.currentItem():
        bound = session.query(Boundary).filter(Boundary.id == get_boundary_id()).first()
        x_prof = json.loads(session.query(Profile.x_pulc).filter(Profile.id == get_profile_id()).first()[0])
        y_prof = json.loads(session.query(Profile.y_pulc).filter(Profile.id == get_profile_id()).first()[0])
        index, dist = closest_point(bound.well.x_coord, bound.well.y_coord, x_prof, y_prof)
        # Получение значения средней скорости в среде
        Vmean = ui.doubleSpinBox_vsr.value()
        # Расчёт значения глубины, которое будет использоваться для отображения точки и текста на графике
        d = ((bound.depth * 100) / Vmean) / 8
        # Создание графического объекта точки с переданными параметрами
        scatter = pg.ScatterPlotItem(x=[index], y=[d], symbol='o', pen=pg.mkPen(None),
                                     brush=pg.mkBrush(color='#FFF500'), size=10)
        radarogramma.addItem(scatter)  # Добавление графического объекта точки на график
        globals()[f'bound_scatter'] = scatter  # Сохранение ссылки на графический объект точки в globals()

        # Создание графического объекта текста с переданными параметрами
        text_item = pg.TextItem(text=f'{bound.title} ({bound.depth})', color='white')
        text_item.setPos(index + 10, d)  # Установка позиции текста на графике
        radarogramma.addItem(text_item)  # Добавление графического объекта текста на график
        globals()[f'bound_text'] = text_item  # Сохранение ссылки на графический объект текста в globals()
        dmin = ((bound.depth * 100) / ui.doubleSpinBox_vmin.value()) / 8
        dmax = ((bound.depth * 100) / ui.doubleSpinBox_vmax.value()) / 8
        lmin = pg.InfiniteLine(pos=dmin, angle=0, pen=pg.mkPen(color='white', width=1, dash=[2, 2]))
        lmax = pg.InfiniteLine(pos=dmax, angle=0, pen=pg.mkPen(color='white', width=1, dash=[2, 2]))
        radarogramma.addItem(lmin)
        radarogramma.addItem(lmax)
        globals()[f'int_bound_min'] = lmin
        globals()[f'int_bound_max'] = lmax

        text_min = pg.TextItem(text=f'{bound.title} Vmin', color='white')
        text_min.setPos(0, dmin - 30)
        radarogramma.addItem(text_min)
        globals()[f'int_bound_text_min'] = text_min
        text_max = pg.TextItem(text=f'{bound.title} Vmax', color='white')
        text_max.setPos(0, dmax - 30)
        radarogramma.addItem(text_max)
        globals()[f'int_bound_text_max'] = text_max

        ui.doubleSpinBox_target_val.setValue(bound.depth)


def profile_wells_to_Excel():
    """Формирование таблицы Excel для профилей и скважин текущего объекта"""
    list_col = ['Объект', 'Профиль', 'Скважина', 'Расстояние', 'Кол-во каротажей', 'Данные скважины']

    object = session.query(GeoradarObject).filter_by(id=get_object_id()).first()
    pd_profile_wells = pd.DataFrame(columns=list_col)
    set_info('Начало создания таблицы профилей и скважин для текущего объекта...', 'blue')
    for research in object.researches:
        ui.progressBar.setMaximum(len(research.profiles))
        for n, profile in enumerate(research.profiles):
            ui.progressBar.setValue(n+1)
            wells = get_list_nearest_well(profile.id)
            if not wells:
                continue
            for well in wells:
                well_dict = dict()
                well_dict['Объект'] = f"{object.title}\n\n{research.date_research}"
                well_dict['Профиль'] = profile.title
                well_dict['Скважина'] = well[0].name
                well_dict['Расстояние'] = round(well[2], 2)
                well_dict['Кол-во каротажей'] = len(well[0].well_logs)
                well_dict['Данные скважины'] = f"X: {well[0].x_coord}\nY: {well[0].y_coord}\nАльтитуда: {well[0].alt}\n"
                for opt in session.query(WellOptionally).filter_by(well_id=well[0].id):
                    well_dict['Данные скважины'] += f"{opt.option}: {opt.value}\n"

                pd_profile_wells = pd.concat([pd_profile_wells, pd.DataFrame(data=well_dict, columns=list_col,
                                                                             index=[0])], axis=0, ignore_index=True)

    set_info('Создание таблицы профилей и скважин для текущего объекта завершено', 'blue')

    file_name = QFileDialog.getSaveFileName(
        None,
        "Сохранить данные профилей и скважин",
        "",
        "Excel Files (*.xlsx)"
    )[0]
    if not file_name:
        return
    # Добавляем .xlsx, если его нет в конце
    if not file_name.lower().endswith('.xlsx'):
        file_name += '.xlsx'
    pd_profile_wells.to_excel(file_name, index=False)

    # Сохранение с фиксированной шириной столбцов
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        pd_profile_wells.to_excel(writer, index=False, sheet_name='Данные')

        workbook = writer.book
        worksheet = writer.sheets['Данные']

        # Создаем форматы
        header_format = workbook.add_format({
            'bold': True,
            'align': 'center',
            'valign': 'vcenter',
            'border': 1
        })

        cell_format = workbook.add_format({
            'text_wrap': True,
            'align': 'left',
            'valign': 'vcenter'
        })

        # Устанавливаем фиксированную ширину для всех столбцов (в символах)
        column_widths = {
            'Объект': 15,
            'Профиль': 15,
            'Скважина': 12,
            'Расстояние': 12,
            'Кол-во каротажей': 17,
            'Данные скважины': 40
        }

        # Применяем к заголовкам
        worksheet.set_row(0, None, header_format)

        # Применяем ширину и формат
        for idx, col_name in enumerate(pd_profile_wells.columns):
            worksheet.set_column(
                idx, idx,
                width=column_widths.get(col_name, 15),  # 15 - ширина по умолчанию
                cell_format=cell_format
            )


def all_profile_wells_to_Excel():
    """Формирование таблицы Excel для профилей и скважин всех объектов"""
    columns = ['Объект', 'Профиль', 'Скважина', 'Расстояние', 'Кол-во каротажей', 'Данные скважины']
    records = []
    objects = session.query(GeoradarObject).all()
    set_info('Начало создания таблицы профилей и скважин для всех объектов...', 'blue')
    for obj in objects:
        for research in tqdm(obj.researches, desc=f'Формирование строк для объекта "{obj.title}"'):
            ui.progressBar.setMaximum(len(research.profiles))
            for n, profile in enumerate(research.profiles):
                ui.progressBar.setValue(n + 1)
                wells = get_list_nearest_well(profile.id)
                if not wells:
                    continue
                for well in wells:
                    well_data = []
                    well_obj = well[0]

                    # Основные данные
                    well_data.append(f"{obj.title}\n\n{research.date_research}")
                    well_data.append(profile.title)
                    well_data.append(well_obj.name)
                    well_data.append(round(well[2], 2))
                    well_data.append(len(well_obj.well_logs))

                    # Данные скважины
                    well_info = f"X: {well_obj.x_coord}\nY: {well_obj.y_coord}\nАльтитуда: {well_obj.alt}"
                    for opt in session.query(WellOptionally).filter_by(well_id=well_obj.id):
                        well_info += f"\n{opt.option}: {opt.value}"

                    well_data.append(well_info)
                    records.append(well_data)

        set_info(f'Сформированы строки для объекта "{obj.title}"', 'green')

    df_profile_wells = pd.DataFrame(records, columns=columns)

    set_info('Создание таблицы профилей и скважин для всех объектов завершено', 'blue')

    file_name = QFileDialog.getSaveFileName(
        None,
        "Сохранить данные профилей и скважин",
        "",
        "Excel Files (*.xlsx)"
    )[0]
    if not file_name:
        return
    # Добавляем .xlsx, если его нет в конце
    if not file_name.lower().endswith('.xlsx'):
        file_name += '.xlsx'

    # Сохранение с фиксированной шириной столбцов
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        df_profile_wells.to_excel(writer, index=False, sheet_name='Данные')

        workbook = writer.book
        worksheet = writer.sheets['Данные']

        # Создаем форматы
        header_format = workbook.add_format({
            'bold': True,
            'align': 'center',
            'valign': 'vcenter',
            'border': 1
        })

        cell_format = workbook.add_format({
            'text_wrap': True,
            'align': 'left',
            'valign': 'vcenter'
        })

        # Устанавливаем фиксированную ширину для всех столбцов (в символах)
        column_widths = {
            'Объект': 15,
            'Профиль': 15,
            'Скважина': 12,
            'Расстояние': 12,
            'Кол-во каротажей': 17,
            'Данные скважины': 40
        }

        # Применяем к заголовкам
        worksheet.set_row(0, None, header_format)

        # Применяем ширину и формат
        for idx, col_name in enumerate(df_profile_wells.columns):
            worksheet.set_column(
                idx, idx,
                width=column_widths.get(col_name, 15),  # 15 - ширина по умолчанию
                cell_format=cell_format
            )



def deduplicate_wells(session, distance_threshold: float = 5.0,
                      name_ratio: float = 0.5) -> None:
    """
    Удаляет дублирующиеся скважины и переносит их зависимости в «каноническую» запись.

    Параметры:
        distance_threshold – максимально допустимое расстояние между координатами (метры);
        name_ratio         – минимальная доля совпадения названий (0..1).
    """
    wells = session.query(Well).order_by(Well.id).all()  # все скважины по возрастанию id
    removed_ids = set()                                  # id скважин, помеченных на удаление

    # Счётчики для итоговой статистики
    processed = duplicates = 0
    boundary_moved = optional_moved = logs_moved = 0
    mlp_moved = reg_moved = 0

    # Основной цикл с прогресс‑баром
    for well in tqdm(wells, desc='Обработка скважин'):
        processed += 1
        if well.id in removed_ids:                       # пропускаем уже удалённые
            continue

        for other in wells:
            if other.id in removed_ids or other.id <= well.id:
                continue

            # Расстояние между координатами
            dist = math.hypot((well.x_coord or 0) - (other.x_coord or 0),
                         (well.y_coord or 0) - (other.y_coord or 0))
            # Сходство названий
            name_sim = SequenceMatcher(None, well.name or "",
                                       other.name or "").ratio()

            if dist <= distance_threshold and name_sim >= name_ratio:
                duplicates += 1

                # --- Boundary ---
                for b in list(other.boundaries):
                    if not any(abs(b.depth - bb.depth) < 1e-6 and b.title == bb.title
                               for bb in well.boundaries):
                        b.well = well
                        boundary_moved += 1

                # --- WellOptionally ---
                for opt in list(other.well_optionally):
                    if not any(opt.option == o.option and opt.value == o.value
                               for o in well.well_optionally):
                        opt.well = well
                        optional_moved += 1

                # --- WellLog ---
                for log in list(other.well_logs):
                    if not any(log.curve_name == l.curve_name for l in well.well_logs):
                        log.well = well
                        logs_moved += 1

                # --- MarkupMLP ---
                for m in list(other.markups_mlp):
                    if not any(
                        m.analysis_id == ex.analysis_id and
                        m.profile_id == ex.profile_id and
                        m.formation_id == ex.formation_id and
                        m.type_markup == ex.type_markup
                        for ex in well.markups_mlp
                    ):
                        m.well = well
                        mlp_moved += 1

                # --- MarkupReg ---
                for m in list(other.markups_reg):
                    if not any(
                        m.analysis_id == ex.analysis_id and
                        m.profile_id == ex.profile_id and
                        m.formation_id == ex.formation_id and
                        m.type_markup == ex.type_markup
                        for ex in well.markups_reg
                    ):
                        m.well = well
                        reg_moved += 1

                removed_ids.add(other.id)                # помечаем дубликат
                session.delete(other)                    # удаляем из сессии

        # Промежуточный коммит и отчёт каждые 500 скважин
        if processed % 500 == 0:
            session.commit()
            summary = (
                f"Обработано {processed} скважин, найдено {duplicates} дублей; "
                f"Boundary: {boundary_moved}, WellOptionally: {optional_moved}, "
                f"WellLog: {logs_moved}, MarkupMLP: {mlp_moved}, MarkupReg: {reg_moved}"
            )
            print(summary)
            set_info(summary, "blue")

    # Финальный коммит и итоговая статистика
    session.commit()
    summary = (
        f"Обработано {processed} скважин, найдено {duplicates} дублей; "
        f"Boundary: {boundary_moved}, WellOptionally: {optional_moved}, "
        f"WellLog: {logs_moved}, MarkupMLP: {mlp_moved}, MarkupReg: {reg_moved}"
    )
    print(summary)
    set_info(summary, "blue")


def remove_duplicate_wells():
    deduplicate_wells(session)


def find_nearest_profiles(well_id: int, max_distance: float) -> dict[int, float]:
    """Возвращает словарь {profile_id: минимальное расстояние}.
    Если профилей в пределах max_distance нет, возвращает ближайший профиль без учёта порога.
    """
    well = session.query(Well).filter(Well.id == well_id).first()
    if not well:
        raise ValueError(f"Well with id {well_id} not found")

    candidates = {}
    profiles = session.query(Profile).all()

    for prof in profiles:
        if not prof.x_pulc or not prof.y_pulc:
            continue
        xs, ys = json.loads(prof.x_pulc), json.loads(prof.y_pulc)
        _, dist = closest_point(well.x_coord, well.y_coord, xs, ys)
        if dist <= max_distance:
            candidates[prof.id] = dist

    if candidates:
        return candidates

    # профили внутри max_distance не найдены – ищем ближайший по всему набору
    nearest_id, min_dist = None, float("inf")
    for prof in profiles:
        if not prof.x_pulc or not prof.y_pulc:
            continue
        xs, ys = json.loads(prof.x_pulc), json.loads(prof.y_pulc)
        _, dist = closest_point(well.x_coord, well.y_coord, xs, ys)
        if dist < min_dist:
            nearest_id, min_dist = prof.id, dist

    return {nearest_id: min_dist} if nearest_id is not None else {}


def find_closest_profile():
    """Поиск ближайшего профиля для выбранной скважины"""
    well_id = get_well_id()
    max_distance = ui.spinBox_well_distance.value()
    dict_profiles = find_nearest_profiles(well_id, max_distance)
    n = 1
    for p_id, dist in dict_profiles.items():
        profile = get_profile_by_id(p_id)
        set_info(f'{n}. {profile.research.object.title} {profile.research.date_research.year} Профиль: {profile.title}  \nрасстояние: {dist:.2f} м.', 'green')
        n += 1