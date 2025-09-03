import pandas as pd

from filter_well import get_names_boundary
from func import *
from krige import draw_map
from regression import update_list_reg, remove_all_param_geovel_reg, update_list_well_markup_reg
from well import show_data_well


def show_well_log():

    if not get_well_id():
        set_info('Скважина не выбрана', 'red')
        QMessageBox.critical(MainWindow, 'Ошибка', 'Скважина не выбрана')
    else:
        WellLogForm = QtWidgets.QDialog()
        ui_wl = Ui_Form_well_log()
        ui_wl.setupUi(WellLogForm)
        WellLogForm.show()
        WellLogForm.setAttribute(Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

        m_width, m_height = get_width_height_monitor()
        WellLogForm.resize(int(m_width/4), m_height - 200)

        ui_wl.label.setText(f'Каротажные кривые скважины {get_well_name()}')
        boundaries = session.query(Boundary).filter_by(well_id=get_well_id()).all()
        for b in boundaries:
            if 'uf' in b.title or 'уф' in b.title or 'ss' in b.title:
                ui_wl.doubleSpinBox_depth.setValue(b.depth)
                break

        def update_list_well_log():
            well_log = session.query(WellLog).filter(WellLog.well_id == get_well_id()).all()
            ui_wl.listWidget_well_log.clear()
            for log in well_log:
                item = QtWidgets.QListWidgetItem(f'{log.curve_name} ID{log.id}')
                item.setToolTip(f'{log.begin} - {log.end}; {log.step}\n{log.description}')
                ui_wl.listWidget_well_log.addItem(item)


        def draw_well_log():
            try:
                well_log_id = ui_wl.listWidget_well_log.currentItem().text().split(' ID')[-1]
                well_log = session.query(WellLog).filter_by(id=well_log_id).first()
            except AttributeError:
                return

            if not well_log:
                return

            try:
                Y = json.loads(well_log.curve_data)
                Y = np.array(Y)

                # Проверка на нулевой шаг
                if well_log.step == 0:
                    print("Ошибка: шаг (step) равен нулю.")
                    well_log.step = 0.1

                # Подсчёт длины массива X, чтобы совпадал с Y
                X = np.arange(well_log.begin, well_log.begin + len(Y) * well_log.step, well_log.step)

                # В случае если длины всё равно не совпадают — подрежем X
                if len(X) > len(Y):
                    X = X[:len(Y)]
                elif len(Y) > len(X):
                    Y = Y[:len(X)]

                curve = pg.PlotCurveItem(Y, X)

                ui_wl.widget_graph_well_log.clear()
                ui_wl.widget_graph_well_log.showGrid(x=True, y=True)
                ui_wl.widget_graph_well_log.invertY(True)
                ui_wl.widget_graph_well_log.addItem(curve)
                draw_depth_spinbox()

            except Exception as e:
                print(f"Ошибка при построении графика: {e}")


        def load_well_log():
            filename = QtWidgets.QFileDialog.getOpenFileName(
                caption='Выберите файл LAS',
                filter='*.las'
            )[0]

            if filename:
                add_well_log_to_db(filename)
            update_list_well_log()


        def load_well_log_xls():
            filename = QtWidgets.QFileDialog.getOpenFileName(
                caption='Выберите файл Excel',
                filter='*.xlsx'
            )[0]

            if filename:
                add_well_log_xls_to_db(filename)
            update_list_well_log()
            update_list_well(select_well=True, selected_well_id=get_well_id())


        def load_well_log_by_dir():
            filename = QtWidgets.QFileDialog.getExistingDirectory(
                caption='Выберите папку',
            )

            if filename:
                for file in os.listdir(filename):
                    if file.endswith('.las'):
                        add_well_log_to_db(f'{filename}/{file}')
            update_list_well_log()


        def add_well_log_to_db(las_file):
            las = ls.read(las_file)
            list_curves = las.keys()

            for curve in list_curves:
                if curve in ['DEPT', 'DEPH', 'MD', 'DEPTH']:
                    continue
                try:
                    description = (f'Скв.: {las.well["WELL"].value}\n'
                                   f'Площ.:' f'{las.well["AREA"].value}\n'
                                   f'Дата: {las.well["DATE"].value}')
                except KeyError:
                    description = (f'Скв.: {las.well["WELL"].value}\n'
                                   f'Площ.:' f'{las.well["FLD"].value}\n'
                                   f'Дата: {las.well["DATE"].value}')
                new_well_log = WellLog(
                    well_id=get_well_id(),
                    curve_name=curve,
                    curve_data=json.dumps(list(las[curve])),
                    begin=las.well["STRT"].value,
                    end=las.well["STOP"].value,
                    step=las.well["STEP"].value,
                    description=description
                )
                session.add(new_well_log)
            session.commit()


        def add_well_log_xls_to_db(xls_file):
            sep = os.path.sep
            well_name = xls_file.split(sep)[-1].split('.')[0]

            pd_table = pd.read_excel(xls_file)
            curves, bound, age = read_well_log_xls(pd_table, 'DEPTH', 'AGE')

            for name, data  in curves.items():
                description = well_name

                new_well_log = WellLog(
                    well_id=get_well_id(),
                    curve_name=name,
                    curve_data=json.dumps(data),
                    begin=bound[name]['start'],
                    end=bound[name]['end'],
                    step=0.1,
                    description=description
                )
                session.add(new_well_log)

            for age_name, age_depth in age.items():
                new_bound = Boundary(
                    well_id=get_well_id(),
                    depth=age_depth,
                    title=f'{age_name}-scan'
                )
                session.add(new_bound)

            session.commit()


        def create_regression_analysis_by_current_well_log():
            try:
                well_log_id = ui_wl.listWidget_well_log.currentItem().text().split(' ID')[-1]
                well_log = session.query(WellLog).filter_by(id=well_log_id).first()
            except AttributeError:
                set_info('Выберите каротаж', 'red')
                QMessageBox.critical(WellLogForm, 'Ошибка', 'Необходимо выбрать каротаж!')
                return

            if not well_log:
                set_info('Каротаж не найден', 'red')
                QMessageBox.critical(WellLogForm, 'Ошибка', 'Каротаж не найден')
                return

            new_regression_analysis(well_log.curve_name)
            update_list_reg()



        def create_regression_analysis_by_all_well_log():
            for row in range(ui_wl.listWidget_well_log.count()):
                well_log_id = ui_wl.listWidget_well_log.item(row).text().split(' ID')[-1]
                well_log = session.query(WellLog).filter_by(id=well_log_id).first()
                if well_log:
                    new_regression_analysis(well_log.curve_name)
            update_list_reg()


        def new_regression_analysis(well_log_name):
            if session.query(AnalysisReg).filter_by(title=well_log_name).first():
                set_info(f'Анализ {well_log_name} уже существует', 'red')
                QMessageBox.information(WellLogForm, 'Создание нового анализа', f'Анализ {well_log_name} уже '
                                                                                f'существует')
                return

            new_regression = AnalysisReg(
                title=well_log_name
            )
            session.add(new_regression)
            session.commit()

            set_info(f'Анализ {well_log_name} создан', 'green')


        def add_current_well_log_to_regression():
            well_log_id = ui_wl.listWidget_well_log.currentItem().text().split(' ID')[-1]
            well_log = session.query(WellLog).filter_by(id=well_log_id).first()

            analysis_id = get_regmod_id()
            well_id = well_log.well_id
            profile_id = get_profile_id()
            formation_id = get_formation_id()

            if analysis_id and well_id and profile_id and formation_id:
                if session.query(MarkupReg).filter_by(analysis_id=analysis_id, well_id=well_id, profile_id=profile_id, formation_id=formation_id).first():
                    set_info(f'Обучающая скважина уже добавлена в анализ {well_log.curve_name}', 'red')
                    return

                remove_all_param_geovel_reg()
                target_value = get_median_value_from_interval(well_log_id, ui_wl.doubleSpinBox_depth.value(), ui_wl.doubleSpinBox_interval.value())
                if not target_value:
                    set_info(f'Каротаж {well_log.curve_name} не представлен в выбранном интервале', 'red')
                    return

                if target_value < 0:
                    set_info(f'Отрицательное значение каротажа {well_log.curve_name} в интервале', 'red')
                    return

                add_well_log_markup_reg(analysis_id, well_id, profile_id, formation_id, target_value)
                set_info(f'Обучающая скважина {well_log.well.name} добавлена в анализ {well_log.curve_name}', 'green')

                update_list_well_markup_reg()
            else:
                set_info('Выбраны не все параметры - скважина и пласт', 'red')
                QMessageBox.critical(MainWindow, 'Ошибка', 'Выберите все параметры для добавления обучающей скважины!')


        def add_all_well_log_to_regression():
            list_analysis, list_info = [], []

            for row in range(ui_wl.listWidget_well_log.count()):
                well_log_id = ui_wl.listWidget_well_log.item(row).text().split(' ID')[-1]
                well_log = session.query(WellLog).filter_by(id=well_log_id).first()

                if well_log:
                    reg_analysis = session.query(AnalysisReg).filter_by(title=well_log.curve_name).first()

                    if not reg_analysis:
                        set_info(f'Анализ {well_log.curve_name} не найден', 'red')
                        continue

                    analysis_id = reg_analysis.id
                    well_id = well_log.well_id
                    profile_id = get_profile_id()
                    formation_id = get_formation_id()

                    if analysis_id and well_id and profile_id and formation_id:
                        if session.query(MarkupReg).filter_by(analysis_id=analysis_id, well_id=well_id, profile_id=profile_id, formation_id=formation_id).first():
                            set_info(f'Обучающая скважина уже добавлена в анализ {well_log.curve_name}', 'red')
                            continue

                        target_value = get_median_value_from_interval(well_log_id, ui_wl.doubleSpinBox_depth.value(), ui_wl.doubleSpinBox_interval.value())
                        if not target_value:
                            set_info(f'Каротаж {well_log.curve_name} не представлен в выбранном интервале', 'red')
                            continue

                        if target_value < 0:
                            set_info(f'Отрицательное значение каротажа {well_log.curve_name} в интервале', 'red')
                            continue

                        list_analysis.append(f'{reg_analysis.title} id{reg_analysis.id}')
                        list_info.append(f'{well_log.curve_name}: {target_value}')
                        add_well_log_markup_reg(analysis_id, well_id, profile_id, formation_id, target_value)
                        set_info(f'Обучающая скважина {well_log.well.name} добавлена в анализ {well_log.curve_name}', 'green')
                    else:
                        set_info('выбраны не все параметры - скважина и пласт', 'red')
                        QMessageBox.critical(MainWindow, 'Ошибка', 'выберите все параметры для добавления обучающей скважины!')
                        return

            for i in list_analysis:
                ui.comboBox_regmod.setCurrentText(i)
                remove_all_param_geovel_reg()
                update_list_well_markup_reg()

            well_name = well_log.well.name if well_log else ''
            set_info(f'Добавлены обучающие скважины {well_name} в {len(list_info)} моделей.', 'blue')
            for i in list_info:
                ui.info.append(i)


        def update_all_well_log_in_regression():
            list_analysis, list_info = [], []

            for row in range(ui_wl.listWidget_well_log.count()):
                well_log_id = ui_wl.listWidget_well_log.item(row).text().split(' ID')[-1]
                well_log = session.query(WellLog).filter_by(id=well_log_id).first()

                if well_log:
                    reg_analysis = session.query(AnalysisReg).filter_by(title=well_log.curve_name).first()

                    if not reg_analysis:
                        set_info(f'Анализ {well_log.curve_name} не найден', 'red')
                        continue

                    analysis_id = reg_analysis.id
                    well_id = well_log.well_id
                    profile_id = get_profile_id()
                    formation_id = get_formation_id()

                    if analysis_id and well_id and profile_id and formation_id:
                        markup_reg = session.query(MarkupReg).filter_by(
                            analysis_id=analysis_id,
                            well_id=well_id,
                            profile_id=profile_id,
                            formation_id=formation_id
                        ).first()

                        target_value = get_median_value_from_interval(well_log_id,
                                                                      ui_wl.doubleSpinBox_depth.value(),
                                                                      ui_wl.doubleSpinBox_interval.value())
                        if not target_value:
                            set_info(f'Каротаж {well_log.curve_name} не представлен в выбранном интервале', 'red')
                            continue

                        if target_value < 0:
                            set_info(f'Отрицательное значение каротажа {well_log.curve_name} в интервале', 'red')
                            continue

                        if not markup_reg:
                            set_info(f'Обучающая скважина {well_log.well.name} не найдена в анализе {well_log.curve_name}', 'red')
                            add_well_log_markup_reg(analysis_id, well_id, profile_id, formation_id, target_value)
                            set_info(f'Обучающая скважина {well_log.well.name} добавлена в анализ {well_log.curve_name}',
                                'green')
                            continue

                        markup_reg.target_value = target_value
                        session.commit()

                        list_analysis.append(f'{reg_analysis.title} id{reg_analysis.id}')
                        list_info.append(f'{well_log.curve_name}: {target_value}')
                        set_info(
                            f'Обучающая скважина {well_log.well.name} обновлена в анализе {well_log.curve_name}',
                            'green')


            for i in list_analysis:
                ui.comboBox_regmod.setCurrentText(i)
                remove_all_param_geovel_reg()
                update_list_well_markup_reg()

            well_name = well_log.well.name if well_log else ''
            set_info(f'Обновлены обучающие скважины {well_name} у {len(list_info)} моделей', 'blue')
            for i in list_info:
                ui.info.append(i)


        def median_to_target_value():
            try:
                well_log_id = ui_wl.listWidget_well_log.currentItem().text().split(' ID')[-1]
                median_value = get_median_value_from_interval(well_log_id, ui_wl.doubleSpinBox_depth.value(), ui_wl.doubleSpinBox_interval.value())
            except AttributeError:
                set_info('Выберите каротаж', 'red')
                QMessageBox.critical(WellLogForm, 'Ошибка', 'Необходимо выбрать каротаж!')
                return

            ui.doubleSpinBox_target_val.setValue(median_value)


        def get_median_value_from_interval(well_log_id, begin, interval):
            well_log = session.query(WellLog).filter_by(id=well_log_id).first()
            if not well_log:
                return
            value = get_median_by_depth(json.loads(well_log.curve_data), well_log.begin, well_log.step, begin, interval)
            return value


        def remove_well_log():
            try:
                session.query(WellLog).filter_by(id=ui_wl.listWidget_well_log.currentItem().text().split(' ID')[-1]).delete()
                session.commit()
                update_list_well_log()
                update_list_well(select_well=True, selected_well_id=get_well_id())
            except AttributeError:
                set_info(f'Необходимо выбрать каротаж для удаления', 'red')
                return


        def remove_all_well_log():
            session.query(WellLog).filter_by(well_id=get_well_id()).delete()
            session.commit()
            update_list_well_log()
            update_list_well(select_well=True, selected_well_id=get_well_id())

        def draw_depth_spinbox():
            """ Добавление линий глубины и интервала на график """
            global hor_line_dep, hor_line_int

            # Удаление предыдущих линий
            if 'hor_line_dep' in globals():
                ui_wl.widget_graph_well_log.removeItem(hor_line_dep)
                ui_wl.widget_graph_well_log.removeItem(hor_line_int)


            # Создание бесконечных линий
            value_dep = ui_wl.doubleSpinBox_depth.value()
            value_int = ui_wl.doubleSpinBox_interval.value()

            hor_line_dep = pg.InfiniteLine(pos=value_dep, angle=0,
                                           pen=pg.mkPen(color='red', width=1.5, dash=[4, 7]))
            hor_line_int = pg.InfiniteLine(pos=value_dep+value_int, angle=0,
                                           pen=pg.mkPen(color='yellow', width=1.5, dash=[4, 7]))


            # Добавление линий на соответствующие графики
            ui_wl.widget_graph_well_log.addItem(hor_line_dep)
            ui_wl.widget_graph_well_log.addItem(hor_line_int)

            try:
                well_log_id = ui_wl.listWidget_well_log.currentItem().text().split(' ID')[-1]
            except AttributeError:
                return
            median_value = get_median_value_from_interval(well_log_id, value_dep, value_int)
            ui_wl.label_value.setText(str(median_value))


        def add_well_log_markup_reg(analysis_id, well_id, profile_id, formation_id, target_value):
            """Добавить новую обучающую скважину для обучения регрессионной модели"""

            well = session.query(Well).filter(Well.id == well_id).first()
            x_prof = json.loads(session.query(Profile.x_pulc).filter(Profile.id == profile_id).first()[0])
            y_prof = json.loads(session.query(Profile.y_pulc).filter(Profile.id == profile_id).first()[0])
            index, _ = closest_point(well.x_coord, well.y_coord, x_prof, y_prof)
            well_dist = ui.spinBox_well_dist_reg.value()
            start = index - well_dist if index - well_dist > 0 else 0
            stop = index + well_dist if index + well_dist < len(x_prof) else len(x_prof)
            list_measure = list(range(start, stop))
            new_markup_reg = MarkupReg(analysis_id=analysis_id, well_id=well_id, profile_id=profile_id,
                                       formation_id=formation_id, target_value=target_value,
                                       list_measure=json.dumps(list_measure))
            session.add(new_markup_reg)
            session.commit()
            set_info(
                f'Добавлена новая обучающая скважина для регрессионной модели - "{well.name} со значенем '
                f'{target_value}"', 'green')

        def map_well_logging():
            MapWellLog = QtWidgets.QDialog()
            ui_mwl = Ui_MapWellLogging()
            ui_mwl.setupUi(MapWellLog)
            MapWellLog.show()
            WellLogForm.setAttribute(Qt.WA_DeleteOnClose)

            m_width, m_height = get_width_height_monitor()
            WellLogForm.resize(int(m_width / 4), m_height - 200)

            for b in get_names_boundary():
                check = QCheckBox(b)
                item = QListWidgetItem()
                if 'uf' in b or 'уф' in b:
                    check.setChecked(True)
                ui_mwl.listWidget_title_layer.addItem(item)
                ui_mwl.listWidget_title_layer.setItemWidget(item, check)

            for l in get_names_logging():
                check = QCheckBox(l)
                item = QListWidgetItem()
                ui_mwl.listWidget_title_log.addItem(item)
                ui_mwl.listWidget_title_log.setItemWidget(item, check)

            def draw_map_well_log():
                min_x, max_x, min_y, max_y = 0, 0, 0, 0
                value_int = ui_wl.doubleSpinBox_interval.value()
                if not ui_mwl.checkBox_all_well.isChecked():
                    list_center = get_center_object_coordinates()
                    list_min_max = get_min_max_object_coordinates()
                    min_x = list_min_max[0] - (list_center[0] - list_min_max[0])
                    max_x = list_min_max[1] + (list_min_max[1] - list_center[0])
                    min_y = list_min_max[2] - (list_center[1] - list_min_max[2])
                    max_y = list_min_max[3] + (list_min_max[3] - list_center[1])

                dict_title_depth = get_dict_check_checkbox(ui_mwl.listWidget_title_layer)
                list_title_depth = [k for k, v in dict_title_depth.items() if v]

                dict_title_log = get_dict_check_checkbox(ui_mwl.listWidget_title_log)
                list_title_log = [k for k, v in dict_title_log.items() if v]

                well_logging = session.query(WellLog).filter(
                    WellLog.curve_name.in_(list_title_log)
                ).all()

                list_x, list_y, list_val, list_well_name = [], [], [], []
                for wl in well_logging:
                    bound = session.query(Boundary).filter(
                        Boundary.well_id == wl.well_id,
                        Boundary.title.in_(list_title_depth)
                    ).first()
                    if bound:
                        well = session.query(Well).filter_by(id=bound.well_id).first()
                        if not ui_mwl.checkBox_all_well.isChecked():
                            try:
                                if well.x_coord < min_x or well.x_coord > max_x or well.y_coord < min_y or well.y_coord > max_y:
                                    continue
                            except AttributeError:
                                print(f'AttributeError {wl.well_id}')
                                continue
                        median_value = get_median_value_from_interval(wl.id, bound.depth, value_int)
                        print(well.name, bound.depth, median_value)
                        if median_value:
                            list_well_name.append(well.name)
                            list_x.append(well.x_coord)
                            list_y.append(well.y_coord)
                            list_val.append(median_value)

                if not ui_mwl.checkBox_all_well.isChecked():
                    draw_map(list_x, list_y, list_val, list_title_log[0], profiles=True, list_name=list_well_name)
                else:
                    draw_map(list_x, list_y, list_val, list_title_log[0], list_name=list_well_name)

            def to_excel_well_log():
                min_x, max_x, min_y, max_y = 0, 0, 0, 0
                value_int = ui_wl.doubleSpinBox_interval.value()
                if not ui_mwl.checkBox_all_well.isChecked():
                    list_center = get_center_object_coordinates()
                    list_min_max = get_min_max_object_coordinates()
                    min_x = list_min_max[0] - (list_center[0] - list_min_max[0])
                    max_x = list_min_max[1] + (list_min_max[1] - list_center[0])
                    min_y = list_min_max[2] - (list_center[1] - list_min_max[2])
                    max_y = list_min_max[3] + (list_min_max[3] - list_center[1])

                dict_title_depth = get_dict_check_checkbox(ui_mwl.listWidget_title_layer)
                list_title_depth = [k for k, v in dict_title_depth.items() if v]

                dict_title_log = get_dict_check_checkbox(ui_mwl.listWidget_title_log)
                list_title_log = [k for k, v in dict_title_log.items() if v]

                well_logging = session.query(WellLog).filter(
                    WellLog.curve_name.in_(list_title_log)
                ).all()

                list_for_pd = []
                for wl in well_logging:
                    bound = session.query(Boundary).filter(
                        Boundary.well_id == wl.well_id,
                        Boundary.title.in_(list_title_depth)
                    ).first()
                    if bound:
                        well = session.query(Well).filter_by(id=bound.well_id).first()
                        if not ui_mwl.checkBox_all_well.isChecked():
                            if well.x_coord < min_x or well.x_coord > max_x or well.y_coord < min_y or well.y_coord > max_y:
                                continue
                        median_value = get_median_value_from_interval(wl.id, bound.depth, value_int)
                        print(well.name, bound.depth, median_value)
                        if median_value:
                            list_for_pd.append([well.name, well.x_coord, well.y_coord, median_value])

                pd_data = pd.DataFrame(list_for_pd, columns=['skv', 'X', 'Y', 'value'])
                if ui_mwl.checkBox_all_well.isChecked():
                    file_name = f'all_well__{list_title_log[0]}.xlsx'
                else:
                    file_name = f'{get_object_name()}_{get_research_name()}__{list_title_log[0]}.xlsx'
                fn = QFileDialog.getSaveFileName(caption="Сохранить значения выбранных каротажей в интервале в таблицу", directory=file_name,
                                                 filter="Excel Files (*.xlsx)")
                pd_data.to_excel(fn[0])

            ui_mwl.pushButton_draw_map.clicked.connect(draw_map_well_log)
            ui_mwl.pushButton_to_excel.clicked.connect(to_excel_well_log)
            MapWellLog.exec_()

        def clean_double_well_logging():
            dict_well_log = {}
            loggings = session.query(WellLog).all()
            ui.progressBar.setMaximum(len(loggings))
            n_rem = 0
            for n, log in enumerate(loggings):
                list_log_id = [key for key, value in dict_well_log.items() if value == log.well.name]
                if list_log_id:
                    log_rem = False
                    for log_id in list_log_id:
                        old_log = session.get(WellLog, log_id)
                        if old_log.curve_data == log.curve_data:
                            log_name = log.curve_name
                            well_name = log.well.name
                            session.delete(log)
                            set_info(f'Удален каротаж {log_name} скважины {well_name}', 'orange')
                            log_rem = True
                            n_rem += 1
                            break
                    if not log_rem:
                        dict_well_log[log.id] = log.well.name
                else:
                    dict_well_log[log.id] = log.well.name
                ui.progressBar.setValue(n+1)
            session.commit()
            update_list_well(select_well=True, selected_well_id=get_well_id())
            set_info(f'Удалено {n_rem} дублей каротажей.', 'blue')

        def uf_to_well_boundary():
            new_bound = Boundary(well_id=get_well_id(), depth=ui_wl.doubleSpinBox_depth.value(), title='P2uf GK')
            session.add(new_bound)
            session.commit()
            update_boundaries()

        def m_ss_to_well_data():
            new_well_opt = WellOptionally(well_id=get_well_id(), option='мощн песч GK', value=str(ui_wl.doubleSpinBox_interval.value()))
            session.add(new_well_opt)
            session.commit()
            show_data_well()


        ui_wl.pushButton_add_well_log.clicked.connect(load_well_log)
        ui_wl.pushButton_add_xls.clicked.connect(load_well_log_xls)
        ui_wl.pushButton_add_dir_well_log.clicked.connect(load_well_log_by_dir)
        ui_wl.pushButton_rm_well_log.clicked.connect(remove_well_log)
        ui_wl.pushButton_rm_all_well_db.clicked.connect(remove_all_well_log)
        ui_wl.listWidget_well_log.currentItemChanged.connect(draw_well_log)
        ui_wl.doubleSpinBox_depth.valueChanged.connect(draw_depth_spinbox)
        ui_wl.doubleSpinBox_interval.valueChanged.connect(draw_depth_spinbox)
        ui_wl.pushButton_create_an.clicked.connect(create_regression_analysis_by_current_well_log)
        ui_wl.pushButton_create_all_an.clicked.connect(create_regression_analysis_by_all_well_log)
        ui_wl.pushButton_add_current.clicked.connect(add_current_well_log_to_regression)
        ui_wl.pushButton_add_all.clicked.connect(add_all_well_log_to_regression)
        ui_wl.pushButton_to_tar_val.clicked.connect(median_to_target_value)
        ui_wl.pushButton_update_all.clicked.connect(update_all_well_log_in_regression)
        ui_wl.pushButton_map_well_log.clicked.connect(map_well_logging)
        ui_wl.pushButton_clean.clicked.connect(clean_double_well_logging)
        ui_wl.pushButton_uf_to_bound.clicked.connect(uf_to_well_boundary)
        ui_wl.pushButton_mosh_pesch_to_data.clicked.connect(m_ss_to_well_data)

        update_list_well_log()

        WellLogForm.exec_()

def get_names_logging():
    logging = session.query(WellLog).all()
    log_name = []
    for l in logging:
        if l.curve_name not in log_name:
            log_name.append(l.curve_name)
    return sorted(log_name)
