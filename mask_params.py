from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QTableView, QPushButton
from func import *
from qt.mask_param_form import *
from qt.mask_info_form import *
from models_db.model import *
from build_table import build_table_train



def mask_param_form():
    """ Форма MaskParam """
    MaskParam = QtWidgets.QDialog()
    ui_mp = Ui_MaskParamForm()
    ui_mp.setupUi(MaskParam)
    MaskParam.show()
    MaskParam.setAttribute(Qt.WA_DeleteOnClose)

    m_width, m_height = get_width_height_monitor()
    MaskParam.resize(int(m_width/1.5), int(m_height/1.5))

    # Загрузка масок
    ui_mp.listWidget_masks.clear()
    for mask in session.query(ParameterMask).all():
        item = QListWidgetItem(f'{mask.count_param} id{mask.id}')
        item.setToolTip(mask.mask_info)
        item.setData(Qt.UserRole, mask)  # Сохраняем объект маски
        ui_mp.listWidget_masks.addItem(item)

    # Создание модели параметров
    model = QStandardItemModel()
    # list_param = get_list_param_numerical_for_train(get_list_param_mlp())
    list_param = universal_expand_parameters(get_unique_parameters_from_mlp())
    # data, list_param_ex = build_table_train(False, 'mlp')
    # list_param_ex = data.columns.tolist()[2:]


    for param in list_param:
        item = QStandardItem(param)
        item.setCheckable(True)
        item.setCheckState(Qt.Unchecked)  # По умолчанию все выключены
        model.appendRow(item)

    ui_mp.listView_mask_params.setModel(model)
    ui_mp.listView_mask_params.setSelectionBehavior(QTableView.SelectRows)

    ui_mp.label_parameters.setText(f"Параметры ({model.rowCount()}):")

    ui_mp.search_edit.setPlaceholderText("Поиск параметров...")

    def filter_parameters(text):
        """ Фильтрация параметров для строки поиска """
        for row in range(model.rowCount()):
            item = model.item(row)
            match = text.lower() in item.text().lower()
            if match:
                index = ui_mp.listView_mask_params.model().index(row, 0)
                ui_mp.listView_mask_params.setCurrentIndex(index)
                return

    ui_mp.search_edit.textChanged.connect(filter_parameters)

    def on_mask_selected():
        """ Отметка чекбоксов параметров, относящихся к текущей маске """
        selected_items = ui_mp.listWidget_masks.selectedItems()
        ui_mp.listWidget_masks.setProperty("user_selected", bool(selected_items))
        if not selected_items:
            return

        selected_mask = selected_items[0].data(Qt.UserRole)
        mask_str = selected_mask.mask.replace("'", '"').strip()

        try:
            mask_parameters = json.loads(mask_str)
        except json.JSONDecodeError:
            mask_parameters = []
            info = f"Ошибка в данных маски id{selected_mask.id}"
            set_info(info, 'red')
            QMessageBox.critical(MaskParam, 'Ошибка', info)

        # Сбрасываем все чекбоксы
        for row in range(model.rowCount()):
            model.item(row).setCheckState(Qt.Unchecked)

        # Отмечаем только параметры из маски
        for param in mask_parameters:
            items = model.findItems(param.strip(), Qt.MatchExactly)
            if items:
                items[0].setCheckState(Qt.Checked)

    def update_mask_info():
        """ Форма для обновления mask_info текущей маски """
        user_selected = ui_mp.listWidget_masks.property("user_selected")
        if not user_selected:
            set_info('Не выбрана маска', 'red')
            QMessageBox.critical(MaskParam, 'Ошибка', 'Не выбрана маска')
            return

        current_item = ui_mp.listWidget_masks.currentItem()
        if not current_item:
            set_info('Не выбрана маска', 'red')
            QMessageBox.critical(MaskParam, 'Ошибка', 'Не выбрана маска')
            return

        try:
            current_mask = current_item.data(Qt.UserRole)
            mask = session.query(ParameterMask).filter_by(id=current_mask.id).first()
        except Exception as e:
            set_info('Ошибка загрузки маски', 'red')
            QMessageBox.critical(MaskParam, 'Ошибка', f'Ошибка при загрузке маски: {e}')
            return

        FormMaskInfo = QtWidgets.QDialog()
        ui_mi = Ui_Form_Mask_Info()
        ui_mi.setupUi(FormMaskInfo)
        FormMaskInfo.show()
        FormMaskInfo.setAttribute(Qt.WA_DeleteOnClose)

        ui_mi.textEdit.setText(mask.mask_info)

        def update_info():
            """ Обновление mask_info """
            new_info = ui_mi.textEdit.toPlainText()
            # Обновляем в базе данных
            session.query(ParameterMask).filter_by(id=current_mask.id).update(
                {'mask_info': new_info},
                synchronize_session='fetch'
            )
            session.commit()

            current_item.setToolTip(new_info)
            FormMaskInfo.close()

        def cancel_update():
            FormMaskInfo.close()

        ui_mi.buttonBox.accepted.connect(update_info)
        ui_mi.buttonBox.rejected.connect(cancel_update)

        FormMaskInfo.exec_()

    def save_mask_changes():
        """ Сохранение изменений для текущей маски """
        selected_items = ui_mp.listWidget_masks.selectedItems()
        if not selected_items:
            info = "Маска не выбрана, изменения не сохранятся"
            set_info(info, 'red')
            QMessageBox.critical(MaskParam, 'Сохранение изменений', info)
            return

        selected_mask = selected_items[0].data(Qt.UserRole)

        # Используем функцию для получения отмеченных параметров
        checked_parameters = get_checked_parameters(ui_mp.listView_mask_params)

        # Обновляем маску
        selected_mask.mask = json.dumps(checked_parameters)
        selected_mask.count_param = len(checked_parameters)

        try:
            session.commit()
            # Обновляем отображение в списке масок
            selected_items[0].setText(f'{selected_mask.count_param} id{selected_mask.id}')

            # Получаем текущую таблицу
            table = ui_mp.verticalLayout_table_mp.itemAt(0).widget()

            if table and table.objectName() == "maskParamsTable":
                # Находим столбец с текущей маской
                mask_header = f"{selected_mask.count_param} id{selected_mask.id}"
                col = -1
                for i in range(table.columnCount()):
                    if table.horizontalHeaderItem(i).text().endswith(f"id{selected_mask.id}"):
                        col = i
                        break

                if col >= 0:
                    # Обновляем заголовок столбца
                    table.horizontalHeaderItem(col).setText(mask_header)

                    # Обновляем все ячейки в столбце
                    for row in range(table.rowCount()):
                        param_name = table.verticalHeaderItem(row).text()
                        is_included = param_name in checked_parameters

                        item = table.item(row, col)
                        if item:
                            item.setBackground(QColor('#ABF37F') if is_included else QColor('#FF8080'))
                            item.setToolTip("Включен" if is_included else "Не включен")

            info = f"Маска id{selected_mask.id} успешно обновлена"
            set_info(info, 'green')
            QMessageBox.information(MaskParam, 'Сохранение изменений', info)
        except Exception as e:
            session.rollback()
            set_info(f"Ошибка при сохранении: {str(e)}", 'red')

    def save_as_new_mask():
        """ Создание новой маски (сохранение изменений текущей маски в новой маске) """
        selected_items = ui_mp.listWidget_masks.selectedItems()

        # Используем функцию для получения отмеченных параметров
        checked_parameters = get_checked_parameters(ui_mp.listView_mask_params)

        if not selected_items:
            new_mask = ParameterMask(
                mask=json.dumps(checked_parameters),
                count_param=len(checked_parameters)
            )
        else:
            selected_mask = selected_items[0].data(Qt.UserRole)
            new_mask = ParameterMask(
                mask=json.dumps(checked_parameters),
                mask_info=selected_mask.mask_info,
                count_param = len(checked_parameters)
            )
        try:
            session.add(new_mask)

            session.commit()

            item = QListWidgetItem(f'{new_mask.count_param} id{new_mask.id}')
            item.setToolTip(new_mask.mask_info)
            item.setData(Qt.UserRole, new_mask)  # Сохраняем объект маски
            ui_mp.listWidget_masks.addItem(item)

            # Делаем новую маску текущей выбранной
            ui_mp.listWidget_masks.setCurrentItem(item)
            ui_mp.listWidget_masks.scrollToItem(item)  # Прокручиваем список к новой маске

            # Получаем текущую таблицу
            table = ui_mp.verticalLayout_table_mp.itemAt(0).widget()

            if table and table.objectName() == "maskParamsTable":
                # Добавляем новый столбец
                col_position = table.columnCount()
                table.insertColumn(col_position)

                # Устанавливаем заголовок
                table.setHorizontalHeaderItem(col_position,
                                              QTableWidgetItem(f'{new_mask.count_param} id{new_mask.id}'))

                # Заполняем ячейки нового столбца
                for row in range(table.rowCount()):
                    param_name = table.verticalHeaderItem(row).text()
                    is_included = param_name in checked_parameters

                    item = QTableWidgetItem()
                    item.setBackground(QColor('#ABF37F') if is_included else QColor('#FF8080'))
                    item.setToolTip("Включен" if is_included else "Не включен")
                    item.setData(Qt.UserRole, (row, col_position,
                                               f'{new_mask.count_param} id{new_mask.id}', param_name))
                    table.setItem(row, col_position, item)

                table.resizeColumnsToContents()
            else:
                # Если таблица не найдена, создаем заново
                table_mask_params()

            # Принудительно вызываем обработчик выбора
            QTimer.singleShot(100, lambda: on_mask_selected())

            info = f"Создана новая маска id{new_mask.id}"
            set_info(info, 'green')
            QMessageBox.information(MaskParam, 'Создание новой маски', info)

        except Exception as e:
            session.rollback()
            info = f'Ошибка при создании новой маски: {str(e)}'
            set_info(info, 'red')
            QMessageBox.critical(MaskParam, 'Создание новой маски', info)


    def delete_mask():
        """ Удаление текущей маски """
        selected_items = ui_mp.listWidget_masks.selectedItems()
        if not selected_items:
            info = "Маска не выбрана"
            set_info(info, 'red')
            QMessageBox.critical(MaskParam, 'Удаление маски', info)
            return

        selected_mask = selected_items[0].data(Qt.UserRole)

        existing_model_mask = session.query(TrainedModelClassMask).filter_by(mask_id=selected_mask.id).first()
        existing_model_reg_mask = session.query(TrainedModelRegMask).filter_by(mask_id=selected_mask.id).first()
        if existing_model_mask or existing_model_reg_mask:
            info = f"Невозможно удалить маску id{selected_mask.id}: она используется в обучении моделей"
            set_info(info, 'red')
            QMessageBox.critical(MaskParam, 'Удаление маски', info)
            return
        else:
            result = QtWidgets.QMessageBox.question(
                MaskParam,
                'Удаление маски',
                f'Вы уверены, что хотите удалить маску id{selected_mask.id}?',
                QtWidgets.QMessageBox.Yes,
                QtWidgets.QMessageBox.No
            )
            if result == QtWidgets.QMessageBox.No:
                return

            if result == QtWidgets.QMessageBox.Yes:
                try:
                    session.query(ParameterMask).filter(ParameterMask.id == selected_mask.id).delete()
                    session.commit()

                    # Удаляем из списка
                    row = ui_mp.listWidget_masks.row(selected_items[0])
                    ui_mp.listWidget_masks.takeItem(row)

                    #  Удаляем маску в таблице
                    table = ui_mp.verticalLayout_table_mp.itemAt(0).widget()  # Получаем таблицу
                    if table and table.objectName() == "maskParamsTable":
                        mask_header = f"{selected_mask.count_param} id{selected_mask.id}"
                        try:
                            col = -1
                            for i in range(table.columnCount()):
                                if table.horizontalHeaderItem(i).text() == mask_header:
                                    col = i
                                    break
                            if col >= 0:
                                table.removeColumn(col)
                        except:
                            table_mask_params()  # Если что-то пошло не так, пересоздаем таблицу полностью

                    info = 'Маска успешно удалена'
                    QMessageBox.information(MaskParam, 'Удаление маски', info)
                    set_info(info, 'red')
                except Exception as e:
                    session.rollback()
                    info = f'Ошибка при удалении маски: {str(e)}'
                    set_info(info, 'red')
                    QMessageBox.critical(MaskParam, 'Удаление маски', info)

    def table_mask_params():
        """Таблица зависимостей масок и параметров с возможностью множественного выделения"""
        # Очищаем предыдущую таблицу
        while ui_mp.verticalLayout_table_mp.count():
            item = ui_mp.verticalLayout_table_mp.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Создаем DataFrame
        data = []
        masks = session.query(ParameterMask).all()

        for param in list_param:
            row_data = {'parameter': param}
            for mask in masks:
                mask_params = json.loads(mask.mask.replace("'", '"')) if mask.mask else []
                row_data[f'{mask.count_param} id{mask.id}'] = param in mask_params
            data.append(row_data)

        df = pd.DataFrame(data)
        df.set_index('parameter', inplace=True)

        # Создаем QTableWidget
        table = QTableWidget(df.shape[0], df.shape[1])
        table.setObjectName("maskParamsTable")

        # Устанавливаем заголовки
        table.setHorizontalHeaderLabels(df.columns)
        table.setVerticalHeaderLabels(df.index)

        # Заполняем таблицу
        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                val = df.iat[row, col]
                item = QTableWidgetItem()
                item.setBackground(QColor('#ABF37F') if val else QColor('#FF8080'))
                item.setToolTip("Включен" if val else "Не включен")
                item.setData(Qt.UserRole,
                             (row, col, df.columns[col], df.index[row]))  # Добавляем информацию о маске и параметре
                table.setItem(row, col, item)

        # Включаем множественное выделение
        table.setSelectionMode(QTableWidget.MultiSelection)

        def table_selection(enable):
            selected_items = table.selectedItems()
            if not selected_items:
                info = 'Не выбраны ячейки в таблице!'
                set_info(info, 'red')
                QMessageBox.critical(MaskParam, 'Ошибка', info)
                return

            # Группируем изменения по маскам
            mask_changes = {}

            for item in selected_items:
                row, col, mask_header, param_name = item.data(Qt.UserRole)
                mask_id = int(mask_header.split('id')[-1])

                if mask_id not in mask_changes:
                    mask_changes[mask_id] = {
                        'mask_obj': session.query(ParameterMask).get(mask_id),
                        'params': set(),
                        'col': col
                    }
                mask_changes[mask_id]['params'].add((param_name, enable))

            # Применяем изменения
            for mask_id, data in mask_changes.items():
                if not data['mask_obj']:
                    continue

                mask = data['mask_obj']
                mask_params = json.loads(mask.mask.replace("'", '"')) if mask.mask else []

                for param_name, new_state in data['params']:
                    if new_state and param_name not in mask_params:
                        mask_params.append(param_name)
                    elif not new_state and param_name in mask_params:
                        mask_params.remove(param_name)

                    # Обновляем цвет ячейки
                    row = df.index.get_loc(param_name)
                    table.item(row, data['col']).setBackground(
                        QColor('#ABF37F') if new_state else QColor('#FF8080')
                    )
                    table.item(row, data['col']).setToolTip("Включен" if new_state else "Не включен")

                # Обновляем чекбоксы в списке параметров
                for param_name, new_state in data['params']:
                    items = model.findItems(param_name, Qt.MatchExactly)
                    if items:
                        items[0].setCheckState(Qt.Checked if new_state else Qt.Unchecked)

                # Обновляем отображение в списке масок
                for i in range(ui_mp.listWidget_masks.count()):
                    item = ui_mp.listWidget_masks.item(i)
                    if item.data(Qt.UserRole).id == mask_id:
                        item.setText(f'{mask.count_param} id{mask.id}')
                        break

            # Снимаем выделение после применения изменений
            table.clearSelection()

        ui_mp.pushButton_on.clicked.connect(lambda: table_selection(True))
        ui_mp.pushButton_off.clicked.connect(lambda: table_selection(False))

        # Настройки таблицы
        table.setSortingEnabled(True)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()

        # Добавляем таблицу в layout
        ui_mp.verticalLayout_table_mp.addWidget(table)

    if model.rowCount() > 0:
        table_mask_params()

    ui_mp.listWidget_masks.itemSelectionChanged.connect(on_mask_selected)
    ui_mp.pushButton_save.clicked.connect(save_mask_changes)
    ui_mp.pushButton_save_as.clicked.connect(save_as_new_mask)
    ui_mp.pushButton_mask_info.clicked.connect(update_mask_info)
    ui_mp.pushButton_delete.clicked.connect(delete_mask)


    MaskParam.exec_()