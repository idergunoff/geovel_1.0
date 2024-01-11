from func import *


def update_combobox_geochem():
    ui.comboBox_geochem.clear()
    for i in session.query(Geochem).order_by(Geochem.title).all():
        ui.comboBox_geochem.addItem(f'{i.title} id{i.id}')
    if ui.comboBox_geochem.count() > 0:
        update_listwidget_geochem_point()
        update_combobox_geochem_well()
        update_listwidget_param_geochem()


def update_combobox_geochem_well():
    ui.comboBox_geochem_well.clear()
    for i in session.query(GeochemWell).filter_by(geochem_id=get_geochem_id()).all():
        ui.comboBox_geochem_well.addItem(f'{i.title} id{i.id}')
        ui.comboBox_geochem_well.setItemData(
            ui.comboBox_geochem_well.findText(f'{i.title} id{i.id}'),
            QBrush(QColor(i.color)),
            Qt.UserRole
        )
    update_listwidget_geochem_well_point()


def update_listwidget_param_geochem():
    ui.listWidget_geochem_param.clear()
    for i in session.query(GeochemParameter).filter_by(geochem_id=get_geochem_id()).all():
        ui.listWidget_geochem_param.addItem(f'{i.title} id{i.id}')


def update_listwidget_geochem_point():
    ui.listWidget_g_point.clear()
    for i in session.query(GeochemPoint).filter_by(geochem_id=get_geochem_id()).order_by(GeochemPoint.title).all():
        ui.listWidget_g_point.addItem(f'{i.title} id{i.id}')


def update_listwidget_geochem_well_point():
    ui.listWidget_g_well_point.clear()
    for i in session.query(GeochemWellPoint).filter_by(g_well_id=get_geochem_well_id()).order_by(GeochemWellPoint.title).all():
        ui.listWidget_g_well_point.addItem(f'{i.title} id{i.id}')


def remove_geochem():
    """ Удалить объект геохимических данных """

    gchm = session.query(Geochem).filter_by(id=get_geochem_id()).first()
    geochem_name = gchm.title

    for gparam in session.query(GeochemParameter).filter_by(geochem_id=gchm.id).all():
        session.delete(gparam)

    for gp in session.query(GeochemPoint).filter_by(geochem_id=gchm.id).all():
        for gpv in session.query(GeochemPointValue).filter_by(g_point_id=gp.id).all():
            session.delete(gpv)
        session.delete(gp)

    for gw in session.query(GeochemWell).filter_by(geochem_id=gchm.id).all():
        for gwp in session.query(GeochemWellPoint).filter_by(g_well_id=gw.id).all():
            for gwpv in session.query(GeochemWellPointValue).filter_by(g_well_point_id=gwp.id).all():
                session.delete(gwpv)
            session.delete(gwp)
        session.delete(gw)
    session.delete(gchm)
    session.commit()
    set_info(f'Геохимический объект {geochem_name} и все данные по нему удалены', 'green')
    update_combobox_geochem()



def load_geochem():
    """ Загрузка геохимических данных из файла Excel """
    try:
        file_name = QFileDialog.getOpenFileName(caption='Выберите файл Excel', filter='*.xls *.xlsx')[0]
        set_info(f'Загружен Excel файл {file_name}', 'blue')
        data_geochem = pd.read_excel(file_name, header=0)
    except FileNotFoundError:
        return

    data_geochem = clean_dataframe(data_geochem)

    GeochemLoader = QtWidgets.QDialog()
    ui_gl = Ui_GeochemLoader()
    ui_gl.setupUi(GeochemLoader)
    GeochemLoader.show()
    GeochemLoader.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия
    m_width, m_height = get_width_height_monitor()
    GeochemLoader.resize(int(m_width/2), 300)

    list_combobox = [ui_gl.comboBox_name, ui_gl.comboBox_x, ui_gl.comboBox_y, ui_gl.comboBox_class]
    for n, cmbx in enumerate(list_combobox):
        for i in data_geochem.columns:
            cmbx.addItem(i)
        cmbx.setCurrentIndex(n)

    def update_list_g_param():
        ui_gl.listWidget_param.clear()
        for i in data_geochem.columns:
            ui_gl.listWidget_param.addItem(i)
        try:
            ui_gl.listWidget_param.takeItem(ui_gl.listWidget_param.row(
                ui_gl.listWidget_param.findItems(ui_gl.comboBox_name.currentText(), Qt.MatchFlag.MatchFixedString)[0]))
            ui_gl.listWidget_param.takeItem(ui_gl.listWidget_param.row(
                ui_gl.listWidget_param.findItems(ui_gl.comboBox_x.currentText(), Qt.MatchFlag.MatchFixedString)[0]))
            ui_gl.listWidget_param.takeItem(ui_gl.listWidget_param.row(
                ui_gl.listWidget_param.findItems(ui_gl.comboBox_y.currentText(), Qt.MatchFlag.MatchFixedString)[0]))
            ui_gl.listWidget_param.takeItem(ui_gl.listWidget_param.row(
                ui_gl.listWidget_param.findItems(ui_gl.comboBox_class.currentText(), Qt.MatchFlag.MatchFixedString)[0]))
        except IndexError:
            pass

    def change_color_geochem():
        button_color = ui_gl.pushButton_color.palette().color(ui_gl.pushButton_color.backgroundRole())
        color = QColorDialog.getColor(button_color)
        ui_gl.pushButton_color.setStyleSheet(f"background-color: {color.name()};")
        ui_gl.pushButton_color.setText(color.name())

    def update_combobox_geochem_classes():
        ui_gl.comboBox_classes.clear()
        for i in data_geochem[ui_gl.comboBox_class.currentText()].unique():
            ui_gl.comboBox_classes.addItem(str(i))

    def add_geochem_well():
        ui_gl.listWidget_well.addItem(ui_gl.comboBox_classes.currentText())
        ui_gl.listWidget_well.findItems(
            ui_gl.comboBox_classes.currentText(), Qt.MatchFlag.MatchFixedString
        )[0].setBackground(QColor(ui_gl.pushButton_color.text()))

    def remove_geochem_well():
        ui_gl.listWidget_well.takeItem(ui_gl.listWidget_well.currentRow())

    def remove_geochem_param():
        ui_gl.listWidget_param.takeItem(ui_gl.listWidget_param.currentRow())

    def get_param_id(title, geochem):
        return session.query(GeochemParameter).filter(
            GeochemParameter.title == title,
            GeochemParameter.geochem_id == geochem
        ).first().id

    def load_geochem_to_db():
        # сохраняем новый геохимический объект
        new_geochem = Geochem(title=file_name.split('/')[-1].split('.')[0])
        session.add(new_geochem)
        session.commit()

        # сохраняем все геохимические параметры из таблицы
        for n_row_param in range(ui_gl.listWidget_param.count()):
            new_param = GeochemParameter(
                geochem_id=new_geochem.id,
                title=ui_gl.listWidget_param.item(n_row_param).text()
            )
            session.add(new_param)
        session.commit()

        # сохраняем геохимические скважины
        for n_row_well in range(ui_gl.listWidget_well.count()):
            new_geochem_well = GeochemWell(
                geochem_id=new_geochem.id,
                title=ui_gl.listWidget_well.item(n_row_well).text(),
                color=ui_gl.listWidget_well.item(n_row_well).background().color().name()
            )
            session.add(new_geochem_well)
            session.commit()


            ui.progressBar.setMaximum(len(data_geochem.loc[
                data_geochem[ui_gl.comboBox_class.currentText()] == new_geochem_well.title
            ].index))

            # сохраняем геохимические точки в текущей скважине
            for pb, point_well in enumerate(data_geochem.loc[
                data_geochem[ui_gl.comboBox_class.currentText()] == new_geochem_well.title
            ].index):
                ui.progressBar.setValue(pb + 1)
                new_geochem_well_point = GeochemWellPoint(
                    g_well_id=new_geochem_well.id,
                    title=data_geochem[ui_gl.comboBox_name.currentText()][point_well],
                    x_coord=float(data_geochem[ui_gl.comboBox_x.currentText()][point_well]),
                    y_coord=float(data_geochem[ui_gl.comboBox_y.currentText()][point_well])
                )
                session.add(new_geochem_well_point)
                session.commit()

                # сохраняем значения геохимических параметров в текущей точке
                for n_row_param in range(ui_gl.listWidget_param.count()):
                    new_geochem_well_point_value = GeochemWellPointValue(
                        g_well_point_id=new_geochem_well_point.id,
                        g_param_id=get_param_id(ui_gl.listWidget_param.item(n_row_param).text(), new_geochem.id),
                        value=float(data_geochem[ui_gl.listWidget_param.item(n_row_param).text()][point_well])
                    )
                    session.add(new_geochem_well_point_value)
                session.commit()

            # удаляем точки текущей скважины из DataFrame
            data_geochem.drop(
                data_geochem.loc[data_geochem[ui_gl.comboBox_class.currentText()] == new_geochem_well.title].index,
                inplace=True
            )

        #  сохраняем оставшиеся точки
        ui.progressBar.setMaximum(len(data_geochem.index))

        for point in data_geochem.index:
            ui.progressBar.setValue(point + 1)
            new_geochem_point = GeochemPoint(
                geochem_id=new_geochem.id,
                title=data_geochem[ui_gl.comboBox_name.currentText()][point],
                x_coord=float(data_geochem[ui_gl.comboBox_x.currentText()][point]),
                y_coord=float(data_geochem[ui_gl.comboBox_y.currentText()][point])
            )
            session.add(new_geochem_point)
            session.commit()

            # сохраняем значения геохимических параметров в текущей точке
            for n_row_param in range(ui_gl.listWidget_param.count()):
                new_geochem_point_value = GeochemPointValue(
                    g_point_id=new_geochem_point.id,
                    g_param_id=get_param_id(ui_gl.listWidget_param.item(n_row_param).text(), new_geochem.id),
                    value=float(data_geochem[ui_gl.listWidget_param.item(n_row_param).text()][point])
                )
                session.add(new_geochem_point_value)
            session.commit()

        set_info(f'Геохимический объект "{new_geochem.title}" успешно загружен в базу данных', 'green')
        update_combobox_geochem()
        GeochemLoader.close()

    update_list_g_param()
    update_combobox_geochem_classes()
    set_random_color(ui_gl.pushButton_color)

    ui_gl.pushButton_del_param.clicked.connect(remove_geochem_param)
    ui_gl.pushButton_del_well.clicked.connect(remove_geochem_well)
    ui_gl.pushButton_add_well.clicked.connect(add_geochem_well)
    ui_gl.comboBox_class.currentTextChanged.connect(update_combobox_geochem_classes)
    ui_gl.pushButton_color.clicked.connect(change_color_geochem)
    ui_gl.comboBox_name.currentTextChanged.connect(update_list_g_param)
    ui_gl.comboBox_x.currentTextChanged.connect(update_list_g_param)
    ui_gl.comboBox_y.currentTextChanged.connect(update_list_g_param)
    ui_gl.comboBox_class.currentTextChanged.connect(update_list_g_param)
    ui_gl.buttonBox.rejected.connect(GeochemLoader.close)
    ui_gl.buttonBox.accepted.connect(load_geochem_to_db)

    GeochemLoader.exec_()