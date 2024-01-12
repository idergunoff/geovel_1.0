import matplotlib.pyplot as plt
import pandas as pd

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
            Qt.BackgroundRole
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


def build_table_geochem_analytic(point_name=False):
    param = session.query(GeochemParameter).filter_by(geochem_id=get_geochem_id()).all()
    list_param = [i.title for i in param]
    if point_name:
        data_geochem = pd.DataFrame(columns=['well', 'point', 'color'] + list_param)
    else:
        data_geochem = pd.DataFrame(columns=['well'] + list_param)
    for g_well in session.query(GeochemWell).filter_by(geochem_id=get_geochem_id()).all():
        well = g_well.title
        for gwp in session.query(GeochemWellPoint).filter_by(g_well_id=g_well.id).all():
            dict_value = {'well': well, 'point': gwp.title, 'color': g_well.color} if point_name else {'well': well}
            for gwpv in session.query(GeochemWellPointValue).filter_by(g_well_point_id=gwp.id).all():
                dict_value[gwpv.g_param.title] = gwpv.value
            data_geochem = pd.concat([data_geochem, pd.DataFrame([dict_value])], ignore_index=True)
    for g_point in session.query(GeochemPoint).filter_by(geochem_id=get_geochem_id()).all():
        dict_value = {'well': 'field', 'point': g_point.title, 'color': 'green'} if point_name else {'well': 'field'}
        for gpv in session.query(GeochemPointValue).filter_by(g_point_id=g_point.id).all():
            dict_value[gpv.g_param.title] = gpv.value
        data_geochem = pd.concat([data_geochem, pd.DataFrame([dict_value])], ignore_index=True)

    return data_geochem




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


def anova_geochem():
    Anova = QtWidgets.QDialog()
    ui_anova = Ui_Anova()
    ui_anova.setupUi(Anova)
    Anova.show()
    Anova.setAttribute(QtCore.Qt.WA_DeleteOnClose) # атрибут удаления виджета после закрытия

    m_width, m_height = get_width_height_monitor()
    Anova.resize(int(m_width/3)*2, int(m_height/3)*2)

    # ui_anova.graphicsView.setBackground('w')
    data_plot = build_table_geochem_analytic()
    markers = list(set(data_plot['well']))
    pallet = {'field': 'green'}
    for m in markers:
        if m == 'field':
            continue
        pallet[m] = session.query(GeochemWell).filter(GeochemWell.title == m, GeochemWell.geochem_id == get_geochem_id()).first().color

    for w in data_plot['well'].unique():
        check_box_widget = QCheckBox(w)
        check_box_widget.setChecked(True)
        list_item = QListWidgetItem()
        ui_anova.listWidget_checkbox_well.addItem(list_item)
        ui_anova.listWidget_checkbox_well.setItemWidget(list_item, check_box_widget)


    figure = plt.figure()
    canvas = FigureCanvas(figure)
    ui_anova.horizontalLayout.addWidget(canvas)


    for i in data_plot.columns.tolist()[2:]:
        ui_anova.listWidget.addItem(i)

    def draw_graph_anova():
        try:
            figure.clear()
            list_well = get_list_check_checkbox(ui_anova.listWidget_checkbox_well)
            data_plot_new = data_plot.loc[data_plot['well'].isin(list_well)]
            param = ui_anova.listWidget.currentItem().text()
            if ui_anova.radioButton_box.isChecked():
                sns.boxplot(data=data_plot_new, y=param, x='well', orient='v', palette=pallet)
            if ui_anova.radioButton_violin.isChecked():
                sns.violinplot(data=data_plot_new, y=param, x='well', orient='v', palette=pallet)
            if ui_anova.radioButton_strip.isChecked():
                sns.stripplot(data=data_plot_new, y=param, x='well', hue='well', orient='v', palette=pallet)
            if ui_anova.radioButton_boxen.isChecked():
                sns.boxenplot(data=data_plot_new, y=param, x='well', orient='v', palette=pallet)
            figure.tight_layout()
            canvas.draw()
        except ValueError:
            pass

    ui_anova.listWidget.currentItemChanged.connect(draw_graph_anova)
    ui_anova.radioButton_boxen.clicked.connect(draw_graph_anova)
    ui_anova.radioButton_strip.clicked.connect(draw_graph_anova)
    ui_anova.radioButton_violin.clicked.connect(draw_graph_anova)
    ui_anova.radioButton_box.clicked.connect(draw_graph_anova)
    for i in range(ui_anova.listWidget_checkbox_well.count()):
        ui_anova.listWidget_checkbox_well.itemWidget(ui_anova.listWidget_checkbox_well.item(i)).stateChanged.connect(draw_graph_anova)


    Anova.exec_()


def tsne_geochem():
    TSNE_form = QtWidgets.QDialog()
    ui_tsne = Ui_TSNE_PCA()
    ui_tsne.setupUi(TSNE_form)
    TSNE_form.show()
    TSNE_form.setAttribute(QtCore.Qt.WA_DeleteOnClose) # атрибут удаления виджета после закрытия

    m_width, m_height = get_width_height_monitor()
    TSNE_form.resize(int(m_width/3)*2, int(m_height/3)*2)

    # ui_anova.graphicsView.setBackground('w')
    data_plot = build_table_geochem_analytic(point_name=True)
    markers = list(set(data_plot['well']))
    pallet = {'field': 'green'}
    for m in markers:
        if m == 'field':
            continue
        pallet[m] = session.query(GeochemWell).filter(GeochemWell.title == m, GeochemWell.geochem_id == get_geochem_id()).first().color

    for w in data_plot['well'].unique():
        check_box_widget = QCheckBox(w)
        check_box_widget.setChecked(True)
        list_item = QListWidgetItem()
        ui_tsne.listWidget_checkbox_well.addItem(list_item)
        ui_tsne.listWidget_checkbox_well.setItemWidget(list_item, check_box_widget)

    def set_list_point():
        ui_tsne.listWidget_point.clear()
        list_well = get_list_check_checkbox(ui_tsne.listWidget_checkbox_well)
        data_plot_new = data_plot.loc[data_plot['well'].isin(list_well)]

        for i in data_plot_new.index:
            ui_tsne.listWidget_point.addItem(str(data_plot_new['point'][i]))
            ui_tsne.listWidget_point.findItems(str(data_plot_new['point'][i]), Qt.MatchExactly)[0].setBackground(
                QColor(data_plot_new['color'][i]))
        # ui_tsne.listWidget_point.setCurrentRow(0)

    figure = plt.figure()
    canvas = FigureCanvas(figure)
    mpl_toolbar = NavigationToolbar(canvas)
    ui_tsne.verticalLayout_graph.addWidget(mpl_toolbar)
    ui_tsne.verticalLayout_graph.addWidget(canvas)

    def draw_graph_tsne():
        list_well = get_list_check_checkbox(ui_tsne.listWidget_checkbox_well)
        data_plot_new = data_plot.loc[data_plot['well'].isin(list_well)].reset_index(drop=False)

        try:
            figure.clear()

            data_tsne = data_plot_new.drop(['well', 'point', 'color'], axis=1)

            if ui_tsne.checkBox_standart.isChecked():
                scaler = StandardScaler()
                data_tsne = scaler.fit_transform(data_tsne)
            if ui_tsne.radioButton_tsne.isChecked():
                name_graph = 't-SNE'
                tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
                data_tsne_result = tsne.fit_transform(data_tsne)
            if ui_tsne.radioButton_pca.isChecked():
                name_graph = 'PCA'
                pca = PCA(n_components=2)
                data_tsne_result = pca.fit_transform(data_tsne)
            data_plot_new = pd.concat([data_plot_new, pd.DataFrame(data_tsne_result, columns=['0', '1'])], axis=1)

            pd.set_option('display.max_rows', None)
            print(data_plot_new)

            sns.scatterplot(data=data_plot_new, x='0', y='1', hue='well', s=100, palette=pallet)
            try:
                plt.vlines(
                    x=data_plot_new['0'].loc[data_plot_new['point'] == ui_tsne.listWidget_point.currentItem().text()],
                    ymin=data_plot_new['1'].min(),
                    ymax=data_plot_new['1'].max(),
                    color='black',
                    linestyle='--'
                )
                plt.hlines(
                    y=data_plot_new['1'].loc[data_plot_new['point'] == ui_tsne.listWidget_point.currentItem().text()],
                    xmin=data_plot_new['0'].min(),
                    xmax=data_plot_new['0'].max(),
                    color='black',
                    linestyle='--'
                )
            except AttributeError:
                print('AttributeError')
                pass
            plt.grid()
            figure.suptitle(name_graph)
            figure.tight_layout()
            canvas.draw()
        except ValueError:
            pass

    ui_tsne.listWidget_point.currentItemChanged.connect(draw_graph_tsne)
    ui_tsne.radioButton_tsne.clicked.connect(draw_graph_tsne)
    ui_tsne.radioButton_pca.clicked.connect(draw_graph_tsne)
    ui_tsne.checkBox_standart.clicked.connect(draw_graph_tsne)
    # ui_tsne.listWidget_point.clicked.connect(set_list_point)
    ui_tsne.listWidget_point.clicked.connect(draw_graph_tsne)
    for i in range(ui_tsne.listWidget_checkbox_well.count()):
        ui_tsne.listWidget_checkbox_well.itemWidget(ui_tsne.listWidget_checkbox_well.item(i)).stateChanged.connect(draw_graph_tsne)

    set_list_point()
    draw_graph_tsne()

    TSNE_form.exec_()
