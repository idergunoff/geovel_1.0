import pdb

import matplotlib.pyplot as plt
import pandas as pd

from func import *
from classification_func import train_classifier
from krige import draw_map


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
    ui.label_29.setText(f'Parameters: {ui.listWidget_geochem_param.count()}')




def update_listwidget_geochem_point():
    ui.listWidget_g_point.clear()
    for i in session.query(GeochemPoint).filter_by(geochem_id=get_geochem_id()).order_by(GeochemPoint.title).all():
        ui.listWidget_g_point.addItem(f'{i.title} id{i.id}')
    update_maket_combobox()
    ui.label_point_expl_3.setText(f'Field Points: {ui.listWidget_g_point.count()}')


def update_listwidget_geochem_well_point():
    ui.listWidget_g_well_point.clear()
    for i in session.query(GeochemWellPoint).filter_by(g_well_id=get_geochem_well_id()).order_by(GeochemWellPoint.title).all():
        ui.listWidget_g_well_point.addItem(f'{i.title} id{i.id}')
    ui.label_point_expl_4.setText(f'Well Points: {ui.listWidget_g_well_point.count()}')


def remove_geochem():
    """ Удалить объект геохимических данных """

    gchm = session.query(Geochem).filter_by(id=get_geochem_id()).first()
    geochem_name = gchm.title

    result = QtWidgets.QMessageBox.question(
        MainWindow,
        'Удаление макета',
        f'Вы уверены, что хотите удалить геохимический объект {geochem_name}?',
        QtWidgets.QMessageBox.Yes,
        QtWidgets.QMessageBox.No
    )

    if result == QtWidgets.QMessageBox.Yes:
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
        for gparam in session.query(GeochemMaket).filter_by(geochem_id=gchm.id).all():
            for c in session.query(GeochemCategory).filter_by(maket_id=gparam.id).all():
                session.delete(c)
        for gm in session.query(GeochemMaket).filter_by(geochem_id=gchm.id).all():
            session.delete(gm)
        for gparam in session.query(GeochemParameter).filter_by(geochem_id=gchm.id).all():
            session.delete(gparam)
        session.delete(gchm)
        session.commit()
        set_info(f'Геохимический объект {geochem_name} и все данные по нему удалены', 'green')
        update_combobox_geochem()
        update_combobox_geochem_well()
        update_listwidget_param_geochem()
        update_listwidget_geochem_point()
        update_listwidget_geochem_well_point()
        update_maket_combobox()
        update_category_combobox()
    else:
        pass



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
                    title=str(data_geochem[ui_gl.comboBox_name.currentText()][point_well]),
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
                title=str(data_geochem[ui_gl.comboBox_name.currentText()][point]),
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


    for i in data_plot.columns.tolist()[1:]:
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
    TSNE_form.resize(m_width - 200, m_height - 200)

    # ui_anova.graphicsView.setBackground('w')
    data_plot = build_table_geochem_analytic(point_name=True)
    markers = list(set(data_plot['well']))
    pallet = {'field': 'green'}
    for m in markers:
        if m == 'field':
            continue
        pallet[m] = session.query(GeochemWell).filter(GeochemWell.title == m, GeochemWell.geochem_id == get_geochem_id()).first().color

    for i in data_plot.columns.tolist()[3:]:
        ui_tsne.listWidget_param.addItem(i)
        ui_tsne.listWidget_param.setCurrentRow(0)

    for i_param in data_plot.columns.tolist()[3:]:
        check_box_widget = QCheckBox(i_param)
        check_box_widget.setChecked(True)
        list_item = QListWidgetItem()
        ui_tsne.listWidget_check_param.addItem(list_item)
        ui_tsne.listWidget_check_param.setItemWidget(list_item, check_box_widget)

    def set_list_point():
        ui_tsne.listWidget_point.clear()
        list_well = get_list_check_checkbox(ui_tsne.listWidget_checkbox_well)
        data_plot_new = data_plot.loc[data_plot['well'].isin(list_well)]


        for i in data_plot_new.index:
            ui_tsne.listWidget_point.addItem(str(data_plot_new['point'][i]))
            ui_tsne.listWidget_point.findItems(str(data_plot_new['point'][i]), Qt.MatchExactly)[0].setBackground(
                QColor(data_plot_new['color'][i]))
        ui_tsne.listWidget_point.setCurrentRow(0)

    def draw_graph_tsne():
        clear_layout(ui_tsne.verticalLayout_graph)
        figure_tsne = plt.figure()
        canvas_tsne = FigureCanvas(figure_tsne)
        mpl_toolbar = NavigationToolbar(canvas_tsne)
        ui_tsne.verticalLayout_graph.addWidget(mpl_toolbar)
        ui_tsne.verticalLayout_graph.addWidget(canvas_tsne)

        list_well = get_list_check_checkbox(ui_tsne.listWidget_checkbox_well)
        data_plot_new = data_plot.loc[data_plot['well'].isin(list_well)]

        data_plot_new.reset_index(inplace=True, drop=True)

        list_drop_point = get_list_check_checkbox(ui_tsne.listWidget_check_point)
        data_plot_new = data_plot_new.loc[data_plot_new['point'].isin(list_drop_point)]

        data_plot_new.reset_index(inplace=True, drop=True)

        list_drop_param = get_list_check_checkbox(ui_tsne.listWidget_check_param, is_checked=False)
        data_plot_new = data_plot_new.drop(list_drop_param, axis=1)

        try:
            data_tsne = data_plot_new.drop(['well', 'point', 'color'], axis=1)

            if ui_tsne.checkBox_standart.isChecked():
                scaler = StandardScaler()
                data_tsne = scaler.fit_transform(data_tsne)
            if ui_tsne.radioButton_tsne.isChecked():
                name_graph = 't-SNE'
                tsne = TSNE(n_components=2, perplexity=ui_tsne.spinBox_perplexity.value(), learning_rate=200, random_state=42)
                data_tsne_result = tsne.fit_transform(data_tsne)
            if ui_tsne.radioButton_pca.isChecked():
                name_graph = 'PCA'
                pca = PCA(n_components=2)
                data_tsne_result = pca.fit_transform(data_tsne)
            data_plot_new = pd.concat([data_plot_new, pd.DataFrame(data_tsne_result, columns=['0', '1'])], axis=1)

            sns.scatterplot(data=data_plot_new, x='0', y='1', hue='well', s=100, palette=pallet)

            if ui_tsne.checkBox_name_point.isChecked():
                # Добавление подписей к точкам
                for i_data in data_plot_new.index:
                    plt.text(data_plot_new['0'][i_data], data_plot_new['1'][i_data],
                            data_plot_new['point'][i_data], horizontalalignment='left',
                            size='medium', color='black', weight='semibold')
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
            figure_tsne.suptitle(f'{name_graph}\n{len(data_plot_new.index)} точек')
            figure_tsne.tight_layout()
            canvas_tsne.draw()
        except ValueError:
            pass

    def draw_graph_anova():
        clear_layout(ui_tsne.verticalLayout_anova)
        figure_anova = plt.figure()
        canvas_anova = FigureCanvas(figure_anova)
        mpl_toolbar = NavigationToolbar(canvas_anova)
        ui_tsne.verticalLayout_anova.addWidget(mpl_toolbar)
        ui_tsne.verticalLayout_anova.addWidget(canvas_anova)

        try:
            list_well = get_list_check_checkbox(ui_tsne.listWidget_checkbox_well)
            data_plot_new = data_plot.loc[data_plot['well'].isin(list_well)]

            data_plot_new.reset_index(inplace=True, drop=True)

            list_drop_point = get_list_check_checkbox(ui_tsne.listWidget_check_point)
            data_plot_new = data_plot_new.loc[data_plot_new['point'].isin(list_drop_point)]

            data_plot_new.reset_index(inplace=True, drop=True)

            param = ui_tsne.listWidget_param.currentItem().text()

            if ui_tsne.checkBox_name_point.isChecked():
                # Добавление подписей к точкам
                for i_data in data_plot_new.index:
                    well_index = list_well.index(data_plot_new['well'][i_data])
                    plt.text(well_index, data_plot_new[param][i_data],
                            data_plot_new['point'][i_data], horizontalalignment='left',
                            size='medium', color='black', weight='semibold')

            if ui_tsne.radioButton_box.isChecked():
                sns.boxplot(data=data_plot_new, y=param, x='well', orient='v', palette=pallet)
            if ui_tsne.radioButton_violin.isChecked():
                sns.violinplot(data=data_plot_new, y=param, x='well', orient='v', palette=pallet, inner='stick')
            if ui_tsne.radioButton_strip.isChecked():
                sns.stripplot(data=data_plot_new, y=param, x='well', hue='well', orient='v', palette=pallet)
            if ui_tsne.radioButton_boxen.isChecked():
                sns.boxenplot(data=data_plot_new, y=param, x='well', orient='v', palette=pallet)

            figure_anova.suptitle(f'ANOVA\n{len(data_plot_new.index)} точек')
            plt.grid()
            figure_anova.tight_layout()
            canvas_anova.draw()
        except ValueError:
            pass
        except AttributeError:
            print('AttributeError_anova')

    def set_list_check_point():
        ui_tsne.listWidget_check_point.clear()
        list_well = get_list_check_checkbox(ui_tsne.listWidget_checkbox_well)
        data_plot_point = data_plot.loc[data_plot['well'].isin(list_well)]
        for point_name in data_plot_point['point']:
            check_box_widget = QCheckBox(point_name)
            check_box_widget.setChecked(True)
            # check_box_widget.stateChanged.connect(draw_graph_tsne)
            list_item = QListWidgetItem()
            ui_tsne.listWidget_check_point.addItem(list_item)
            ui_tsne.listWidget_check_point.setItemWidget(list_item, check_box_widget)


    for w in data_plot['well'].unique():
        check_box_widget = QCheckBox(w)
        check_box_widget.setChecked(True)
        # check_box_widget.stateChanged.connect(draw_graph_tsne)
        list_item = QListWidgetItem()
        ui_tsne.listWidget_checkbox_well.addItem(list_item)
        ui_tsne.listWidget_checkbox_well.setItemWidget(list_item, check_box_widget)


    ui_tsne.listWidget_point.currentItemChanged.connect(draw_graph_tsne)
    ui_tsne.listWidget_point.currentItemChanged.connect(draw_graph_anova)
    # ui_tsne.radioButton_tsne.clicked.connect(draw_graph_tsne)
    # ui_tsne.radioButton_pca.clicked.connect(draw_graph_tsne)
    # ui_tsne.checkBox_standart.clicked.connect(draw_graph_tsne)
    # ui_tsne.spinBox_perplexity.valueChanged.connect(draw_graph_tsne)
    # ui_tsne.listWidget_point.clicked.connect(set_list_point)
    ui_tsne.checkBox_name_point.stateChanged.connect(draw_graph_tsne)
    ui_tsne.checkBox_name_point.stateChanged.connect(draw_graph_anova)
    ui_tsne.listWidget_param.currentItemChanged.connect(draw_graph_anova)
    ui_tsne.radioButton_boxen.clicked.connect(draw_graph_anova)
    ui_tsne.radioButton_strip.clicked.connect(draw_graph_anova)
    ui_tsne.radioButton_violin.clicked.connect(draw_graph_anova)
    ui_tsne.radioButton_box.clicked.connect(draw_graph_anova)
    ui_tsne.pushButton_apply.clicked.connect(draw_graph_tsne)
    ui_tsne.pushButton_apply.clicked.connect(draw_graph_anova)

    for i in range(ui_tsne.listWidget_checkbox_well.count()):
        ui_tsne.listWidget_checkbox_well.itemWidget(ui_tsne.listWidget_checkbox_well.item(i)).stateChanged.connect(set_list_check_point)
        # ui_tsne.listWidget_checkbox_well.itemWidget(ui_tsne.listWidget_checkbox_well.item(i)).stateChanged.connect(draw_graph_tsne)
        # ui_tsne.listWidget_checkbox_well.itemWidget(ui_tsne.listWidget_checkbox_well.item(i)).stateChanged.connect(draw_graph_anova)

    # for i in range(ui_tsne.listWidget_check_param.count()):
    #     ui_tsne.listWidget_check_param.itemWidget(ui_tsne.listWidget_check_param.item(i)).stateChanged.connect(draw_graph_tsne)

    # for i in range(ui_tsne.listWidget_check_point.count()):
    #     ui_tsne.listWidget_check_point.itemWidget(ui_tsne.listWidget_check_point.item(i)).stateChanged.connect(draw_graph_tsne)

    set_list_point()
    set_list_check_point()
    draw_graph_tsne()
    draw_graph_anova()

    TSNE_form.exec_()


def update_maket_combobox():
    ui.comboBox_geochem_maket.clear()
    for i in session.query(GeochemMaket).filter_by(geochem_id=get_geochem_id()).all():
        ui.comboBox_geochem_maket.addItem(f'{i.title} id{i.id}')
        ui.comboBox_geochem_maket.setItemData(ui.comboBox_geochem_maket.count() - 1, {'id': i.id})
    update_geochem_param_train_list()
    update_category_combobox()
    update_g_model_list()


def update_category_combobox():
    ui.comboBox_geochem_cat.clear()
    for i in session.query(GeochemCategory).filter_by(maket_id=get_maket_id()).all():
        ui.comboBox_geochem_cat.addItem(f'{i.title} id{i.id}')
        ui.comboBox_geochem_cat.setItemData(
            ui.comboBox_geochem_cat.findText(f'{i.title} id{i.id}'),
            QBrush(QColor(i.color)),
            Qt.BackgroundRole
        )
    update_g_train_point_list()



def add_maket():
    """Добавить новый макет"""
    if ui.lineEdit_string.text() == '':
        return
    new_maket = GeochemMaket(title=ui.lineEdit_string.text(), geochem_id=get_geochem_id())
    session.add(new_maket)
    session.commit()
    update_maket_combobox()
    update_geochem_param_train_list()
    set_info(f'Макет {new_maket.title} добавлен', 'green')

def remove_maket():
    """Удалить макет"""
    maket_title = ui.comboBox_geochem_maket.currentText()
    result = QtWidgets.QMessageBox.question(
        MainWindow,
        'Удаление макета',
        f'Вы уверены, что хотите удалить макет {maket_title}?',
        QtWidgets.QMessageBox.Yes,
        QtWidgets.QMessageBox.No
    )
    if result == QtWidgets.QMessageBox.Yes:
        for i in session.query(GeochemMaket).filter_by(id=get_maket_id()).first().categories:
            session.query(GeochemTrainPoint).filter_by(cat_id=i.id).delete()
        session.query(GeochemTrainParameter).filter_by(maket_id=get_maket_id()).delete()
        session.query(GeochemCategory).filter_by(maket_id=get_maket_id()).delete()
        session.query(GeochemMaket).filter_by(id=get_maket_id()).delete()
        session.commit()
        set_info(f'Макет "{maket_title}" удален', 'green')
        update_maket_combobox()
    else:
        pass

def add_category():
    """Добавить новую категорию"""
    if ui.lineEdit_string.text() == '':
        return
    new_cat = GeochemCategory(title=ui.lineEdit_string.text(), maket_id=get_maket_id(), color=ui.pushButton_color.text())
    session.add(new_cat)
    session.commit()
    update_category_combobox()
    update_geochem_param_train_list()
    set_info(f'Категория {new_cat.title} добавлена', 'green')

def remove_category():
    """Удалить категорию"""
    cat_title = ui.comboBox_geochem_cat.currentText()
    result = QtWidgets.QMessageBox.question(
        MainWindow,
        'Удаление категории',
        f'Вы уверены, что хотите удалить категорию {cat_title}?',
        QtWidgets.QMessageBox.Yes,
        QtWidgets.QMessageBox.No
    )
    if result == QtWidgets.QMessageBox.Yes:
        session.query(GeochemTrainPoint).filter_by(cat_id=get_category_id()).delete()
        session.query(GeochemCategory).filter_by(id=get_category_id()).delete()
        session.commit()
        set_info(f'Категория "{cat_title}" удалена', 'green')
        update_category_combobox()
    else:
        pass

def update_geochem_param_train_list():
    ui.listWidget_geochem_param_train.clear()
    for i in session.query(GeochemTrainParameter).filter_by(maket_id=get_maket_id()).all():
        try:
            item_text = (f'{i.param.title} id{i.id}')
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, i.id)
            ui.listWidget_geochem_param_train.addItem(item)
        except AttributeError:
            session.query(GeochemTrainParameter).filter_by(id=i.id).delete()
    session.commit()
    ui.label_30.setText(f'Parameters: {ui.listWidget_geochem_param_train.count()}')
    # update_category_combobox()

def add_all_geochem_param_train():
    """ Добавляет все параметры геохимии в список для тренировки """
    for i in session.query(GeochemParameter).filter_by(geochem_id=get_geochem_id()).all():
        if get_maket_id():
            maket = session.query(GeochemTrainParameter).filter_by(maket_id=get_maket_id()).all()
            flag = 0
            for mk in maket:
                if mk.param_id == i.id:
                    flag = 1
                    break
            if flag == 0:
                param = GeochemTrainParameter(maket_id=get_maket_id(), param_id=i.id)
                session.add(param)
        else:
            set_info(f'Для добавления параметра создайте макет', 'red')
    update_geochem_param_train_list()

def add_geochem_param_train():
    item = session.query(GeochemParameter).filter_by(id=ui.listWidget_geochem_param.currentItem().text().split(' id')[-1]).first()
    if get_maket_id():
        maket = session.query(GeochemTrainParameter).filter_by(maket_id=get_maket_id()).all()
        for mk in maket:
            if mk.param_id == item.id:
                return
        param = GeochemTrainParameter(param_id=item.id, maket_id=get_maket_id())
        session.add(param)
    else:
        set_info(f'Для добавления тренировочного параметра создайте макет', 'red')
    session.commit()
    update_geochem_param_train_list()


def remove_geochem_param_train():
    if ui.listWidget_geochem_param_train.currentItem():
        item = session.query(GeochemTrainParameter).filter_by(
            id=ui.listWidget_geochem_param_train.currentItem().text().split(' id')[-1]).first()
        if item:
            session.delete(item)
            session.commit()
            update_geochem_param_train_list()


def add_whole_well_to_list():
    points = session.query(GeochemTrainPoint).join(GeochemCategory).filter(
        GeochemCategory.maket_id == get_maket_id()
    ).all()
    list_points = [p.point_well_id for p in points]
    for i in session.query(GeochemWellPoint).filter_by(g_well_id=get_geochem_well_id()).all():
        if i.id not in list_points:
            point = GeochemTrainPoint(cat_id=get_category_id(), type_point='well', point_well_id=i.id, title=i.title)
            session.add(point)
            session.commit()
    update_g_train_point_list()


def add_field_point_to_list():
    try:
        item = session.query(GeochemPoint).filter_by(id=ui.listWidget_g_point.currentItem().text().split(' id')[-1]).first()
    except AttributeError:
        set_info(f'Выберите точку', 'red')
        return
    if get_category_id():
        cat = session.query(GeochemTrainPoint).join(GeochemCategory).filter_by(maket_id=get_maket_id()).all()
        for c in cat:
            if c.point_id == item.id:
                set_info(f'Точка "{item.title}" уже добавлена', 'red')
                return
        point = GeochemTrainPoint(cat_id=get_category_id(), type_point='field', point_id=item.id, title=item.title)
        session.add(point)
        session.commit()
    update_g_train_point_list()


def del_g_train_point():
    point_name = ui.listWidget_g_train_point.currentItem().text()
    result = QtWidgets.QMessageBox.question(
        MainWindow,
        'Удаление точки',
        f'Вы уверены, что хотите удалить точку {point_name}?',
        QtWidgets.QMessageBox.Yes,
        QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        session.query(GeochemTrainPoint).filter_by(
            id=ui.listWidget_g_train_point.currentItem().text().split(' id')[-1]).delete()
        session.commit()
        set_info(f'Точка "{point_name}" удалена', 'green')
        update_g_train_point_list()
    else:
        pass

def build_geochem_table():
    parameters = session.query(GeochemTrainParameter).filter_by(maket_id=get_maket_id()).all()
    list_param = [i.param.title for i in parameters]
    data = pd.DataFrame(columns=['title', 'category'] + list_param)

    for point in session.query(GeochemTrainPoint).join(GeochemCategory).join(GeochemMaket).filter(
            GeochemMaket.id == get_maket_id()).all():
        dict_point = {}
        if point.fake:
            continue
        dict_point['title'] = point.title
        dict_point['category'] = point.category.title
        for p in parameters:
            if point.type_point == 'field':
                value = session.query(GeochemPointValue).filter_by(g_point_id=point.point_id,
                                                                       g_param_id=p.param_id).first()
            else:
                value = session.query(GeochemWellPointValue).filter_by(g_well_point_id=point.point_well_id,
                                                                         g_param_id=p.param_id).first()
            dict_point[p.param.title] = value.value

        data = pd.concat([data, pd.DataFrame([dict_point])], ignore_index=True)
    return data, list_param


def build_geochem_table_field():
    parameters = session.query(GeochemParameter).filter_by(geochem_id=get_geochem_id()).all()
    list_param = [i.title for i in parameters]
    data = pd.DataFrame(columns=['title', 'X', 'Y'] + list_param)

    for point in session.query(GeochemPoint).filter_by(geochem_id=get_geochem_id()).all():
        dict_point = {'title': point.title, 'X': point.x_coord, 'Y': point.y_coord}
        for p in parameters:
            dict_point[p.title] = session.query(GeochemPointValue).filter_by(g_point_id=point.id, g_param_id=p.id).first().value
        data = pd.concat([data, pd.DataFrame([dict_point])], ignore_index=True)
    return data, list_param


def train_model_geochem():
    data_train, list_param = build_geochem_table()
    colors = {}
    if session.query(GeochemTrainParameter).filter_by(maket_id=get_maket_id()).first():
        for c in session.query(GeochemCategory).filter_by(maket_id=get_maket_id()).all():
            colors[c.title] = c.color
        train_classifier(data_train, list_param, list_param, colors, 'category', 'title', 'geochem')


def calc_geochem_classification():
    data_test, list_param = build_geochem_table_field()

    Choose_RegModel = QtWidgets.QDialog()
    ui_rm = Ui_FormRegMod()
    ui_rm.setupUi(Choose_RegModel)
    Choose_RegModel.show()
    Choose_RegModel.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    def calc_class_model():

        model = session.query(GeochemTrainedModel).filter_by(
            id=ui.listWidget_g_trained_model.currentItem().data(Qt.UserRole)).first()

        list_param_model = json.loads(model.list_params)

        with open(model.path_model, 'rb') as f:
            class_model = pickle.load(f)

        working_sample = data_test[list_param_model].values.tolist()

        list_cat = list(class_model.classes_)

        try:
            mark = class_model.predict(working_sample)
            probability = class_model.predict_proba(working_sample)
        except ValueError:
            working_sample = [[np.nan if np.isinf(x) else x for x in y] for y in working_sample]
            data = imputer.fit_transform(working_sample)
            mark = class_model.predict(data)
            probability = class_model.predict_proba(data)

            for i in working_sample.index:
                p_nan = [working_sample.columns[ic + 3] for ic, v in
                         enumerate(working_sample.iloc[i, 3:].tolist()) if
                         np.isnan(v)]
                if len(p_nan) > 0:
                    set_info(
                        f'Внимание для измерения "{i}" отсутствуют параметры "{", ".join(p_nan)}", поэтому расчёт для'
                        f' этого измерения может быть не корректен', 'red')

        # Добавление предсказанных меток и вероятностей в рабочие данные
        working_data_result = pd.concat([data_test, pd.DataFrame(probability, columns=list_cat)], axis=1)
        working_data_result['mark'] = mark

        x = list(working_data_result['X'])
        y = list(working_data_result['Y'])
        if len(set(mark)) == 2 and not ui_rm.checkBox_color_marker.isChecked():
            z = list(working_data_result[list_cat[0]])
            color_marker = False
            z_number = string_to_unique_number(list(working_data_result['mark']), 'geochem')
            working_data_result['mark_number'] = z_number
        else:
            z = string_to_unique_number(list(working_data_result['mark']), 'geochem')
            color_marker = True
            working_data_result['mark_number'] = z
        draw_map(x, y, z, f'Geochem {ui.listWidget_g_trained_model.currentItem().text()}', color_marker)
        result1 = QMessageBox.question(MainWindow, 'Сохранение', 'Сохранить результаты расчёта MLP?', QMessageBox.Yes, QMessageBox.No)
        if result1 == QMessageBox.Yes:
            result2 = QMessageBox.question(MainWindow, 'Сохранение', 'Сохранить только результаты расчёта?', QMessageBox.Yes, QMessageBox.No)
            if result2 == QMessageBox.Yes:
                list_col = [i.title for i in session.query(GeochemCategory).filter_by(maket_id=get_maket_id()).all()]
                list_col += ['title', 'X', 'Y', 'mark', 'mark_number']
                working_data_result = working_data_result[list_col]
            else:
                pass
            try:
                file_name = f'{get_object_name()}_{get_research_name()}__модель_{get_mlp_title()}.xlsx'
                fn = QFileDialog.getSaveFileName(caption=f'Сохранить результат MLP "{get_object_name()}_{get_research_name()}" в таблицу', directory=file_name,
                                                 filter="Excel Files (*.xlsx)")
                working_data_result.to_excel(fn[0])
                set_info(f'Таблица сохранена в файл: {fn[0]}', 'green')
            except ValueError:
                pass
        else:
            pass
    ui_rm.pushButton_calc_model.clicked.connect(calc_class_model)
    Choose_RegModel.exec_()


def drop_fake_geochem():
    for cat in session.query(GeochemCategory).filter_by(maket_id=get_maket_id()).all():
        session.query(GeochemTrainPoint).filter_by(cat_id=cat.id).update({'fake': 0}, synchronize_session='fetch')
    session.commit()
    update_g_train_point_list()


def draw_point_graph():
    PointGraph = QtWidgets.QDialog()
    ui_pg = Ui_GraphParam()
    ui_pg.setupUi(PointGraph)
    PointGraph.show()
    PointGraph.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    m_width, m_height = get_width_height_monitor()
    PointGraph.resize(m_width - 200, m_height - 200)

    data_plot = build_table_geochem_analytic(point_name=True)
    markers = list(set(data_plot['well']))
    # pallet = {'field': 'green'}
    # for m in markers:
    #     if m == 'field':
    #         continue
    #     pallet[m] = session.query(GeochemWell).filter(GeochemWell.title == m, GeochemWell.geochem_id == get_geochem_id()).first().color

    def draw_graph():
        clear_layout(ui_pg.verticalLayout)
        figure = plt.figure()
        canvas = FigureCanvas(figure)
        mpl_toolbar = NavigationToolbar(canvas)
        ui_pg.verticalLayout.addWidget(mpl_toolbar)
        ui_pg.verticalLayout.addWidget(canvas)

        list_point = get_list_check_checkbox(ui_pg.listWidget_point)
        list_param = get_list_check_checkbox(ui_pg.listWidget_param)

        for p in list_point:
            plt.plot(
                data_plot.loc[data_plot['point'] == p][list_param].values.tolist()[0],
                label=p
            )
        num_param = range(len(list_param))
        plt.xticks(num_param, list_param, rotation=90)
        plt.grid()
        plt.legend()
        figure.tight_layout()
        figure.subplots_adjust(bottom=0.25)
        canvas.draw()

    for i_param in data_plot.columns.tolist()[3:]:
        check_box_widget = QCheckBox(i_param)
        check_box_widget.setChecked(True)
        check_box_widget.clicked.connect(draw_graph)
        list_item = QListWidgetItem()
        ui_pg.listWidget_param.addItem(list_item)
        ui_pg.listWidget_param.setItemWidget(list_item, check_box_widget)

    def set_list_point():
        ui_pg.listWidget_point.clear()
        list_well = get_list_check_checkbox(ui_pg.listWidget_well)
        data_plot_point = data_plot.loc[data_plot['well'].isin(list_well)]
        for n, point_name in enumerate(data_plot_point['point']):
            check_box_widget = QCheckBox(point_name)
            if n < 2:
                check_box_widget.setChecked(True)
            check_box_widget.clicked.connect(draw_graph)
            list_item = QListWidgetItem()
            ui_pg.listWidget_point.addItem(list_item)
            ui_pg.listWidget_point.setItemWidget(list_item, check_box_widget)


    for w in data_plot['well'].unique():
        check_box_widget = QCheckBox(w)
        check_box_widget.setChecked(True)
        check_box_widget.clicked.connect(set_list_point)
        check_box_widget.clicked.connect(draw_graph)
        list_item = QListWidgetItem()
        ui_pg.listWidget_well.addItem(list_item)
        ui_pg.listWidget_well.setItemWidget(list_item, check_box_widget)

    def all_check_param():
        all_check(ui_pg.listWidget_param, ui_pg.checkBox_param_all)
        draw_graph()

    def all_check_point():
        all_check(ui_pg.listWidget_point, ui_pg.checkBox_point_all)
        draw_graph()

    def all_check_well():
        all_check(ui_pg.listWidget_well, ui_pg.checkBox_well_all)
        draw_graph()
        set_list_point()

    def all_check(widget, check):
        check = True if check.isChecked() else False
        for i in range(widget.count()):
            item = widget.item(i)
            if isinstance(item, QListWidgetItem):
                checkbox = widget.itemWidget(item)
                if isinstance(checkbox, QCheckBox):
                    checkbox.setChecked(check)

    set_list_point()
    draw_graph()

    ui_pg.checkBox_param_all.stateChanged.connect(all_check_param)
    ui_pg.checkBox_point_all.stateChanged.connect(all_check_point)
    ui_pg.checkBox_well_all.stateChanged.connect(all_check_well)
    PointGraph.exec_()