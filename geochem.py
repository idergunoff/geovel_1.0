import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special
from scipy import stats

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

    ui_tsne.spinBox_lof.setValue(int(len(data_plot) * 0.1))

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
                print(ui_tsne.spinBox_random_stat.value())
                name_graph = 't-SNE'
                tsne = TSNE(
                    n_components=2,
                    perplexity=ui_tsne.spinBox_perplexity.value(),
                    learning_rate=200,
                    random_state=ui_tsne.spinBox_random_stat.value()
                )
                data_tsne_result = tsne.fit_transform(data_tsne)
            if ui_tsne.radioButton_pca.isChecked():
                name_graph = 'PCA'
                pca = PCA(
                    n_components=2,
                    random_state=ui_tsne.spinBox_random_stat.value()
                )
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
        ui_tsne.spinBox_lof.setValue(int(len(data_plot_point) * 0.1))

    for w in data_plot['well'].unique():
        check_box_widget = QCheckBox(w)
        check_box_widget.setChecked(True)
        # check_box_widget.stateChanged.connect(draw_graph_tsne)
        list_item = QListWidgetItem()
        ui_tsne.listWidget_checkbox_well.addItem(list_item)
        ui_tsne.listWidget_checkbox_well.setItemWidget(list_item, check_box_widget)

    def find_best_param():
        list_well = get_list_check_checkbox(ui_tsne.listWidget_checkbox_well)
        list_param = get_list_check_checkbox(ui_tsne.listWidget_check_param)
        list_point = get_list_check_checkbox(ui_tsne.listWidget_check_point)
        data_find = data_plot.loc[data_plot['point'].isin(list_point)]
        data_find.reset_index(inplace=True, drop=True)
        # print(distance_between_centers(data_plot, list_well[0], list_well[1], list_param))
        # print(find_optimal_params_add_one_at_a_time(data_plot, list_well[0], list_well[1], list_param))
        best_param = find_optimal_params_remove_one_at_a_time(data_find, list_well[0], list_well[1], list_param)
        print(best_param)
        check_param(best_param)
        draw_graph_tsne()

    def all_check_param():
        for i in range(ui_tsne.listWidget_check_param.count()):
            checkbox = ui_tsne.listWidget_check_param.itemWidget(ui_tsne.listWidget_check_param.item(i))
            checkbox.setChecked(True)

    def check_param(param):
        for i in range(ui_tsne.listWidget_check_param.count()):
            checkbox = ui_tsne.listWidget_check_param.itemWidget(ui_tsne.listWidget_check_param.item(i))
            if checkbox.text() in param:
                checkbox.setChecked(True)
            else:
                checkbox.setChecked(False)
        # draw_graph_tsne()
        # draw_graph_anova()

    def distance_between_centers(data, well1, well2, params):
        data_param = data[params]
        if ui_tsne.checkBox_standart.isChecked():
            scaler = StandardScaler()
            data_param = scaler.fit_transform(data_param)
        if ui_tsne.radioButton_tsne.isChecked():
            tsne = TSNE(
                n_components=2,
                perplexity=ui_tsne.spinBox_perplexity.value(),
                learning_rate=200,
                random_state=ui_tsne.spinBox_random_stat.value()
            )
            projected_data = pd.DataFrame(tsne.fit_transform(data_param), columns=['tsne1', 'tsne2'])
        if ui_tsne.radioButton_pca.isChecked():
            pca = PCA(n_components=2, random_state=ui_tsne.spinBox_random_stat.value())
            projected_data = pd.DataFrame(pca.fit_transform(data_param), columns=['tsne1', 'tsne2'])

        projected_data['well'] = data['well']

        center1 = np.array([projected_data.loc[projected_data['well'] == well1]['tsne1'].median(),
                            projected_data.loc[projected_data['well'] == well1]['tsne2'].median()])
        center2 = np.array([projected_data.loc[projected_data['well'] == well2]['tsne1'].median(),
                            projected_data.loc[projected_data['well'] == well2]['tsne2'].median()])

        # print(np.linalg.norm(center1 - center2))
        return np.sqrt((center2[0] - center1[0]) ** 2 + (center2[1] - center1[1]) ** 2)

    def find_optimal_params_remove_one_at_a_time(data, well1, well2, initial_params):
        current_params, best_params = initial_params.copy(), initial_params.copy()
        current_distance = distance_between_centers(data, well1, well2, current_params)
        len_params = len(current_params)
        ui_tsne.progressBar_1.setMaximum(len_params)
        while True:
            max_distance = current_distance
            ui_tsne.lcdNumber.display(len(best_params))
            print(f'max_distance={max_distance},current={len(current_params)}, best={len(best_params)}')
            param_to_remove = None
            list_dist, list_param = [], []
            ui_tsne.progressBar_1.setValue(len_params - len(current_params))
            ui_tsne.progressBar_2.setMaximum(len(current_params))
            for np, param in enumerate(current_params):
                ui_tsne.progressBar_2.setValue(np)
                temp_params = current_params.copy()
                temp_params.remove(param)
                temp_distance = distance_between_centers(data, well1, well2, temp_params)
                list_dist.append(temp_distance)
                list_param.append(param)

                if temp_distance > max_distance:
                    max_distance = temp_distance
                    param_to_remove = param

            if param_to_remove is not None:
                print(param_to_remove)
                current_params.remove(param_to_remove)
                current_distance = max_distance
                best_params = current_params.copy()
                check_param(best_params)
                draw_graph_tsne()
            else:
                i_rem = list_dist.index(max(list_dist))
                current_params.remove(list_param[i_rem])
            if len(current_params) == 2:
                break

        return best_params

    def calc_lof():
        list_param = get_list_check_checkbox(ui_tsne.listWidget_check_param)
        list_point = get_list_check_checkbox(ui_tsne.listWidget_check_point)
        data_param = data_plot.loc[data_plot['point'].isin(list_point)][list_param]
        data_param.reset_index(inplace=True, drop=True)
        data_point = data_plot.loc[data_plot['point'].isin(list_point)]
        data_point.reset_index(inplace=True, drop=True)
        scaler = StandardScaler()
        data_param = scaler.fit_transform(data_param)
        if ui_tsne.radioButton_tsne.isChecked():
            try:
                tsne = TSNE(
                    n_components=2,
                    perplexity=ui_tsne.spinBox_perplexity.value(),
                    learning_rate=200,
                    random_state=ui_tsne.spinBox_random_stat.value()
                )
                projected_data = pd.DataFrame(tsne.fit_transform(data_param), columns=['tsne1', 'tsne2'])
            except ValueError:
                set_info('ValueError, параметр perplexity должен быть меньше количества точек', 'red')
                QMessageBox.critical(MainWindow, 'Ошибка', 'Параметр perplexity должен быть меньше количества точек')
                return
        if ui_tsne.radioButton_pca.isChecked():
            pca = PCA(n_components=2, random_state=ui_tsne.spinBox_random_stat.value())
            projected_data = pd.DataFrame(pca.fit_transform(data_param), columns=['tsne1', 'tsne2'])
        lof = LocalOutlierFactor(n_neighbors=ui_tsne.spinBox_lof.value())
        label_lof = lof.fit_predict(data_param)
        factor_lof = -lof.negative_outlier_factor_

        data_lof = data_point.copy()
        data_lof['lof'] = label_lof
        data_lof = pd.concat([projected_data, data_lof], axis=1)

        colors_lof = ['red' if label == -1 else 'blue' for label in label_lof]

        clear_layout(ui_tsne.verticalLayout_anova)
        figure_bar = plt.figure()
        canvas_bar = FigureCanvas(figure_bar)
        mpl_toolbar = NavigationToolbar(canvas_bar)
        ui_tsne.verticalLayout_anova.addWidget(mpl_toolbar)
        ui_tsne.verticalLayout_anova.addWidget(canvas_bar)
        plt.bar(range(len(label_lof)), factor_lof, color=colors_lof)
        plt.xticks(range(len(label_lof)), list_point, rotation=90)
        figure_bar.suptitle(f'коэффициенты LOF')
        figure_bar.tight_layout()
        canvas_bar.show()

        clear_layout(ui_tsne.verticalLayout_graph)
        figure_tsne = plt.figure()
        canvas_tsne = FigureCanvas(figure_tsne)
        mpl_toolbar = NavigationToolbar(canvas_tsne)
        ui_tsne.verticalLayout_graph.addWidget(mpl_toolbar)
        ui_tsne.verticalLayout_graph.addWidget(canvas_tsne)
        sns.scatterplot(data=data_lof, x='tsne1', y='tsne2', hue='lof', s=100, palette={-1: 'red', 1: 'blue'})
        for i_data in data_lof.index:
            plt.text(data_lof['tsne1'][i_data], data_lof['tsne2'][i_data],
                data_lof['point'][i_data], horizontalalignment='left',
                size='medium', color='black', weight='semibold')
        plt.grid()
        figure_tsne.suptitle(f't-SNE')
        figure_tsne.tight_layout()
        canvas_tsne.draw()

    def clear_lof():
        list_param = get_list_check_checkbox(ui_tsne.listWidget_check_param)
        list_point = get_list_check_checkbox(ui_tsne.listWidget_check_point)
        data_param = data_plot.loc[data_plot['point'].isin(list_point)][list_param]
        data_param.reset_index(inplace=True, drop=True)
        data_point = data_plot.loc[data_plot['point'].isin(list_point)]
        data_point.reset_index(inplace=True, drop=True)
        scaler = StandardScaler()
        data_param = scaler.fit_transform(data_param)
        lof = LocalOutlierFactor(n_neighbors=ui_tsne.spinBox_lof.value())
        label_lof = lof.fit_predict(data_param)
        data_lof = data_point.copy()
        data_lof['lof'] = label_lof
        for i in range(ui_tsne.listWidget_check_point.count()):
            checkbox = ui_tsne.listWidget_check_point.itemWidget(ui_tsne.listWidget_check_point.item(i))
            try:
                if data_lof['lof'].loc[data_lof['point'] == checkbox.text()].values[0] == -1:
                    checkbox.setChecked(False)
                else:
                    checkbox.setChecked(True)
            except IndexError:
                pass

    def set_param_maket():
        session.query(GeochemTrainParameter).filter_by(maket_id=get_maket_id()).delete()
        for i in range(ui_tsne.listWidget_check_param.count()):
            checkbox = ui_tsne.listWidget_check_param.itemWidget(ui_tsne.listWidget_check_param.item(i))
            if checkbox.isChecked():
                param_id = session.query(GeochemParameter).filter_by(
                    title=checkbox.text(),
                    geochem_id=get_geochem_id()
                ).first().id
                new_train_param = GeochemTrainParameter(
                    maket_id=get_maket_id(),
                    param_id=param_id
                )
                session.add(new_train_param)
        session.commit()
        update_geochem_param_train_list()

    def set_point_fake():
        """ Установить флаг fake на отключенные точки """
        for i in range(ui_tsne.listWidget_check_point.count()):
            checkbox = ui_tsne.listWidget_check_point.itemWidget(ui_tsne.listWidget_check_point.item(i))
            fake = False if checkbox.isChecked() else True
            point = session.query(GeochemTrainPoint).join(GeochemCategory).filter(
                GeochemTrainPoint.title == checkbox.text(),
                GeochemCategory.maket_id == get_maket_id()
            ).first()
            if point:
                session.query(GeochemTrainPoint).filter_by(id=point.id).update(
                    {'fake': fake}, synchronize_session='fetch')
        session.commit()
        update_g_train_point_list()

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
    ui_tsne.toolButton.clicked.connect(all_check_param)
    ui_tsne.pushButton_best_param.clicked.connect(find_best_param)
    ui_tsne.pushButton_lof.clicked.connect(calc_lof)
    ui_tsne.toolButton_lof_clean.clicked.connect(clear_lof)
    ui_tsne.toolButton_lof_all.clicked.connect(set_list_check_point)
    ui_tsne.pushButton_to_maket.clicked.connect(set_param_maket)
    ui_tsne.pushButton_point_to_fake.clicked.connect(set_point_fake)

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
    ui.label_30.setText(f'Train parameters: {ui.listWidget_geochem_param_train.count()}')
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

    def calc_mean_well(well_name: str, list_param: list):
        data_mean =  data_plot.loc[data_plot['well'] == well_name][list_param]
        list_result = []
        for param in list_param:
            if ui_pg.radioButton_mean.isChecked():
                list_result.append(data_mean[param].mean())
            if ui_pg.radioButton_median.isChecked():
                list_result.append(data_mean[param].median())
        return list_result

    def calc_conf_interval_well(well_name: str, list_param: list):
        data_mean =  data_plot.loc[data_plot['well'] == well_name][list_param]
        list_conf_top, list_conf_bottom = [], []
        for param in list_param:
            param_mean = data_mean[param].mean()
            param_std = data_mean[param].std()
            param_conf_interval = stats.norm.interval(0.95, loc=param_mean, scale=param_std)
            if param_conf_interval[0] < 0:
                list_conf_top.append(0)
            else:
                list_conf_top.append(param_conf_interval[0])
            if param_conf_interval[1] < 0:
                list_conf_bottom.append(0)
            else:
                list_conf_bottom.append(param_conf_interval[1])
        return list_conf_top, list_conf_bottom



    def draw_graph():
        clear_layout(ui_pg.verticalLayout)
        figure = plt.figure()
        canvas = FigureCanvas(figure)
        mpl_toolbar = NavigationToolbar(canvas)
        ui_pg.verticalLayout.addWidget(mpl_toolbar)
        ui_pg.verticalLayout.addWidget(canvas)

        list_point = get_list_check_checkbox(ui_pg.listWidget_point)
        list_param = get_list_check_checkbox(ui_pg.listWidget_param)
        list_well_graph = get_list_check_checkbox(ui_pg.listWidget_well_graph)

        for p in list_point:
            if ui_pg.checkBox_marker.isChecked():
                plt.plot(data_plot.loc[data_plot['point'] == p][list_param].values.tolist()[0], label=p, marker='o')
            else:
                plt.plot(data_plot.loc[data_plot['point'] == p][list_param].values.tolist()[0], label=p)
        num_param = range(len(list_param))

        for wg in list_well_graph:
            if ui_pg.checkBox_marker.isChecked():
                plt.plot(calc_mean_well(wg, list_param), label=f'mean_{wg}', marker='o', linestyle='--', linewidth=3)
            else:
                plt.plot(calc_mean_well(wg, list_param), label=f'mean_{wg}', linestyle='--', linewidth=3)

        if ui_pg.checkBox_conf_int.isChecked():
            for wg in list_well_graph:
                list_conf_top, list_conf_bottom = calc_conf_interval_well(wg, list_param)
                plt.fill_between(num_param, list_conf_top, list_conf_bottom, alpha=0.2, label=f'conf_int_{wg}')


        if ui_pg.checkBox_log.isChecked():
            plt.yscale('log')
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

    for n_w, w in enumerate(data_plot['well'].unique()):
        check_box_widget = QCheckBox(w)
        if n_w < 1:
            check_box_widget.setChecked(True)
        check_box_widget.clicked.connect(draw_graph)
        list_item = QListWidgetItem()
        ui_pg.listWidget_well_graph.addItem(list_item)
        ui_pg.listWidget_well_graph.setItemWidget(list_item, check_box_widget)

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
    ui_pg.checkBox_marker.clicked.connect(draw_graph)
    ui_pg.checkBox_log.clicked.connect(draw_graph)
    ui_pg.radioButton_mean.clicked.connect(draw_graph)
    ui_pg.radioButton_median.clicked.connect(draw_graph)
    ui_pg.checkBox_conf_int.clicked.connect(draw_graph)
    PointGraph.exec_()




# def find_optimal_params_add_one_at_a_time(data, well1, well2, initial_params):
#     current_params = [initial_params[0], initial_params[1]]
#     current_distance, dop = 0, 0
#
#     while len(current_params) < len(initial_params):
#         max_distance = current_distance
#         param_to_add = None
#         print(max_distance, len(current_params), dop)
#         list_dist, list_param = [], []
#
#         for param in initial_params:
#             if param not in current_params:
#                 temp_params = current_params.copy()
#                 temp_params.append(param)
#                 temp_distance = distance_between_centers(data, well1, well2, temp_params)
#                 list_dist.append(temp_distance)
#                 list_param.append(param)
#
#                 if temp_distance > max_distance:
#                     max_distance = temp_distance
#                     param_to_add = param
#
#         if param_to_add is not None:
#             current_params.append(param_to_add)
#             print(param_to_add)
#             current_distance = max_distance
#             dop = 0
#         else:
#             i_add = list_dist.index(max(list_dist))
#             current_params.append(list_param[i_add])
#             dop += 1
#             if dop > 10:
#                 break
#
#     return current_params
#
def try_func():
    ui.label_31.setText("Enter Pressed")




def update_trained_model_geochem_comment():
    """ Изменить комментарий обученной модели """
    try:
        g_model = session.query(GeochemTrainedModel).filter_by(id=ui.listWidget_g_trained_model.currentItem().data(Qt.UserRole)).first()
    except AttributeError:
        QMessageBox.critical(MainWindow, 'Не выбрана модель', 'Не выбрана модель.')
        set_info('Не выбрана модель', 'red')
        return

    FormComment = QtWidgets.QDialog()
    ui_cmt = Ui_Form_comment()
    ui_cmt.setupUi(FormComment)
    FormComment.show()
    FormComment.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    ui_cmt.textEdit.setText(g_model.comment)

    def update_comment():
        g_model.comment = ui_cmt.textEdit.toPlainText()
        session.commit()
        update_g_model_list()
        FormComment.close()

    ui_cmt.buttonBox.accepted.connect(update_comment)

    FormComment.exec_()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the cosine similarity between two vectors a and b.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def softmax_new(x):
    new_x = []
    for i in x:
        new_x.append((np.pi - np.arccos(i)) / np.pi)
    return new_x


def norm_akhmet(x):
    new_x = []
    for i in x:
        new_x.append((i - np.min(x)) / (np.max(x) - np.min(x)))
    return new_x

def calc_vector():
    data_train, list_param = build_geochem_table()
    data_test, _ = build_geochem_table_field()

    data_vector_train = data_train[list_param]
    data_vector_test = data_test[list_param]

    if ui.checkBox_std_sc.isChecked():
        scaler = StandardScaler()
        data_vector_train = scaler.fit_transform(data_vector_train)
        data_vector_train = pd.DataFrame(data_vector_train, columns=list_param)
        data_vector_test = scaler.transform(data_vector_test)
        data_vector_test = pd.DataFrame(data_vector_test, columns=list_param)

    list_cos_mean = []
    for i in data_vector_test.index:
        list_cos = []
        vector = data_vector_test.loc[i].to_numpy()
        for j in data_vector_train.index:
            if ui.checkBox_geochem_vector_corr.isChecked():
                list_cos.append(np.corrcoef(data_vector_train.loc[j], vector)[0][1])
            else:
                list_cos.append(cosine_similarity(data_vector_train.loc[j], vector))
            # list_cos.append(np.linalg.norm(data_vector_train.loc[j] - vector))
        list_cos_mean.append(np.median(list_cos))
    if ui.checkBox_softmax.isChecked():
        list_cos_mean = softmax_new(list_cos_mean)
        list_cos_mean = norm_akhmet(list_cos_mean)
    data_test['cos_mean'] = list_cos_mean
    x, y, z = data_test['X'], data_test['Y'], data_test['cos_mean']

    draw_map(x, y, z, f'Geochem vector {ui.comboBox_geochem_maket.currentText().split(" id")[0]}', False)


    result1 = QMessageBox.question(MainWindow, 'Сохранение', 'Сохранить результаты расчёта MLP?', QMessageBox.Yes,
                                   QMessageBox.No)
    if result1 == QMessageBox.Yes:
        result2 = QMessageBox.question(MainWindow, 'Сохранение', 'Сохранить только результаты расчёта?', QMessageBox.Yes,
                                       QMessageBox.No)
        if result2 == QMessageBox.Yes:
            list_col = ['title', 'X', 'Y', 'cos_mean']
            data_test = data_test[list_col]
        else:
            pass
        try:
            file_name = f'{get_object_name()}_{get_research_name()}__модель_{get_mlp_title()}.xlsx'
            fn = QFileDialog.getSaveFileName(
                caption=f'Сохранить результат MLP "{get_object_name()}_{get_research_name()}" в таблицу',
                directory=file_name,
                filter="Excel Files (*.xlsx)")
            data_test.to_excel(fn[0])
            set_info(f'Таблица сохранена в файл: {fn[0]}', 'green')
        except ValueError:
            pass
    else:
        pass
