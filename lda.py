import json
import profile

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor

from draw import draw_radarogram, draw_formation, draw_fill, draw_fake, draw_fill_result, remove_poly_item
from func import *
from qt.choose_formation_lda import *


def add_lda():
    """Добавить новый анализ LDA"""
    if ui.lineEdit_string.text() == '':
        set_info('Введите название анализа', 'red')
        return
    new_lda = AnalysisLDA(title=ui.lineEdit_string.text())
    session.add(new_lda)
    session.commit()
    update_list_lda()
    set_info(f'Добавлен новый анализ LDA - "{ui.lineEdit_string.text()}"', 'green')


def remove_lda():
    """Удалить анализ LDA"""
    lda_title = get_lda_title()
    result = QtWidgets.QMessageBox.question(ui.listWidget_well_lda, 'Remove markup LDA',
                                            f'Вы уверены, что хотите удалить модель LDA "{lda_title}"?',
                                            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        session.query(ParameterLDA).filter_by(analysis_id=get_LDA_id()).delete()
        session.query(MarkerLDA).filter_by(analysis_id=get_LDA_id()).delete()
        session.query(MarkupLDA).filter_by(analysis_id=get_LDA_id()).delete()
        session.query(AnalysisLDA).filter_by(id=get_LDA_id()).delete()
        session.commit()
        set_info(f'Удалена модель LDA - "{lda_title}"', 'green')
        update_list_lda()
    elif result == QtWidgets.QMessageBox.No:
        pass


def update_list_lda():
    """Обновить список анализов LDA"""
    ui.comboBox_lda_analysis.clear()
    for i in session.query(AnalysisLDA).order_by(AnalysisLDA.title).all():
        ui.comboBox_lda_analysis.addItem(f'{i.title} id{i.id}')
    update_list_marker_lda()


def add_marker_lda():
    """Добавить новый маркер LDA"""
    if ui.lineEdit_string.text() == '':
        set_info('Введите название маркера', 'red')
        return
    new_marker = MarkerLDA(title=ui.lineEdit_string.text(), analysis_id=get_LDA_id(), color=ui.pushButton_color.text())
    session.add(new_marker)
    session.commit()
    update_list_marker_lda()
    set_info(f'Добавлен новый маркер LDA - "{ui.lineEdit_string.text()}"', 'green')


def remove_marker_lda():
    """Удалить маркер LDA"""
    # todo: подтверждение удаления, потом удаление всех составляющих анализа
    pass


def update_list_marker_lda():
    """Обновить список маркеров LDA"""
    ui.comboBox_mark_lda.clear()
    for i in session.query(MarkerLDA).filter(MarkerLDA.analysis_id == get_LDA_id()).order_by(MarkerLDA.title).all():
        item = f'{i.title} id{i.id}'
        ui.comboBox_mark_lda.addItem(f'{i.title} id{i.id}')
        ui.comboBox_mark_lda.setItemData(ui.comboBox_mark_lda.findText(item), QBrush(QColor(i.color)),
                                         Qt.BackgroundRole)
    update_list_well_markup_lda()
    update_list_param_lda()


def add_well_markup_lda():
    """Добавить новую обучающую скважину для LDA"""
    analysis_id = get_LDA_id()
    well_id = get_well_id()
    profile_id = get_profile_id()
    formation_id = get_formation_id()
    marker_id = get_marker_id()

    if analysis_id and well_id and profile_id and marker_id and formation_id:
        for param in get_list_param_lda():
            if not session.query(literal_column(f'Formation.{param}')).filter(Formation.id == formation_id).first()[0]:
                set_info(f'Параметр {param} отсутствует для профиля {get_profile_name()}', 'red')
                return
        well = session.query(Well).filter(Well.id == well_id).first()
        x_prof = json.loads(session.query(Profile.x_pulc).filter(Profile.id == profile_id).first()[0])
        y_prof = json.loads(session.query(Profile.y_pulc).filter(Profile.id == profile_id).first()[0])
        index, _ = closest_point(well.x_coord, well.y_coord, x_prof, y_prof)
        well_dist = ui.spinBox_well_dist.value()
        start = index - well_dist if index - well_dist > 0 else 0
        stop = index + well_dist if index + well_dist < len(x_prof) else len(x_prof)

        list_measure = list(range(start, stop))
        new_markup_lda = MarkupLDA(analysis_id=analysis_id, well_id=well_id, profile_id=profile_id,
                                   marker_id=marker_id, formation_id=formation_id,
                                   list_measure=json.dumps(list_measure))
        session.add(new_markup_lda)
        session.commit()
        set_info(f'Добавлена новая обучающая скважина для LDA - "{ui.lineEdit_string.text()}"', 'green')
        update_list_well_markup_lda()
    else:
        set_info('выбраны не все параметры', 'red')


def remove_well_markup_lda():
    markup = session.query(MarkupLDA).filter(MarkupLDA.id == get_markup_id()).first()
    if not markup:
        return
    skv_name = session.query(Well.name).filter(Well.id == markup.well_id).first()[0]
    prof_name = session.query(Profile.title).filter(Profile.id == markup.profile_id).first()[0]
    lda_name = session.query(AnalysisLDA.title).filter(AnalysisLDA.id == markup.analysis_id).first()[0]
    result = QtWidgets.QMessageBox.question(ui.listWidget_well_lda, 'Remove markup LDA',
                                            f'Вы уверены, что хотите удалить скважину "{skv_name}" на '
                                            f'профиле "{prof_name}" из обучающей модели LDA-анализа "{lda_name}"?',
                                            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        session.delete(markup)
        session.commit()
        set_info(f'Удалена обучающая скважина для LDA - "{ui.listWidget_well_lda.currentItem().text()}"', 'green')
        update_list_well_markup_lda()
    elif result == QtWidgets.QMessageBox.No:
        pass


def update_list_well_markup_lda():
    """Обновление списка обучающих скважин LDA"""
    ui.listWidget_well_lda.clear()
    for i in session.query(MarkupLDA).filter(MarkupLDA.analysis_id == get_LDA_id()).all():
        fake = len(json.loads(i.list_fake)) if i.list_fake else 0
        measure = len(json.loads(i.list_measure))
        item = f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | {i.well.name} | {measure - fake} из {measure} | id{i.id}'
        ui.listWidget_well_lda.addItem(item)
        i_item = ui.listWidget_well_lda.findItems(item, Qt.MatchContains)[0]
        i_item.setBackground(QBrush(QColor(i.marker.color)))
        # ui.listWidget_well_lda.setItemData(ui.listWidget_well_lda.findText(item), QBrush(QColor(i.marker.color)), Qt.BackgroundRole)


def choose_marker_lda():
    # Функция выбора маркера LDA
    # Выбирает маркер, на основе сохраненных данных из базы данных, и затем обновляет все соответствующие виджеты
    # пользовательского интерфейса

    # Получение информации о маркере из БД по его ID
    markup = session.query(MarkupLDA).filter(MarkupLDA.id == get_markup_id()).first()
    # Если ID маркера не найден в БД, то функция завершается
    if not markup:
        return

    # Установка соответствующих значений виджетов пользовательского интерфейса
    ui.comboBox_object.setCurrentText(f'{markup.profile.research.object.title} id{markup.profile.research.object_id}')
    update_research_combobox()
    ui.comboBox_research.setCurrentText(
        f'{markup.profile.research.date_research.strftime("%m.%Y")} id{markup.profile.research_id}')
    update_profile_combobox()
    count_measure = len(json.loads(session.query(Profile.signal).filter(Profile.id == markup.profile_id).first()[0]))
    ui.comboBox_profile.setCurrentText(f'{markup.profile.title} ({count_measure} измерений) id{markup.profile_id}')
    draw_radarogram()
    ui.comboBox_plast.setCurrentText(f'{markup.formation.title} id{markup.formation_id}')
    draw_formation()
    draw_well(markup.well_id)
    list_measure = json.loads(markup.list_measure)  # Получение списка измерений
    list_fake = json.loads(markup.list_fake) if markup.list_fake else []  # Получение списка пропущенных измерений
    list_up = json.loads(markup.formation.layer_up.layer_line)  # Получение списка с верхними границами формации
    list_down = json.loads(markup.formation.layer_down.layer_line)  # Получение списка со снижными границами формации
    y_up = [list_up[i] for i in list_measure]  # Создание списка верхних границ для отображения
    y_down = [list_down[i] for i in list_measure]  # Создание списка нижних границ для отображения
    # Обновление маркера с конкретными данными о верхней и нижней границах и цветом
    draw_fill(list_measure, y_up, y_down, markup.marker.color)
    draw_fake(list_fake, list_up, list_down)


def add_param_geovel_lda():
    param = ui.comboBox_geovel_param_lda.currentText()
    for m in session.query(MarkupLDA).filter(MarkupLDA.analysis_id == get_LDA_id()).all():
        if not session.query(literal_column(f'Formation.{param}')).filter(Formation.id == m.formation_id).first()[0]:
            set_info(f'Параметр {param} отсутствует для профиля {m.profile.title}', 'red')
            return
    if session.query(ParameterLDA).filter_by(
            analysis_id=get_LDA_id(),
            parameter= param
    ).count() == 0:
        add_param_lda(param)
        update_list_param_lda()
    else:
        set_info(f'Параметр {param} уже добавлен', 'red')


def add_all_param_geovel_lda():
    new_list_param = list_param_geovel.copy()
    for param in list_param_geovel:
        for m in session.query(MarkupLDA).filter(MarkupLDA.analysis_id == get_LDA_id()).all():
            if not session.query(literal_column(f'Formation.{param}')).filter(Formation.id == m.formation_id).first()[0]:
                if param in new_list_param:
                    set_info(f'Параметр {param} отсутствует для профиля {m.profile.title}', 'red')
                    new_list_param.remove(param)
    for param in new_list_param:
        if session.query(ParameterLDA).filter(ParameterLDA.analysis_id == get_LDA_id()).filter(
                ParameterLDA.parameter == param).count() > 0:
            set_info(f'Параметр {param} уже добавлен', 'red')
            continue
        add_param_lda(param)
    update_list_param_lda()


def remove_param_geovel_lda():
    if ui.listWidget_param_lda.currentItem():
        session.query(ParameterLDA).filter_by(analysis_id=get_LDA_id(),
                                              parameter=ui.listWidget_param_lda.currentItem().text().split(' ')[
                                                  0]).delete()
        session.commit()
        update_list_param_lda()


def remove_all_param_geovel_lda():
    session.query(ParameterLDA).filter_by(analysis_id=get_LDA_id()).delete()
    session.commit()
    update_list_param_lda()


def update_list_param_lda():
    calc_f_test()
    # ui.listWidget_param_lda.clear()
    # for i in session.query(ParameterLDA).filter(ParameterLDA.analysis_id == get_LDA_id()).all():
    #     ui.listWidget_param_lda.addItem(i.parameter)


def draw_LDA():
    """ Построить диаграмму рассеяния для модели анализа LDA """
    data_train, list_param = build_table_train_lda()
    colors = {}
    for m in session.query(MarkerLDA).filter(MarkerLDA.analysis_id == get_LDA_id()).all():
        colors[m.title] = m.color
    training_sample = data_train[list_param].values.tolist()
    markup = sum(data_train[['mark']].values.tolist(), [])
    clf = LinearDiscriminantAnalysis()
    try:
        trans_coef = clf.fit(training_sample, markup).transform(training_sample)
    except ValueError:
        ui.label_info.setText(
            f'Ошибка в расчетах LDA! Возможно значения одного из параметров отсутствуют в интервале обучающей выборки.')
        ui.label_info.setStyleSheet('color: red')
        return
    data_trans_coef = pd.DataFrame(trans_coef)
    data_trans_coef['mark'] = data_train[['mark']]

    fig = plt.figure(figsize=(10, 10), dpi=80)
    ax = plt.subplot()
    if ui.listWidget_param_lda.count() < 3:
        sns.scatterplot(data=data_trans_coef, x=0, y=100, hue='mark', palette=colors)
    else:
        sns.scatterplot(data=data_trans_coef, x=0, y=1, hue='mark', palette=colors)
    ax.grid()
    ax.xaxis.grid(True, "minor", linewidth=.25)
    ax.yaxis.grid(True, "minor", linewidth=.25)
    # title_graph = f'Диаграмма рассеяния для канонических значений для обучающей выборки' \
    #               f'\n{ui.comboBox_class_lda.currentText().split(". ")[1]}' \
    #               f'\nПараметры: {" ".join(list_param)}' f'\nКоличество образцов: {str(len(data_trans_coef.index))}'
    # plt.title(title_graph, fontsize=16)
    fig.show()


def calc_verify_lda():
    data_train, list_param = build_table_train_lda()
    # colors = [m.color for m in session.query(MarkerLDA).filter(MarkerLDA.analysis_id == get_LDA_id()).all()]
    training_sample = data_train[list_param].values.tolist()
    markup = sum(data_train[['mark']].values.tolist(), [])
    clf = LinearDiscriminantAnalysis()
    try:
        clf.fit(training_sample, markup)
    except ValueError:
        ui.label_info.setText(
            f'Ошибка в расчетах LDA! Возможно значения одного из параметров отсутствуют в интервале обучающей выборки.')
        ui.label_info.setStyleSheet('color: red')
        return
    n, k = 0, 0
    ui.progressBar.setMaximum(len(data_train.index))
    for i in data_train.index:
        new_mark = clf.predict([data_train.loc[i].loc[list_param].tolist()])[0]
        if data_train['mark'][i] != new_mark:
            prof_id = data_train['prof_well_index'][i].split('_')[0]
            well_id = data_train['prof_well_index'][i].split('_')[1]
            ix = int(data_train['prof_well_index'][i].split('_')[2])
            old_list_fake = session.query(MarkupLDA.list_fake).filter(
                MarkupLDA.analysis_id == get_LDA_id(),
                MarkupLDA.profile_id == prof_id,
                MarkupLDA.well_id == well_id
            ).first()[0]
            if old_list_fake:
                new_list_fake = json.loads(old_list_fake)
                new_list_fake.append(ix)
            else:
                new_list_fake = [ix]
            session.query(MarkupLDA).filter(
                MarkupLDA.analysis_id == get_LDA_id(),
                MarkupLDA.profile_id == prof_id,
                MarkupLDA.well_id == well_id
            ).update({'list_fake': json.dumps(new_list_fake)}, synchronize_session='fetch')
            session.commit()
            n += 1
        k += 1
        ui.progressBar.setValue(k)
    session.commit()
    set_info(f'Из обучающей выборки удалено {n} измерений.', 'blue')
    update_list_well_markup_lda()
    update_list_param_lda()


def reset_verify_lda():
    session.query(MarkupLDA).filter(MarkupLDA.analysis_id == get_LDA_id()).update({'list_fake': None},
                                                                                  synchronize_session='fetch')
    session.commit()
    set_info(f'Выбросы для анализа "{ui.comboBox_lda_analysis.currentText()}" очищены.', 'green')
    update_list_well_markup_lda()
    update_list_param_lda()


def calc_LDA():
    colors = {}
    for m in session.query(MarkerLDA).filter(MarkerLDA.analysis_id == get_LDA_id()).all():
        colors[m.title] = m.color
    colors['test'] = '#999999'
    working_data, data_trans_coef, curr_form = get_working_data_lda()
    fig = plt.figure(figsize=(10, 10), dpi=80)
    ax = plt.subplot()
    if ui.listWidget_param_lda.count() < 3:
        sns.scatterplot(data=data_trans_coef, x=0, y=100, hue='mark', palette=colors)
    else:
        sns.scatterplot(data=data_trans_coef, x=0, y=1, hue='mark', palette=colors)
    ax.grid()
    ax.xaxis.grid(True, "minor", linewidth=.25)
    ax.yaxis.grid(True, "minor", linewidth=.25)
    # title_graph = f'Диаграмма рассеяния для канонических значений для обучающей выборки' \
    #               f'\n{ui.comboBox_class_lda.currentText().split(". ")[1]}' \
    #               f'\nПараметры: {" ".join(list_param)}' f'\nКоличество образцов: {str(len(data_trans_coef.index))}'
    # plt.title(title_graph, fontsize=16)
    fig.show()
    remove_poly_item()
    list_up = json.loads(curr_form.layer_up.layer_line)  # Получение списка с верхними границами формации
    list_down = json.loads(curr_form.layer_down.layer_line)  # Получение списка со снижными границами формации

    previous_element = None
    list_dupl = []
    for index, current_element in enumerate(working_data['mark']):
        if current_element == previous_element:
            list_dupl.append(index)
        else:
            if list_dupl:
                list_dupl.append(list_dupl[-1] + 1)
                y_up = [list_up[i] for i in list_dupl]
                y_down = [list_down[i] for i in list_dupl]
                draw_fill_result(list_dupl, y_up, y_down, colors[previous_element])
            list_dupl = [index]
        previous_element = current_element
    if len(list_dupl) > 0:
        y_up = [list_up[i] for i in list_dupl]
        y_down = [list_down[i] for i in list_dupl]
        draw_fill_result(list_dupl, y_up, y_down, colors[previous_element])
    if ui.checkBox_save_prof_lda.isChecked():
        try:
            file_name = f'{get_object_name()}_{get_research_name()}_{get_profile_name()}__модель_{get_lda_title()}.xlsx'
            fn = QFileDialog.getSaveFileName(caption="Сохранить выборку в таблицу", directory=file_name,
                                             filter="Excel Files (*.xlsx)")
            working_data.to_excel(fn[0])
            set_info(f'Таблица сохранена в файл: {fn[0]}', 'green')
        except ValueError:
            pass


def calc_f_test():
    data_train, list_param = build_table_train_lda()
    list_marker = get_list_marker()
    ui.listWidget_param_lda.clear()
    for param in list_param:
        groups = []
        for mark in list_marker:
            groups.append(data_train[data_train['mark'] == mark][param].values.tolist())
        F, p = f_oneway(*groups)
        ui.listWidget_param_lda.addItem(f'{param} \t\tF={round(F, 2)} p={round(p, 3)}')
        if F < 1 or p > 0.05:
            i_item = ui.listWidget_param_lda.findItems(f'{param} \t\tF={round(F, 2)} p={round(p, 3)}', Qt.MatchContains)[0]
            i_item.setBackground(QBrush(QColor('red')))


def calc_obj_lda():
    working_data_result = pd.DataFrame(columns=['prof_index', 'x_pulc', 'y_pulc', 'mark'] + get_list_param_lda() + get_list_marker())
    for n, prof in enumerate(session.query(Profile).filter(Profile.research_id == get_research_id()).all()):
        count_measure = len(json.loads(session.query(Profile.signal).filter(Profile.id == prof.id).first()[0]))
        ui.comboBox_profile.setCurrentText(f'{prof.title} ({count_measure} измерений) id{prof.id}')
        update_formation_combobox()
        if len(prof.formations) == 1:
            ui.comboBox_plast.setCurrentText(f'{prof.formations[0].title} id{prof.formations[0].id}')
        elif len(prof.formations) > 1:
            Choose_Formation = QtWidgets.QDialog()
            ui_cf = Ui_FormationLDA()
            ui_cf.setupUi(Choose_Formation)
            Choose_Formation.show()
            Choose_Formation.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия
            for f in prof.formations:
                ui_cf.listWidget_form_lda.addItem(f'{f.title} id{f.id}')
            ui_cf.listWidget_form_lda.setCurrentRow(0)

            def form_lda_ok():
                ui.comboBox_plast.setCurrentText(ui_cf.listWidget_form_lda.currentItem().text())
                Choose_Formation.close()
            ui_cf.pushButton_ok_form_lda.clicked.connect(form_lda_ok)
            Choose_Formation.exec_()
        working_data, data_trans_coef, curr_form = get_working_data_lda()
        working_data['x_pulc'] = json.loads(prof.x_pulc)
        working_data['y_pulc'] = json.loads(prof.y_pulc)
        working_data_result = pd.concat([working_data_result, working_data], axis=0, ignore_index=True)
    try:
        file_name = f'{get_object_name()}_{get_research_name()}__модель_{get_lda_title()}.xlsx'
        fn = QFileDialog.getSaveFileName(caption=f'Сохранить результат LDA "{get_object_name()}_{get_research_name()}" в таблицу', directory=file_name,
                                         filter="Excel Files (*.xlsx)")
        working_data_result.to_excel(fn[0])
        set_info(f'Таблица сохранена в файл: {fn[0]}', 'green')
    except ValueError:
        pass

