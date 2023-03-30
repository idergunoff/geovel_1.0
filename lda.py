import json

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor

from draw import draw_radarogram, draw_formation, draw_fill
from func import *


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
    # todo: подтверждение удаления, потом удаление всех составляющих анализа
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
        ui.comboBox_mark_lda.setItemData(ui.comboBox_mark_lda.findText(item), QBrush(QColor(i.color)), Qt.BackgroundRole)
    update_list_well_markup_lda()


def add_well_markup_lda():
    """Добавить новую обучающую скважину для LDA"""
    analysis_id = get_LDA_id()
    well_id = get_well_id()
    profile_id = get_profile_id()
    formation_id = get_formation_id()
    marker_id = get_marker_id()

    if analysis_id and well_id and profile_id and marker_id and formation_id:

        well = session.query(Well).filter(Well.id == well_id).first()
        x_prof = json.loads(session.query(Profile.x_pulc).filter(Profile.id == profile_id).first()[0])
        y_prof = json.loads(session.query(Profile.y_pulc).filter(Profile.id == profile_id).first()[0])
        index, _ = closest_point(well.x_coord, well.y_coord, x_prof, y_prof)
        well_dist = ui.spinBox_well_dist.value()
        start = index - well_dist if index - well_dist > 0 else 0
        stop = index + well_dist if index + well_dist < len(x_prof) else len(x_prof) + 1

        list_measure = list(range(start, stop))
        new_markup_lda = MarkupLDA(analysis_id=analysis_id, well_id=well_id, profile_id=profile_id,
                                            marker_id=marker_id, formation_id=formation_id, list_measure=json.dumps(list_measure))
        session.add(new_markup_lda)
        session.commit()
        set_info(f'Добавлена новая обучающая скважина для LDA - "{ui.lineEdit_string.text()}"', 'green')
        update_list_well_markup_lda()
    else:
        set_info('выбраны не все параметры', 'red')


def update_list_well_markup_lda():
    """Обновление списка обучающих скважин LDA"""
    ui.listWidget_well_lda.clear()
    for i in session.query(MarkupLDA).filter(MarkupLDA.analysis_id == get_LDA_id()).all():
        item = f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | {i.well.name} | id{i.id}'
        ui.listWidget_well_lda.addItem(item)
        i_item = ui.listWidget_well_lda.findItems(item, Qt.MatchContains)[0]
        i_item.setBackground(QBrush(QColor(i.marker.color)))
        # ui.listWidget_well_lda.setItemData(ui.listWidget_well_lda.findText(item), QBrush(QColor(i.marker.color)), Qt.BackgroundRole)


def choose_marker_lda():
    """Выбор маркера LDA"""
    markup = session.query(MarkupLDA).filter(MarkupLDA.id == get_markup_id()).first()
    if not markup:
        return
    ui.comboBox_object.setCurrentText(f'{markup.profile.research.object.title} id{markup.profile.research.object_id}')
    update_research_combobox()
    ui.comboBox_research.setCurrentText(f'{markup.profile.research.date_research.strftime("%m.%Y")} id{markup.profile.research_id}')
    update_profile_combobox()
    count_measure = len(json.loads(session.query(Profile.signal).filter(Profile.id == markup.profile_id).first()[0]))
    ui.comboBox_profile.setCurrentText(f'{markup.profile.title} ({count_measure} измерений) id{markup.profile_id}')
    draw_radarogram()
    ui.comboBox_plast.setCurrentText(f'{markup.formation.title} id{markup.formation_id}')
    draw_formation()
    draw_well(markup.well_id)
    list_measure = json.loads(markup.list_measure)
    list_up = json.loads(markup.formation.layer_up.layer_line)
    list_down = json.loads(markup.formation.layer_down.layer_line)
    y_up = [list_up[i] for i in list_measure]
    y_down = [list_down[i] for i in list_measure]
    draw_fill(list_measure, y_up, y_down, markup.marker.color)