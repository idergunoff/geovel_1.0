from func import *
from qt.formation_ai_form import *


def add_regression_model():
    """Добавить новый набор для обучения регрессионной модели"""
    if ui.lineEdit_string.text() == '':
        set_info('Введите название модели в поле ввода текста в верхней части главного окна', 'red')
        QMessageBox.critical(MainWindow, 'Ошибка', 'Введите название модели!')
        return
    new_reg_mod = AnalysisReg(title=ui.lineEdit_string.text())
    session.add(new_reg_mod)
    session.commit()
    update_list_reg()
    set_info(f'Добавлен новый анализ MLP - "{ui.lineEdit_string.text()}"', 'green')


def remove_reg():
    """Удалить анализ MLP"""
    mlp_title = get_regmod_title()
    result = QtWidgets.QMessageBox.question(ui.listWidget_well_mlp, 'Remove Regression Model',
                                            f'Вы уверены, что хотите удалить модель "{mlp_title}"?',
                                            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
    if result == QtWidgets.QMessageBox.Yes:
        session.query(ParameterReg).filter_by(analysis_id=get_regmod_id()).delete()
        session.query(MarkupReg).filter_by(analysis_id=get_regmod_id()).delete()
        session.query(AnalysisReg).filter_by(id=get_regmod_id()).delete()
        session.commit()
        set_info(f'Удалена модель - "{mlp_title}"', 'green')
        update_list_reg()
    elif result == QtWidgets.QMessageBox.No:
        pass


def update_list_reg(db=False):
    """Обновить список наборов для обучения регрессионной модели"""
    ui.comboBox_regmod.clear()
    for i in session.query(AnalysisReg).order_by(AnalysisReg.title).all():
        ui.comboBox_regmod.addItem(f'{i.title} id{i.id}')
    update_list_well_markup_reg()
    # if db:
    #     update_list_marker_mlp_db()
    # else:
    #     update_list_marker_mlp()



def add_well_markup_reg():
    """Добавить новую обучающую скважину для MLP"""
    analysis_id = get_regmod_id()
    well_id = get_well_id()
    profile_id = get_profile_id()
    formation_id = get_formation_id()

    if analysis_id and well_id and profile_id and formation_id:
        # remove_all_param_geovel_mlp()

        if ui.checkBox_profile_intersec.isChecked():
            x_prof = json.loads(session.query(Profile.x_pulc).filter(Profile.id == profile_id).first()[0])
            inter = session.query(Intersection).filter(Intersection.id == well_id).first()
            well_dist = ui.spinBox_well_dist.value()
            start = inter.i_profile - well_dist if inter.i_profile - well_dist > 0 else 0
            stop = inter.i_profile + well_dist if inter.i_profile + well_dist < len(x_prof) else len(x_prof)
            list_measure = list(range(start, stop))
            new_markup_reg = MarkupReg(analysis_id=analysis_id, well_id=well_id, profile_id=profile_id,
                                       formation_id=formation_id, target_value=inter.temperature,
                                       list_measure=json.dumps(list_measure), type_markup='intersection')
        else:
            target_value = ui.doubleSpinBox_target_val.value()
            well = session.query(Well).filter(Well.id == well_id).first()
            x_prof = json.loads(session.query(Profile.x_pulc).filter(Profile.id == profile_id).first()[0])
            y_prof = json.loads(session.query(Profile.y_pulc).filter(Profile.id == profile_id).first()[0])
            index, _ = closest_point(well.x_coord, well.y_coord, x_prof, y_prof)
            well_dist = ui.spinBox_well_dist.value()
            start = index - well_dist if index - well_dist > 0 else 0
            stop = index + well_dist if index + well_dist < len(x_prof) else len(x_prof)
            list_measure = list(range(start, stop))
            new_markup_reg = MarkupReg(analysis_id=analysis_id, well_id=well_id, profile_id=profile_id,
                                       formation_id=formation_id, target_value=target_value,
                                       list_measure=json.dumps(list_measure))
        session.add(new_markup_reg)
        session.commit()
        set_info(f'Добавлена новая обучающая скважина для регрессионной модели - "{get_well_name()} со значенем '
                 f'{new_markup_reg.target_value}"', 'green')
        update_list_well_markup_reg()
    else:
        set_info('выбраны не все параметры - скважина/пересечение и пласт', 'red')
        QMessageBox.critical(MainWindow, 'Ошибка', 'выберите все параметры для добавления обучающей скважины!')


def update_list_well_markup_reg():
    """Обновить список обучающих скважин"""
    ui.listWidget_well_regmod.clear()
    count_markup, count_measure = 0, 0
    for i in session.query(MarkupReg).filter(MarkupReg.analysis_id == get_regmod_id()).all():
        try:
            fake = len(json.loads(i.list_fake)) if i.list_fake else 0
            measure = len(json.loads(i.list_measure))
            if i.type_markup == 'intersection':
                inter_name = session.query(Intersection.name).filter(Intersection.id == i.well_id).first()[0]
                item = (f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | {inter_name.split("_")[0]} | '
                        f'{measure - fake} | {i.target_value} |id{i.id}')
            else:
                item = (f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | {i.well.name} | '
                        f'{measure - fake} | {i.target_value} |id{i.id}')
            ui.listWidget_well_regmod.addItem(item)
            count_markup += 1
            count_measure += measure - fake
        except AttributeError:
            set_info(f'Параметр для профиля {i.profile.title} удален из-за отсутствия одного из параметров', 'red')
            session.delete(i)
            session.commit()
    ui.label_count_markup_reg.setText(f'<i><u>{count_markup}</u></i> обучающих скважин; <i><u>{count_measure}</u></i> измерений')
