import json
import pickle
from collections import Counter

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QListWidgetItem, QMessageBox

from build_table import _model_input_columns, _model_table_sources_key, build_table_test
from func import get_research_id, imputer, MainWindow, session, set_info, ui, update_list_model_prediction
from models_db.model import (CalcObject, Formation, GeoradarObject, Profile, ProfileModelPrediction, Research,
                             TrainedModelClass, TrainedModelReg)
from qt.calc_object import Ui_Dialog_calc_object
from qt.choose_formation_lda import Ui_FormationLDA
from qt.work_windows import ask_in_window, show_work_window


def calc_object():
    CalcObjectWindow = QtWidgets.QDialog()
    ui_co = Ui_Dialog_calc_object()
    ui_co.setupUi(CalcObjectWindow)

    model_classes = {'cls': TrainedModelClass, 'reg': TrainedModelReg}

    def show_progress(completed, total, interrupted=False):
        status = 'Расчёт прерван. ' if interrupted else ''
        ui_co.label_models_progress.setText(f'{status}Посчитано моделей: {completed} из {total}')
        ui_co.progressBar_models.setRange(0, max(1, total))
        ui_co.progressBar_models.setValue(completed)
        ui_co.progressBar_models.repaint()
        ui_co.label_models_progress.repaint()

    def update_co_models_list():
        ui_co.listWidget_objects.clear()
        for i in session.query(CalcObject).all():
            try:
                research = session.query(Research).filter_by(id=i.research_id).first()
                object = session.query(GeoradarObject).filter_by(id=research.object_id).first()
                formations = list(json.loads(i.list_formation))

                if i.type_ml == 'cls':
                    model = session.query(TrainedModelClass).filter_by(id=i.model_id).first()
                    item_text = f'{object.title} {research.date_research} {i.type_ml} {model.title} formations: {len(formations)}'
                    item = QListWidgetItem(item_text)
                    item.setData(Qt.UserRole, i.id)
                    ui_co.listWidget_objects.addItem(item)

                if i.type_ml == 'reg':
                    model = session.query(TrainedModelReg).filter_by(id=i.model_id).first()
                    item_text = f'{object.title} {research.date_research} {i.type_ml} {model.title} formations: {len(formations)}'
                    item = QListWidgetItem(item_text)
                    item.setData(Qt.UserRole, i.id)
                    ui_co.listWidget_objects.addItem(item)

                ui_co.listWidget_objects.setCurrentRow(0)
            except AttributeError:
                pass
        show_progress(0, ui_co.listWidget_objects.count())

    def remove_co_model():
        item = ui_co.listWidget_objects.currentItem()
        if item is None:
            return
        co = session.query(CalcObject).filter_by(id=item.data(Qt.UserRole)).first()
        if co is None:
            update_co_models_list()
            return
        session.delete(co)
        session.commit()
        update_co_models_list()
        set_info(f'Запись id{co.id} удалена из списка', 'green')

    def clear_co_list():
        for co in session.query(CalcObject).all():
            session.delete(co)
            session.commit()
        update_co_models_list()
        set_info('Список очищен', 'green')

    def add_cls_model():

        current_item = ui.listWidget_trained_model_class.currentItem()
        if not current_item:
            set_info('Модель не выбрана!', 'red')
            return

        model = session.query(TrainedModelClass).filter_by(
            id=current_item.data(Qt.UserRole)).first()

        research_id = get_research_id()
        profiles = session.query(Profile).filter(Profile.research_id == research_id).all()
        if len(profiles) == 0:
            set_info(f'У текущего исследования нет профилей', 'red')
        else:
            list_formation = []
            flag_break = []
            no_formation = 0
            for n, prof in enumerate(profiles):
                if flag_break:
                    if flag_break[0] == 'stop':
                        break
                    else:
                        set_info(f'Нет пласта с названием {flag_break[1]} для профиля {flag_break[0]}', 'red')
                        QMessageBox.critical(MainWindow, 'Ошибка', f'Нет пласта с названием {flag_break[1]} для профиля '
                                                                   f'{flag_break[0]}, выберите пласты для каждого профиля.')
                        return
                if len(prof.formations) == 0:
                    no_formation += 1
                if len(prof.formations) == 1:
                    list_formation.append(prof.formations[0].id)
                elif len(prof.formations) > 1:
                    Choose_Formation = QtWidgets.QDialog()
                    ui_cf = Ui_FormationLDA()
                    ui_cf.setupUi(Choose_Formation)
                    Choose_Formation.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия
                    for f in prof.formations:
                        ui_cf.listWidget_form_lda.addItem(f'{f.title} id{f.id}')
                    ui_cf.listWidget_form_lda.setCurrentRow(0)

                    def form_mlp_ok():
                        nonlocal flag_break
                        if ui_cf.checkBox_to_all.isChecked():
                            title_form = ui_cf.listWidget_form_lda.currentItem().text().split(' id')[0]
                            for prof in profiles:
                                prof_form = session.query(Formation).filter_by(
                                    profile_id=prof.id,
                                    title=title_form
                                ).first()
                                if prof_form:
                                    list_formation.append(prof_form.id)
                                else:
                                    flag_break = [prof.title, title_form]
                                    Choose_Formation.close()
                                    return
                            flag_break = ['stop', 'stop']
                            Choose_Formation.close()
                        else:
                            list_formation.append(int(ui_cf.listWidget_form_lda.currentItem().text().split('id')[-1]))
                            Choose_Formation.close()

                    ui_cf.pushButton_ok_form_lda.clicked.connect(form_mlp_ok)
                    ask_in_window(Choose_Formation, CalcObjectWindow)

            if no_formation > 0:
                if no_formation == 1:
                    prof = 'профиля'
                else:
                    prof = 'профилей'
                set_info(f'У {no_formation} {prof} текущего исследования отсутствуют пласты. '
                         f'Исследование не добавлено в список CalcObject', 'red')
            else:
                new_co_cls = CalcObject(
                    research_id=research_id,
                    type_ml='cls',
                    model_id=model.id,
                    list_formation=json.dumps(list_formation)
                )
                session.add(new_co_cls)
                session.commit()
                update_co_models_list()


    def add_reg_model():

        current_item = ui.listWidget_trained_model_reg.currentItem()
        if not current_item:
            set_info('Модель не выбрана!', 'red')
            return

        model = session.query(TrainedModelReg).filter_by(
            id=current_item.data(Qt.UserRole)).first()

        research_id = get_research_id()
        profiles = session.query(Profile).filter(Profile.research_id == research_id).all()
        if len(profiles) == 0:
            set_info(f'У текущего исследования нет профилей', 'red')
        else:
            list_formation = []
            flag_break = []
            no_formation = 0
            for n, prof in enumerate(profiles):
                if flag_break:
                    if flag_break[0] == 'stop':
                        break
                    else:
                        set_info(f'Нет пласта с названием {flag_break[1]} для профиля {flag_break[0]}', 'red')
                        QMessageBox.critical(MainWindow, 'Ошибка', f'Нет пласта с названием {flag_break[1]} для профиля '
                                                                   f'{flag_break[0]}, выберите пласты для каждого профиля.')
                        return
                if len(prof.formations) == 0:
                    no_formation += 1
                elif len(prof.formations) == 1:
                    list_formation.append(prof.formations[0].id)
                elif len(prof.formations) > 1:
                    Choose_Formation = QtWidgets.QDialog()
                    ui_cf = Ui_FormationLDA()
                    ui_cf.setupUi(Choose_Formation)
                    Choose_Formation.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия
                    for f in prof.formations:
                        ui_cf.listWidget_form_lda.addItem(f'{f.title} id{f.id}')
                    ui_cf.listWidget_form_lda.setCurrentRow(0)

                    def form_mlp_ok():
                        nonlocal flag_break
                        if ui_cf.checkBox_to_all.isChecked():
                            title_form = ui_cf.listWidget_form_lda.currentItem().text().split(' id')[0]
                            for prof in profiles:
                                prof_form = session.query(Formation).filter_by(
                                    profile_id=prof.id,
                                    title=title_form
                                ).first()
                                if prof_form:
                                    list_formation.append(prof_form.id)
                                else:
                                    flag_break = [prof.title, title_form]
                                    Choose_Formation.close()
                                    return
                            flag_break = ['stop', 'stop']
                            Choose_Formation.close()
                        else:
                            list_formation.append(int(ui_cf.listWidget_form_lda.currentItem().text().split('id')[-1]))
                            Choose_Formation.close()

                    ui_cf.pushButton_ok_form_lda.clicked.connect(form_mlp_ok)
                    ask_in_window(Choose_Formation, CalcObjectWindow)

            if no_formation > 0:
                if no_formation == 1:
                    prof = 'профиля'
                else:
                    prof = 'профилей'
                set_info(f'У {no_formation} {prof} текущего исследования отсутствуют пласты. '
                         f'Исследование не добавлено в список CalcObject', 'red')
            else:
                new_co_reg = CalcObject(
                    research_id=research_id,
                    type_ml = 'reg',
                    model_id = model.id,
                    list_formation = json.dumps(list_formation)
                )
                session.add(new_co_reg)
                session.commit()
                update_co_models_list()


    def calculate_model(co, model, model_table_cache):
        type_model = co.type_ml
        is_classification = type_model == 'cls'
        analysis = 'mlp' if is_classification else 'regmod'

        with open(model.path_model, 'rb') as f:
            estimator = pickle.load(f)

        predict = estimator.predict_proba if is_classification else estimator.predict
        input_columns = _model_input_columns(type_model, model)
        formation_ids = json.loads(co.list_formation)
        profiles = session.query(Profile).filter_by(research_id=co.research_id).all()

        for index, profile in enumerate(profiles):
            if session.query(ProfileModelPrediction).filter_by(
                    profile_id=profile.id, type_model=type_model, model_id=model.id).count():
                research = profile.research
                set_info(f'Прогноз для исследования от {research.date_research} объекта {research.object.title} '
                         'уже есть в БД', 'red')
                continue

            formation = session.get(Formation, formation_ids[index])
            if formation is None:
                raise ValueError(f'Не найден пласт для профиля {profile.title}')

            working_data, _ = build_table_test(
                analysis, model=model, curr_form=formation,
                model_table_cache=model_table_cache, use_model_mask=True,
            )
            sample = working_data[input_columns].values.tolist()

            try:
                prediction = predict(sample)
            except ValueError:
                sample = [[np.nan if np.isinf(value) else value for value in row] for row in sample]
                prediction = predict(imputer.fit_transform(sample))

            result = [round(row[0], 6) for row in prediction] if is_classification else prediction.tolist()
            session.add(ProfileModelPrediction(
                profile_id=profile.id, type_model=type_model, model_id=model.id, prediction=json.dumps(result),
            ))
            session.commit()
            set_info(f'Результат расчета модели "{model.title}" для профиля {profile.title} сохранен', 'green')

        update_list_model_prediction()

    def start_co():
        queue = ui_co.listWidget_objects
        total = queue.count()
        if total == 0:
            return

        completed = 0
        show_progress(completed, total)
        set_info('Начало расчета объектов по моделям', 'blue')

        try:
            jobs = []
            for index in range(total):
                co = session.get(CalcObject, queue.item(index).data(Qt.UserRole))
                if co is None:
                    jobs.append((None, None, None))
                    continue
                model = session.get(model_classes[co.type_ml], co.model_id)
                source = (co.research_id, _model_table_sources_key(model)) if model is not None else None
                jobs.append((co, model, source))

            remaining = Counter(source for _, _, source in jobs if source is not None)
            table_caches = {source: {} for source, count in remaining.items() if count > 1}

            for co, model, source in jobs:
                if co is None:
                    queue.takeItem(0)
                    continue
                if model is None:
                    raise ValueError(f'Модель для записи id{co.id} не найдена')

                calculate_model(co, model, table_caches.get(source))
                session.delete(co)
                session.commit()
                queue.takeItem(0)
                remaining[source] -= 1
                if remaining[source] == 0:
                    table_caches.pop(source, None)
                completed += 1
                show_progress(completed, total)
        except Exception as error:
            session.rollback()
            show_progress(completed, total, interrupted=True)
            set_info('Расчет объектов по моделям прерван', 'red')
            QMessageBox.critical(CalcObjectWindow, 'Ошибка расчёта', str(error))
            return

        set_info('Расчет объектов по моделям завершен', 'blue')

    update_co_models_list()
    ui_co.pushButton_start_co.clicked.connect(start_co)
    ui_co.pushButton_delete_co.clicked.connect(remove_co_model)
    ui_co.pushButton_clear_co.clicked.connect(clear_co_list)
    ui_co.pushButton_calc_cls.clicked.connect(add_cls_model)
    ui_co.pushButton_calc_reg.clicked.connect(add_reg_model)

    show_work_window(CalcObjectWindow)
