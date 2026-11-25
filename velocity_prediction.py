import json
import random
import statistics
from html import escape

from sqlalchemy.exc import SQLAlchemyError


from func import *
from velocity_depth_signal import build_velocity_depth_signal
from velocity_prediction_binding import bind_object_predictions


def update_list_bind_vel_prediction():
    bindings = session.query(BindingLayerPrediction).join(Layers).filter(Layers.profile_id == get_profile_id()).all()
    ui.listWidget_bind_vel.clear()
    for i in bindings:
        try:
            if i.prediction.type_model == 'cls':
                model = session.query(TrainedModelClass).filter_by(id=i.prediction.model_id).first()
            else:
                model = session.query(TrainedModelReg).filter_by(id=i.prediction.model_id).first()
            item = f'{i.layer.layer_title}_{i.prediction.type_model}-{model.title}_id{i.id}'
            ui.listWidget_bind_vel.addItem(item)
        except AttributeError:
            pass


def add_bind_vel_prediction():
    try:
        pred_id = ui.listWidget_model_pred.currentItem().text().split(' id')[-1]
    except AttributeError:
        set_info('Выберите модель', 'red')
        return
    layer_id = get_layer_id()
    if not layer_id:
        set_info('Выберите слой', 'red')
        return
    session.add(BindingLayerPrediction(prediction_id=pred_id, layer_id=layer_id))
    session.commit()
    update_list_bind_vel_prediction()


def rmv_bind_vel_prediction():
    try:
        session.query(BindingLayerPrediction).filter_by(id=ui.listWidget_bind_vel.currentItem().text().split('_id')[-1]).delete()
    except AttributeError:
        set_info('Выберите привязку которую хотите удалить', 'red')
    session.commit()
    update_list_bind_vel_prediction()


def add_bind_vel_prediction_all_profiles():
    object_id, profile_id = get_object_id(), get_profile_id()
    if not object_id or not profile_id:
        set_info('Выберите объект и текущий профиль с настроенными привязками.', 'red')
        return
    try:
        count = bind_object_predictions(session, object_id, profile_id)
        session.commit()
    except ValueError as error:
        set_info(escape(str(error)).replace('\n', '<br>'), 'red')
        return
    except SQLAlchemyError:
        session.rollback()
        set_info('Не удалось сохранить привязки. Изменения отменены.', 'red')
        return
    update_list_bind_vel_prediction()
    set_info(f'Привязки текущего профиля применены к остальным профилям объекта: {count}.', 'green')


def calc_deep_predict_current_profile():
    profile_id = get_profile_id()
    if ui.checkBox_minmax.isChecked():
        curr_prof = session.query(CurrentProfileMinMax).first()
    else:
        curr_prof = session.query(CurrentProfile).first()
    if curr_prof is None or curr_prof.profile_id != profile_id:
        set_info('Не выбран профиль для пересчёта глубины.', 'red')
        return
    try:
        signal = json.loads(curr_prof.signal)
        bindings = session.query(BindingLayerPrediction).join(Layers).filter(
            Layers.profile_id == profile_id,
        ).order_by(BindingLayerPrediction.id).all()
        boundaries = []
        for binding in bindings:
            if binding.prediction is None or binding.prediction.profile_id != profile_id:
                raise ValueError(f'Слой «{binding.layer.layer_title}»: нет prediction этого профиля; проверьте привязку.')
            prediction = binding.prediction
            raw_depths = (prediction.corrected[0].correct
                          if ui.checkBox_corr_pred.isChecked() and prediction.corrected else prediction.prediction)
            try:
                boundaries.append((binding.layer.layer_title, json.loads(binding.layer.layer_line), json.loads(raw_depths)))
            except (TypeError, ValueError) as error:
                raise ValueError(f'Слой «{binding.layer.layer_title}»: не сохранены корректные границы или prediction.') from error
        return build_velocity_depth_signal(signal, boundaries, interpolate_list, savgol_line)
    except (TypeError, ValueError) as error:
        set_info(f'Профиль id{profile_id}: {escape(str(error))}', 'red')
        return



def draw_radar_vel_pred():

    deep_signal = calc_deep_predict_current_profile()
    if deep_signal is None:
        return
    l_max = 0
    for i in deep_signal:
        l_max = len(i) if len(i) > l_max else l_max
    deep_signal = [i + [0 for _ in range(l_max - len(i))] for i in deep_signal]
    deep_signal = [interpolate_list(i, 512) for i in deep_signal]

    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(deep_signal))
    session.add(new_current)
    session.commit()
    # clear_current_velocity_model()
    # new_curr_vel_mod = CurrentVelocityModel(
    #     vel_model_id=ui.listWidget_vel_model.currentItem().text().split(' id')[-1],
    #     active=True,
    #     scale=l_max/10/512
    # )
    # session.add(new_curr_vel_mod)
    # session.commit()
    # save_max_min(deep_signal)
    # if ui.checkBox_minmax.isChecked():
    #     deep_signal = json.loads(session.query(CurrentProfileMinMax).filter_by(id=1).first().signal)
    draw_image_deep_prof(deep_signal, l_max/512)


def calc_list_velocity():
    list_vel = []
    bindings = session.query(BindingLayerPrediction).join(Layers).filter(Layers.profile_id == get_profile_id()).all()
    for b in range(len(bindings)):
        if len(list_vel) == 0:
            l = json.loads(bindings[b].layer.layer_line)
            p = savgol_line(json.loads(bindings[b].prediction.prediction), 175)
            list_vel.append([(p[i] * 100) / (l[i] * 8) for i in range(len(l))])
        else:
            l = [a - b for a, b in zip(json.loads(bindings[b].layer.layer_line), json.loads(bindings[b - 1].layer.layer_line))]
            p = [a - b for a, b in zip(savgol_line(json.loads(bindings[b].prediction.prediction), 175),
                 savgol_line(json.loads(bindings[b - 1].prediction.prediction), 175))]
            list_vel.append([(p[i] * 100) / (l[i] * 8) for i in range(len(l))])

    return list_vel


def update_list_model_nn():
    ui.listWidget_model_nn.clear()
    if not ui.checkBox_model_nn.isChecked():
        return
    for p in session.query(ProfileModelPrediction).filter_by(profile_id=get_profile_id(), type_model='reg').all():

        model = session.query(TrainedModelReg).filter_by(id=p.model_id).first()
        item = QtWidgets.QListWidgetItem(f'{p.type_model} {model.title} id{p.id}')
        item.setToolTip(f'{round(os.path.getsize(model.path_model) / 1048576, 4)} МБ\n'
                        f'{model.comment}\n'
                        f'Количество параметров: '
                        f'{len(get_list_param_numerical(json.loads(model.list_params), model))}\n')
        ui.listWidget_model_nn.addItem(item)
    ui.listWidget_model_nn.setCurrentRow(0)


def correct_profile_model_predict():
    profile_id, research_id = get_profile_id(), get_research_id()
    prediction_item = ui.listWidget_model_pred.currentItem()
    prediction_id = prediction_item.text().split(' id')[-1] if prediction_item else None
    use_corrected = ui.checkBox_corr_pred.isChecked()
    CorrModelPred = QtWidgets.QDialog()
    ui_cmp = Ui_FormCorrectedModel()
    ui_cmp.setupUi(CorrModelPred)

    def update_list_model_prediction_cmp():
        ui_cmp.listWidget_model_pred.clear()
        for p in session.query(ProfileModelPrediction).filter_by(profile_id=profile_id).all():
            if p.type_model == 'cls':
                model = session.query(TrainedModelClass).filter_by(id=p.model_id).first()
            else:
                model = session.query(TrainedModelReg).filter_by(id=p.model_id).first()
            try:
                item = QtWidgets.QListWidgetItem(f'{p.type_model} {model.title} id{p.id}')
                item.setToolTip(f'{round(os.path.getsize(model.path_model) / 1048576, 4)} МБ\n'
                                f'{model.comment}\n'
                                f'Количество параметров: '
                                f'{len(get_list_param_numerical(json.loads(model.list_params), model))}\n')
                ui_cmp.listWidget_model_pred.addItem(item)
            except (AttributeError, FileNotFoundError):
                session.delete(p)
                session.commit()
                set_info('Модель удалена', 'red')

    update_list_model_prediction_cmp()

    def get_current_prediction():
        return session.get(ProfileModelPrediction, prediction_id) if prediction_id is not None else None

    pred = get_current_prediction()
    if not pred:
        set_info('Не выбран прогноз для коррекции', 'red')
        CorrModelPred.close()
        return

    def get_pred_with_correct(prediction_obj):
        result_pred = json.loads(prediction_obj.prediction)

        if prediction_obj.corrected:
            if use_corrected:
                result_pred = json.loads(prediction_obj.corrected[0].correct)
            # else:
            #     session.query(PredictionCorrect).filter_by(
            #         prediction_id=prediction_obj.id
            #     ).update({'correct': prediction_obj.prediction})
        else:
            session.add(PredictionCorrect(prediction_id=prediction_obj.id, correct=prediction_obj.prediction))
            session.commit()
        return result_pred

    list_pred = get_pred_with_correct(pred)
    # session.commit()

    if pred.type_model == 'reg':
        ui_cmp.doubleSpinBox_pred_min.setMaximum(max(list_pred) * 3)
        ui_cmp.doubleSpinBox_pred_max.setMaximum(max(list_pred) * 3)

    ui_cmp.spinBox_int_max.setMaximum(len(list_pred) - 1)
    ui_cmp.spinBox_int_min.setMaximum(len(list_pred) - 1)
    ui_cmp.spinBox_int_min.setValue(0)
    # ui_cmp.doubleSpinBox_value.setValue(0.5)

    def update_spinBox_value():
        if ui_cmp.checkBox_compare.isChecked():
            ui_cmp.spinBox_int_max.setValue(len(list_pred) - 1)
        else:
            ui_cmp.spinBox_int_max.setValue(40)

    update_spinBox_value()

    def update_median_value():
        try:
            compare_pred = json.loads(session.query(ProfileModelPrediction).filter_by(
                id=ui_cmp.listWidget_model_pred.currentItem().text().split(' id')[-1]).first().prediction)
        except AttributeError:
            return
        data_slice = compare_pred[ui_cmp.spinBox_int_min.value():ui_cmp.spinBox_int_max.value()]
        median_value = (max(data_slice) + min(data_slice)) / 2
        ui_cmp.doubleSpinBox_value.setValue(median_value)

    update_median_value()

    line_up = ui_cmp.spinBox_int_min.value()
    line_down = ui_cmp.spinBox_int_max.value()
    l_int_up = pg.InfiniteLine(pos=line_up, angle=90, pen=pg.mkPen(color='darkred', width=4, dash=[8, 2]))
    l_int_down = pg.InfiniteLine(pos=line_down, angle=90, pen=pg.mkPen(color='darkgreen', width=4, dash=[8, 2]))
    ui.graph.addItem(l_int_up)
    ui.graph.addItem(l_int_down)

    l_int_up_r = pg.InfiniteLine(pos=line_up, angle=90, pen=pg.mkPen(color='darkred', width=4, dash=[8, 2]))
    l_int_down_r = pg.InfiniteLine(pos=line_down, angle=90, pen=pg.mkPen(color='darkgreen', width=4, dash=[8, 2]))
    radarogramma.addItem(l_int_up_r)
    radarogramma.addItem(l_int_down_r)

    def draw_int_line():
        nonlocal l_int_up, l_int_down, l_int_up_r, l_int_down_r
        ui.graph.removeItem(l_int_up)
        ui.graph.removeItem(l_int_down)
        radarogramma.removeItem(l_int_up_r)
        radarogramma.removeItem(l_int_down_r)
        if get_profile_id() != profile_id:
            return
        line_up = ui_cmp.spinBox_int_min.value()
        line_down = ui_cmp.spinBox_int_max.value()
        l_int_up = pg.InfiniteLine(pos=line_up, angle=90, pen=pg.mkPen(color='darkred', width=4, dash=[8, 2]))
        l_int_down = pg.InfiniteLine(pos=line_down, angle=90, pen=pg.mkPen(color='darkgreen', width=4, dash=[8, 2]))
        ui.graph.addItem(l_int_up)
        ui.graph.addItem(l_int_down)

        l_int_up_r = pg.InfiniteLine(pos=line_up, angle=90, pen=pg.mkPen(color='darkred', width=4, dash=[8, 2]))
        l_int_down_r = pg.InfiniteLine(pos=line_down, angle=90, pen=pg.mkPen(color='darkgreen', width=4, dash=[8, 2]))
        radarogramma.addItem(l_int_up_r)
        radarogramma.addItem(l_int_down_r)

        ui_cmp.spinBox_int_min.setMaximum(ui_cmp.spinBox_int_max.value() - 1)
        ui_cmp.spinBox_int_max.setMinimum(ui_cmp.spinBox_int_min.value() + 1)

        ui_cmp.doubleSpinBox_pred_min.setValue(min(list_pred[ui_cmp.spinBox_int_min.value():ui_cmp.spinBox_int_max.value()]))
        ui_cmp.doubleSpinBox_pred_max.setValue(max(list_pred[ui_cmp.spinBox_int_min.value():ui_cmp.spinBox_int_max.value()]))

    draw_int_line()

    def correct_predict():

        current_pred = get_current_prediction()
        if not current_pred:
           set_info('Не выбран прогноз для коррекции', 'red')
           return
        working_prediction = get_pred_with_correct(current_pred)

        if ui_cmp.checkBox_compare.isChecked():

            if not ui_cmp.listWidget_model_pred.currentItem():
                set_info('Не выбран прогноз для сравнения', 'red')
                return

            pred_cmp = session.query(ProfileModelPrediction).filter_by(
                    id=ui_cmp.listWidget_model_pred.currentItem().text().split(' id')[-1]).first()

            if not pred_cmp:
                set_info('Прогноз для сравнения не найден', 'red')
                return

            value_cmp = ui_cmp.doubleSpinBox_value.value()

            def apply_correction(list_pred_target, list_pred_cmp):
                for i in range(len(list_pred_target)):
                    if ui_cmp.radioButton_less.isChecked() and list_pred_cmp[i] < value_cmp:
                        list_pred_target[i] = round(
                            random.uniform(ui_cmp.doubleSpinBox_pred_min.value(), ui_cmp.doubleSpinBox_pred_max.value()), 5)
                    if ui_cmp.radioButton_more.isChecked() and list_pred_cmp[i] > value_cmp:
                        list_pred_target[i] = round(
                            random.uniform(ui_cmp.doubleSpinBox_pred_min.value(), ui_cmp.doubleSpinBox_pred_max.value()), 5)

            if ui_cmp.checkBox_for_obj.isChecked():
                profiles = session.query(Profile).filter(Profile.research_id == research_id).all()

                for prf in profiles:
                    obj_pred = session.query(ProfileModelPrediction).filter_by(
                        profile_id=prf.id,
                        model_id=current_pred.model_id,
                        type_model=current_pred.type_model
                    ).first()
                    obj_pred_cmp = session.query(ProfileModelPrediction).filter_by(
                        profile_id=prf.id,
                        model_id=pred_cmp.model_id,
                        type_model=pred_cmp.type_model
                    ).first()

                    if not obj_pred or not obj_pred_cmp:
                        continue

                    list_pred_obj = get_pred_with_correct(obj_pred)
                    list_pred_cmp_obj = get_pred_with_correct(obj_pred_cmp)

                    apply_correction(list_pred_obj, list_pred_cmp_obj)

                    session.query(PredictionCorrect).filter_by(prediction_id=obj_pred.id).update({
                        'correct': json.dumps(list_pred_obj)
                    })

                session.commit()
                CorrModelPred.close()
                return

            list_pred_cmp = get_pred_with_correct(pred_cmp)
            apply_correction(working_prediction, list_pred_cmp)

        else:
            for i in range(ui_cmp.spinBox_int_min.value(), ui_cmp.spinBox_int_max.value() + 1):
                working_prediction[i] = round(random.uniform(ui_cmp.doubleSpinBox_pred_min.value(), ui_cmp.doubleSpinBox_pred_max.value()), 5)

        session.query(PredictionCorrect).filter_by(prediction_id=current_pred.id).update({'correct': json.dumps(working_prediction)})
        session.commit()
        CorrModelPred.close()

    def delete_pred():
        if pred.corrected:
            session.query(PredictionCorrect).filter_by(prediction_id=pred.id).update({'correct': pred.prediction})
        session.commit()
        ui.checkBox_corr_pred.setChecked(False)


    ui_cmp.buttonBox.accepted.connect(correct_predict)
    ui_cmp.buttonBox.rejected.connect(CorrModelPred.close)
    ui_cmp.spinBox_int_max.valueChanged.connect(draw_int_line)
    ui_cmp.spinBox_int_min.valueChanged.connect(draw_int_line)
    ui_cmp.checkBox_compare.stateChanged.connect(update_spinBox_value)
    ui_cmp.spinBox_int_max.valueChanged.connect(update_median_value)
    ui_cmp.spinBox_int_min.valueChanged.connect(update_median_value)
    ui_cmp.listWidget_model_pred.currentItemChanged.connect(update_median_value)
    ui_cmp.pushButton_delete_pred.clicked.connect(delete_pred)

    def cleanup_correction_lines():
        try:
            # Удаляем линии из основного графика
            if hasattr(l_int_up, 'scene') and l_int_up.scene():
                ui.graph.removeItem(l_int_up)
            if hasattr(l_int_down, 'scene') and l_int_down.scene():
                ui.graph.removeItem(l_int_down)

            # Удаляем линии из радарограммы
            if hasattr(l_int_up_r, 'scene') and l_int_up_r.scene():
                radarogramma.removeItem(l_int_up_r)
            if hasattr(l_int_down_r, 'scene') and l_int_down_r.scene():
                radarogramma.removeItem(l_int_down_r)
        except Exception as e:
            print(f"Ошибка при удалении линий: {str(e)}")

    # Подключаем очистку к сигналу закрытия диалога
    CorrModelPred.finished.connect(cleanup_correction_lines)

    show_work_window(CorrModelPred)


def check_object_prediction_wells():
    """Compare saved predictions of the selected model throughout the object."""
    from PyQt5 import QtCore
    from prediction_well_check import collect_prediction_wells
    from qt.prediction_well_check_dialog import PredictionWellCheckDialog

    item = ui.listWidget_model_pred.currentItem()
    if item is None:
        QMessageBox.information(MainWindow, 'Нет прогноза', 'Выберите сохранённый прогноз в Formation AI.')
        return
    prediction = session.query(ProfileModelPrediction).filter_by(
        id=item.text().split(' id')[-1]).first()
    object_id = get_object_id()
    if prediction is None or object_id is None:
        QMessageBox.information(MainWindow, 'Нет данных', 'Выберите объект и сохранённый прогноз.')
        return
    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
    try:
        report = collect_prediction_wells(
            session, object_id, prediction, get_list_nearest_well,
            use_corrected=ui.checkBox_corr_pred.isChecked())
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()
    description = (
        f'Выбран прогноз: {item.text()}. Модель {prediction.type_model} id{prediction.model_id}. '
        f'Расстояние до профиля ≤ {ui.spinBox_well_distance.value()} м. '
        'Значение берётся в ближайшей точке профиля. '
        'Отдельная строка для каждой пары скважина / сохранённый прогноз. '
        f'Профилей без прогноза этой модели: {report.profiles_without_prediction}. '
        f'Ошибок данных: {len(report.issues)}. '
        'Единицы прогноза и параметра скважины должны совпадать; преобразование не выполняется.')
    if not report.samples:
        message = QMessageBox(QMessageBox.Information, 'Нет данных для сравнения', description, parent=MainWindow)
        if report.issues:
            message.setDetailedText('\n'.join(report.issues))
        show_work_window(message)
        return

    def open_candidate_log(well_id, details):
        from well_log import show_well_log
        update_list_well(select_well=True, selected_well_id=well_id)
        selected = details.get('selected', {}).get('details', {}) if details else {}
        show_well_log(selected_curve_id=selected.get('well_log_id'),
                      selected_depth=selected.get('depth'),
                      selected_interval=selected.get('interval') or [None, None], parent=dialog)

    dialog = PredictionWellCheckDialog(session, report.samples, description, MainWindow, open_candidate_log,
                                       prediction_type=prediction.type_model)
    if report.issues:
        issues_button = QtWidgets.QPushButton('Показать ошибки данных')
        def show_issues():
            box = QMessageBox(QMessageBox.Information, 'Пропущенные данные',
                              f'Ошибок: {len(report.issues)}', parent=dialog)
            box.setDetailedText('\n'.join(report.issues))
            show_work_window(box)
        issues_button.clicked.connect(show_issues)
        dialog.layout().insertWidget(1, issues_button)
    dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    dialog.show()


def check_uf():
    predict = session.query(ProfileModelPrediction).filter_by(id=ui.listWidget_model_pred.currentItem().text().split(' id')[-1]).first()
    list_predict_uf = json.loads(predict.prediction)

    formation = session.query(Formation).filter_by(profile_id=get_profile_id()).first()
    list_land = ensure_formation_terrain(session, formation)
    list_top = json.loads(formation.top)

    graph = [list_predict_uf[i] - (list_land[i] - list_top[i]) for i in range(len(list_predict_uf))]

    number = list(range(1, len(graph) + 1))
    ui.graph.clear()
    cc = (120, 120, 120, 255)
    curve = pg.PlotCurveItem(x=number, y=graph, pen=cc)  # создаем объект класса PlotCurveIte
    curve_filter = pg.PlotCurveItem(x=number, y=savgol_filter(graph, 31, 3), pen=pg.mkPen(color='red', width=2.4))
    ui.graph.addItem(curve)  # добавляем график данных на график
    ui.graph.addItem(curve_filter)  # добавляем фильтрованный график данных на график
    fit_profile_graph_y(ui.graph)
    ui.graph.showGrid(x=True, y=True)  # отображаем сетку на графике
    ui.graph.getAxis('bottom').setScale(2.5)
    ui.graph.getAxis('bottom').setLabel('Профиль, м')
