import json
import random
import statistics

from torch.cuda import graph

from func import *
from velocity_model import check_intersection


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


def calc_deep_predict_current_profile():
    list_layer_line, list_predict = [], []
    for i in session.query(BindingLayerPrediction).join(Layers).filter(Layers.profile_id == get_profile_id()).all():
        try:
            list_predict.append(savgol_line(json.loads(i.prediction.prediction), 175))
            list_layer_line.append(json.loads(i.layer.layer_line))
        except AttributeError:
            session.delete(i)
            session.commit()
            continue

    if not check_intersection(list_layer_line):
        set_info('Слои не должны пересекаться', 'red')
        return

    if ui.checkBox_minmax.isChecked():
        curr_prof = session.query(CurrentProfileMinMax).first()
    else:
        curr_prof = session.query(CurrentProfile).first()
    try:
        signal = json.loads(curr_prof.signal)
    except AttributeError:
        set_info('Не выбран профиль', 'red')
        return
    deep_signal = [[] for _ in range(len(signal))]
    
    for i in range(len(list_predict) + 1):
        try:
            if i == 0:
                list_layer_top = [0 for _ in range(len(signal))]
            else:
                list_layer_top = list_layer_line[i - 1]

            if i == len(list_predict):
                list_layer_bottom = [511 for _ in range(len(signal))]
            else:
                list_layer_bottom = list_layer_line[i]

            if i == 0:
                list_deep_line = list_predict[i]
            elif i == len(list_predict):
                list_deep_line = [(5 * 8 * (511 - list_layer_top[nv])) / 100  for nv in range(len(signal))]
            else:
                list_deep_line = [list_predict[i][j] - list_predict[i-1][j] for j in range(len(signal))]
        except IndexError:
            break

        for nm, measure in enumerate(signal):
            try:
                deep_signal[nm] = deep_signal[nm] + interpolate_list(
                    measure[list_layer_top[nm]:list_layer_bottom[nm]],
                    int(list_deep_line[nm]))
            except IndexError:
                break
    return deep_signal



def draw_radar_vel_pred():

    deep_signal = calc_deep_predict_current_profile()
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
    global l_int_up, l_int_down
    CorrModelPred = QtWidgets.QDialog()
    ui_cmp = Ui_FormCorrectedModel()
    ui_cmp.setupUi(CorrModelPred)
    CorrModelPred.show()
    CorrModelPred.setAttribute(Qt.WA_DeleteOnClose)

    def update_list_model_prediction_cmp():
        ui_cmp.listWidget_model_pred.clear()
        for p in session.query(ProfileModelPrediction).filter_by(profile_id=get_profile_id()).all():
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
        if not ui.listWidget_model_pred.currentItem():
            return None
        return session.query(ProfileModelPrediction).filter_by(id=ui.listWidget_model_pred.currentItem().text().split(' id')[-1]).first()

    pred = get_current_prediction()
    if not pred:
        set_info('Не выбран прогноз для коррекции', 'red')
        CorrModelPred.close()
        return

    def get_pred_with_correct(prediction_obj):
        result_pred = json.loads(prediction_obj.prediction)

        if prediction_obj.corrected:
            if ui.checkBox_corr_pred.isChecked():
                result_pred = json.loads(prediction_obj.corrected[0].correct)
            else:
                session.query(PredictionCorrect).filter_by(
                    prediction_id=prediction_obj.id
                ).update({'correct': prediction_obj.prediction})
        else:
            session.add(PredictionCorrect(prediction_id=prediction_obj.id, correct=prediction_obj.prediction))
            session.commit()
        return result_pred

    list_pred = get_pred_with_correct(pred)
    session.commit()

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

    def draw_int_line():
        global l_int_up, l_int_down
        ui.graph.removeItem(l_int_up)
        ui.graph.removeItem(l_int_down)
        line_up = ui_cmp.spinBox_int_min.value()
        line_down = ui_cmp.spinBox_int_max.value()
        l_int_up = pg.InfiniteLine(pos=line_up, angle=90, pen=pg.mkPen(color='darkred', width=4, dash=[8, 2]))
        l_int_down = pg.InfiniteLine(pos=line_down, angle=90, pen=pg.mkPen(color='darkgreen', width=4, dash=[8, 2]))
        ui.graph.addItem(l_int_up)
        ui.graph.addItem(l_int_down)

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
                profiles = session.query(Profile).filter(Profile.research_id == get_research_id()).all()

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
            apply_correction(list_pred, list_pred_cmp)

        else:
            for i in range(ui_cmp.spinBox_int_min.value(), ui_cmp.spinBox_int_max.value() + 1):
                list_pred[i] = round(random.uniform(ui_cmp.doubleSpinBox_pred_min.value(), ui_cmp.doubleSpinBox_pred_max.value()), 5)

        session.query(PredictionCorrect).filter_by(prediction_id=current_pred.id).update({'correct': json.dumps(list_pred)})
        session.commit()
        CorrModelPred.close()

    def delete_pred():
        if pred.corrected:
            session.query(PredictionCorrect).filter_by(prediction_id=pred.id).update({'correct': pred.prediction})
        session.commit()


    ui_cmp.buttonBox.accepted.connect(correct_predict)
    ui_cmp.buttonBox.rejected.connect(CorrModelPred.close)
    ui_cmp.spinBox_int_max.valueChanged.connect(draw_int_line)
    ui_cmp.spinBox_int_min.valueChanged.connect(draw_int_line)
    ui_cmp.checkBox_compare.stateChanged.connect(update_spinBox_value)
    ui_cmp.spinBox_int_max.valueChanged.connect(update_median_value)
    ui_cmp.spinBox_int_min.valueChanged.connect(update_median_value)
    ui_cmp.listWidget_model_pred.currentItemChanged.connect(update_median_value)
    ui_cmp.pushButton_delete_pred.clicked.connect(delete_pred)

    CorrModelPred.exec_()


def check_uf():
    predict = session.query(ProfileModelPrediction).filter_by(id=ui.listWidget_model_pred.currentItem().text().split(' id')[-1]).first()
    list_predict_uf = json.loads(predict.prediction)

    formation = session.query(Formation).filter_by(profile_id=get_profile_id()).first()
    list_land = json.loads(formation.land)
    list_top = json.loads(formation.top)

    graph = [list_predict_uf[i] - (list_land[i] - list_top[i]) for i in range(len(list_predict_uf))]

    number = list(range(1, len(graph) + 1))
    ui.graph.clear()
    cc = (120, 120, 120, 255)
    curve = pg.PlotCurveItem(x=number, y=graph, pen=cc)  # создаем объект класса PlotCurveIte
    curve_filter = pg.PlotCurveItem(x=number, y=savgol_filter(graph, 31, 3), pen=pg.mkPen(color='red', width=2.4))
    ui.graph.addItem(curve)  # добавляем график данных на график
    ui.graph.addItem(curve_filter)  # добавляем фильтрованный график данных на график
    ui.graph.showGrid(x=True, y=True)  # отображаем сетку на графике
    ui.graph.getAxis('bottom').setScale(2.5)
    ui.graph.getAxis('bottom').setLabel('Профиль, м')