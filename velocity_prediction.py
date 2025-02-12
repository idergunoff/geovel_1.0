import json
import random

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

        list_layer_line.append(json.loads(i.layer.layer_line))

        list_predict.append(savgol_line(json.loads(i.prediction.prediction), 175))

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

        for nm, measure in enumerate(signal):
            deep_signal[nm] = deep_signal[nm] + interpolate_list(
                measure[list_layer_top[nm]:list_layer_bottom[nm]],
                int(list_deep_line[nm]))
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

    pred = session.query(ProfileModelPrediction).filter_by(id=ui.listWidget_model_pred.currentItem().text().split(' id')[-1]).first()

    list_pred = json.loads(pred.prediction)

    if pred.corrected:
        if ui.checkBox_corr_pred.isChecked():
            list_pred = json.loads(pred.corrected[0].correct)
        else:
            session.query(PredictionCorrect).filter_by(prediction_id=pred.id).update({'correct': pred.prediction})
    else:
        session.add(PredictionCorrect(prediction_id=pred.id, correct=pred.prediction))
    session.commit()

    if pred.type_model == 'reg':
        ui_cmp.doubleSpinBox_pred_min.setMaximum(max(list_pred) * 3)
        ui_cmp.doubleSpinBox_pred_max.setMaximum(max(list_pred) * 3)

    ui_cmp.spinBox_int_max.setMaximum(len(list_pred) - 1)
    ui_cmp.spinBox_int_min.setMaximum(len(list_pred) - 1)
    ui_cmp.spinBox_int_min.setValue(0)
    ui_cmp.spinBox_int_max.setValue(40)

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
        for i in range(ui_cmp.spinBox_int_min.value(), ui_cmp.spinBox_int_max.value() + 1):
            list_pred[i] = round(random.uniform(ui_cmp.doubleSpinBox_pred_min.value(), ui_cmp.doubleSpinBox_pred_max.value()), 5)

        session.query(PredictionCorrect).filter_by(prediction_id=pred.id).update({'correct': json.dumps(list_pred)})
        session.commit()

        CorrModelPred.close()

    ui_cmp.buttonBox.accepted.connect(correct_predict)
    ui_cmp.buttonBox.rejected.connect(CorrModelPred.close)
    ui_cmp.spinBox_int_max.valueChanged.connect(draw_int_line)
    ui_cmp.spinBox_int_min.valueChanged.connect(draw_int_line)

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