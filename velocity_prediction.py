import json

from func import *
from velocity_model import check_intersection


def update_list_bind_vel_prediction():
    bindings = session.query(BindingLayerPrediction).join(Layers).filter(Layers.profile_id == get_profile_id()).all()
    ui.listWidget_bind_vel.clear()
    for i in bindings:
        if i.prediction.type_model == 'cls':
            model = session.query(TrainedModelClass).filter_by(id=i.prediction.model_id).first()
        else:
            model = session.query(TrainedModelReg).filter_by(id=i.prediction.model_id).first()
        item = f'{i.layer.layer_title}_{i.prediction.type_model}-{model.title}_id{i.id}'
        ui.listWidget_bind_vel.addItem(item)


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
        list_predict.append(savgol_filter(json.loads(i.prediction.prediction), 175, 3))

    if not check_intersection(list_layer_line):
        set_info('Слои не должны пересекаться', 'red')
        return

    if ui.checkBox_minmax.isChecked():
        curr_prof = session.query(CurrentProfileMinMax).first()
    else:
        curr_prof = session.query(CurrentProfile).first()

    signal = json.loads(curr_prof.signal)
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
            p = savgol_filter(json.loads(bindings[b].prediction.prediction), 175, 3)
            list_vel.append([(p[i] * 100) / (l[i] * 8) for i in range(len(l))])
        else:
            l = [a - b for a, b in zip(json.loads(bindings[b].layer.layer_line), json.loads(bindings[b - 1].layer.layer_line))]
            p = [a - b for a, b in zip(savgol_filter(json.loads(bindings[b].prediction.prediction), 175, 3),
                 savgol_filter(json.loads(bindings[b - 1].prediction.prediction), 175, 3))]
            list_vel.append([(p[i] * 100) / (l[i] * 8) for i in range(len(l))])

    return list_vel



