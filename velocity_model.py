from draw import remove_fill_form, draw_fill_model
from func import *


def update_list_binding():
    ui.listWidget_bind.clear()
    for i in session.query(Binding).filter_by(profile_id=get_profile_id()).all():
        ui.listWidget_bind.addItem(f'{i.layer.layer_title} - {i.boundary.title} ({i.boundary.depth} м.) id{i.id}')


def add_binding():
    profile = session.query(Profile).filter_by(id=get_profile_id()).first()
    layer_id = 0
    for i in ui.widget_layer.findChildren(QtWidgets.QCheckBox):
        if i.isChecked():
            layer_id = int(i.text().split(' id')[-1])
            break
    if layer_id == 0:
        set_info('Выберите слой к которому нужно привязать границу', 'red')
        QMessageBox.critical(MainWindow, 'Ошибка', 'Выберите слой к которому нужно привязать границу')
        return
    if not get_boundary_id():
        set_info('Выберите границу', 'red')
        QMessageBox.critical(MainWindow, 'Ошибка', 'Выберите границу')
        return
    w = session.query(Well).filter_by(id=get_well_id()).first()

    index, dist = closest_point(w.x_coord, w.y_coord, json.loads(profile.x_pulc), json.loads(profile.y_pulc))

    new_bind = Binding(profile_id=profile.id, layer_id=layer_id, boundary_id=get_boundary_id(), index_measure=index)
    session.add(new_bind)
    session.commit()
    update_list_binding()


def remove_bind_point():
    for key, value in globals().items():
        if key == 'bind_scatter_id{}'.format(ui.listWidget_bind.currentItem().text().split(' id')[-1]):
            radarogramma.removeItem(globals()[key])

def draw_bind_point():
    for key, value in globals().items():
        if key.startswith('bind_'):
            radarogramma.removeItem(globals()[key])

    bind = session.query(Binding).filter_by(id=ui.listWidget_bind.currentItem().text().split(' id')[-1]).first()
    layer = session.query(Layers).filter_by(id=bind.layer_id).first()

    layer_line = json.loads(layer.layer_line)
    x_coord = bind.index_measure
    y_coord = layer_line[bind.index_measure]

    scatter = pg.ScatterPlotItem(x=[x_coord], y=[y_coord], pen=pg.mkPen('r', width=3), brush=pg.mkBrush('r'),
                                 symbol='o', size=15, hoverable=True)
    radarogramma.addItem(scatter)
    globals()[f'bind_scatter_id{bind.id}'] = scatter


def remove_binding():
    session.query(Binding).filter(Binding.id == int(ui.listWidget_bind.currentItem().text().split(' id')[-1])).delete()
    session.commit()
    update_list_binding()


def check_intersection(list_line):
    for line1 in range(len(list_line) - 1):
        for line2 in range(line1 + 1, len(list_line)):
            diff_line = [y - x for x, y in zip(list_line[line1], list_line[line2])]
            for i in diff_line:
                if i <= 0:
                    return False
    return True


def calc_velocity_formation(type_form, lt_id, lb_id):
    if type_form == 'top':
        # линии кровли и подошвы слоя
        list_line_bottom = json.loads(session.query(Layers).filter_by(id=lb_id).first().layer_line)
        list_line_top = [0] * len(list_line_bottom)
        # точки измерения скорости
        list_bind_bottom = session.query(Binding).filter_by(layer_id=lb_id).order_by(Binding.index_measure).all()
        list_i_bing = [b.index_measure for b in list_bind_bottom]
        # интерполяция скорости по слою
        list_ib_vel = []
        for n, ib in enumerate(list_i_bing):
            dist = (list_bind_bottom[n].boundary.depth) * 100
            t_ns = list_line_bottom[ib] * 8
            list_ib_vel.append(dist / t_ns)

    elif type_form == 'bottom':
        # линии кровли и подошвы слоя
        list_line_top = json.loads(session.query(Layers).filter_by(id=lt_id).first().layer_line)
        list_line_bottom = [511] * len(list_line_top)
        # точки измерения скорости
        list_bind_top = session.query(Binding).filter_by(layer_id=lt_id).order_by(Binding.index_measure).all()
        list_i_bing = [b.index_measure for b in list_bind_top]
        # интерполяция скорости по слою
        list_ib_vel = [ui.doubleSpinBox_vsr.value()] * len(list_bind_top)

    else:
        # линии кровли и подошвы слоя
        list_line_top = json.loads(session.query(Layers).filter_by(id=lt_id).first().layer_line)
        list_line_bottom = json.loads(session.query(Layers).filter_by(id=lb_id).first().layer_line)
        # точки измерения скорости
        list_bind_top = session.query(Binding).filter_by(layer_id=lt_id).order_by(Binding.index_measure).all()
        list_bind_bottom = session.query(Binding).filter_by(layer_id=lb_id).order_by(Binding.index_measure).all()
        list_ib_top = [b.index_measure for b in list_bind_top]
        list_ib_bottom = [b.index_measure for b in list_bind_bottom]
        # проверка наличия точек измерения скорости по кровле и подошве
        list_del = []
        for n, i in enumerate(list_ib_top):
            if i not in list_ib_bottom:
                list_del.append(list_bind_top[n])
        for i in list_del:
            list_bind_top.remove(i)
        list_del = []
        for n, i in enumerate(list_ib_bottom):
            if i not in list_ib_top:
                list_del.append(list_bind_bottom[n])
        for i in list_del:
            list_bind_bottom.remove(i)
        list_i_bing = [b.index_measure for b in list_bind_top]
        # интерполяция скорости по слоям
        list_ib_vel = []
        for n, ib in enumerate(list_i_bing):
            dist = (list_bind_bottom[n].boundary.depth - list_bind_top[n].boundary.depth) * 100
            t_ns = (list_line_bottom[ib] - list_line_top[ib]) * 8
            list_ib_vel.append(dist / t_ns)

    return list_line_top, list_line_bottom, list_i_bing, list_ib_vel



def interpolate_speed(L, index_list, speed_list):
    result = [0] * L

    # Проверка наличия точек измерения скорости
    if not index_list or not speed_list:
        return result

    # Сортировка индексов точек
    sorted_indices = sorted(index_list)

    # Первая точка - копирование скорости
    result[sorted_indices[0]] = speed_list[0]

    # Интерполяция между точками
    for i in range(1, len(sorted_indices)):
        start_index = sorted_indices[i - 1]
        end_index = sorted_indices[i]
        start_speed = speed_list[i - 1]
        end_speed = speed_list[i]

        # Интерполяция скорости между точками
        interpolation = np.linspace(start_speed, end_speed, end_index - start_index)

        # Заполнение результата
        result[start_index:end_index] = interpolation

    # Последняя точка - копирование скорости
    result[sorted_indices[-1]:] = [speed_list[-1]] * (L - sorted_indices[-1])
    result = [speed_list[0] if x == 0 else x for x in result ]

    return result




def calc_velocity_model():
    list_layer_id = []
    for i in ui.widget_layer.findChildren(QtWidgets.QCheckBox):
        if i.isChecked():
            list_layer_id.append(int(i.text().split(' id')[-1]))

    if not list_layer_id:
        set_info('Должен быть выбран хотя бы один слой', 'red')
        QMessageBox.critical(MainWindow, 'Ошибка', 'Должен быть выбран хотя бы один слой')
        return

    list_layer_line = []
    for i in list_layer_id:
        list_layer_line.append(json.loads(session.query(Layers).filter_by(id=i).first().layer_line))
    list_layer_line = sorted(list_layer_line, key=lambda x: x[0])
    if not check_intersection(list_layer_line):
        set_info('Слои не должны пересекаться', 'red')
        QMessageBox.critical(MainWindow, 'Ошибка', 'Слои не должны пересекаться')

    if ui.lineEdit_string.text() == '':
        set_info('Введите название скоростной модели', 'red')
        QMessageBox.critical(MainWindow, 'Ошибка', 'Введите название скоростной модели')
        return

    new_vel_model = VelocityModel(profile_id = get_profile_id(), title=ui.lineEdit_string.text())
    session.add(new_vel_model)
    session.commit()

    list_line_top, list_line_bottom, list_i_bing, list_ib_vel = calc_velocity_formation('top', False, list_layer_id[0])
    list_velocity = interpolate_speed(len(list_line_top), list_i_bing, list_ib_vel)
    index_form = 1

    new_vel_form = VelocityFormation(
        profile_id=get_profile_id(),
        vel_model_id=new_vel_model.id,
        layer_top=json.dumps(list_line_top),
        layer_bottom=json.dumps(list_line_bottom),
        color=get_rnd_color(),
        velocity=json.dumps(list_velocity),
        index=index_form
    )
    session.add(new_vel_form)
    session.commit()

    for n, l_id in enumerate(list_layer_id):
        if n == len(list_layer_id) - 1:
            type_form = 'bottom'
            list_line_top, list_line_bottom, list_i_bing, list_ib_vel = calc_velocity_formation(type_form, l_id, False)
        else:
            type_form = 'middle'
            list_line_top, list_line_bottom, list_i_bing, list_ib_vel = calc_velocity_formation(type_form, l_id, l_id + 1)

        list_velocity = interpolate_speed(len(list_line_top), list_i_bing, list_ib_vel)

        index_form += 1
        new_vel_form = VelocityFormation(
            profile_id=get_profile_id(),
            vel_model_id=new_vel_model.id,
            layer_top=json.dumps(list_line_top),
            layer_bottom=json.dumps(list_line_bottom),
            color=get_rnd_color(),
            velocity=json.dumps(list_velocity),
            index=index_form
        )
        session.add(new_vel_form)
        session.commit()

    update_list_velocity_model()

    calc_deep_layers(new_vel_model.id)


def calc_deep_layers(vel_model_id):
    """ Расчет и сохранение глубинного профиля по скоростной модели """
    new_deep_prof = DeepProfile(profile_id=get_profile_id(), vel_model_id=vel_model_id)
    session.add(new_deep_prof)
    session.commit()
    signal = json.loads(new_deep_prof.profile.signal)
    deep_signal = [[] for _ in range(len(signal))]
    index_deep_layer = 1
    for vm in session.query(VelocityFormation).filter_by(vel_model_id=vel_model_id).order_by(VelocityFormation.index).all():
        list_layer_top = json.loads(vm.layer_top)
        list_layer_bottom = json.loads(vm.layer_bottom)
        list_velocity = json.loads(vm.velocity)
        list_deep_line = [v * 8 * (list_layer_bottom[nv] - list_layer_top[nv]) for nv, v in enumerate(list_velocity)]
        new_deep_layer = DeepLayer(
            deep_profile_id=new_deep_prof.id,
            vel_form_id = vm.id,
            layer_line_thick=json.dumps(list_deep_line),
            index=index_deep_layer
        )
        session.add(new_deep_layer)
        session.commit()
        index_deep_layer += 1
        for nm, measure in enumerate(signal):
            deep_signal[nm] = deep_signal[nm] + interpolate_list(
                measure[list_layer_top[nm]:list_layer_bottom[nm]],
                int(list_deep_line[nm] / 10))

    new_deep_prof.signal = json.dumps(deep_signal)
    session.commit()


def calc_deep_layers_to_current_profile(vel_model_id):
    """ Расчет глубинного профиля по текущему профилю """
    curr_prof = session.query(CurrentProfile).filter_by(id=1).first()
    vel_mod = session.query(VelocityModel).filter_by(id=vel_model_id).first()
    if curr_prof.profile_id != vel_mod.profile_id:
        set_info('Текущий профиль и профиль скоростной модели не соответствуют', 'red')
        QMessageBox.critical(MainWindow, 'Ошибка', 'Текущий профиль и профиль скоростной модели не соответствуют')
        return
    signal = json.loads(curr_prof.signal)
    deep_signal = [[] for _ in range(len(signal))]
    for vf in session.query(VelocityFormation).filter_by(vel_model_id=vel_model_id).order_by(VelocityFormation.index).all():
        list_layer_top = json.loads(vf.layer_top)
        list_layer_bottom = json.loads(vf.layer_bottom)
        list_velocity = json.loads(vf.velocity)
        list_deep_line = [v * 8 * (list_layer_bottom[nv] - list_layer_top[nv]) for nv, v in enumerate(list_velocity)]

        for nm, measure in enumerate(signal):
            deep_signal[nm] = deep_signal[nm] + interpolate_list(
                measure[list_layer_top[nm]:list_layer_bottom[nm]],
                int(list_deep_line[nm] / 10))

    return deep_signal


def update_list_velocity_model():
    ui.listWidget_vel_model.clear()
    for i in session.query(VelocityModel).filter_by(profile_id=get_profile_id()):
        ui.listWidget_vel_model.addItem(f'{i.title} id{i.id}')


def draw_deep_profile():
    # deep_prof = session.query(DeepProfile).filter_by(
    #     vel_model_id=ui.listWidget_vel_model.currentItem().text().split(' id')[-1]).first()
    #
    deep_signal = calc_deep_layers_to_current_profile(ui.listWidget_vel_model.currentItem().text().split(' id')[-1])
    l_max = 0
    for i in deep_signal:
        l_max = len(i) if len(i) > l_max else l_max
    deep_signal = [i + [0 for _ in range(l_max - len(i))] for i in deep_signal]
    deep_signal = [interpolate_list(i, 512) for i in deep_signal]
    # draw_image_deep_prof(deep_signal)

    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(deep_signal))
    session.add(new_current)
    clear_current_velocity_model()
    new_curr_vel_mod = CurrentVelocityModel(
        vel_model_id=ui.listWidget_vel_model.currentItem().text().split(' id')[-1],
        active=True,
        scale=l_max/10/512
    )
    session.add(new_curr_vel_mod)
    session.commit()
    save_max_min(deep_signal)
    if ui.checkBox_minmax.isChecked():
        deep_signal = json.loads(session.query(CurrentProfileMinMax).filter_by(id=1).first().signal)
    draw_image_deep_prof(deep_signal, l_max/10/512)
#

def remove_velocity_model():
    if not ui.listWidget_vel_model.currentItem():
        set_info('Выберите скоростную модель для удаления', 'red')
        QMessageBox.critical(MainWindow, 'Ошибка', 'Выберите скоростную модель для удаления')
        return
    vel_model = session.query(VelocityModel).filter_by(id=ui.listWidget_vel_model.currentItem().text().split(' id')[-1]).first()
    title_model = vel_model.title
    for i in vel_model.velocity_formations:
        session.delete(i)
    deep_prof = session.query(DeepProfile).filter_by(vel_model_id=vel_model.id).first()
    for i in deep_prof.deep_layers:
        session.delete(i)
    session.delete(vel_model)
    session.delete(deep_prof)
    session.commit()
    set_info(f'Скоростная модель "{title_model}" удалена', 'green')
    update_list_velocity_model()


def draw_vel_model_point():
    remove_fill_form()
    try:
        vel_model = session.query(VelocityModel).filter_by(id=ui.listWidget_vel_model.currentItem().text().split(' id')[-1]).first()
    except AttributeError:
        return

    curr_vel_model = session.query(CurrentVelocityModel).first()
    for i in vel_model.velocity_formations:
        list_top = json.loads(i.layer_top)
        list_bottom = json.loads(i.layer_bottom)
        if curr_vel_model:
            if curr_vel_model.active:
                list_top = calc_line_by_vel_model(curr_vel_model.vel_model_id, list_top, curr_vel_model.scale)
                list_bottom = calc_line_by_vel_model(curr_vel_model.vel_model_id, list_bottom, curr_vel_model.scale)
        list_vel = json.loads(i.velocity)
        if ui.checkBox_vel_color.isChecked():
            list_color = [rainbow_colors[int(i)] if int(i) < len(rainbow_colors) else rainbow_colors[-1] for i in list_vel]
            previous_element = None
            list_dupl = []
            for index, current_element in enumerate(list_color):
                if current_element == previous_element:
                    list_dupl.append(index)
                else:
                    if list_dupl:
                        list_dupl.append(list_dupl[-1] + 1)
                        y_up = [list_top[i] for i in list_dupl]
                        y_down = [list_bottom[i] for i in list_dupl]
                        draw_fill_model(list_dupl, y_up, y_down, previous_element)
                    list_dupl = [index]
                previous_element = current_element
            if len(list_dupl) > 0:
                y_up = [list_top[i] for i in list_dupl]
                y_down = [list_bottom[i] for i in list_dupl]
                draw_fill_model(list_dupl, y_up, y_down, previous_element)
        else:
            draw_fill_model(list(range(len(list_top))), list_top, list_bottom, i.color)


def draw_relief():
    if ui.checkBox_relief.isChecked():
        curr_prof = session.query(CurrentProfile).first()
        prof = session.query(Profile).filter(Profile.id == curr_prof.profile_id).first()
        depth_relief = json.loads(prof.depth_relief)
        prof_signal = json.loads(prof.signal)
        relief_signal = [[0 for _ in range(int((depth_relief[i] * 100) / 40))] + prof_signal[i] for i in range(len(prof_signal))]
        relief_signal = [interpolate_list(i, 512) for i in relief_signal]
        draw_image(relief_signal)
    else:
        draw_image(json.loads(session.query(CurrentProfileMinMax).first().signal))
