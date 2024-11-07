from fileinput import filename

import pandas as pd

from func import *
from pyqtgraph.exporters import ImageExporter
from PIL import Image

from krige import draw_map


def draw_radarogram():
    global l_up, l_down
    if 'curve_up' in globals():
        radarogramma.removeItem(globals()['curve_up'])
    if 'curve_down' in globals():
        radarogramma.removeItem(globals()['curve_down'])
    if 'text_item' in globals():
        radarogramma.removeItem(globals()['text_item'])
    if 'poly_item' in globals():
        radarogramma.removeItem(globals()['poly_item'])
    remove_fill_form()
    remove_poly_item()
    remove_curve_fake()
    ui.info.clear()
    clear_current_velocity_model()
    clear_current_profile()
    prof = session.query(Profile).filter(Profile.id == get_profile_id()).first()
    rad = json.loads(prof.signal)
    ui.progressBar.setMaximum(len(rad))
    radar = calc_atrib(rad, ui.comboBox_atrib.currentText())
    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(radar))
    session.add(new_current)
    session.commit()
    save_max_min(radar)
    ui.checkBox_minmax.setCheckState(0)
    draw_image(radar)

    calc_relief_profile(prof)

    set_info(f'Отрисовка "{ui.comboBox_atrib.currentText()}" профиля ({get_object_name()}, {get_profile_name()})', 'blue')
    updatePlot()
    line_up = ui.spinBox_rad_up.value()
    line_down = ui.spinBox_rad_down.value()
    l_up = pg.InfiniteLine(pos=line_up, angle=90, pen=pg.mkPen(color='darkred',width=1, dash=[8, 2]))
    l_down = pg.InfiniteLine(pos=line_down, angle=90, pen=pg.mkPen(color='darkgreen', width=1, dash=[8, 2]))
    radarogramma.addItem(l_up)
    radarogramma.addItem(l_down)
    update_layers()
    draw_layers()
    update_formation_combobox()


def set_scale():
    radarogramma.setXRange(0, 400)
    radarogramma.setYRange(0, 512)


def change_background():
    if ui.checkBox_black_white.isChecked():
        ui.radarogram.setBackground('w')
        ui.graph.setBackground('w')
        ui.signal.setBackground('w')
    else:
        ui.radarogram.setBackground('k')
        ui.graph.setBackground('k')
        ui.signal.setBackground('k')


def on_range_changed():
    X, Y = radarogramma.viewRange()
    ui.graph.setXRange(X[0], X[1])


def crop_from_right(image):
    """Обрезает изображение справа до первого ненулевого (не черного) пикселя."""
    width, height = image.size
    color = (255, 255, 255, 255) if ui.checkBox_black_white.isChecked() else (0, 0, 0, 255)

    for x in range(width - 1, -1, -1):
        if image.getpixel((x, 5)) != color:
            back_break = x
            combined_image = image.crop((0, 0, back_break, height))
            break
    return combined_image

def resize_image(image, width, new_height):
    """Изменить размер изображения до заданной ширины, сохраняя пропорции."""
    return image.resize((width, new_height), Image.Resampling.LANCZOS)

def concatenate_images_vertically(image1, image2):
    """Объединить два изображения по вертикали."""
    new_width = max(image1.width, image2.width)
    new_height = image1.height + image2.height
    new_image = Image.new('RGB', (new_width, new_height))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (0, image1.height))
    return new_image

def process_images(images, graphs):
    """Изменить размеры изображений в graphs и объединить каждую пару с изображениями из images."""
    combined_images = []
    for i in range(len(images)):
        img1 = images[i]
        img2 = graphs[i]
        img2_cropped = crop_from_right(img2)
        aspect_ratio = crop_from_right(graphs[1]).height / crop_from_right(graphs[1]).width
        new_height = int(aspect_ratio * images[1].width)
        img2_resized = resize_image(img2_cropped, img1.width, new_height)
        combined_image = concatenate_images_vertically(img1, img2_resized)
        combined_images.append(combined_image)
    return combined_images


def save_image():
    exporter = ImageExporter(radarogramma)
    exporter.parameters()['height'] = 610
    count_measure = len(
        json.loads(session.query(Profile.signal).filter(Profile.id == get_profile_id()).first()[0]))

    if ui.checkBox_save_graph.isChecked():
        graph_exporter = ImageExporter(ui.graph.scene())
        graph_exporter.parameters()['width'] = 868

        list_paths = []
        list_graphs = []
        N = (count_measure + 399) // 400
        for i in range(N):
            n = i * 400
            m = n + 400
            radarogramma.setXRange(n, m, padding=0)
            ui.graph.setXRange(n, m, padding=0)
            exporter.export(f'{i}_part.png')
            graph_exporter.export(f'{i}_graph.png')
            list_paths.append(f'{i}_part.png')
            list_graphs.append(f'{i}_graph.png')

        images = [Image.open(path) for path in list_paths]
        graphs = [Image.open(path) for path in list_graphs]

        color = (255, 255, 255, 255) if ui.checkBox_black_white.isChecked() else (0, 0, 0, 255)
        color_short = (255, 255, 255) if ui.checkBox_black_white.isChecked() else (0, 0, 0)
        color_break = 0
        while images[0].getpixel((color_break, 200)) == color:
            color_break += 1

        graph_break = 0
        while graphs[0].getpixel((graph_break, 2)) == color:
            graph_break += 1

        for i in range(len(images)):
            width, height = images[i].size
            left = color_break + 1 if i != 0 else 0
            images[i] = images[i].crop((left, 0, width, height))

        for i in range(len(graphs)):
            width, height = graphs[i].size
            left = graph_break + 1 if i != 0 else 0
            graphs[i] = graphs[i].crop((left, 0, width, height))

        combined_images = process_images(images, graphs)

        total_width = sum(img.width for img in combined_images)
        max_height = max(img.height for img in combined_images)
        combined_image = Image.new('RGB', (total_width, max_height), color_short)

        x_offset = 0
        for img in combined_images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width

        back_break = 0
        comb_width, comb_height = combined_image.size
        for x in range(comb_width - 1, -1, -1):
            if combined_image.getpixel((x, 100)) != color \
                    and combined_image.getpixel((x, 100)) != color_short:
                back_break = x
                combined_image = combined_image.crop((0, 0, back_break + 23, comb_height))
                break

        save_path, _ = QFileDialog.getSaveFileName(None, 'Сохранить изображение', '', 'PNG (*.png)')
        if save_path == '':
            return
        if not save_path.endswith('.png'):
            save_path += '.png'
        combined_image.save(save_path)

        for file in list_paths:
            os.remove(file)
        for file in list_graphs:
            os.remove(file)

    else:
        list_paths = []
        N = (count_measure + 399) // 400
        for i in range(N):
            n = i * 400
            m = n + 400
            radarogramma.setXRange(n, m, padding=0)
            exporter.export(f'{i}_part.png')
            list_paths.append(f'{i}_part.png')

        images = [Image.open(path) for path in list_paths]

        color = (255, 255, 255, 255) if ui.checkBox_black_white.isChecked() else (0, 0, 0, 255)
        color_short = (255, 255, 255) if ui.checkBox_black_white.isChecked() else (0, 0, 0)
        color_break = 0
        while images[0].getpixel((color_break, 200)) == color:
            color_break += 1

        for i in range(len(images)):
            width, height = images[i].size
            left = color_break + 1 if i != 0 else 0
            images[i] = images[i].crop((left, 0, width, height))

        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        combined_image = Image.new('RGB', (total_width, max_height), color_short)

        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width

        back_break = 0
        comb_width, comb_height = combined_image.size
        for x in range(comb_width - 1, -1, -1):
            if combined_image.getpixel((x, 100)) != color and combined_image.getpixel((x, 100)) != color_short:
                back_break = x
                combined_image = combined_image.crop((0, 0, back_break + 23, comb_height))
                break

        save_path, _ = QFileDialog.getSaveFileName(None, 'Сохранить изображение', '', 'PNG (*.png)')
        if save_path == '':
            return
        if not save_path.endswith('.png'):
            save_path += '.png'
        combined_image.save(save_path)
        for file in list_paths:
            os.remove(file)

    # img.save('radarogramma.png')
    #
    # # Открываем изображение
    # result_image = Image.open("radarogramma.png")
    #
    # # Получаем размеры изображения
    # width, height = result_image.size
    #
    # # Задаем отступы для увеличения изображения
    # lp = 50
    # bp = 25
    #
    # # Создаем новое изображение с учетом отступов
    # new_width = width + lp
    # new_height = height + bp
    # new_image = Image.new("RGB", (new_width, new_height), "black")
    #
    # # Вставляем текущее изображение с учетом отступов
    # new_image.paste(result_image, (lp, 0))
    #
    # # Создаем объект для рисования на новом изображении
    # draw = ImageDraw.Draw(new_image)
    # # Создаем объект для рисования
    # # draw = ImageDraw.Draw(result_image)
    #
    # font = ImageFont.load_default()
    #
    # # Рисуем ось x (горизонтальную линию внизу)
    # draw.line([(0 + lp, height - 1), (width + lp, height - 1)], fill="white", width=1)
    #
    # # Рисуем ось y (вертикальную линию слева)
    # draw.line([(0 + lp, 0), (0 + lp, height)], fill="white", width=1)
    #
    # # Добавляем деления и подписи на оси X
    # for x in range(0, width, 40):
    #     draw.line([(x + lp, height - 5), (x + lp, height + 5)], fill="white", width=1)
    #     tick_label = f"{int(x * 2.5)}"
    #     draw.text((x + lp - 10, height + 10), tick_label, fill="white", font=font)
    #
    # # Добавляем деления и подписи на оси Y
    # for y in range(0, height, 64):
    #     draw.line([(-5 + lp, y), (5 + lp, y)], fill="white", width=2)
    #     tick_label = f"{y * 8}"
    #     draw.text((-30 + lp, y + 5), tick_label, fill="white", font=font)
    #
    # # Добавляем тонкую светлую сетку грида
    # dash_length = 5
    # for x in range(0, width, 40):
    #     for y in range(0, height, dash_length * 2):
    #         draw.point((x + lp, y), fill=(200, 200, 200))
    #
    # for y in range(0, height, 64):
    #     for x in range(0, width, dash_length * 2):
    #         draw.point((x + lp, y), fill=(200, 200, 200))
    #
    # # Подписываем оси
    # draw.text((width + lp - 20, height - 20), "X", fill="white", font=font)
    # draw.text((10 + lp, 10), "Y", fill="white", font=font)
    #
    # # Сохраняем результат
    # try:
    #     file_name = f"{get_profile_name()}.png"
    #     fn = QFileDialog.getSaveFileName(
    #         caption=f'Сохранить файл "{get_profile_name()}"',
    #         directory=file_name,
    #         filter="Изображения (*.png)")
    #     print(fn)
    #     new_image.save(fn[0])
    #     set_info(f'Сохранено в файл: {fn[0]}', 'green')
    # except ValueError:
    #     pass
    #
    # # Удаляем файл
    # del_img = 'radarogramma.png'
    # if os.path.exists(del_img):
    #     os.remove(del_img)


def show_grid():
    if ui.checkBox_grid.isChecked():
        radarogramma.showGrid(x=True, y=True)
    else:
        radarogramma.showGrid(x=False, y=False)

# def draw_axis():
#
#     axis_x = radarogramma.getAxis('bottom')
#     ticks_x = axis_x.tickValues(axis_x.range[0], axis_x.range[1], axis_x.width())
#     major_tick_x_str = axis_x.tickStrings(ticks_x[0][1], 2.5, ticks_x[0][0])
#     minor_tick_x_str = axis_x.tickStrings(ticks_x[2][1], 2.5, ticks_x[2][0])

def draw_current_radarogram():
    global l_up, l_down
    if 'curve_up' in globals():
        radarogramma.removeItem(globals()['curve_up'])
    if 'curve_down' in globals():
        radarogramma.removeItem(globals()['curve_down'])
    if 'text_item' in globals():
        radarogramma.removeItem(globals()['text_item'])
    if 'poly_item' in globals():
        radarogramma.removeItem(globals()['poly_item'])
    remove_poly_item()
    remove_curve_fake()
    rad = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])
    ui.progressBar.setMaximum(len(rad))
    radar = calc_atrib(rad, ui.comboBox_atrib.currentText())
    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(radar))
    session.add(new_current)
    session.commit()
    save_max_min(radar)
    if ui.checkBox_minmax.isChecked():
        radar = json.loads(session.query(CurrentProfileMinMax.signal).filter(CurrentProfileMinMax.id == 1).first()[0])
    draw_image(radar)
    set_info(f'Отрисовка "{ui.comboBox_atrib.currentText()}" для текущего профиля', 'blue')
    updatePlot()
    line_up = ui.spinBox_rad_up.value()
    line_down = ui.spinBox_rad_down.value()
    l_up = pg.InfiniteLine(pos=line_up, angle=90, pen=pg.mkPen(color='darkred',width=1, dash=[8, 2]))
    l_down = pg.InfiniteLine(pos=line_down, angle=90, pen=pg.mkPen(color='darkgreen', width=1, dash=[8, 2]))
    radarogramma.addItem(l_up)
    radarogramma.addItem(l_down)
    update_formation_combobox()


def draw_max_min():
    rad = session.query(CurrentProfile.signal).first()
    radar = json.loads(rad[0])
    radar_max_min = []
    ui.progressBar.setMaximum(len(radar))
    for n, sig in enumerate(radar):
        # max_points, _ = find_peaks(np.array(sig))
        # min_points, _ = find_peaks(-np.array(sig))
        diff_signal = np.diff(sig) #возможно 2min/max
        max_points = argrelmax(diff_signal)[0]
        min_points = argrelmin(diff_signal)[0]
        signal_max_min = build_list(max_points, min_points)

        radar_max_min.append(signal_max_min)
        ui.progressBar.setValue(n + 1)
    clear_current_profile()
    new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(radar_max_min))
    session.add(new_current)
    session.commit()

    draw_image(radar_max_min)
    updatePlot()
    set_info(f'Отрисовка "max/min" для текущего профиля', 'blue')


def draw_rad_line():
    global l_up, l_down
    radarogramma.removeItem(l_up)
    radarogramma.removeItem(l_down)
    line_up = ui.spinBox_rad_up.value()
    line_down = ui.spinBox_rad_down.value()
    l_up = pg.InfiniteLine(pos=line_up, angle=90, pen=pg.mkPen(color='darkred',width=1, dash=[8, 2]))
    l_down = pg.InfiniteLine(pos=line_down, angle=90, pen=pg.mkPen(color='darkgreen', width=1, dash=[8, 2]))
    radarogramma.addItem(l_up)
    radarogramma.addItem(l_down)


def choose_minmax():
    remove_poly_item()
    remove_curve_fake()
    if ui.checkBox_minmax.isChecked():
        radar = json.loads(session.query(CurrentProfileMinMax.signal).filter(CurrentProfileMinMax.id == 1).first()[0])
    else:
        radar = json.loads(session.query(CurrentProfile.signal).filter(CurrentProfile.id == 1).first()[0])
    vel_mod = session.query(CurrentVelocityModel).first()
    if vel_mod:
        if vel_mod.active:
            draw_image_deep_prof(radar, vel_mod.scale)
    else:
        draw_image(radar)
    updatePlot()


def draw_formation():
    remove_poly_item()
    remove_curve_fake()
    if 'curve_up' in globals():
        radarogramma.removeItem(globals()['curve_up'])
    if 'curve_down' in globals():
        radarogramma.removeItem(globals()['curve_down'])
    if 'text_item' in globals():
        radarogramma.removeItem(globals()['text_item'])
    if ui.comboBox_plast.currentText() == '-----':
        return
    # elif ui.comboBox_plast.currentText() == 'KROT':
    #     t_top = json.loads(session.query(Profile.T_top).filter(Profile.id == get_profile_id()).first()[0])
    #     t_bot = json.loads(session.query(Profile.T_bottom).filter(Profile.id == get_profile_id()).first()[0])
    #     layer_up = [x / 8 for x in t_top]
    #     layer_down = [x / 8 for x in t_bot]
    #     title_text = 'KROT'
    else:
        formation = session.query(Formation).filter(Formation.id == get_formation_id()).first()
        layer_up = json.loads(session.query(Layers.layer_line).filter(Layers.id == formation.up).first()[0])
        layer_down = json.loads(session.query(Layers.layer_line).filter(Layers.id == formation.down).first()[0])
        title_text = formation.title
    x = list(range(len(layer_up)))
    vel_mod = session.query(CurrentVelocityModel).first()
    if vel_mod:
        if vel_mod.active:
            layer_up = calc_line_by_vel_model(vel_mod.vel_model_id, layer_up, vel_mod.scale)
            layer_down = calc_line_by_vel_model(vel_mod.vel_model_id, layer_down, vel_mod.scale)

    if ui.checkBox_relief.isChecked():
        profile = session.query(Profile).filter(Profile.id == get_profile_id()).first()
        if profile.depth_relief:
            depth = [i * 100 / 40 for i in json.loads(profile.depth_relief)]
            coeff = 512 / (512 + np.max(depth))
            layer_up = [int((x + y) * coeff) for x, y in zip(layer_up, depth)]
            layer_down = [int((x + y) * coeff) for x, y in zip(layer_down, depth)]

    # Создаем объект линии и добавляем его на радарограмму
    curve_up = pg.PlotCurveItem(x=x, y=layer_up, pen=pg.mkPen(width=2))
    curve_down = pg.PlotCurveItem(x=x, y=layer_down, pen=pg.mkPen(width=2))
    radarogramma.addItem(curve_up)
    radarogramma.addItem(curve_down)
    # Создаем объект текста для отображения id слоя и добавляем его на радарограмму
    text_item = pg.TextItem(text=f'{title_text}', color='white')
    text_item.setPos(min(x) - int(max(x) - min(x)) / 50, int(layer_down[x.index(min(x))]))
    radarogramma.addItem(text_item)
    # Добавляем созданные объекты в глобальные переменные для возможности последующего удаления
    globals()['curve_up'] = curve_up
    globals()['curve_down'] = curve_down
    globals()['text_item'] = text_item


def draw_fill(x, y1, y2, color):
    remove_poly_item()
    remove_curve_fake()
    curve_up = pg.PlotCurveItem(x=x, y=y1)
    curve_down = pg.PlotCurveItem(x=x, y=y2)
    poly_item = pg.FillBetweenItem(curve1=curve_down, curve2=curve_up, brush=color)
    radarogramma.addItem(poly_item)
    poly_item.setOpacity(0.5)
    poly_item.setZValue(1)
    globals()['poly_item'] = poly_item


def draw_fake(list_fake, list_up, list_down):
    remove_curve_fake()
    for f in list_fake:
        curve_fake = pg.PlotCurveItem(x=[f, f], y=[list_up[f], list_down[f]], pen=pg.mkPen(color='white', width=1))
        radarogramma.addItem(curve_fake)
        curve_fake.setZValue(1)
        globals()[f'curve_fake_{f}'] = curve_fake


def remove_poly_item():
    for key, value in globals().items():
        if key.startswith('poly_item'):
            radarogramma.removeItem(globals()[key])


def remove_fill_form():
    for key, value in globals().items():
        if key.startswith('fill_form_'):
            radarogramma.removeItem(globals()[key])


def remove_curve_fake():
    for key, value in globals().items():
        if key.startswith('curve_fake_'):
            radarogramma.removeItem(globals()[key])


def draw_fill_result(x, y1, y2, color):
    curve_up = pg.PlotCurveItem(x=x, y=y1)
    curve_down = pg.PlotCurveItem(x=x, y=y2)
    poly_item = pg.FillBetweenItem(curve1=curve_down, curve2=curve_up, brush=color)
    radarogramma.addItem(poly_item)
    poly_item.setOpacity(0.5)
    poly_item.setZValue(1)
    globals()[f'poly_item{x[0]}'] = poly_item


def draw_fill_model(x, y1, y2, color):
    curve_up = pg.PlotCurveItem(x=x, y=y1)
    curve_down = pg.PlotCurveItem(x=x, y=y2)
    poly_item = pg.FillBetweenItem(curve1=curve_down, curve2=curve_up, brush=color)
    radarogramma.addItem(poly_item)
    poly_item.setOpacity(0.5)
    poly_item.setZValue(1)
    globals()[f'fill_form_{color}_{y1[0]}_{y2[0]}_{x[0]}'] = poly_item

def build_table_profile_model_predict():
    predict = session.query(ProfileModelPrediction).filter_by(id=ui.listWidget_model_pred.currentItem().text().split(' id')[-1]).first()
    if not predict:
        set_info('Выберите модель в Model Prediction', 'red')
        return
    pd_predict = pd.DataFrame(columns=['x_pulc', 'y_pulc', 'prediction'])
    for prof in session.query(Profile).filter_by(research_id=predict.profile.research_id).all():
        pred = session.query(ProfileModelPrediction).filter_by(profile_id=prof.id, model_id=predict.model_id).first()
        if not pred:
            set_info(f'Модель не расчитана для профиля {prof.title}', 'red')
            QtWidgets.QMessageBox.critical(MainWindow, 'Error', f'Модель не расчитана для профиля {prof.title}')
            return
        x_pulc = json.loads(prof.x_pulc)
        y_pulc = json.loads(prof.y_pulc)
        value = json.loads(pred.prediction)
        pd_predict = pd.concat([pd_predict, pd.DataFrame({'x_pulc': x_pulc, 'y_pulc': y_pulc, 'prediction': value})], ignore_index=True)

    return pd_predict


def draw_profile_model_predict():
    pd_predict = build_table_profile_model_predict()
    if pd_predict.empty:
        return
    draw_map(pd_predict['x_pulc'], pd_predict['y_pulc'], pd_predict['prediction'], ui.listWidget_model_pred.currentItem().text().split(' id')[0])


def save_excel_profile_model_predict():
    pd_predict = build_table_profile_model_predict()
    if pd_predict.empty:
        return
    filename = QtWidgets.QFileDialog.getSaveFileName(
        MainWindow,
        caption='Save file',
        directory=f'{get_object_name()}_{ui.listWidget_model_pred.currentItem().text().split(" id")[0]}.xlsx',
        filter='Excel files (*.xlsx)'
    )[0]
    try:
        pd_predict.to_excel(filename)
        set_info(f'Файл {filename} сохранен', 'green')
    except ValueError:
        set_info('Файл не сохранен', 'red')