import json
from fileinput import filename

import pandas as pd
from sympy.physics.units import velocity

from func import *
from pyqtgraph.exporters import ImageExporter
from PIL import Image

from krige import draw_map
from velocity_prediction import calc_deep_predict_current_profile, calc_list_velocity


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
    # ui.info.clear()
    ui.checkBox_relief.setChecked(False)
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

def set_marks_scale(image, width, width_2, length, bottom_break, bottom_comb=None):
    draw = ImageDraw.Draw(image)

    for i in range(length):
        # font = ImageFont.truetype("arial.ttf", 11)
        # font_graph = ImageFont.truetype("arial.ttf", 10)
        font = get_system_font(13)
        font_graph = get_system_font(12)

        text = f'{i+1}000'
        text_bbox = draw.textbbox((0, 0), text, font=font)  # Вычисляем размеры текста
        text_width = text_bbox[2] - text_bbox[0]
        if i == 0:
            text_x = width - text_width // 2
        else:
            text_x = (width + width_2 * i) - text_width // 2
        draw.text((text_x, bottom_break + 2), text, fill="grey", font=font)
        if ui.checkBox_save_graph.isChecked():
            draw.text((text_x, bottom_comb + 2), text, fill="grey", font=font_graph)

    return image


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

        bottom_break = images[0].height - 1
        while images[0].getpixel((images[0].width - 1, bottom_break)) == color:
            bottom_break -= 1

        cropped_images = []
        for img in images:
            width, height = img.size
            cropped_img = img.crop((0, 0, width, height - 10))
            cropped_images.append(cropped_img)

        images = cropped_images

        color_break, color_break_0, count_color = 0, 0, 0
        while images[0].getpixel((color_break, 200)) == color or count_color < 5:
            if images[0].getpixel((color_break, 200)) != color:
                count_color += 1
                if count_color == 1:
                    color_break_0 = color_break
            else:
                count_color = 0
            color_break += 1
        color_break = color_break_0

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

        bottom_comb = combined_image.height - 1
        color_pix = combined_image.getpixel((combined_image.width - 5, bottom_comb))
        while combined_image.getpixel((combined_image.width - 5, bottom_comb)) == color_pix:
            bottom_comb -= 1
        print(bottom_comb)
        final_image = set_marks_scale(combined_image, combined_images[0].width, combined_images[1].width,
                                      len(combined_images), bottom_break, bottom_comb)
        save_path, _ = QFileDialog.getSaveFileName(None, 'Сохранить изображение', '', 'PNG (*.png)')
        if save_path == '':
            return
        if not save_path.endswith('.png'):
            save_path += '.png'
        final_image.save(save_path)

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
        cropped_images = []
        for img in images:
            width, height = img.size
            cropped_img = img.crop((0, 0, width, height - 10))
            cropped_images.append(cropped_img)

        images = cropped_images

        color = (255, 255, 255, 255) if ui.checkBox_black_white.isChecked() else (0, 0, 0, 255)
        color_short = (255, 255, 255) if ui.checkBox_black_white.isChecked() else (0, 0, 0)

        bottom_break = images[0].height - 1
        while images[0].getpixel((images[0].width - 1, bottom_break)) == color:
            bottom_break -= 1

        color_break, color_break_0, count_color = 0, 0, 0
        while images[0].getpixel((color_break, 200)) == color or count_color < 5:
            if images[0].getpixel((color_break, 200)) != color:
                count_color += 1
                if count_color == 1:
                    color_break_0 = color_break
            else:
                count_color = 0
            color_break += 1
        color_break = color_break_0

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

        final_image = set_marks_scale(combined_image, images[0].width, images[1].width,
                                      len(images), bottom_break)
        save_path, _ = QFileDialog.getSaveFileName(None, 'Сохранить изображение', '', 'PNG (*.png)')
        if save_path == '':
            return
        if not save_path.endswith('.png'):
            save_path += '.png'
        final_image.save(save_path)
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
        pd_dict = {'x_pulc': x_pulc, 'y_pulc': y_pulc, 'prediction': value}
        if ui.checkBox_use_land.isChecked():
            land = json.loads(prof.formations[0].land)
            pd_dict['land'] = land
            pd_dict['abs_uf'] = [i - j for i, j in zip(land, value)]
        pd_predict = pd.concat([pd_predict, pd.DataFrame(pd_dict)], ignore_index=True)

    return pd_predict


def draw_profile_model_predict():
    pd_predict = build_table_profile_model_predict()
    if pd_predict.empty:
        return
    if ui.checkBox_use_land.isChecked():
        draw_map(pd_predict['x_pulc'], pd_predict['y_pulc'], pd_predict['abs_uf'], ui.listWidget_model_pred.currentItem().text().split(' id')[0])
    else:
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
    remove_poly_item()
    remove_curve_fake()
    remove_fill_form()
    if 'curve_up' in globals():
        radarogramma.removeItem(globals()['curve_up'])
    if 'curve_down' in globals():
        radarogramma.removeItem(globals()['curve_down'])
    if 'text_item' in globals():
        radarogramma.removeItem(globals()['text_item'])

    if ui.checkBox_relief.isChecked():
        if ui.checkBox_minmax.isChecked():
            curr_prof = session.query(CurrentProfileMinMax).first()
        else:
            curr_prof = session.query(CurrentProfile).first()
        if not curr_prof:
            return

        prof = session.query(Profile).filter(Profile.id == curr_prof.profile_id).first()
        if ui.checkBox_vel.isChecked():
            velocity_signal = calc_deep_predict_current_profile()
            depth_relief = json.loads(prof.depth_relief)
            relief_velocity_signal = [[-128 for _ in range(int(depth_relief[i]))] + velocity_signal[i] for i in range(len(depth_relief))]
            l_max = 0
            for i in relief_velocity_signal:
                l_max = len(i) if len(i) > l_max else l_max
            result_signal = [i + [-128 for _ in range(l_max - len(i))] for i in relief_velocity_signal]
            result_signal = [interpolate_list(i, 512) for i in result_signal]
            draw_image_deep_prof(result_signal, l_max / 512)

            bindings = session.query(BindingLayerPrediction).join(Layers).filter(
                Layers.profile_id == get_profile_id()).all()
            for n, b in enumerate(bindings):
                layer = savgol_filter(json.loads(b.prediction.prediction), 175, 3)
                line = [(layer[i] + depth_relief[i]) / (l_max / 512) for i in range(len(layer))]
                x = list(range(len(line)))
                curve = pg.PlotCurveItem(x=x, y=line, pen=pg.mkPen(width=2))
                radarogramma.addItem(curve)
                globals()[f'curve_fake_{n}'] = curve
        else:
            prof_signal = json.loads(curr_prof.signal)

            depth_relief = json.loads(prof.depth_relief)
            bottom_relief = [np.max(depth_relief) - i for i in depth_relief]
            relief_signal = [[-128 for _ in range(int((depth_relief[i] * 100) / 40))] + prof_signal[i] + [-128 for _ in range(int((bottom_relief[i] * 100) / 40))] for i in range(len(prof_signal))]
            max_sig = len(relief_signal)
            relief_signal = [interpolate_list(i, 512) for i in relief_signal]
            draw_image(relief_signal)
        # if ui.checkBox_vel.isChecked():
        #     depth = [i * 100 / 40 for i in json.loads(prof.depth_relief)]
        #     print(depth)
        #     # l_max = 0
        #     # for rs in relief_signal:
        #     #     l_max = len(rs) if len(rs) > l_max else l_max
        #
        #     coeff = 512 / (512 + np.max(depth))
        #     bindings = session.query(BindingLayerPrediction).join(Layers).filter(Layers.profile_id == get_profile_id()).all()
        #     for n, b in enumerate(bindings):
        #         l = json.loads(b.layer.layer_line)
        #         p = savgol_filter(json.loads(b.prediction.prediction), 175, 3)
        #         v = [(p[i]*100)/(l[i]*8) for i in range(len(l))]
        #         print('v', v)
        #         layer = [p[i] * 100 / 8*v[i] for i in range(len(p))]
        #         print(layer)
        #         line = [int(x + y) * coeff for x, y in zip(layer, depth)]
        #         x1 = list(range(len(line)))
        #         curve = pg.PlotCurveItem(x=x1, y=line, pen=pg.mkPen(width=2))
        #         radarogramma.addItem(curve)
        #         globals()[f'curve_fake_{n}'] = curve
    else:
        if ui.checkBox_vel.isChecked():
            deep_signal = calc_deep_predict_current_profile()
            l_max = 0
            try:
                for i in deep_signal:
                    l_max = len(i) if len(i) > l_max else l_max
            except:
                return
            deep_signal = [i + [0 for _ in range(l_max - len(i))] for i in deep_signal]
            deep_signal = [interpolate_list(i, 512) for i in deep_signal]
            draw_image_deep_prof(deep_signal, l_max / 512)

            bindings = session.query(BindingLayerPrediction).join(Layers).filter(Layers.profile_id == get_profile_id()).all()
            for n, b in enumerate(bindings):
                line = [i / (l_max/512) for i in savgol_filter(json.loads(b.prediction.prediction), 175, 3)]
                x = list(range(len(line)))
                curve = pg.PlotCurveItem(x=x, y=line, pen=pg.mkPen(width=2))
                radarogramma.addItem(curve)
                globals()[f'curve_fake_{n}'] = curve

        else:
            try:
                draw_image(json.loads(session.query(CurrentProfile).first().signal))
            except:
                return

def get_color_rainbow(probability):

    rainbow_colors =[
        "#0000FF",  # Синий
        "#0066FF",  # Голубой
        "#00CCFF",  # Светло-голубой
        "#00FFCC",  # Бирюзовый
        "#00FF66",  # Зеленовато-голубой
        "#33FF33",  # Ярко-зеленый
        "#99FF33",  # Желто-зеленый
        "#FFFF00",  # Желтый
        "#FF6600",  # Оранжевый
        "#FF0000"   # Красный
    ]
    try:
        return rainbow_colors[int(probability * 10)]
    except (IndexError, ValueError):
        return '#FF0000'


def draw_profile_model_prediction():
    ui.graph.clear()
    remove_poly_item()
    if not ui.checkBox_velmod.isChecked():
        remove_fill_form()
    try:
        model = session.query(ProfileModelPrediction).filter_by(id=ui.listWidget_model_pred.currentItem().text().split(' id')[-1]).first()
        graph = json.loads(model.prediction)
    except AttributeError:
        return

    if ui.checkBox_vel.isChecked() and model.type_model == 'cls' and not ui.checkBox_velmod.isChecked():
        if ui.checkBox_relief.isChecked():
            curr_prof = session.query(CurrentProfile).first()
            prof = session.query(Profile).filter(Profile.id == curr_prof.profile_id).first()

            velocity_signal = calc_deep_predict_current_profile()
            depth_relief = json.loads(prof.depth_relief)
            relief_velocity_signal = [[-128 for _ in range(int(depth_relief[i]))] + velocity_signal[i] for i in
                                      range(len(depth_relief))]
            l_max = 0
            for i in relief_velocity_signal:
                l_max = len(i) if len(i) > l_max else l_max
            result_signal = [i + [-128 for _ in range(l_max - len(i))] for i in relief_velocity_signal]
            result_signal = [interpolate_list(i, 512) for i in result_signal]
            draw_image_deep_prof(result_signal, l_max / 512)

            bindings = session.query(BindingLayerPrediction).join(Layers).filter(
                Layers.profile_id == get_profile_id()).all()

            layer_up = savgol_filter(json.loads(bindings[0].prediction.prediction), 175, 3)
            list_up = [(layer_up[i] + depth_relief[i]) / (l_max / 512) for i in range(len(layer_up))]
            layer_down = savgol_filter(json.loads(bindings[1].prediction.prediction), 175, 3)
            list_down = [(layer_down[i] + depth_relief[i]) / (l_max / 512) for i in range(len(layer_down))]

        else:
            deep_signal = calc_deep_predict_current_profile()
            l_max = 0
            for i in deep_signal:
                l_max = len(i) if len(i) > l_max else l_max
            deep_signal = [i + [0 for _ in range(l_max - len(i))] for i in deep_signal]
            deep_signal = [interpolate_list(i, 512) for i in deep_signal]
            draw_image_deep_prof(deep_signal, l_max / 512)

            bindings = session.query(BindingLayerPrediction).join(Layers).filter(
                Layers.profile_id == get_profile_id()).all()

            list_up = [i / (l_max / 512) for i in savgol_filter(json.loads(bindings[0].prediction.prediction), 175, 3)]
            list_down = [i / (l_max / 512) for i in savgol_filter(json.loads(bindings[1].prediction.prediction), 175, 3)]

        previous_element = None
        list_dupl = []
        for index, pred in enumerate(graph):
            color = get_color_rainbow(pred)
            if color == previous_element:
                list_dupl.append(index)
            else:
                if list_dupl:
                    list_dupl.append(list_dupl[-1] + 1)
                    y_up = [list_up[i] for i in list_dupl]
                    y_down = [list_down[i] for i in list_dupl]
                    draw_fill_result(list_dupl, y_up, y_down, previous_element)
                list_dupl = [index]
            previous_element = color
        if len(list_dupl) > 0:
            y_up = [list_up[i] for i in list_dupl]
            y_down = [list_down[i] for i in list_dupl]
            draw_fill_result(list_dupl, y_up, y_down, get_color_rainbow(pred))

    try:
        number = list(range(1, len(graph) + 1))  # создаем список номеров элементов данных
    except Exception as e:
        set_info(e, 'red')
        return

    cc = (120, 120, 120, 255)
    curve = pg.PlotCurveItem(x=number, y=graph, pen=cc)  # создаем объект класса PlotCurveIte
    # m для отображения графика данных
    # создаем объект класса PlotCurveItem для отображения фильтрованных данных с помощью savgol_filter()
    curve_filter = pg.PlotCurveItem(x=number, y=savgol_filter(graph, 31, 3), pen=pg.mkPen(color='red', width=2.4))
    ui.graph.addItem(curve)  # добавляем график данных на график
    ui.graph.addItem(curve_filter)  # добавляем фильтрованный график данных на график
    ui.graph.showGrid(x=True, y=True)  # отображаем сетку на графике
    ui.graph.getAxis('bottom').setScale(2.5)
    ui.graph.getAxis('bottom').setLabel('Профиль, м')
    set_info(f'Отрисовка предсказания модели "{ui.listWidget_model_pred.currentItem().text().split(" id")[0]}" '
             f'для текущего профиля', 'blue')


def draw_velocity_model_color():
    if ui.checkBox_velmod.isChecked():
        if ui.checkBox_vel.isChecked():
            draw_relief()
            list_vel = calc_list_velocity()

            # clear_current_profile()
            # new_current = CurrentProfile(profile_id=get_profile_id(), signal=json.dumps(list_vel))
            # session.add(new_current)
            # session.commit()
            curr_prof = session.query(CurrentProfile).first()
            if not curr_prof:
                return

            if ui.checkBox_relief.isChecked():
                prof = session.query(Profile).filter(Profile.id == curr_prof.profile_id).first()
                velocity_signal = calc_deep_predict_current_profile()
                depth_relief = json.loads(prof.depth_relief)
                relief_velocity_signal = [[-128 for _ in range(int(depth_relief[i]))] + velocity_signal[i] for i in
                                          range(len(depth_relief))]
                l_max = 0
                for i in relief_velocity_signal:
                    l_max = len(i) if len(i) > l_max else l_max
            else:
                deep_signal = calc_deep_predict_current_profile()
                l_max = 0
                for i in deep_signal:
                    l_max = len(i) if len(i) > l_max else l_max
            bindings = session.query(BindingLayerPrediction).join(Layers).filter(
                Layers.profile_id == get_profile_id()).all()

            for index, b in enumerate(bindings):
                if ui.checkBox_relief.isChecked():
                    if index == 0:
                        layer_down = savgol_filter(json.loads(bindings[index].prediction.prediction), 175, 3)
                        list_down = [(layer_down[i] + depth_relief[i]) / (l_max / 512) for i in range(len(layer_down))]

                        layer_up = [0 for _ in range(len(list_down))]
                        list_up = [(layer_up[i] + depth_relief[i]) / (l_max / 512) for i in range(len(layer_up))]

                    else:
                        layer_down = savgol_filter(json.loads(bindings[index].prediction.prediction), 175, 3)
                        list_down = [(layer_down[i] + depth_relief[i]) / (l_max / 512) for i in range(len(layer_down))]

                        layer_up = savgol_filter(json.loads(bindings[index - 1].prediction.prediction), 175, 3)
                        list_up = [(layer_up[i] + depth_relief[i]) / (l_max / 512) for i in range(len(layer_up))]

                else:
                    list_down = [i / (l_max / 512) for i in
                                 savgol_filter(json.loads(bindings[index].prediction.prediction), 175, 3)]
                    if index == 0:
                        list_up = [0 for _ in range(len(list_down))]
                    else:
                        list_up = [i / (l_max / 512) for i in
                                 savgol_filter(json.loads(bindings[index - 1].prediction.prediction), 175, 3)]

                list_color = [rainbow_colors[int(i)] if int(i) < len(rainbow_colors) else rainbow_colors[-1] for i in
                              list_vel[index]]
                previous_element = None
                list_dupl = []
                for index, current_element in enumerate(list_color):
                    if current_element == previous_element:
                        list_dupl.append(index)
                    else:
                        if list_dupl:
                            list_dupl.append(list_dupl[-1] + 1)
                            y_up = [list_up[i] for i in list_dupl]
                            y_down = [list_down[i] for i in list_dupl]
                            draw_fill_model(list_dupl, y_up, y_down, previous_element)
                        list_dupl = [index]
                    previous_element = current_element
                if len(list_dupl) > 0:
                    y_up = [list_up[i] for i in list_dupl]
                    y_down = [list_down[i] for i in list_dupl]
                    draw_fill_model(list_dupl, y_up, y_down, previous_element)


        else:
            remove_fill_form()
    else:
        remove_fill_form()