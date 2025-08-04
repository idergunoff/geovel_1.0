import pandas as pd

from draw import draw_noise
from func import *
from krige import draw_map


def check_noise_profile():
    n_lof = ui.spinBox_lof_noise.value()
    profile = session.query(Profile).filter_by(id=get_profile_id()).first()

    pd_signal = pd.DataFrame(json.loads(profile.signal), columns=[f'sig{n}' for n in range(512)])
    print(pd_signal)

    scaler = StandardScaler()
    training_sample_lof = scaler.fit_transform(pd_signal)

    lof = LocalOutlierFactor(n_neighbors=n_lof)
    label_lof = lof.fit_predict(training_sample_lof)
    factor_lof = -lof.negative_outlier_factor_
    print(label_lof)
    print(factor_lof)

    ui.graph.clear()
    number = list(range(1, len(label_lof) + 1))  # создаем список номеров элементов данных

    curve = pg.PlotCurveItem(x=number, y=label_lof, pen=pg.mkPen(color='red', width=2.4))

    ui.graph.addItem(curve)  # добавляем график данных на график
    ui.graph.showGrid(x=True, y=True)  # отображаем сетку на графике
    ui.graph.getAxis('bottom').setScale(2.5)
    ui.graph.getAxis('bottom').setLabel('Профиль, м')
    draw_noise(label_lof)


def check_noise_object():
    n_lof = ui.spinBox_lof_noise.value()
    profiles = session.query(Profile).filter_by(research_id=get_research_id()).all()
    signal, list_x, list_y = [], [], []
    for profile in profiles:
        signal.extend(json.loads(profile.signal))
        list_x.extend(json.loads(profile.x_pulc))
        list_y.extend(json.loads(profile.y_pulc))

    pd_signal = pd.DataFrame(signal, columns=[f'sig{n}' for n in range(512)])
    print(pd_signal)

    scaler = StandardScaler()
    training_sample_lof = scaler.fit_transform(pd_signal)

    lof = LocalOutlierFactor(n_neighbors=n_lof)
    label_lof = lof.fit_predict(training_sample_lof)
    factor_lof = -lof.negative_outlier_factor_
    print(label_lof)
    print(factor_lof)

    draw_map(list_x, list_y, label_lof, 'NOISE', profiles=True)
    result = QMessageBox.question(MainWindow, 'Сохранение', 'Сохранить результаты расчёта зоны помех?',
                                   QMessageBox.Yes, QMessageBox.No)

    if result == QMessageBox.Yes:
        pd_result = pd.DataFrame(columns=['x', 'y', 'noise'])
        pd_result['x'] = list_x
        pd_result['y'] = list_y
        pd_result['noise'] = label_lof
    else:
        return
    try:
        file_name = f'{get_object_name()}_{get_research_name()}__NOISE.xlsx'
        fn = QFileDialog.getSaveFileName(
            caption=f'Сохранить результат расчета зоны помех "{get_object_name()}_{get_research_name()}" в таблицу',
            directory=file_name,
            filter="Excel Files (*.xlsx)")
        pd_result.to_excel(fn[0])
        set_info(f'Таблица сохранена в файл: {fn[0]}', 'green')
    except ValueError:
        pass
