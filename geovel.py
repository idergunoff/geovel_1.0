import traceback

from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QColorDialog

from load import *
from filtering import *
from draw import *
from layer import *
from well import *
from lda import *


MainWindow.show()


# def show_globals():
#     print(globals().keys())


def mouse_moved_to_signal(evt):
    """ Отслеживаем координаты курсора и отображение на графике сигнала """
    global hor_line_sig, hor_line_rad, vert_line_rad, vert_line_graph
    # Удаление предыдущих линий при движении мыши
    if 'hor_line_sig' in globals():
        ui.signal.removeItem(hor_line_sig)
        radarogramma.removeItem(hor_line_rad)
        radarogramma.removeItem(vert_line_rad)
        ui.graph.removeItem(vert_line_graph)
    # Получение координат курсора
    pos = evt[0]
    vb = radarogramma.vb
    # Проверка, находится ли курсор в пределах области графика
    if radarogramma.sceneBoundingRect().contains(pos):
        mousePoint = vb.mapSceneToView(pos)
        # Создание бесконечных линий
        hor_line_sig = pg.InfiniteLine(pos=mousePoint.y(), angle=0, pen=pg.mkPen(color='w', width=0.5, dash=[4, 7]))
        hor_line_rad = pg.InfiniteLine(pos=mousePoint.y(), angle=0, pen=pg.mkPen(color='w', width=0.5, dash=[4, 7]))
        vert_line_rad = pg.InfiniteLine(pos=mousePoint.x(), angle=90, pen=pg.mkPen(color='w', width=0.5, dash=[4, 7]))
        vert_line_graph = pg.InfiniteLine(pos=mousePoint.x(), angle=90, pen=pg.mkPen(color='w', width=0.5, dash=[4, 7]))
        # Добавление линий на соответствующие графики
        ui.signal.addItem(hor_line_sig)
        radarogramma.addItem(hor_line_rad)
        radarogramma.addItem(vert_line_rad)
        ui.graph.addItem(vert_line_graph)


proxy = pg.SignalProxy(radarogramma.scene().sigMouseMoved, rateLimit=60, slot=mouse_moved_to_signal)


def log_uncaught_exceptions(ex_cls, ex, tb):
    """ Вывод ошибок программы """
    text = '{}: {}:\n'.format(ex_cls.__name__, ex)
    text += ''.join(traceback.format_tb(tb))
    print(text)
    QtWidgets.QMessageBox.critical(None, 'Error', text)
    sys.exit()


def change_color():
    button_color = ui.pushButton_color.palette().color(ui.pushButton_color.backgroundRole())
    color = QColorDialog.getColor(button_color)
    ui.pushButton_color.setStyleSheet(f"background-color: {color.name()};")
    ui.pushButton_color.setText(color.name())


    # result = QtWidgets.QMessageBox.question(None, 'Вопрос', 'Вы уверены, что хотите выйти?',
    #                                         QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.Cancel)
    #
    # if result == QtWidgets.QMessageBox.Yes:
    #     print('Да')
    #
    # elif result == QtWidgets.QMessageBox.Cancel:
    #     print('Отмена')


img.scene().sigMouseClicked.connect(mouseClicked)

ui.pushButton_save_signal.clicked.connect(save_signal)
ui.pushButton_draw_rad.clicked.connect(draw_radarogram)
ui.pushButton_draw_cur.clicked.connect(draw_current_radarogram)
ui.pushButton_vacuum.clicked.connect(vacuum)
ui.pushButton_uf.clicked.connect(load_uf_grid)
ui.pushButton_m.clicked.connect(load_m_grid)
ui.pushButton_r.clicked.connect(load_r_grid)
ui.pushButton_fft.clicked.connect(calc_fft)
ui.pushButton_add_fft.clicked.connect(calc_add_fft)
ui.pushButton_ifft.clicked.connect(calc_ifft)
ui.pushButton_medfilt.clicked.connect(calc_medfilt)
ui.pushButton_wiener.clicked.connect(calc_wiener)
ui.pushButton_savgol.clicked.connect(calc_savgol)
ui.pushButton_filtfilt.clicked.connect(calc_filtfilt)
ui.pushButton_reset.clicked.connect(reset_spinbox_fft)
ui.pushButton_maxmin.clicked.connect(draw_max_min)
ui.pushButton_rfft2.clicked.connect(calc_rfft2)
ui.pushButton_irfft2.clicked.connect(calc_irfft2)
ui.pushButton_dct.clicked.connect(calc_dctn)
ui.pushButton_idct.clicked.connect(calc_idctn)
ui.pushButton_log.clicked.connect(calc_log)
ui.pushButton_rang.clicked.connect(calc_rang)
ui.pushButton_add_window.clicked.connect(add_window)
ui.pushButton_add_layer.clicked.connect(add_layer)
ui.pushButton_add_formation.clicked.connect(add_formation)
ui.pushButton_remove_layer.clicked.connect(remove_layer)
ui.pushButton_edges_layer.clicked.connect(save_layer)
# ui.pushButton_find_oil.clicked.connect(show_globals)
ui.pushButton_add_well.clicked.connect(add_well)
ui.pushButton_edit_well.clicked.connect(edit_well)
ui.pushButton_add_wells.clicked.connect(add_wells)
ui.pushButton_add_bound.clicked.connect(add_boundary)
ui.pushButton_rem_bound.clicked.connect(remove_boundary)
ui.pushButton_color.clicked.connect(change_color)
ui.pushButton_add_lda.clicked.connect(add_lda)
ui.pushButton_add_mark_lda.clicked.connect(add_marker_lda)
ui.pushButton_add_well_lda.clicked.connect(add_well_markup_lda)


ui.toolButton_add_obj.clicked.connect(add_object)
ui.toolButton_load_prof.clicked.connect(load_profile)
ui.toolButton_del_prof.clicked.connect(delete_profile)
ui.toolButton_load_plast.clicked.connect(load_param)
ui.toolButton_del_plast.clicked.connect(remove_formation)
ui.toolButton_crop_up.clicked.connect(crop_up)
ui.toolButton_crop_down.clicked.connect(crop_down)


ui.comboBox_object.activated.connect(update_research_combobox)
ui.comboBox_research.activated.connect(update_profile_combobox)
ui.comboBox_profile.activated.connect(update_formation_combobox)
ui.comboBox_plast.activated.connect(update_param_combobox)
ui.comboBox_plast.activated.connect(draw_formation)
ui.comboBox_param_plast.activated.connect(draw_param)
ui.comboBox_lda_analysis.activated.connect(update_list_marker_lda)

ui.checkBox_minmax.stateChanged.connect(choose_minmax)
ui.checkBox_draw_layer.stateChanged.connect(draw_layers)
ui.checkBox_all_formation.stateChanged.connect(draw_param)
ui.checkBox_profile_well.stateChanged.connect(update_list_well)


ui.spinBox_ftt_up.valueChanged.connect(draw_fft_spectr)
ui.spinBox_fft_down.valueChanged.connect(draw_fft_spectr)
ui.spinBox_roi.valueChanged.connect(changeSpinBox)
ui.spinBox_rad_up.valueChanged.connect(draw_rad_line)
ui.spinBox_rad_down.valueChanged.connect(draw_rad_line)
ui.spinBox_well_distance.valueChanged.connect(update_list_well)
ui.doubleSpinBox_vmin.valueChanged.connect(draw_bound_int)
ui.doubleSpinBox_vmax.valueChanged.connect(draw_bound_int)
ui.doubleSpinBox_vsr.valueChanged.connect(draw_wells)


ui.listWidget_well.currentItemChanged.connect(show_data_well)
ui.listWidget_well.currentItemChanged.connect(update_boundaries)
ui.listWidget_bound.currentItemChanged.connect(draw_bound_int)
ui.listWidget_well_lda.currentItemChanged.connect(choose_marker_lda)

roi.sigRegionChanged.connect(updatePlot)

ui.pushButton_find_oil.clicked.connect(filter19)

update_object()
clear_current_profile()
clear_current_profile_min_max()
clear_spectr()
clear_window_profile()
# update_layers()
update_list_well()
set_info('Старт...', 'green')
set_random_color()
update_list_lda()




sys.excepthook = log_uncaught_exceptions

sys.exit(app.exec_())