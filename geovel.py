import traceback

from object import *
from load import *
from filtering import *
from draw import *


MainWindow.show()


# def mouseMoved(evt):
#     """ Отслеживаем координаты курсора и прокрутка таблицы до выбранного курсором значения """
#     pos = evt[0]
#     vb = radarogramma.vb
#     if radarogramma.sceneBoundingRect().contains(pos):
#         mousePoint = vb.mapSceneToView(pos)
#         print(mousePoint.x(), mousePoint.y())


def log_uncaught_exceptions(ex_cls, ex, tb):
    """ Вывод ошибок программы """
    text = '{}: {}:\n'.format(ex_cls.__name__, ex)
    text += ''.join(traceback.format_tb(tb))
    print(text)
    QtWidgets.QMessageBox.critical(None, 'Error', text)
    sys.exit()

# proxy = pg.SignalProxy(radarogramma.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)

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


ui.toolButton_add_obj.clicked.connect(add_object)
ui.toolButton_load_prof.clicked.connect(load_profile)
ui.toolButton_del_prof.clicked.connect(delete_profile)
ui.toolButton_load_plast.clicked.connect(load_param)


ui.comboBox_object.activated.connect(update_profile_combobox)
ui.comboBox_profile.activated.connect(update_param_combobox)
ui.comboBox_param_plast.activated.connect(draw_param)

ui.checkBox_minmax.stateChanged.connect(choose_minmax)

ui.spinBox_ftt_up.valueChanged.connect(draw_fft_spectr)
ui.spinBox_fft_down.valueChanged.connect(draw_fft_spectr)
ui.spinBox_roi.valueChanged.connect(changeSpinBox)
ui.spinBox_rad_up.valueChanged.connect(draw_rad_line)
ui.spinBox_rad_down.valueChanged.connect(draw_rad_line)

roi.sigRegionChanged.connect(updatePlot)


update_object()
clear_current_profile()
clear_current_profile_min_max()
clear_spectr()
clear_window_profile()
set_info('Старт...', 'green')



sys.excepthook = log_uncaught_exceptions

sys.exit(app.exec_())