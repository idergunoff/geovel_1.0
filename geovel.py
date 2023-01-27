import traceback

from object import *
from load import *
from filtering import *


MainWindow.show()


def log_uncaught_exceptions(ex_cls, ex, tb):
    """ Вывод ошибок программы """
    text = '{}: {}:\n'.format(ex_cls.__name__, ex)
    text += ''.join(traceback.format_tb(tb))
    print(text)
    QtWidgets.QMessageBox.critical(None, 'Error', text)
    sys.exit()


ui.pushButton_save_signal.clicked.connect(save_signal)
ui.pushButton_draw_rad.clicked.connect(draw_radarogram)
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


ui.toolButton_add_obj.clicked.connect(add_object)
ui.toolButton_load_prof.clicked.connect(load_profile)
ui.toolButton_del_prof.clicked.connect(delete_profile)
ui.toolButton_load_plast.clicked.connect(load_param)


ui.comboBox_object.activated.connect(update_profile_combobox)
ui.comboBox_profile.activated.connect(update_param_combobox)
ui.comboBox_param_plast.activated.connect(draw_param)

ui.spinBox_ftt_up.valueChanged.connect(draw_fft_spectr)
ui.spinBox_fft_down.valueChanged.connect(draw_fft_spectr)

roi.sigRegionChanged.connect(updatePlot)


update_object()
clear_current_profile()
clear_spectr()



sys.excepthook = log_uncaught_exceptions

sys.exit(app.exec_())