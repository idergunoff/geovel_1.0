import traceback

from object import *
from load import *


MainWindow.show()


def log_uncaught_exceptions(ex_cls, ex, tb):
    """ Вывод ошибок программы """
    text = '{}: {}:\n'.format(ex_cls.__name__, ex)
    text += ''.join(traceback.format_tb(tb))
    print(text)
    QtWidgets.QMessageBox.critical(None, 'Error', text)
    sys.exit()


ui.pushButton_draw_rad.clicked.connect(draw_radarogram)
ui.pushButton_vacuum.clicked.connect(vacuum)

ui.toolButton_add_obj.clicked.connect(add_object)
ui.toolButton_load_prof.clicked.connect(load_profile)
ui.toolButton_del_prof.clicked.connect(delete_profile)
ui.toolButton_load_plast.clicked.connect(load_param)


ui.comboBox_object.activated.connect(update_profile_combobox)
# ui.comboBox_profile.activated.connect(update_param_combobox)

roi.sigRegionChanged.connect(updatePlot)


update_object()
clear_current_profile()


sys.excepthook = log_uncaught_exceptions

sys.exit(app.exec_())