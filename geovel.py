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


ui.toolButton_add_obj.clicked.connect(add_object)
ui.toolButton_load_prof.clicked.connect(load_profile)


update_object()


sys.excepthook = log_uncaught_exceptions

sys.exit(app.exec_())