from PyQt5 import QtCore, QtWidgets


_open_windows = {}


def show_work_window(window):
    """Show a working window and keep its Python object alive until Qt deletes it."""
    key = id(window)
    if key not in _open_windows:
        _open_windows[key] = window
        window.destroyed.connect(lambda: _open_windows.pop(key, None))
    window.setWindowModality(QtCore.Qt.NonModal)
    window.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    window.setAttribute(QtCore.Qt.WA_QuitOnClose, False)
    window.show()


def ask_in_window(dialog, owner):
    """Collect an immediate answer while leaving other working windows available."""
    if dialog.parentWidget() is not owner:
        dialog.setParent(owner, dialog.windowFlags())
    dialog.setWindowModality(QtCore.Qt.WindowModal)
    return dialog.exec_()


def confirm_in_window(owner, title, text):
    box = QtWidgets.QMessageBox(
        QtWidgets.QMessageBox.Question, title, text,
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, owner,
    )
    box.setDefaultButton(QtWidgets.QMessageBox.No)
    return ask_in_window(box, owner) == QtWidgets.QMessageBox.Yes
