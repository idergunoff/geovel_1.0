"""Persistent UI preferences shared by GeoVel windows.

The module deliberately works with widget object names, so adding a new input to
one of the supported forms does not require maintaining another settings list.
"""

import os

from PyQt5 import QtCore, QtWidgets


ORGANIZATION = "GeoVel"
APPLICATION = "GeoVel"
_DIALOG_KEY = "files/last_directory"
_FORMS_GROUP = "forms_v2"


def settings():
    return QtCore.QSettings(ORGANIZATION, APPLICATION)


def _editable_widgets(root):
    types = (
        QtWidgets.QAbstractButton,
        QtWidgets.QComboBox,
        QtWidgets.QSpinBox,
        QtWidgets.QDoubleSpinBox,
        QtWidgets.QLineEdit,
        QtWidgets.QTabWidget,
    )
    widgets = []
    for widget_type in types:
        widgets.extend(root.findChildren(widget_type))
    # A subclass can be returned for more than one requested base class.
    return (widget for widget in dict.fromkeys(widgets) if widget.objectName())


def _widget_value(widget):
    if isinstance(widget, QtWidgets.QAbstractButton) and widget.isCheckable():
        return widget.isChecked()
    if isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
        return widget.value()
    if isinstance(widget, QtWidgets.QComboBox):
        return widget.currentText()
    if isinstance(widget, QtWidgets.QLineEdit):
        return widget.text()
    if isinstance(widget, QtWidgets.QTabWidget):
        return widget.currentIndex()
    return None


def _widget_key(root, widget):
    """Return a stable, unambiguous key for a widget inside *root*.

    Object names are usually unique in a Qt Designer form, but that is not a
    Qt requirement.  Using only ``objectName`` made different spin boxes share
    one QSettings entry in forms containing repeated/nested controls.  Include
    the named parent hierarchy and the widget type to prevent such collisions.
    """
    parts = [f"{type(widget).__name__}:{widget.objectName()}"]
    parent = widget.parentWidget()
    while parent is not None and parent is not root:
        if parent.objectName():
            parts.append(f"{type(parent).__name__}:{parent.objectName()}")
        parent = parent.parentWidget()
    return "/".join(reversed(parts))


def _restore_widget(widget, value):
    if isinstance(widget, QtWidgets.QAbstractButton) and widget.isCheckable():
        widget.setChecked(str(value).lower() in ("1", "true", "yes"))
    elif isinstance(widget, QtWidgets.QSpinBox):
        widget.setValue(int(value))
    elif isinstance(widget, QtWidgets.QDoubleSpinBox):
        widget.setValue(float(value))
    elif isinstance(widget, QtWidgets.QComboBox):
        index = widget.findText(str(value))
        if index >= 0:
            widget.setCurrentIndex(index)
    elif isinstance(widget, QtWidgets.QLineEdit):
        widget.setText(str(value))
    elif isinstance(widget, QtWidgets.QTabWidget):
        widget.setCurrentIndex(int(value))


def save_form(root, group, name_filter=None):
    """Save editable child widgets of *root* under a stable settings group."""
    store = settings()
    store.beginGroup(f"{_FORMS_GROUP}/{group}")
    for widget in _editable_widgets(root):
        if name_filter is not None and not name_filter(widget.objectName()):
            continue
        value = _widget_value(widget)
        if value is not None:
            store.setValue(_widget_key(root, widget), value)
    store.endGroup()
    store.sync()


def restore_form(root, group, name_filter=None):
    """Restore values which are still valid for the current version of a form."""
    store = settings()
    store.beginGroup(f"{_FORMS_GROUP}/{group}")
    for widget in _editable_widgets(root):
        if name_filter is not None and not name_filter(widget.objectName()):
            continue
        key = _widget_key(root, widget)
        if not store.contains(key):
            continue
        blocker = QtCore.QSignalBlocker(widget)
        try:
            _restore_widget(widget, store.value(key))
        except (TypeError, ValueError):
            # Obsolete/corrupt values must not prevent a window from opening.
            pass
        finally:
            del blocker
    store.endGroup()


def bind_form(root, group, name_filter=None):
    """Restore a form now and save it whenever the window is closed."""
    restore_form(root, group, name_filter)
    root.finished.connect(lambda _result: save_form(root, group, name_filter))


def _dialog_directory(suggested, save=False):
    remembered = str(settings().value(_DIALOG_KEY, "") or "")
    if not remembered or not os.path.isdir(remembered):
        return suggested
    if save and suggested:
        basename = os.path.basename(os.path.normpath(str(suggested)))
        return os.path.join(remembered, basename)
    return remembered


def _remember_selection(selection):
    path = selection[0] if isinstance(selection, tuple) else selection
    if not path:
        return selection
    directory = path if os.path.isdir(path) else os.path.dirname(path)
    if directory:
        settings().setValue(_DIALOG_KEY, directory)
    return selection


def install_file_dialog_history():
    """Make all static open/save/directory dialogs reuse their last directory."""
    if getattr(QtWidgets.QFileDialog, "_geovel_history_installed", False):
        return

    def wrap(method_name, save=False):
        original = getattr(QtWidgets.QFileDialog, method_name)

        def remembered_dialog(*args, **kwargs):
            args = list(args)
            if "directory" in kwargs:
                kwargs["directory"] = _dialog_directory(kwargs["directory"], save)
            elif len(args) >= 3:
                args[2] = _dialog_directory(args[2], save)
            else:
                kwargs["directory"] = _dialog_directory("", save)
            return _remember_selection(original(*args, **kwargs))

        setattr(QtWidgets.QFileDialog, method_name, staticmethod(remembered_dialog))

    wrap("getOpenFileName")
    wrap("getOpenFileNames")
    wrap("getSaveFileName", save=True)
    wrap("getExistingDirectory")
    QtWidgets.QFileDialog._geovel_history_installed = True
