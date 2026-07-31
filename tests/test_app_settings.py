import os

import pytest

pytest.importorskip("PyQt5")
from PyQt5 import QtCore, QtWidgets

import app_settings


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path):
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(tmp_path))
    app_settings.settings().clear()


def test_form_values_round_trip(app):
    dialog = QtWidgets.QDialog()
    layout = QtWidgets.QVBoxLayout(dialog)
    spin = QtWidgets.QSpinBox(objectName="spinBox_count")
    check = QtWidgets.QCheckBox(objectName="checkBox_enabled")
    combo = QtWidgets.QComboBox(objectName="comboBox_method")
    combo.addItems(["first", "second"])
    line = QtWidgets.QLineEdit(objectName="lineEdit_value")
    for widget in (spin, check, combo, line):
        layout.addWidget(widget)

    spin.setValue(7)
    check.setChecked(True)
    combo.setCurrentText("second")
    line.setText("saved")
    app_settings.save_form(dialog, "test")

    spin.setValue(1)
    check.setChecked(False)
    combo.setCurrentText("first")
    line.clear()
    app_settings.restore_form(dialog, "test")

    assert (spin.value(), check.isChecked()) == (7, True)
    assert combo.currentText() == "second"
    assert line.text() == "saved"


def test_dialog_directory_preserves_save_filename(app, tmp_path):
    app_settings.settings().setValue(app_settings._DIALOG_KEY, str(tmp_path))

    assert app_settings._dialog_directory("result.xlsx", save=True) == str(tmp_path / "result.xlsx")
    assert app_settings._dialog_directory("ignored", save=False) == str(tmp_path)


def test_name_filter_limits_persisted_widgets(app):
    root = QtWidgets.QWidget()
    kept = QtWidgets.QSpinBox(root, objectName="spinBox_cluster_count")
    skipped = QtWidgets.QSpinBox(root, objectName="spinBox_other")
    kept.setValue(4)
    skipped.setValue(9)

    app_settings.save_form(root, "filtered", lambda name: "cluster" in name)

    store = app_settings.settings()
    assert store.contains("forms/filtered/spinBox_cluster_count")
    assert not store.contains("forms/filtered/spinBox_other")
