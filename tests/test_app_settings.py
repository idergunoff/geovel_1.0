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
    store.beginGroup(f"{app_settings._FORMS_GROUP}/filtered")
    assert store.contains(app_settings._widget_key(root, kept))
    assert not store.contains(app_settings._widget_key(root, skipped))
    store.endGroup()


def test_widgets_with_repeated_object_names_have_independent_values(app):
    root = QtWidgets.QWidget()
    left_group = QtWidgets.QGroupBox(root, objectName="left_settings")
    right_group = QtWidgets.QGroupBox(root, objectName="right_settings")
    left = QtWidgets.QSpinBox(left_group, objectName="spinBox_count")
    right = QtWidgets.QSpinBox(right_group, objectName="spinBox_count")
    left.setValue(3)
    right.setValue(17)

    app_settings.save_form(root, "duplicates")
    left.setValue(0)
    right.setValue(0)
    app_settings.restore_form(root, "duplicates")

    assert left.value() == 3
    assert right.value() == 17


def test_legacy_flat_values_are_not_restored(app):
    root = QtWidgets.QWidget()
    spin = QtWidgets.QSpinBox(root, objectName="spinBox_count")
    app_settings.settings().setValue("forms/test/spinBox_count", 99)

    app_settings.restore_form(root, "test")

    assert spin.value() == 0
