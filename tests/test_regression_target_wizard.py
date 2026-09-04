import os

import pytest

pytest.importorskip("PyQt5")
from PyQt5 import QtCore, QtWidgets

import app_settings
import qt.regression_target_wizard as wizard_module
from qt.regression_target_wizard import RegressionTargetWizard, WizardCandidate


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _EmptyQuery:
    def order_by(self, *_args):
        return self

    def all(self):
        return []


class _Session:
    def query(self, *_args):
        return _EmptyQuery()


@pytest.fixture(scope="module")
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path):
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(tmp_path))
    app_settings.settings().clear()


def test_open_well_log_button_is_available_for_every_target_source(application, monkeypatch):
    monkeypatch.setattr(wizard_module, "list_canonical_targets", lambda *_args: [])
    candidate = WizardCandidate(1, "Скважина 1", 2, "Профиль 1", 3, 0.0, [])
    opened = []
    dialog = RegressionTargetWizard(
        _Session(), [candidate], open_well_log=lambda well_id, details: opened.append((well_id, details))
    )

    for index in range(dialog.source_combo.count()):
        dialog.source_combo.setCurrentIndex(index)
        assert not dialog.open_log_button.isHidden()

    dialog._render()
    dialog.table.selectRow(0)
    dialog.open_log_button.click()
    assert opened == [(candidate.well_id, {})]
    dialog.close()


def test_check_window_is_modeless_while_add_window_remains_modal(application, monkeypatch):
    monkeypatch.setattr(wizard_module, "list_canonical_targets", lambda *_args: [])
    candidate = WizardCandidate(1, "Скважина 1", 2, "Профиль 1", 3, 0.0, [])

    check_dialog = RegressionTargetWizard(_Session(), [candidate], mode="check")
    add_dialog = RegressionTargetWizard(_Session(), [candidate], mode="add")

    assert not check_dialog.isModal()
    assert add_dialog.isModal()

    check_dialog.close()
    add_dialog.close()


def test_all_target_options_are_restored_including_source_dependent_target(application, monkeypatch):
    class Target:
        def __init__(self, canonical_name, target_id):
            self.canonical_name = canonical_name
            self.id = target_id

    targets = {
        "boundary": [Target("Кровля", 1)],
        "well_data": [Target("Температура", 2), Target("Давление", 3)],
        "well_log": [Target("GR", 4)],
    }
    monkeypatch.setattr(wizard_module, "list_canonical_targets",
                        lambda _session, source: targets[source])
    candidate = WizardCandidate(1, "Скважина 1", 2, "Профиль 1", 3, 0.0, [])
    dialog = RegressionTargetWizard(_Session(), [candidate], mode="check")
    dialog.source_combo.setCurrentIndex(dialog.source_combo.findData("well_data"))
    dialog.canonical_combo.setCurrentText("Давление")
    dialog.strict_check.setChecked(True)
    dialog.sum_check.setChecked(False)
    dialog.hide_missing_check.setChecked(True)
    dialog.absolute_tolerance.setValue(2.75)
    dialog.aggregation_combo.setCurrentIndex(dialog.aggregation_combo.findData("mean"))
    dialog.operation_combo.setCurrentIndex(dialog.operation_combo.findData("difference"))
    dialog.depth_mode_combo.setCurrentIndex(dialog.depth_mode_combo.findData("fixed"))
    dialog.fixed_depth.setValue(1234.5)
    dialog.interval.setValue(17.25)
    dialog.position_combo.setCurrentIndex(dialog.position_combo.findData("centered"))
    dialog.show()
    dialog.close()
    application.processEvents()

    restored = RegressionTargetWizard(_Session(), [candidate], mode="check")

    assert restored.source_combo.currentData() == "well_data"
    assert restored.canonical_combo.currentText() == "Давление"
    assert restored.strict_check.isChecked()
    assert not restored.sum_check.isChecked()
    assert restored.hide_missing_check.isChecked()
    assert restored.absolute_tolerance.value() == 2.75
    assert restored.aggregation_combo.currentData() == "mean"
    assert restored.operation_combo.currentData() == "difference"
    assert restored.depth_mode_combo.currentData() == "fixed"
    assert restored.fixed_depth.value() == 1234.5
    assert restored.interval.value() == 17.25
    assert restored.position_combo.currentData() == "centered"
    restored.close()
