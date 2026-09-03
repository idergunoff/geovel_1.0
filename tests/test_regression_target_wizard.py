import os

import pytest

pytest.importorskip("PyQt5")
from PyQt5 import QtWidgets

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
