import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")
from PyQt5 import QtWidgets

from qt.classification_well_wizard import ClassificationWellCandidate, ClassificationWellWizard


class _Query:
    def order_by(self, *_args):
        return self

    def all(self):
        return []


class _Session:
    def query(self, *_args):
        return _Query()


@pytest.fixture(scope="module", autouse=True)
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_class_and_skip_choices_are_mutually_exclusive():
    candidate = ClassificationWellCandidate(1, "W-1", 2, "P-1", 3, 12.5, [4, 5])
    markers = [SimpleNamespace(id=10, title="Class A"), SimpleNamespace(id=20, title="Class B")]
    dialog = ClassificationWellWizard(_Session(), [candidate], markers)

    assert dialog.assignments() == [(candidate, None)]
    candidate._button_group.button(0).setChecked(True)
    assert dialog.assignments() == [(candidate, 10)]
    assert not candidate._button_group.button(2).isChecked()


def test_parameter_columns_precede_three_decision_columns():
    candidate = ClassificationWellCandidate(1, "W-1", 2, "P-1", 3, 0, [])
    markers = [SimpleNamespace(id=10, title="Class A"), SimpleNamespace(id=20, title="Class B")]
    dialog = ClassificationWellWizard(_Session(), [candidate], markers, mode="check")

    headers = [dialog.table.horizontalHeaderItem(i).text() for i in range(dialog.table.columnCount())]
    assert headers[-3:] == ["Class A", "Class B", "Не добавлять"]
    assert headers[:4] == ["Скважина", "Профиль", "Расстояние", "Пласт ID"]
