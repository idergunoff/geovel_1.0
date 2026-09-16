import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")
from PyQt5 import QtCore, QtWidgets

import app_settings
import qt.classification_well_wizard as wizard_module
from qt.classification_well_wizard import ClassificationWellCandidate, ClassificationWellWizard
from regression_target.service import Resolution, ResolutionCandidate, TargetSettings


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


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path):
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(tmp_path))
    app_settings.settings().clear()


def test_class_and_skip_choices_are_mutually_exclusive():
    candidate = ClassificationWellCandidate(1, "W-1", 2, "P-1", 3, 12.5, [4, 5])
    markers = [SimpleNamespace(id=10, title="Class A"), SimpleNamespace(id=20, title="Class B")]
    dialog = ClassificationWellWizard(_Session(), [candidate], markers)

    assert dialog.assignments() == [(candidate, None)]
    candidate._button_group.button(0).setChecked(True)
    assert dialog.assignments() == [(candidate, 10)]
    assert not candidate._button_group.button(2).isChecked()


def test_parameter_columns_precede_three_decision_columns():
    candidate = ClassificationWellCandidate(1, "W-1", 2, "P-1", 3, 0, [], object_name="Object 1")
    markers = [SimpleNamespace(id=10, title="Class A"), SimpleNamespace(id=20, title="Class B")]
    dialog = ClassificationWellWizard(_Session(), [candidate], markers, mode="check")

    headers = [dialog.table.horizontalHeaderItem(i).text() for i in range(dialog.table.columnCount())]
    assert headers[-3:] == ["Class A", "Class B", "Не добавлять"]
    assert headers[:5] == ["Скважина", "Профиль", "Объект", "Расстояние", "Пласт ID"]
    assert headers[5] == "Уже в классификации"
    assert dialog.table.horizontalHeaderItem(3).toolTip() == "Расстояние до скважины от профиля"
    assert dialog.table.item(0, 2).text() == "Object 1"


def test_existing_markup_is_visibly_marked_with_its_class():
    candidate = ClassificationWellCandidate(
        1, "W-1", 2, "P-1", 3, 0, [], markup_id=42, current_marker_id=20)
    markers = [SimpleNamespace(id=10, title="Class A"), SimpleNamespace(id=20, title="Class B")]

    dialog = ClassificationWellWizard(_Session(), [candidate], markers)

    status = dialog.table.item(0, 5)
    assert status.text() == "Да — Class B"
    assert "уже добавлена" in status.toolTip()
    assert status.background().color().name() == "#d9f2df"
    assert dialog.assignments() == [(candidate, 20)]


def test_class_choice_survives_table_rerender():
    candidate = ClassificationWellCandidate(1, "W-1", 2, "P-1", 3, 0, [])
    markers = [SimpleNamespace(id=10, title="A"), SimpleNamespace(id=20, title="B")]
    dialog = ClassificationWellWizard(_Session(), [candidate], markers)
    candidate._button_group.button(1).setChecked(True)

    dialog._render()

    assert dialog.assignments() == [(candidate, 20)]


def test_double_click_resolves_multiple_source_values(monkeypatch):
    candidate = ClassificationWellCandidate(1, "W-1", 2, "P-1", 3, 0, [])
    candidate.values = [Resolution("ambiguous", candidates=[
        ResolutionCandidate(11.0, 1, "first", "11"),
        ResolutionCandidate(22.0, 2, "second", "22"),
    ])]
    markers = [SimpleNamespace(id=10, title="A"), SimpleNamespace(id=20, title="B")]
    dialog = ClassificationWellWizard(_Session(), [candidate], markers)
    dialog.parameters = [("Depth", TargetSettings("boundary", 1))]
    dialog._render()
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem",
                        lambda *_args: ("second: 22 → 22", True))

    dialog._resolve_ambiguity(0, 6)

    assert candidate.values[0].status == "resolved"
    assert candidate.values[0].value == 22.0


def test_parameter_header_uses_canonical_name_and_detailed_tooltip(monkeypatch):
    class Target:
        canonical_name = "GR"
        id = 4

    monkeypatch.setattr(wizard_module, "list_canonical_targets",
                        lambda _session, source: [Target()] if source == "well_log" else [])
    monkeypatch.setattr(wizard_module, "resolve_target",
                        lambda *_args, **_kwargs: Resolution("resolved", 42.0))
    candidate = ClassificationWellCandidate(1, "W-1", 2, "P-1", 3, 0, [])
    markers = [SimpleNamespace(id=10, title="A"), SimpleNamespace(id=20, title="B")]
    dialog = ClassificationWellWizard(_Session(), [candidate], markers)
    dialog.source.setCurrentIndex(dialog.source.findData("well_log"))
    dialog.aggregation.setCurrentIndex(dialog.aggregation.findData("max"))

    dialog._add_parameter()

    header = dialog.table.horizontalHeaderItem(6)
    assert header.text() == "GR"
    assert "Параметр: GR (каноническое название)" in header.toolTip()
    assert "Источник: каротажная кривая" in header.toolTip()
    assert "Агрегация отсчётов кривой: максимум" in header.toolTip()
    assert "Интервал:" in header.toolTip()
    dialog.close()


def test_source_choice_is_applied_to_same_parameter_columns(monkeypatch):
    candidate = ClassificationWellCandidate(1, "W-1", 2, "P-1", 3, 0, [])
    choices = [
        ResolutionCandidate(11.0, 1, "first", "11"),
        ResolutionCandidate(22.0, 2, "second", "22"),
    ]
    candidate.values = [
        Resolution("ambiguous", candidates=choices.copy()),
        Resolution("ambiguous", candidates=choices.copy()),
        Resolution("ambiguous", candidates=choices.copy()),
    ]
    markers = [SimpleNamespace(id=10, title="A"), SimpleNamespace(id=20, title="B")]
    dialog = ClassificationWellWizard(_Session(), [candidate], markers)
    dialog.parameters = [
        ("Depth 1", TargetSettings("boundary", 1)),
        ("Depth 2", TargetSettings("boundary", 1, aggregation="mean")),
        ("Other", TargetSettings("boundary", 2)),
    ]
    dialog._render()
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem",
                        lambda *_args: ("second: 22 → 22", True))

    dialog._resolve_ambiguity(0, 6)

    assert candidate.values[0].value == 22.0
    assert candidate.values[1].value == 22.0
    assert candidate.values[1].details["selected"]["source_id"] == 2
    assert candidate.values[2].status == "ambiguous"


def test_boundary_choice_is_reused_for_dependent_log_columns(monkeypatch):
    candidate = ClassificationWellCandidate(1, "W-1", 2, "P-1", 3, 0, [])
    choices = [
        ResolutionCandidate(100.0, 5, "top", "100"),
        ResolutionCandidate(200.0, 6, "bottom", "200"),
    ]
    candidate.values = [
        Resolution("ambiguous", candidates=choices.copy(),
                   details={"pending_selection": "boundary_depth"}),
        Resolution("ambiguous", candidates=choices.copy(),
                   details={"pending_selection": "boundary_depth"}),
    ]
    markers = [SimpleNamespace(id=10, title="A"), SimpleNamespace(id=20, title="B")]
    dialog = ClassificationWellWizard(_Session(), [candidate], markers)
    dialog.parameters = [
        ("GR", TargetSettings("well_log", 1, boundary_canonical_id=9)),
        ("SP", TargetSettings("well_log", 2, boundary_canonical_id=9)),
    ]
    dialog._render()
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem",
                        lambda *_args: ("bottom: 200 → 200", True))
    calls = []

    def resolve(_session, well_id, settings, *, boundary_candidate):
        calls.append((well_id, settings.canonical_id, boundary_candidate.source_id))
        return Resolution("resolved", float(settings.canonical_id))

    monkeypatch.setattr(wizard_module, "resolve_target", resolve)

    dialog._resolve_ambiguity(0, 6)

    assert calls == [(1, 1, 6), (1, 2, 6)]
    assert [value.status for value in candidate.values] == ["resolved", "resolved"]


def test_open_log_uses_selected_parameter_details():
    opened = []
    candidate = ClassificationWellCandidate(7, "W-7", 2, "P-1", 3, 0, [])
    candidate.values = [Resolution("resolved", 5.0, details={"selected": {"details": {"well_log_id": 9}}})]
    markers = [SimpleNamespace(id=10, title="A"), SimpleNamespace(id=20, title="B")]
    dialog = ClassificationWellWizard(
        _Session(), [candidate], markers, open_well_log=lambda well_id, details: opened.append((well_id, details)))
    dialog.parameters = [("Log", TargetSettings("well_log", 1))]
    dialog._render()
    dialog.table.setCurrentCell(0, 6)

    dialog._open_well_log()

    assert opened == [(7, candidate.values[0].details)]


def test_input_settings_are_restored_for_each_window_mode(application, monkeypatch):
    class Target:
        def __init__(self, name, target_id):
            self.canonical_name = name
            self.id = target_id

    targets = {
        "boundary": [Target("Кровля", 1)],
        "well_data": [Target("Температура", 2), Target("Давление", 3)],
        "well_log": [Target("GR", 4)],
    }
    monkeypatch.setattr(wizard_module, "list_canonical_targets",
                        lambda _session, source: targets[source])
    markers = [SimpleNamespace(id=10, title="A"), SimpleNamespace(id=20, title="B")]

    check_dialog = ClassificationWellWizard(_Session(), [], markers, mode="check")
    check_dialog.source.setCurrentIndex(check_dialog.source.findData("well_log"))
    check_dialog.canonical.setCurrentText("GR")
    check_dialog.aggregation.setCurrentIndex(check_dialog.aggregation.findData("mean"))
    check_dialog.operation.setCurrentIndex(check_dialog.operation.findData("difference"))
    check_dialog.depth_mode.setCurrentIndex(check_dialog.depth_mode.findData("fixed"))
    check_dialog.fixed_depth.setValue(1234.5)
    check_dialog.interval.setValue(17.25)
    check_dialog.position.setCurrentIndex(check_dialog.position.findData("centered"))
    check_dialog.show()
    check_dialog.close()
    application.processEvents()

    add_dialog = ClassificationWellWizard(_Session(), [], markers, mode="add")
    assert add_dialog.source.currentData() == "boundary"
    add_dialog.source.setCurrentIndex(add_dialog.source.findData("well_data"))
    add_dialog.canonical.setCurrentText("Давление")
    add_dialog.strict.setChecked(True)
    add_dialog.explicit_sum.setChecked(False)
    add_dialog.show()
    add_dialog.close()
    application.processEvents()

    restored_check = ClassificationWellWizard(_Session(), [], markers, mode="check")
    assert restored_check.source.currentData() == "well_log"
    assert restored_check.canonical.currentText() == "GR"
    assert restored_check.aggregation.currentData() == "mean"
    assert restored_check.operation.currentData() == "difference"
    assert restored_check.depth_mode.currentData() == "fixed"
    assert restored_check.fixed_depth.value() == 1234.5
    assert restored_check.interval.value() == 17.25
    assert restored_check.position.currentData() == "centered"

    restored_add = ClassificationWellWizard(_Session(), [], markers, mode="add")
    assert restored_add.source.currentData() == "well_data"
    assert restored_add.canonical.currentText() == "Давление"
    assert restored_add.strict.isChecked()
    assert not restored_add.explicit_sum.isChecked()
    restored_check.close()
    restored_add.close()
