"""Dictionary CRUD, draft preview and importer integration through Qt widgets."""

import os
from copy import deepcopy

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")
from PyQt5 import QtCore, QtTest, QtWidgets

from core_description import semantics
from qt.core_description_dictionary import CoreDescriptionDictionaryDialog, DictionaryRuleDialog


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def dictionary_file(tmp_path, monkeypatch):
    target = tmp_path / "dictionary.json"
    monkeypatch.setattr(semantics, "user_dictionary_path", lambda: target)
    return target


def test_words_can_be_added_edited_and_moved_between_sections(app, monkeypatch, dictionary_file):
    dialog = CoreDescriptionDictionaryDialog()
    dialog.category_combo.setCurrentIndex(dialog.category_combo.findData("rock"))
    original_count = len(dialog.payload["rules"])

    def add(editor):
        assert editor.category_combo.currentData() == "rock"
        editor.value_combo.setEditText("Глауконитит")
        editor.patterns_edit.setPlainText("глауконитит\nглауконитита\nглауконитит")
        editor.accept()
        return editor.result()

    monkeypatch.setattr(DictionaryRuleDialog, "exec_", add)
    dialog.add_button.click()
    assert len(dialog.payload["rules"]) == original_count + 1
    rule = dialog.payload["rules"][-1]
    assert rule["patterns"] == ["глауконитит", "глауконитита"]
    assert dialog._selected_id() == rule["id"]
    assert dialog.save_button.isEnabled()
    assert not dictionary_file.exists()
    dialog.test_edit.setPlainText("Глауконитит")
    dialog.test_button.click()
    assert "Глауконитит (основная)" in dialog.test_result.text()

    def edit(editor):
        editor.category_combo.setCurrentIndex(editor.category_combo.findData("uncertainty"))
        editor.value_combo.setCurrentText("Неопределённость")
        editor.patterns_edit.setPlainText("допускается\nпредполагается")
        editor.accept()
        return editor.result()

    monkeypatch.setattr(DictionaryRuleDialog, "exec_", edit)
    dialog.edit_button.click()
    assert dialog.category_combo.currentData() == "uncertainty"
    assert dialog.payload["rules"][-1]["category"] == "uncertainty"
    assert dialog.payload["rules"][-1]["canonical_value"] == "uncertain"
    assert dialog._selected_id() == rule["id"]
    dialog.search_edit.setText("предполагается")
    assert dialog.table.rowCount() == 1
    dialog.search_edit.setText("несуществующее слово")
    assert dialog.table.rowCount() == 0
    assert not dialog.delete_button.isEnabled()


def test_edit_existing_rule_preserves_id_comment_and_canonical_code(app):
    rule = deepcopy(semantics.load_dictionary_payload()["rules"][0])
    rule["comment"] = "Проверено"
    editor = DictionaryRuleDialog(rule)
    assert editor.value_combo.currentText() == "Песчаник"
    editor.patterns_edit.setPlainText("песчаник\nпесчаники\nновая словоформа")
    editor.enabled_check.setChecked(False)
    editor.accept()
    assert editor.result() == QtWidgets.QDialog.Accepted
    assert editor.rule["id"] == rule["id"]
    assert editor.rule["comment"] == "Проверено"
    assert editor.rule["canonical_value"] == "sandstone"
    assert not editor.rule["enabled"]
    assert len(rule["patterns"]) != len(editor.rule["patterns"])


@pytest.mark.parametrize("value, pattern, pattern_type", [("", "порода", "word"), ("Порода", " ", "word"), ("Порода", "[", "regex")])
def test_invalid_rule_stays_open_without_mutating_original(app, value, pattern, pattern_type):
    editor = DictionaryRuleDialog()
    before = deepcopy(editor.rule)
    editor.value_combo.setEditText(value)
    editor.patterns_edit.setPlainText(pattern)
    editor.pattern_type.setCurrentIndex(editor.pattern_type.findData(pattern_type))
    editor.accept()
    assert editor.result() != QtWidgets.QDialog.Accepted
    assert editor.error_label.text()
    assert editor.rule == before


def _answer_dialog(monkeypatch, answer):
    def respond(question):
        next(button for button in question.buttons() if button.text() == answer).click()
        return 0
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec_", respond)


def test_delete_rule_can_be_cancelled_and_committed_then_survives_reopening(app, monkeypatch):
    dialog = CoreDescriptionDictionaryDialog()
    dialog.table.setCurrentCell(0, 0)
    removed = dialog._selected_id()
    _answer_dialog(monkeypatch, "Отмена")
    dialog.delete_button.click()
    assert any(rule["id"] == removed for rule in dialog.payload["rules"])
    _answer_dialog(monkeypatch, "Удалить")
    dialog.delete_button.click()
    assert not any(rule["id"] == removed for rule in dialog.payload["rules"])
    signal = QtTest.QSignalSpy(dialog.dictionary_saved)
    dialog.save_button.click()
    assert len(signal) == 1
    assert not dialog.save_button.isEnabled()
    reopened = CoreDescriptionDictionaryDialog()
    assert not any(rule["id"] == removed for rule in reopened.payload["rules"])
    assert reopened.payload["version"] == signal[0][0]


@pytest.mark.parametrize("answer, saved, visible", [("Отмена", False, True), ("Не сохранять", False, False), ("Сохранить", True, False)])
def test_closing_dirty_window_respects_save_discard_cancel(app, monkeypatch, dictionary_file, answer, saved, visible):
    dialog = CoreDescriptionDictionaryDialog()
    dialog.payload["rules"][0]["patterns"].append("новая словоформа")
    dialog._update_dirty()
    dialog.show()
    _answer_dialog(monkeypatch, answer)
    dialog.close()
    assert dictionary_file.exists() is saved
    assert dialog.isVisible() is visible
    if visible:
        dialog._update_dirty(False)
        dialog.close()
    if not saved:
        assert "новая словоформа" not in semantics.load_dictionary_payload()["rules"][0]["patterns"]


def test_save_failure_keeps_changes_and_prevents_close(app, monkeypatch):
    import qt.core_description_dictionary as module
    dialog = CoreDescriptionDictionaryDialog()
    dialog._update_dirty()

    def fail_save(_payload):
        raise OSError("No space")

    monkeypatch.setattr(module, "save_dictionary", fail_save)
    _answer_dialog(monkeypatch, "Сохранить")
    dialog.show()
    dialog.close()
    assert dialog.isVisible()
    assert dialog._dirty
    assert "No space" in dialog.status_label.text()
    assert dialog.save_button.isEnabled()
    dialog._update_dirty(False)
    dialog.close()


def test_importer_opens_separate_editor_and_reports_saved_changes(app, monkeypatch):
    pytest.importorskip("sqlalchemy")
    from qt.core_description_import import CoreDescriptionImportDialog

    def edit(editor):
        assert isinstance(editor.parent(), CoreDescriptionImportDialog)
        assert editor.isWindow()
        assert editor.windowModality() == QtCore.Qt.WindowModal
        editor.payload["rules"][0]["patterns"].append("новая словоформа")
        editor._update_dirty()
        assert editor.save()
        return QtWidgets.QDialog.Rejected

    monkeypatch.setattr(CoreDescriptionDictionaryDialog, "exec_", edit)
    importer = CoreDescriptionImportDialog(None)
    importer.dictionary_button.click()
    assert "Словари сохранены" in importer.dictionary_notice.text()
    assert "новая словоформа" in semantics.load_dictionary_payload()["rules"][0]["patterns"]
    importer._thread = object()
    monkeypatch.setattr(CoreDescriptionDictionaryDialog, "exec_", lambda _self: pytest.fail("Opened during parsing"))
    importer.edit_dictionary()
    importer._thread = None
