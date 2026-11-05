"""Literal learning and the complete selection → dictionary → reviewed reparse flow."""

from copy import deepcopy
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5 import QtCore, QtGui, QtWidgets
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core_description import semantics
from core_description.database import save_core_import
from core_description.dictionary_learning import build_dictionary_draft
from core_description.models import ParsedCoreDocument, ParsedCoreInterval
from core_description.reparse import preview_reparse, recognition_details
from models_db.model import Base, CoreDescriptionInterval, Well
import models_db.model_cluster
import models_db.model_profile_features
from qt.core_description_learning import LearnDescriptionDialog
from qt.core_description_reparse import ReparsePreviewDialog, ReparseWellScopeDialog
from qt.core_description_view import CoreDescriptionViewDialog


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def dictionary_file(tmp_path, monkeypatch):
    target = tmp_path / "dictionary.json"
    monkeypatch.setattr(semantics, "user_dictionary_path", lambda: target)
    return target


@pytest.mark.parametrize("category,value,expression,expected", [
    ("rock", "Глауконитит", "Глауконитит", "Глауконитит"),
    ("oil_indicator", "present", "пропитан нефтью", "present"),
    ("negation", "none", "углеводороды не обнаружены", "none"),
    ("oil_saturation", "weak", "нефть в малом количестве", "weak"),
])
def test_all_learning_categories_affect_actual_parser(category, value, expression, expected):
    original = semantics.load_dictionary_payload()
    untouched = deepcopy(original)
    draft = build_dictionary_draft(original, category=category, canonical_value=value,
                                   expressions=[expression, expression.upper()])
    assert original == untouched
    assert len(draft["rules"][-1]["patterns"]) == 1
    rocks, oil, _, matches, _ = semantics.analyze_description(expression, semantics.validate_dictionary(draft))
    assert (rocks[0].canonical_value if category == "rock" else oil) == expected
    assert any(m.matched_text == expression for m in matches)


def test_learning_checks_conflicts_and_deduplicates_overlapping_regex_and_literal_matches():
    payload = semantics.load_dictionary_payload()
    with pytest.raises(semantics.DictionaryValidationError, match="другому значению"):
        build_dictionary_draft(payload, category="rock", canonical_value="clay", expressions=["песчаник"])
    draft = build_dictionary_draft(payload, category="rock", canonical_value="sandstone", expressions=["песчаник"])
    rocks, _, _, matches, _ = semantics.analyze_description("Песчаник", semantics.validate_dictionary(draft))
    assert len(rocks) == 1
    assert len([m for m in matches if m.category == "rock"]) == 1
    phrase = build_dictionary_draft(payload, category="rock", canonical_value="sandstone", expressions=["песчаник плотный"])
    rocks, _, _, matches, _ = semantics.analyze_description("Песчаник плотный", semantics.validate_dictionary(phrase))
    assert len(rocks) == 1 and rocks[0].matched_text == "Песчаник плотный"


@pytest.mark.parametrize("expression", ["слабо", "сильно", "интенсивно", "средне", "слабая", "сильного"])
def test_ambiguous_degree_needs_oil_context(expression):
    with pytest.raises(semantics.DictionaryValidationError, match="вместе с признаком нефти"):
        build_dictionary_draft(semantics.load_dictionary_payload(), category="oil_saturation",
                               canonical_value="weak", expressions=[expression])


def test_literal_rules_preserve_word_boundaries_and_support_source_whitespace():
    draft = build_dictionary_draft(semantics.load_dictionary_payload(), category="oil_indicator", canonical_value="present",
                                   expressions=["пропитан нефтью"])
    dictionary = semantics.validate_dictionary(draft)
    assert semantics.analyze_description("пропитан\n   нефтью", dictionary)[1] == "present"
    assert semantics.analyze_description("непропитан нефтью", dictionary)[1] == "unknown"


def test_learning_dialog_previews_without_writing_and_saves_only_on_accept(app, dictionary_file):
    dialog = LearnDescriptionDialog("глауконитит", "Глауконитит, плотный")
    assert not dialog.save_button.isEnabled()
    dialog.value_combo.setEditText("Глауконитит")
    assert "Глауконитит" in dialog.after_label.text()
    assert not dictionary_file.exists()
    dialog.accept()
    assert dialog.result() == QtWidgets.QDialog.Accepted
    assert semantics.analyze_description("Глауконитит")[0][0].canonical_value == "Глауконитит"


def test_existing_literal_rule_is_opened_with_all_forms_and_stale_dictionary_is_rejected(app):
    payload = build_dictionary_draft(semantics.load_dictionary_payload(), category="rock", canonical_value="Глауконитит",
                                     expressions=["глауконитит", "глауконитита"])
    semantics.save_dictionary(payload)
    dialog = LearnDescriptionDialog("ГЛАУКОНИТИТ", "Глауконитит")
    assert dialog.rule_id
    assert "глауконитита" in dialog.patterns_edit.toPlainText()
    count = len(dialog.payload["rules"])
    assert len(dialog.draft["rules"]) == count
    payload["rules"][-1]["comment"] = "Изменение в другом окне"
    semantics.save_dictionary(payload)
    dialog.accept()
    assert dialog.result() != QtWidgets.QDialog.Accepted
    assert "другом окне" in dialog.error_label.text()


def test_save_failure_keeps_learning_dialog_open(app, monkeypatch):
    import qt.core_description_learning as module
    dialog = LearnDescriptionDialog("глауконитит", "Глауконитит")
    dialog.value_combo.setEditText("Глауконитит")

    def fail(_payload):
        raise OSError("no space")

    monkeypatch.setattr(module, "save_dictionary", fail)
    dialog.accept()
    assert dialog.result() != QtWidgets.QDialog.Accepted
    assert "no space" in dialog.error_label.text()


@pytest.fixture()
def dataset():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    well = Well(name="12")
    db.add(well)
    db.commit()
    document = ParsedCoreDocument(
        source_path="source.docx", source_format="docx", file_hash="a" * 64,
        well_name_raw="12", well_name="12", area_name_raw=None, area_name=None,
        described_by_raw=None, described_by=None,
        intervals=[ParsedCoreInterval(100, 101, "🪨 Глауконитит, плотный", 0, 1),
                   ParsedCoreInterval(101, 102, "Песчаник", 0, 2)])
    save_core_import(db, document, well_id=well.id, selected_interval_ids={"0:1", "0:2"})
    yield db, well
    db.close()
    engine.dispose()


def test_selection_learning_and_reviewed_reparse_refresh_view_and_record_history(app, dataset, monkeypatch):
    db, well = dataset
    viewer = CoreDescriptionViewDialog(db, well)
    assert viewer.source_text.toPlainText() == "🪨 Глауконитит, плотный"
    cursor = viewer.source_text.textCursor()
    cursor.setPosition(3)  # UTF-16: emoji occupies two positions, then a space.
    cursor.setPosition(14, QtGui.QTextCursor.KeepAnchor)
    viewer.source_text.setTextCursor(cursor)
    assert viewer.learn_button.isEnabled()

    def learn(dialog):
        assert "Глауконитит" in dialog.patterns_edit.toPlainText()
        dialog.value_combo.setEditText("Глауконитит")
        dialog.accept()
        return dialog.result()

    def apply(dialog):
        first = dialog.preview.proposals[0]
        assert first.text.startswith("🪨")
        assert "Глауконитит" in dialog.table.item(0, 4).text()
        dialog.accept()
        return dialog.result()

    monkeypatch.setattr(LearnDescriptionDialog, "exec_", learn)
    monkeypatch.setattr(ReparsePreviewDialog, "exec_", apply)
    viewer.learn_button.click()
    assert viewer.table.item(0, 4).text() == "Глауконитит"
    row = db.query(CoreDescriptionInterval).filter_by(top_depth=100).one()
    assert recognition_details(row)["history"][0]["action"] == "reparse"
    assert row.rocks[0].match_start == 2  # Parser positions are Unicode code points.
    viewer.close()


def test_preview_cancel_does_not_write_and_hidden_or_protected_rows_are_not_applied(app, dataset):
    db, well = dataset
    rows = db.query(CoreDescriptionInterval).order_by(CoreDescriptionInterval.top_depth).all()
    rows[0].manually_edited = True
    db.commit()
    preview = preview_reparse(db, well_ids=[well.id])
    dialog = ReparsePreviewDialog(db, preview)
    assert rows[0].id not in dialog.selected_ids()
    dialog.allow_manual.setChecked(True)
    dialog._select(QtCore.Qt.Checked)
    assert rows[0].id in dialog.selected_ids()
    dialog.allow_manual.setChecked(False)
    assert rows[0].id not in dialog.selected_ids()
    dialog.reject()
    assert not recognition_details(rows[1]).get("history")
    applied = ReparsePreviewDialog(db, preview)
    applied.accept()
    fresh = preview_reparse(db, well_ids=[well.id])
    filtered = ReparsePreviewDialog(db, fresh, changed_only=True)
    assert rows[1].id not in filtered.selected_ids()
    assert not filtered.apply_button.isEnabled()


def test_learning_cancel_leaves_dictionary_and_database_untouched(app, dataset, dictionary_file, monkeypatch):
    db, well = dataset
    viewer = CoreDescriptionViewDialog(db, well)
    viewer.source_text.selectAll()
    monkeypatch.setattr(LearnDescriptionDialog, "exec_", lambda _dialog: QtWidgets.QDialog.Rejected)
    viewer.learn_selection()
    assert not dictionary_file.exists()
    assert all(not recognition_details(row).get("history") for row in db.query(CoreDescriptionInterval))
    viewer.close()


def test_well_scope_requires_explicit_selection_and_search_keeps_checked_wells(app, dataset):
    db, well = dataset
    scope = ReparseWellScopeDialog(db, well.id)
    assert scope.selected_well_ids() == {well.id}
    scope.search.setText("не существующая скважина")
    assert scope.selected_well_ids() == {well.id}
    scope.clear_all.click()
    assert not scope.selected_well_ids() and not scope.ok_button.isEnabled()
    scope.accept()
    assert scope.result() != QtWidgets.QDialog.Accepted
    scope.select_all.click()
    assert scope.selected_well_ids() == {well.id}


def test_history_window_undo_restores_table_and_updates_other_open_viewers(app, dataset, monkeypatch):
    from core_description.reparse import apply_reparse
    db, well = dataset
    parent = QtWidgets.QWidget()
    first = CoreDescriptionViewDialog(db, well, parent)
    second = CoreDescriptionViewDialog(db, well, parent)
    preview = preview_reparse(db, well_ids=[well.id])
    apply_reparse(db, preview, {p.interval_id for p in preview.proposals})
    first._changed({well.id})
    first.table.setCurrentCell(1, 0)
    second.table.setCurrentCell(1, 0)
    assert first.table.item(1, 4).text() == second.table.item(1, 4).text() == "Песчаник"

    def undo(dialog):
        button = next(b for b in dialog.findChildren(QtWidgets.QPushButton) if b.text() == "Отменить последний переразбор")
        button.click()
        return dialog.result()

    monkeypatch.setattr(QtWidgets.QDialog, "exec_", undo)
    first.show_history()
    assert first.table.item(1, 4).text() == second.table.item(1, 4).text() == "Не определена"
    parent.close()
