"""UI contract tests for the single-document core preview."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")
pytest.importorskip("sqlalchemy")

from PyQt5 import QtCore, QtWidgets
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core_description.models import ParsedCoreDocument, ParsedCoreInterval, RockMention
from models_db.model import Base, Well
import models_db.model_cluster  # noqa: E402,F401
import models_db.model_profile_features  # noqa: E402,F401
from qt.core_description_import import CoreDescriptionImportDialog


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add_all([Well(name="12"), Well(name="13")]); db.commit()
    yield db
    db.close()


def _document():
    return ParsedCoreDocument(
        source_path="core.docx", source_format="docx", file_hash="a" * 64,
        well_name_raw="12", well_name="12", area_name_raw="Северная", area_name="Северная",
        described_by_raw="Описал: Иванов И.И.", described_by="Иванов И.И.",
        intervals=[
            ParsedCoreInterval(100, 102, "Песчаник нефтенасыщенный", 0, 2,
                               rocks=[RockMention("sandstone", "Песчаник")],
                               oil_saturation="present", confidence=.95),
            ParsedCoreInterval(102, 103, "Неизвестная порода", 0, 3,
                               warnings=["unrecognized"], confidence=.4),
        ], dictionary_version="1.0")


def test_preview_matches_well_and_selects_only_confident_rows(app, session):
    dialog = CoreDescriptionImportDialog(session)
    dialog.set_document(_document())

    assert dialog.well_combo.currentData() is not None
    assert dialog.table.rowCount() == 2
    assert dialog.table.item(0, 0).checkState() == QtCore.Qt.Checked
    assert dialog.table.item(1, 0).checkState() == QtCore.Qt.Unchecked
    assert "интервалов: 1" in dialog.summary.text()
    assert dialog.save_button.isEnabled()


def test_filter_does_not_change_hidden_row_selection(app, session):
    dialog = CoreDescriptionImportDialog(session)
    dialog.set_document(_document())
    dialog.table.item(1, 0).setCheckState(QtCore.Qt.Checked)
    dialog.rock_filter.setText("sandstone")

    assert dialog.table.isRowHidden(1)
    dialog._select_visible(False)
    assert dialog.table.item(0, 0).checkState() == QtCore.Qt.Unchecked
    assert dialog.table.item(1, 0).checkState() == QtCore.Qt.Checked


def test_edited_values_are_passed_to_persistence(app, session, monkeypatch):
    dialog = CoreDescriptionImportDialog(session)
    dialog.set_document(_document())
    dialog.table.item(0, 3).setText("edited sandstone")
    dialog.table.item(0, 5).setText("medium")
    dialog.table.item(0, 9).setText("1,5")
    dialog.table.item(0, 10).setText("J₂")
    captured = {}

    class Result:
        intervals_saved = 1

    monkeypatch.setattr("qt.core_description_import.save_core_import",
                        lambda *args, **kwargs: captured.update(kwargs) or Result())
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args: None)
    dialog.save()

    assert captured["selected_interval_ids"] == {"0:2"}
    assert captured["edits"]["0:2"]["oil_saturation"] == "medium"
    assert captured["edits"]["0:2"]["rocks"][0]["canonical_value"] == "edited sandstone"
    assert captured["edits"]["0:2"]["thickness"] == 1.5
    assert captured["edits"]["0:2"]["geological_index_raw"] == "J₂"


def test_highlighting_preserves_text_and_marks_exact_classifier_spans(app, session):
    from PyQt5 import QtGui
    from core_description.semantics import enrich_intervals
    from qt.core_description_import import DescriptionDelegate

    document = _document()
    interval = document.intervals[0]
    interval.raw_description = "🪨 <Песчаник> & глина, без признаков нефти.\n" + "Длинное описание. " * 40
    enrich_intervals(document.intervals)
    dialog = CoreDescriptionImportDialog(session)
    dialog.set_document(document)
    rich = QtGui.QTextDocument()
    rich.setHtml(dialog.table.item(0, 7).data(DescriptionDelegate.HTML_ROLE))
    assert rich.toPlainText() == interval.raw_description

    for match in interval.semantic_matches:
        if match.category not in {"rock", "negation"}:
            continue
        cursor = QtGui.QTextCursor(rich)
        # Qt uses UTF-16 positions, classifier offsets use Python code points.
        position = len(interval.raw_description[:match.start].encode("utf-16-le")) // 2
        cursor.setPosition(position)
        cursor.movePosition(QtGui.QTextCursor.NextCharacter, QtGui.QTextCursor.KeepAnchor)
        expected = DescriptionDelegate.ROCK_COLOR if match.category == "rock" else DescriptionDelegate.OIL_COLOR
        assert cursor.charFormat().background().color().name() == expected

    dialog.table.setColumnWidth(7, 500)
    dialog.table.resizeRowsToContents()
    wide_height = dialog.table.rowHeight(0)
    dialog.table.setColumnWidth(7, 250)
    dialog.table.resizeRowsToContents()
    assert dialog.table.rowHeight(0) > wide_height > dialog.table.rowHeight(1)
    dialog.table.selectRow(0)
    assert not dialog.table.grab().isNull()


def test_restore_includes_thickness_and_geological_index(app, session):
    document = _document()
    document.intervals[0].thickness = 1.5
    document.intervals[0].geological_index_raw = "J₂"
    dialog = CoreDescriptionImportDialog(session)
    dialog.set_document(document)
    dialog.table.item(0, 9).setText("9")
    dialog.table.item(0, 10).setText("изменён")
    dialog.table.selectRow(0)
    dialog._restore_selected()
    assert dialog._edits(0)["thickness"] == 1.5
    assert dialog._edits(0)["geological_index_raw"] == "J₂"


@pytest.mark.parametrize("saturation", ["weak_to_medium", "medium_to_intense", "weak_to_intense"])
def test_saturation_ranges_can_be_filtered_and_selected(app, session, saturation):
    document = _document()
    document.intervals[0].oil_saturation = saturation
    dialog = CoreDescriptionImportDialog(session)
    dialog.set_document(document)

    assert dialog.table.item(0, 5).text() == saturation
    assert dialog.saturation_filter.findText(saturation) >= 0
    dialog.saturation_filter.setCurrentText(saturation)
    assert not dialog.table.isRowHidden(0)
    assert dialog.table.isRowHidden(1)

    dialog._select_visible(False)
    dialog._select_oil()
    assert dialog.table.item(0, 0).checkState() == QtCore.Qt.Checked
    assert dialog._edits(0)["oil_saturation"] == saturation
