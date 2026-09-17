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
