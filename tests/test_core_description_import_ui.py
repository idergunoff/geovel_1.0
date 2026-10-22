"""UI contract tests for the single-document core preview."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")
pytest.importorskip("sqlalchemy")

from PyQt5 import QtCore, QtTest, QtWidgets
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core_description.models import ParsedCoreDocument, ParsedCoreInterval, RockMention
from core_description.terminology import SATURATION_LABELS
from models_db.model import Base, Well, WellOptionally
from core_description.batch import BatchParseItem, BatchParseResult
import models_db.model_cluster  # noqa: E402,F401
import models_db.model_profile_features  # noqa: E402,F401
from qt.core_description_import import CoreDescriptionImportDialog, open_core_description_import


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


def test_preview_needs_three_matches_and_selects_only_confident_intervals(app, session):
    dialog = CoreDescriptionImportDialog(session)
    dialog.set_document(_document())

    assert dialog.well_combo.currentData() is None
    assert dialog.table.rowCount() == 2
    assert dialog.table.item(0, 0).checkState() == QtCore.Qt.Checked
    assert dialog.table.item(1, 0).checkState() == QtCore.Qt.Unchecked
    assert "интервалов: 1" in dialog.summary.text()
    assert not dialog.save_button.isEnabled()


def test_import_window_is_modeless_and_reused(app, session):
    main = QtWidgets.QMainWindow()
    editor = QtWidgets.QLineEdit()
    main.setCentralWidget(editor)
    main.show()
    dialog = open_core_description_import(session, main)
    try:
        assert dialog.isVisible()
        assert dialog.windowModality() == QtCore.Qt.NonModal
        assert app.activeModalWidget() is None
        assert open_core_description_import(session, main) is dialog
        QtTest.QTest.mouseClick(editor, QtCore.Qt.LeftButton)
        QtTest.QTest.keyClicks(editor, "working")
        assert editor.text() == "working"
    finally:
        dialog.close()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        main.close()
    reopened = open_core_description_import(session)
    assert reopened is not dialog
    reopened.close()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_search_filters_by_well_number_not_database_id(app, session):
    dialog = CoreDescriptionImportDialog(session)
    dialog.set_document(_document())
    dialog.well_search.setText("13")
    assert dialog.well_combo.count() == 2
    assert dialog.well_combo.itemData(1) == session.query(Well).filter_by(name="13").one().id
    assert dialog.well_combo.currentData() is None
    assert not dialog.save_button.isEnabled()
    dialog.well_search.setText(" 1 ")
    assert dialog.well_combo.count() == 3
    dialog.well_search.setText("999")
    assert dialog.well_combo.count() == 1
    assert "Нет скважин" in dialog.well_count_label.text()
    dialog.well_search.clear()
    assert dialog.well_combo.count() == 3


@pytest.mark.parametrize("file_area, db_area, search", [
    ("Сарабикуловское поднятие", "Сарабикуловский", "сарабикуловское поднятие"),
    ("Нефтеперспективный объект №58", "Объект 58", "58"),
])
def test_area_choices_and_recommended_wells_are_visible(app, session, file_area, db_area, search):
    other = Well(name="12")
    session.add(other)
    session.flush()
    session.add(WellOptionally(well_id=other.id, option="Площадь", value=db_area))
    session.commit()
    document = _document()
    document.area_name = document.area_name_raw = file_area
    dialog = CoreDescriptionImportDialog(session)
    dialog.set_document(document)
    assert dialog.area_combo.itemText(1) == db_area
    assert dialog.well_combo.itemData(1) == other.id
    assert db_area in dialog.well_combo.itemText(1)
    assert dialog.well_combo.currentData() is None
    dialog.area_combo.setEditText(search)
    assert dialog.well_combo.count() == 2
    assert dialog.well_combo.itemData(1) == other.id
    dialog.well_combo.setCurrentIndex(1)
    assert dialog.save_button.isEnabled()
    dialog.area_combo.setCurrentIndex(0)
    assert dialog.well_combo.count() == 4
    assert dialog.well_combo.currentData() == other.id


def test_manual_choice_is_available_without_automatic_matches(app, session):
    document = _document()
    document.well_name = "UNRECOGNIZED"
    dialog = CoreDescriptionImportDialog(session)
    dialog.set_document(document)
    assert dialog.well_combo.currentData() is None
    assert dialog.well_combo.count() == 3
    dialog.well_combo.setCurrentIndex(2)
    assert dialog.match_method == "manual"
    assert dialog.save_button.isEnabled()


def test_refresh_loads_wells_added_in_main_window(app, session):
    dialog = CoreDescriptionImportDialog(session)
    dialog.set_document(_document())
    dialog.well_combo.setCurrentIndex(1)
    selected_id = dialog.well_combo.currentData()
    well = Well(name="new well")
    session.add(well)
    session.commit()
    dialog.refresh_wells_button.click()
    assert dialog.well_combo.findData(well.id) >= 0
    assert dialog.well_combo.currentData() == selected_id


def test_refresh_disables_save_when_selected_well_was_deleted(app, session):
    dialog = CoreDescriptionImportDialog(session)
    dialog.set_document(_document())
    dialog.well_combo.setCurrentIndex(1)
    selected_id = dialog.well_combo.currentData()
    session.delete(session.get(Well, selected_id))
    session.commit()
    dialog.refresh_wells_button.click()
    assert dialog.well_combo.currentData() is None
    assert not dialog.save_button.isEnabled()


def test_identity_details_are_visible_and_compared_when_selecting(app, session):
    existing = session.query(Well).filter_by(name="12").one()
    existing.alt = 135
    wanted = Well(name="12", alt=132.43)
    session.add(wanted)
    session.flush()
    session.add_all([
        WellOptionally(well_id=wanted.id, option="Залежь", value="Сарабикуловский"),
        WellOptionally(well_id=wanted.id, option="Начало бурения", value="28.08.2019"),
        WellOptionally(well_id=wanted.id, option="Окончание бурения", value="02.09.2019"),
    ])
    session.commit()
    source = _document()
    source.area_name_raw = source.area_name = "Сарабикуловское поднятие"
    source.drilling_start, source.drilling_start_raw = "2019-08-28", "28.08.19"
    source.drilling_end, source.drilling_end_raw = "2019-09-02", "02.09.2019"
    source.wellhead_elevation, source.wellhead_elevation_raw = 132.432, "132,432"
    dialog = CoreDescriptionImportDialog(session)
    dialog.set_document(source)
    assert dialog.well_combo.currentData() == wanted.id
    assert "28.08.2019" in dialog.well_combo.currentText()
    assert "02.09.2019" in dialog.well_combo.currentText()
    assert "132.43" in dialog.well_combo.currentText()
    assert "Залежь: Сарабикуловский" in dialog.comparison_table.item(0, 2).text()
    assert all(dialog.comparison_table.item(row, 3).text() == "Совпадает" for row in range(4))
    dialog.well_combo.setCurrentIndex(dialog.well_combo.findData(existing.id))
    assert dialog.comparison_table.item(3, 3).text() == "Различается"
    assert dialog.comparison_table.item(1, 3).text() == "Нет данных для сравнения"
    assert dialog.match_method == "manual"


def test_area_filter_searches_arbitrary_categories_even_without_source_match(app, session):
    wanted = session.query(Well).filter_by(name="13").one()
    session.add(WellOptionally(well_id=wanted.id, option="Свободная категория", value="Сарабикуловский"))
    session.commit()
    dialog = CoreDescriptionImportDialog(session)
    dialog.set_document(_document())
    dialog.area_combo.setEditText("сараб")
    assert dialog.well_combo.count() == 2
    assert dialog.well_combo.itemData(1) == wanted.id
    assert "Сарабикуловский" in dialog.well_combo.itemText(1)


def test_missing_database_details_never_display_a_match(app, session):
    source = _document()
    source.drilling_start_raw, source.drilling_start = "28.08.2019", "2019-08-28"
    dialog = CoreDescriptionImportDialog(session)
    dialog.set_document(source)
    assert dialog.comparison_table.item(1, 1).text() == "28.08.2019"
    assert dialog.comparison_table.item(1, 2).text() == "—"
    assert dialog.comparison_table.item(1, 3).text() == "Нет данных для сравнения"


def test_three_relaxed_matches_are_selected_and_saved_in_batch(app, session, monkeypatch):
    wanted = session.query(Well).filter_by(name="12").one()
    wanted.alt = 134.432
    session.add_all([
        WellOptionally(well_id=wanted.id, option="Участок", value="Мельничный уч"),
        WellOptionally(well_id=wanted.id, option="Начало бурения", value="28.08.2019"),
    ])
    session.commit()
    source = _document()
    source.area_name_raw = source.area_name = "Мелничное поднятие"
    source.drilling_start = "2019-08-28"
    source.wellhead_elevation = 132.432
    dialog = CoreDescriptionImportDialog(session)
    dialog._batch_finished(BatchParseResult(items=[BatchParseItem(source.source_path, "ready", source)]))
    assert dialog.well_combo.currentData() == wanted.id
    assert "3 из 4" in dialog.match_label.text()
    assert "сходство" in dialog.comparison_table.item(0, 3).text()
    assert dialog.comparison_table.item(3, 3).text() == "Совпадает"
    captured = []
    class Result:
        intervals_saved = 1
    monkeypatch.setattr("qt.core_description_import.save_core_import",
                        lambda *_args, **kwargs: captured.append(kwargs) or Result())
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *_args: None)
    dialog.save()
    assert captured[0]["well_id"] == wanted.id
    assert captured[0]["match_method"] == "exact_name_identity"


def test_batch_keeps_manual_choices_when_switching_and_saving(app, session, monkeypatch):
    first, second = _document(), _document()
    first.source_path, second.source_path = "first.docx", "second.docx"
    first.well_name = second.well_name = "UNKNOWN"
    dialog = CoreDescriptionImportDialog(session)
    dialog._batch_finished(BatchParseResult(items=[
        BatchParseItem(first.source_path, "ready", first),
        BatchParseItem(second.source_path, "ready", second),
    ]))
    ids = [well.id for well in session.query(Well).order_by(Well.id)]
    dialog.well_combo.setCurrentIndex(dialog.well_combo.findData(ids[0]))
    dialog.documents_table.setCurrentCell(1, 1)
    dialog.well_combo.setCurrentIndex(dialog.well_combo.findData(ids[1]))
    dialog.documents_table.setCurrentCell(0, 1)
    assert dialog.well_combo.currentData() == ids[0]
    assert "12" in dialog.documents_table.item(0, 5).text()
    captured = []

    class Result:
        intervals_saved = 1

    def save(_session, document, **kwargs):
        captured.append((document.source_path, kwargs["well_id"], kwargs["match_method"]))
        return Result()

    monkeypatch.setattr("qt.core_description_import.save_core_import", save)
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args: None)
    dialog.save()
    assert captured == [(first.source_path, ids[0], "manual"), (second.source_path, ids[1], "manual")]


def test_closing_waits_for_worker_without_blocking_gui(app, session, monkeypatch):
    import threading

    release = threading.Event()

    def parse(*_args, **_kwargs):
        release.wait(timeout=2)
        return BatchParseResult()

    monkeypatch.setattr("qt.core_description_import.parse_batch", parse)
    dialog = CoreDescriptionImportDialog(session)
    dialog.show()
    dialog.start_batch(["core.docx"])
    dialog.close()
    assert dialog._close_pending
    assert dialog.isVisible()
    assert dialog._thread is not None
    loop = QtCore.QEventLoop()
    dialog.finished.connect(loop.quit)
    QtCore.QTimer.singleShot(0, release.set)
    QtCore.QTimer.singleShot(3000, loop.quit)
    loop.exec_()
    assert dialog._thread is None
    assert not dialog.isVisible()


def test_filter_does_not_change_hidden_row_selection(app, session):
    dialog = CoreDescriptionImportDialog(session)
    dialog.set_document(_document())
    dialog.table.item(1, 0).setCheckState(QtCore.Qt.Checked)
    dialog.rock_filter.setText("песчаник")

    assert dialog.table.isRowHidden(1)
    dialog._select_visible(False)
    assert dialog.table.item(0, 0).checkState() == QtCore.Qt.Unchecked
    assert dialog.table.item(1, 0).checkState() == QtCore.Qt.Checked


def test_edited_values_are_passed_to_persistence(app, session, monkeypatch):
    dialog = CoreDescriptionImportDialog(session)
    dialog.set_document(_document())
    dialog.well_combo.setCurrentIndex(1)
    dialog.table.item(0, 3).setText("Известняк")
    dialog.table.item(0, 5).setText("Средняя")
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
    assert captured["edits"]["0:2"]["rocks"][0]["canonical_value"] == "limestone"
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


@pytest.mark.parametrize("saturation", list(SATURATION_LABELS))
def test_saturation_ranges_can_be_filtered_and_selected(app, session, saturation):
    document = _document()
    document.intervals[0].oil_saturation = saturation
    dialog = CoreDescriptionImportDialog(session)
    dialog.set_document(document)

    label = SATURATION_LABELS[saturation]
    assert dialog.table.item(0, 5).text() == label
    assert dialog.saturation_filter.findData(saturation) >= 0
    dialog.saturation_filter.setCurrentText(label)
    assert not dialog.table.isRowHidden(0)
    assert dialog.table.isRowHidden(1) == (saturation != "unknown")

    dialog._select_visible(False)
    dialog._select_oil()
    expected = QtCore.Qt.Unchecked if saturation in {"none", "unknown"} else QtCore.Qt.Checked
    assert dialog.table.item(0, 0).checkState() == expected
    assert dialog._edits(0)["oil_saturation"] == saturation


def test_russian_rocks_relations_edits_and_restore(app, session):
    document = _document()
    document.intervals[0].rocks.append(RockMention("clay", "глины", "interlayer"))
    dialog = CoreDescriptionImportDialog(session)
    dialog.set_document(document)
    assert dialog.table.item(0, 3).text() == "Песчаник"
    assert dialog.table.item(0, 4).text() == "Глина (прослой)"
    assert "rocks" not in dialog._edits(0)

    dialog.rock_filter.setText("глина")
    assert not dialog.table.isRowHidden(0)
    assert dialog.table.isRowHidden(1)
    dialog.table.item(0, 3).setText("Известняк")
    dialog.table.item(0, 4).setText("Доломит (включение), Алевролит (переслаивание)")
    rocks = dialog._edits(0)["rocks"]
    assert [(r["canonical_value"], r["relation"]) for r in rocks] == [
        ("limestone", "primary"), ("dolomite", "inclusion"), ("siltstone", "interbedding")]

    dialog.table.item(0, 5).setText("Слабая")
    dialog.rock_filter.clear()
    dialog.table.selectRow(0)
    dialog._restore_selected()
    assert dialog.table.item(0, 3).text() == "Песчаник"
    assert dialog.table.item(0, 4).text() == "Глина (прослой)"
    assert dialog.table.item(0, 5).text() == SATURATION_LABELS["present"]
    assert "rocks" not in dialog._edits(0)
    assert dialog._edits(0)["oil_saturation"] == "present"


def test_all_dictionary_rock_terms_are_translated_and_reversible():
    from core_description import load_dictionary
    from core_description.terminology import ROCK_LABELS, RELATION_LABELS, canonical_term

    dictionary = load_dictionary()
    for category, labels in (("rock", ROCK_LABELS), ("rock_relation", RELATION_LABELS)):
        codes = {rule.canonical_value for rule in dictionary.rules if rule.category == category}
        assert codes <= labels.keys()
        for code in codes:
            assert labels[code] != code
            assert canonical_term(labels[code].lower(), labels) == code
            assert canonical_term(code, labels) == code
    assert canonical_term("Особая порода", ROCK_LABELS) == "Особая порода"
