"""Saved core descriptions in the well card, list and read-only table."""

import ast
import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
from sqlalchemy import create_engine, event
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import selectinload, sessionmaker

from core_description.database import save_core_import
from core_description.models import ParsedCoreDocument, ParsedCoreInterval, RockMention
from core_description.presentation import (
    CORE_COLOR, CORE_MARKER, OIL_LAYER_COLOR, core_summary_html, core_well_ids,
    load_core_intervals, secondary_rock_text,
)
from core_description.semantics import enrich_intervals
from core_description.terminology import SATURATION_LABELS
from models_db.model import (
    Base, CoreDescriptionDocument, CoreDescriptionInterval, CoreDescriptionRock, Well, WellLog, WellOptionally,
)
import models_db.model_cluster  # register relationships
import models_db.model_profile_features
from qt.core_description_view import CoreDescriptionViewDialog, open_core_description_view, saved_description_html
from qt.core_description_import import DescriptionDelegate
from well_selection import find_well_row, well_id_from_text


@pytest.fixture(scope="session")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def dataset():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    wells = [Well(id=i + 1, name=f"{i + 1:02}") for i in range(4)]
    db.add_all(wells)
    db.flush()
    db.add_all([WellLog(well_id=wells[2].id), WellLog(well_id=wells[3].id),
                WellOptionally(well_id=wells[1].id, option="Площадь", value="Северная")])
    for index in (1, 3):
        document = ParsedCoreDocument(
            source_path=f"core-{index}.docx", source_format="docx", file_hash=str(index) * 64,
            well_name_raw=wells[index].name, well_name=wells[index].name,
            area_name_raw="Северная", area_name="Северная",
            described_by_raw="Описал: Иванов", described_by="Иванов",
            dictionary_version="1.0", intervals=[
                ParsedCoreInterval(102, 103, "Аргиллит", 0, 3, confidence=1,
                                   rocks=[RockMention("claystone", "Аргиллит")]),
                ParsedCoreInterval(100, 102, "Песчаник <слой> & нефть", 0, 2,
                                   geological_index_raw="C1 <a>", thickness=1.5,
                                   oil_saturation="intense", confidence=1,
                                   rocks=[RockMention("sandstone", "Песчаник")]),
            ])
        save_core_import(db, document, well_id=wells[index].id,
                         selected_interval_ids={"0:2", "0:3"})
    yield db, wells
    db.close()
    engine.dispose()


def _functions(file_name, names, namespace):
    """Exercise legacy UI handlers without starting the whole application."""
    source = Path(__file__).resolve().parents[1] / file_name
    tree = ast.parse(source.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace


def test_summary_orders_depths_escapes_index_and_preserves_reported_thickness(dataset):
    db, wells = dataset
    intervals = load_core_intervals(db, wells[1].id)
    assert [i.top_depth for i in intervals] == [100, 102]
    html = core_summary_html(intervals)
    assert "C1 &lt;a&gt; — 100 — 1.5 — Песчаник — Интенсивная" in html
    assert "— — 102 — — — Аргиллит</p>" in html
    assert "Не определена" not in html
    assert "100–102" not in html
    assert core_well_ids(db) == {wells[1].id, wells[3].id}
    assert load_core_intervals(db, wells[0].id) == []


def test_view_contains_only_selected_well_is_readonly_and_has_full_details(app, dataset):
    db, wells = dataset
    parent = QtWidgets.QWidget()
    dialog = open_core_description_view(db, wells[1], parent)
    assert dialog.isVisible() and not dialog.isModal()
    assert dialog.table.rowCount() == 2
    assert dialog.table.editTriggers() == QtWidgets.QAbstractItemView.NoEditTriggers
    assert dialog.table.item(0, 0).text() == "C1 <a>"
    assert dialog.table.item(0, 4).text() == "Песчаник"
    assert dialog.table.item(0, 6).text() == "Интенсивная"
    assert dialog.table.columnCount() == 8
    assert dialog.table.item(0, 5).text() == ""
    assert dialog.table.item(1, 6).text() == ""
    assert dialog.table.item(0, 0).background().color().name() == OIL_LAYER_COLOR.lower()
    assert "Песчаник <слой> & нефть" in dialog.details.toPlainText()
    assert "Описал: Иванов" in dialog.details.toPlainText()
    assert "core-1.docx" in dialog.details.toPlainText()
    assert "Уверенность: 1" in dialog.details.toPlainText()
    assert "Правки вручную: Нет" in dialog.details.toPlainText()
    dialog.table.selectRow(1)
    assert "Аргиллит" in dialog.details.toPlainText()
    empty = CoreDescriptionViewDialog(db, wells[0], parent)
    assert empty.table.rowCount() == 0
    assert "не загружено" in empty.summary.text()
    parent.close()
    parent.deleteLater()


@pytest.mark.parametrize("mode", ["all", "profile", "object"])
def test_list_prioritizes_core_and_logs_and_keeps_exact_selection(app, dataset, mode):
    db, wells = dataset
    ui = SimpleNamespace(
        listWidget_well=QtWidgets.QListWidget(), label_11=QtWidgets.QLabel(),
        checkBox_profile_intersec=QtWidgets.QCheckBox(),
        checkBox_profile_well=QtWidgets.QCheckBox(), checkBox_object_well=QtWidgets.QCheckBox())
    ui.checkBox_profile_well.setChecked(mode == "profile")
    ui.checkBox_object_well.setChecked(mode == "object")
    ns = _functions("func.py", {"update_list_well"}, dict(
        ui=ui, session=db, Well=Well, selectinload=selectinload,
        QListWidgetItem=QtWidgets.QListWidgetItem, QBrush=QtGui.QBrush, QColor=QtGui.QColor,
        Qt=QtCore.Qt, CORE_COLOR=CORE_COLOR, CORE_MARKER=CORE_MARKER,
        core_well_ids=core_well_ids, find_well_row=find_well_row,
        get_profile_id=lambda: 1, get_object_id=lambda: 1, draw_wells=lambda: None,
        get_list_nearest_well=lambda _: [(w, 0, 2) for w in wells],
        get_list_nearest_well_for_object=lambda _: [(w, 0, 2, None) for w in wells]))
    ns["update_list_well"](select_well=True, selected_well_id=wells[1].id)
    items = [ui.listWidget_well.item(i) for i in range(4)]
    assert [well_id_from_text(item.text()) for item in items] == [2, 3, 4, 1]
    assert items[0].text().endswith(" ▤")
    assert not items[1].text().endswith(" ▤")
    assert "Есть описание керна" in items[2].toolTip()
    assert "каротажных кривых" in items[2].toolTip()
    assert well_id_from_text(ui.listWidget_well.currentItem().text()) == 2
    assert ui.label_11.text() == "Wells: 2 / 4"


def test_well_card_presence_summary_order_and_empty_selection(app, dataset):
    db, wells = dataset
    ui = SimpleNamespace(textEdit_datawell=QtWidgets.QTextEdit(),
                         checkBox_profile_intersec=QtWidgets.QCheckBox(),
                         pushButton_view_core_description=QtWidgets.QPushButton(),
                         pushButton_remove_core_description=QtWidgets.QPushButton())
    current = [wells[1].id]
    ns = _functions("well.py", {"show_data_well"}, dict(
        ui=ui, session=db, Well=Well, WellLog=WellLog, CoreDescriptionDocument=CoreDescriptionDocument,
        WellOptionally=WellOptionally, get_well_id=lambda: current[0], CORE_COLOR=CORE_COLOR,
        load_core_intervals=load_core_intervals, core_summary_html=core_summary_html))
    ns["show_data_well"]()
    text = ui.textEdit_datawell.toPlainText()
    assert text.index("Есть описание керна") < text.index("Скважина №")
    assert text.index("Площадь") < text.index("Описание керна")
    assert "C1 <a> — 100 — 1.5 — Песчаник — Интенсивная" in text
    assert ui.pushButton_view_core_description.isEnabled()
    assert ui.pushButton_remove_core_description.isEnabled()
    current[0] = wells[3].id
    ns["show_data_well"]()
    text = ui.textEdit_datawell.toPlainText()
    assert text.index("Количество каротажных кривых") < text.index("Есть описание керна")
    current[0] = wells[0].id
    ns["show_data_well"]()
    assert "Описание керна" not in ui.textEdit_datawell.toPlainText()
    assert not ui.pushButton_view_core_description.isEnabled()
    assert not ui.pushButton_remove_core_description.isEnabled()
    current[0] = wells[1].id
    ns["show_data_well"]()
    current[0] = None
    ns["show_data_well"]()
    assert ui.textEdit_datawell.toPlainText() == ""
    assert not ui.pushButton_remove_core_description.isEnabled()


@pytest.fixture()
def removal(app, dataset, monkeypatch):
    db, wells = dataset
    parent = QtWidgets.QWidget()
    ui = SimpleNamespace(checkBox_profile_intersec=QtWidgets.QCheckBox())
    current = [wells[1].id]
    confirmations, refreshes, errors, messages = [], [], [], []

    def confirm(*args):
        confirmations.append(args)
        return QtWidgets.QMessageBox.Yes

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", confirm)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args: errors.append(args))
    ns = _functions("well.py", {"remove_core_description"}, dict(
        ui=ui, session=db, Well=Well, CoreDescriptionDocument=CoreDescriptionDocument,
        CoreDescriptionInterval=CoreDescriptionInterval, selectinload=selectinload,
        SQLAlchemyError=SQLAlchemyError, QMessageBox=QtWidgets.QMessageBox,
        MainWindow=parent, CoreDescriptionViewDialog=CoreDescriptionViewDialog,
        get_well_id=lambda: current[0],
        refresh_core_descriptions=lambda: refreshes.append(current[0]),
        set_info=lambda *args: messages.append(args)))
    yield SimpleNamespace(
        run=ns["remove_core_description"], db=db, wells=wells, parent=parent,
        ui=ui, current=current, confirmations=confirmations, refreshes=refreshes,
        errors=errors, messages=messages)
    parent.close()
    parent.deleteLater()


def test_removal_deletes_all_core_records_only_for_selected_well(removal):
    r = removal
    # Production SQLite may have foreign keys disabled; ORM deletion must still
    # remove intervals and rocks, including descriptions from multiple documents.
    assert r.db.connection().exec_driver_sql("PRAGMA foreign_keys").scalar() == 0
    r.wells[3].name = r.wells[1].name
    r.db.add(CoreDescriptionDocument(
        well_id=r.current[0], source_file_name="another.docx", source_file_hash="a" * 64,
        source_format="docx", parser_version="1", match_method="manual", match_confidence=1))
    r.db.commit()
    selected_view = CoreDescriptionViewDialog(r.db, r.wells[1], r.parent)
    other_view = CoreDescriptionViewDialog(r.db, r.wells[3], r.parent)
    selected_view.show()
    other_view.show()

    r.run()

    assert core_well_ids(r.db) == {r.wells[3].id}
    assert r.db.query(CoreDescriptionDocument).count() == 1
    assert r.db.query(CoreDescriptionInterval).count() == 2
    assert r.db.query(CoreDescriptionRock).count() == 2
    assert load_core_intervals(r.db, r.wells[1].id) == []
    assert len(load_core_intervals(r.db, r.wells[3].id)) == 2
    assert r.db.query(Well).count() == 4
    assert r.db.query(WellLog).count() == 2
    assert r.db.query(WellOptionally).count() == 1
    assert not selected_view.isVisible()
    assert other_view.isVisible()
    assert r.refreshes == [r.wells[1].id]
    assert r.confirmations[0][-1] == QtWidgets.QMessageBox.No
    assert r.errors == []


def test_removal_cancel_keeps_descriptions_and_windows(removal, monkeypatch):
    r = removal
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args: QtWidgets.QMessageBox.No)
    viewer = CoreDescriptionViewDialog(r.db, r.wells[1], r.parent)
    viewer.show()

    r.run()

    assert r.db.query(CoreDescriptionDocument).count() == 2
    assert r.db.query(CoreDescriptionInterval).count() == 4
    assert r.db.query(CoreDescriptionRock).count() == 4
    assert viewer.isVisible()
    assert r.refreshes == []


def test_removal_failure_rolls_back_all_records(removal, monkeypatch):
    r = removal

    def fail_commit():
        r.db.flush()
        raise SQLAlchemyError("test database error")

    monkeypatch.setattr(r.db, "commit", fail_commit)
    r.run()

    assert r.db.query(CoreDescriptionDocument).count() == 2
    assert r.db.query(CoreDescriptionInterval).count() == 4
    assert r.db.query(CoreDescriptionRock).count() == 4
    assert len(r.errors) == 1
    assert r.refreshes == []
    assert r.messages[-1][1] == "red"


@pytest.mark.parametrize("well_index,intersections", [(None, False), (0, False), (1, True)])
def test_removal_ignores_missing_selection_empty_core_and_intersections(removal, well_index, intersections):
    r = removal
    r.current[0] = r.wells[well_index].id if well_index is not None else None
    r.ui.checkBox_profile_intersec.setChecked(intersections)

    r.run()

    assert r.db.query(CoreDescriptionDocument).count() == 2
    assert r.db.query(CoreDescriptionInterval).count() == 4
    assert r.db.query(CoreDescriptionRock).count() == 4
    assert r.confirmations == []
    assert r.refreshes == []


def test_availability_and_interval_loading_use_bounded_queries(dataset):
    db, wells = dataset
    well_id = wells[1].id
    db.expire_all()
    statements = []
    def count(*args):
        statements.append(args[2])
    event.listen(db.bind, "before_cursor_execute", count)
    try:
        core_well_ids(db)
        intervals = load_core_intervals(db, well_id)
        core_summary_html(intervals)
        assert len(statements) == 3  # availability, intervals + documents, rocks
    finally:
        event.remove(db.bind, "before_cursor_execute", count)


def test_marked_wells_still_draw_by_numeric_id(app):
    widget = QtWidgets.QListWidget()
    widget.addItems(["скв.№ 12 id2 ▤", "скв.№ 13 id3"])
    drawn = []
    ns = _functions("func.py", {"draw_wells"}, dict(
        ui=SimpleNamespace(listWidget_well=widget, checkBox_profile_intersec=QtWidgets.QCheckBox()),
        well_id_from_text=well_id_from_text, draw_well=drawn.append))
    ns["draw_wells"]()
    assert drawn == [2, 3]


def test_marked_well_is_identified_when_adding_to_cluster_dataset(app, dataset):
    db, wells = dataset
    widget = QtWidgets.QListWidget()
    widget.addItem(f"скв.№ {wells[1].name} id{wells[1].id} ▤")
    widget.setCurrentRow(0)
    combo = QtWidgets.QComboBox()
    combo.addItem("Набор", 1)
    messages = []
    ns = _functions("cluster/well_dataset.py", {"add_selected_well_to_cluster_dataset"}, dict(
        ui=SimpleNamespace(listWidget_well=widget, comboBox_cluster_well_set=combo),
        session=db, Well=Well, WellLog=WellLog, well_id_from_text=well_id_from_text,
        MainWindow=None, QMessageBox=SimpleNamespace(warning=lambda *args: messages.append(args[-1]))))
    ns["add_selected_well_to_cluster_dataset"]()
    assert messages == [f'У скважины "{wells[1].name}" отсутствует каротаж (well_log).']


@pytest.mark.parametrize("saturation", list(SATURATION_LABELS))
def test_any_oil_indication_including_odor_highlights_summary_and_table(app, dataset, saturation):
    db, wells = dataset
    interval = load_core_intervals(db, wells[1].id)[0]
    interval.oil_saturation = saturation
    db.flush()
    positive = saturation not in {"unknown", "none"}
    assert (OIL_LAYER_COLOR in core_summary_html([interval])) == positive
    dialog = CoreDescriptionViewDialog(db, wells[1])
    color = dialog.table.item(0, 0).background().color().name()
    assert (color == OIL_LAYER_COLOR.lower()) == positive
    if saturation == "unknown":
        assert dialog.table.item(0, 6).text() == ""
    dialog.close()


def test_secondary_rocks_are_short_and_rock_only_oil_indications_are_highlighted(dataset):
    db, wells = dataset
    interval = load_core_intervals(db, wells[1].id)[0]
    interval.oil_saturation = "unknown"
    interval.rocks[0].oil_saturation = "odor"
    assert OIL_LAYER_COLOR in core_summary_html([interval])
    interval = SimpleNamespace(rocks=[
        SimpleNamespace(id=1, rock_name="sandstone", role="primary"),
        SimpleNamespace(id=2, rock_name="clay", role="secondary"),
        SimpleNamespace(id=3, rock_name="siltstone", role="interlayer"),
        SimpleNamespace(id=4, rock_name="clay", role="secondary"),
    ])
    assert secondary_rock_text(interval) == "Глина, Алевролит (прослой)"


def test_saved_description_highlighting_preserves_text_and_wraps_to_full_height(app, dataset):
    db, wells = dataset
    interval = load_core_intervals(db, wells[1].id)[0]
    text = "🪨 <Песчаник> & глина, запах нефти.\n" + "Полное длинное описание слоя. " * 40
    parsed = ParsedCoreInterval(100, 102, text, 0, 2)
    enrich_intervals([parsed])
    interval.raw_description = text
    interval.recognition_details = json.dumps({"matches": [asdict(m) for m in parsed.semantic_matches]})
    db.commit()
    dialog = CoreDescriptionViewDialog(db, wells[1])
    rich = QtGui.QTextDocument()
    rich.setHtml(dialog.table.item(0, 7).data(DescriptionDelegate.HTML_ROLE))
    assert rich.toPlainText() == text
    assert {m.category for m in parsed.semantic_matches} >= {"rock", "oil_indicator"}
    for match in parsed.semantic_matches:
        if match.category not in {"rock", "oil_indicator", "oil_saturation"}:
            continue
        cursor = QtGui.QTextCursor(rich)
        cursor.setPosition(len(text[:match.start].encode("utf-16-le")) // 2)
        cursor.movePosition(QtGui.QTextCursor.NextCharacter, QtGui.QTextCursor.KeepAnchor)
        expected = DescriptionDelegate.ROCK_COLOR if match.category == "rock" else DescriptionDelegate.OIL_COLOR
        assert cursor.charFormat().background().color().name() == expected
    dialog.table.horizontalHeader().setStretchLastSection(False)
    dialog.table.setColumnWidth(7, 560)
    dialog.table.resizeRowsToContents()
    wide = dialog.table.rowHeight(0)
    dialog.table.setColumnWidth(7, 260)
    QtTest.QTest.qWait(100)
    assert dialog.table.rowHeight(0) > wide
    delegate = dialog.table.itemDelegateForColumn(7)
    option = QtWidgets.QStyleOptionViewItem()
    option.widget = dialog.table
    option.font = dialog.table.font()
    option.rect.setWidth(dialog.table.columnWidth(7))
    index = dialog.table.model().index(0, 7)
    assert dialog.table.rowHeight(0) >= delegate.sizeHint(option, index).height()
    assert not dialog.table.grab().isNull()
    dialog.close()


@pytest.mark.parametrize("details", ["{}", "broken json", "null", '{"matches": null}',
                                      '{"matches": [null, {"category": "rock", "start": "0", "end": 5}]}'])
def test_missing_or_invalid_saved_spans_keep_plain_description(details):
    interval = SimpleNamespace(raw_description="<Песчаник> & нефть", recognition_details=details)
    rich = QtGui.QTextDocument()
    rich.setHtml(saved_description_html(interval))
    assert rich.toPlainText() == interval.raw_description
