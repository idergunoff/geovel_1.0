"""Saved core descriptions in the well card, list and read-only table."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5 import QtCore, QtGui, QtWidgets
from sqlalchemy import create_engine, event
from sqlalchemy.orm import selectinload, sessionmaker

from core_description.database import save_core_import
from core_description.models import ParsedCoreDocument, ParsedCoreInterval, RockMention
from core_description.presentation import (
    CORE_COLOR, CORE_MARKER, core_summary_html, core_well_ids, load_core_intervals,
)
from models_db.model import (
    Base, CoreDescriptionDocument, CoreDescriptionInterval, Well, WellLog, WellOptionally,
)
import models_db.model_cluster  # register relationships
import models_db.model_profile_features
from qt.core_description_view import CoreDescriptionViewDialog, open_core_description_view
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
    assert "C1 &lt;a&gt; — 100–102 — 1.5 — Песчаник — Интенсивная" in html
    assert "— — 102–103 — — — Аргиллит — Не определена" in html
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
    assert dialog.table.item(0, 12).text() == "core-1.docx"
    assert "Песчаник <слой> & нефть" in dialog.details.toPlainText()
    assert "Описал: Иванов" in dialog.details.toPlainText()
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
                         pushButton_view_core_description=QtWidgets.QPushButton())
    current = [wells[1].id]
    ns = _functions("well.py", {"show_data_well"}, dict(
        ui=ui, session=db, Well=Well, WellLog=WellLog, CoreDescriptionDocument=CoreDescriptionDocument,
        WellOptionally=WellOptionally, get_well_id=lambda: current[0], CORE_COLOR=CORE_COLOR,
        load_core_intervals=load_core_intervals, core_summary_html=core_summary_html))
    ns["show_data_well"]()
    text = ui.textEdit_datawell.toPlainText()
    assert text.index("Есть описание керна") < text.index("Скважина №")
    assert text.index("Площадь") < text.index("Описание керна")
    assert "C1 <a> — 100–102 — 1.5 — Песчаник — Интенсивная" in text
    assert ui.pushButton_view_core_description.isEnabled()
    current[0] = wells[3].id
    ns["show_data_well"]()
    text = ui.textEdit_datawell.toPlainText()
    assert text.index("Количество каротажных кривых") < text.index("Есть описание керна")
    current[0] = wells[0].id
    ns["show_data_well"]()
    assert "Описание керна" not in ui.textEdit_datawell.toPlainText()
    assert not ui.pushButton_view_core_description.isEnabled()
    current[0] = None
    ns["show_data_well"]()
    assert ui.textEdit_datawell.toPlainText() == ""


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
