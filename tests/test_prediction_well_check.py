import datetime
import json
import os
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models_db.model import (Base, GeoradarObject, Research, Profile,
                             ProfileModelPrediction, PredictionCorrect, Well)
import models_db.model_cluster  # register related mappings
import models_db.model_profile_features
from prediction_well_check import collect_prediction_wells

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


@pytest.fixture
def dataset():
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    obj = GeoradarObject(title='Object')
    other = GeoradarObject(title='Other')
    profiles = []
    for i, owner in enumerate([obj, obj, obj, other]):
        research = Research(object=owner, date_research=datetime.date(2020 + i, 1, 1))
        profiles.append(Profile(research=research, title=f'P{i}',
                                x_pulc='[0, 10, 20]', y_pulc='[0, 0, 0]'))
    well = Well(name='W', x_coord=10, y_coord=1)
    db.add_all(profiles + [well]); db.flush()
    selected = ProfileModelPrediction(profile=profiles[0], model_id=7, type_model='reg', prediction='[1, 2, 3]')
    second = ProfileModelPrediction(profile=profiles[1], model_id=7, type_model='reg', prediction='[4, 5, 6]')
    db.add_all([selected, second,
                ProfileModelPrediction(profile=profiles[1], model_id=7, type_model='cls', prediction='[9, 9, 9]'),
                ProfileModelPrediction(profile=profiles[1], model_id=8, type_model='reg', prediction='[8, 8, 8]'),
                ProfileModelPrediction(profile=profiles[3], model_id=7, type_model='reg', prediction='[7, 7, 7]'),
                PredictionCorrect(prediction=selected, correct='[10, 20, 30]')])
    db.commit()
    try:
        yield db, obj, profiles, well, selected, second
    finally:
        db.close()
        engine.dispose()


@pytest.fixture
def choose_export_columns(monkeypatch):
    from PyQt5 import QtCore, QtWidgets

    def choose(titles=None, *, checked=None):
        def exec_picker(picker):
            columns = picker.findChild(QtWidgets.QListWidget)
            items = [columns.item(index) for index in range(columns.count())]
            button_box = picker.findChild(QtWidgets.QDialogButtonBox)
            confirm = button_box.button(QtWidgets.QDialogButtonBox.Ok)
            initially_checked = [item.text() for item in items if item.checkState() == QtCore.Qt.Checked]
            assert initially_checked == (checked if checked is not None else [item.text() for item in items])
            if titles is not None:
                actions = {button.text(): button for button in picker.findChildren(QtWidgets.QPushButton)}
                actions['Снять все'].click()
                assert not confirm.isEnabled()
                actions['Отметить все'].click()
                assert all(item.checkState() == QtCore.Qt.Checked for item in items)
                actions['Снять все'].click()
                assert set(titles) <= {item.text() for item in items}
                for item in items:
                    if item.text() in titles:
                        item.setCheckState(QtCore.Qt.Checked)
            if titles == []:
                button_box.button(QtWidgets.QDialogButtonBox.Cancel).click()
            else:
                assert confirm.isEnabled()
                confirm.click()
            return picker.result()

        monkeypatch.setattr(QtWidgets.QDialog, 'exec_', exec_picker)

    return choose


@pytest.fixture(autouse=True)
def isolate_column_preferences(monkeypatch, tmp_path):
    from PyQt5 import QtCore
    from qt import prediction_well_preferences

    path = str(tmp_path / 'well_report.ini')
    monkeypatch.setattr(prediction_well_preferences, 'settings',
                        lambda: QtCore.QSettings(path, QtCore.QSettings.IniFormat))


def test_same_model_all_researches_and_one_row_per_well_profile(dataset):
    db, obj, profiles, well, selected, second = dataset
    searched = []
    def nearby(profile_id):
        searched.append(profile_id)
        return [(well, 1, 1.0)]
    report = collect_prediction_wells(db, obj.id, selected, nearby)
    assert [s.value for s in report.samples] == [2, 5]
    assert [s.profile.id for s in report.samples] == [p.id for p in profiles[:2]]
    assert searched == [p.id for p in profiles[:2]]
    assert report.profiles_without_prediction == 1
    assert report.issues == []
    assert not db.dirty and not db.new and not db.deleted


def test_corrected_prediction_and_original_fallback(dataset):
    db, obj, profiles, well, selected, second = dataset
    report = collect_prediction_wells(db, obj.id, selected, lambda _: [(well, 1, 1)], True)
    assert [s.value for s in report.samples] == [20, 5]
    assert [s.corrected for s in report.samples] == [True, False]


@pytest.mark.parametrize('raw', ['null', '{}', '[1]', 'bad json', '[1, null, 3]', '[1, NaN, 3]'])
def test_bad_predictions_reported_without_aborting_other_profiles(dataset, raw):
    db, obj, profiles, well, selected, second = dataset
    selected.prediction = raw
    report = collect_prediction_wells(db, obj.id, selected, lambda _: [(well, 1, 1)])
    assert [s.value for s in report.samples] == [5]
    assert len(report.issues) == 1


def test_proximity_search_is_authoritative(dataset):
    db, obj, profiles, well, selected, second = dataset
    report = collect_prediction_wells(db, obj.id, selected, lambda _: [])
    assert report.samples == []
    assert report.issues == []


def test_invalid_geometry_does_not_call_search(dataset):
    db, obj, profiles, well, selected, second = dataset
    profiles[0].y_pulc = '[]'
    searched = []
    def nearby(pid):
        searched.append(pid)
        return [(well, 1, 1)]
    report = collect_prediction_wells(db, obj.id, selected, nearby)
    assert searched == [profiles[1].id]
    assert len(report.issues) == 1


def test_comparison_dialog_reuses_sources_and_never_exposes_markup_actions(
        dataset, monkeypatch, tmp_path, choose_export_columns):
    from PyQt5 import QtWidgets
    from models_db.model import CanonicalBoundary, AliasBoundary, Boundary
    import qt.regression_target_wizard as wizard
    from qt.prediction_well_check_dialog import PredictionWellCheckDialog

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(wizard, 'restore_form', lambda *_args: None)
    monkeypatch.setattr(wizard, 'save_form', lambda *_args: None)
    monkeypatch.setattr(wizard, 'list_canonical_targets', lambda *_args: [
        SimpleNamespace(canonical_name='Target', id=1)])
    db, obj, profiles, well, selected, second = dataset
    db.add(CanonicalBoundary(id=1, canonical_name='Target'))
    db.flush()
    db.add_all([AliasBoundary(alias_name='Target', canonical_id=1),
                Boundary(well_id=well.id, title='Target', depth=5)])
    db.commit()
    report = collect_prediction_wells(db, obj.id, selected, lambda _: [(well, 1, 1)])
    dialog = PredictionWellCheckDialog(db, report.samples, 'Test', prediction_type=selected.type_model)
    assert [dialog.source_combo.itemData(i) for i in range(3)] == ['boundary', 'well_data', 'well_log']
    assert not dialog.isModal()
    assert dialog.replace_button.isHidden()
    assert dialog.add_selected_button.isHidden()
    assert dialog.delete_selected_button.isHidden()
    assert dialog.table.isColumnHidden(0)
    assert dialog.table.horizontalHeaderItem(9).text() == 'Прогноз регрессии'
    assert dialog.table.item(0, 9).text() == '2'
    dialog.calculate_button.click()
    assert all(c.resolution.value == 5 for c in dialog.candidates)
    assert dialog.table.item(0, 11).text() == '+3'
    path = tmp_path / 'comparison.csv'
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', lambda *_a: (str(path), ''))
    choose_export_columns()
    dialog._export_csv()
    import csv
    with path.open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f, delimiter=';'))
    assert rows[0]['prediction'] == '2.0'
    assert rows[0]['well_minus_prediction'] == '3.0'
    assert json.loads(rows[0]['settings'])['source'] == 'boundary'
    dialog.source_combo.setCurrentIndex(2)
    assert dialog.table.item(0, 7).text() == 'Target'
    assert dialog.table.item(0, 10).text() == '5'
    assert dialog.statistics.count == 2
    dialog.close()


def test_analysis_columns_and_excel_report_keep_numeric_and_original_values(
        dataset, monkeypatch, tmp_path, choose_export_columns):
    import csv

    from openpyxl import load_workbook
    from PyQt5 import QtWidgets

    from models_db.model import AliasWellOption, Boundary, CanonicalBoundary, CanonicalWellOption, WellOptionally
    import qt.regression_target_wizard as wizard
    from qt.prediction_well_check_dialog import PredictionWellCheckDialog

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(wizard, 'restore_form', lambda *_args: None)
    monkeypatch.setattr(wizard, 'save_form', lambda *_args: None)
    db, obj, profiles, well, _, _ = dataset
    selected = db.query(ProfileModelPrediction).filter_by(type_model='cls').one()
    selected.prediction = '[0.15, 0.5, 0.75]'
    other_well = Well(name='Без данных', x_coord=20, y_coord=1)
    db.add_all([CanonicalBoundary(canonical_name='Глубина'),
                CanonicalWellOption(id=1, canonical_name='Приток'), other_well])
    db.flush()
    raw_text = '=Описание\n 1,5 + 2,5 м3/сут  '
    db.add_all([AliasWellOption(alias_name='Дебит', canonical_id=1),
                Boundary(well_id=well.id, title='Глубина', depth=0.75),
                WellOptionally(well_id=well.id, option='ДЕБИТ', value='=Описание'),
                WellOptionally(well_id=well.id, option='Приток', value=' 1,5 + 2,5 м3/сут  '),
                WellOptionally(well_id=well.id, option='Другая категория', value='не включать')])
    db.commit()
    report = collect_prediction_wells(db, obj.id, selected, lambda _: [(well, 1, 1), (other_well, 2, 1)])
    dialog = PredictionWellCheckDialog(db, report.samples, 'Test', prediction_type=selected.type_model)
    dialog.hide_missing_check.setChecked(True)
    assert dialog.table.rowCount() == 2
    assert dialog.add_text_column_button.isHidden()
    dialog.source_combo.setCurrentIndex(dialog.source_combo.findData('well_data'))
    assert not dialog.add_text_column_button.isHidden()
    dialog.add_column_button.click()
    dialog.add_text_column_button.click()
    assert dialog.table.horizontalHeaderItem(9).text() == 'Вероятность'
    assert dialog.table.item(0, 13).text() == '4'
    assert dialog.table.item(0, 14).text() == raw_text
    assert dialog.table.item(1, 14).text() == ''
    assert dialog.statistics.count == 0

    dialog.source_combo.setCurrentIndex(dialog.source_combo.findData('boundary'))
    dialog.add_column_button.click()
    dialog.calculate_button.click()
    assert dialog.table.rowCount() == 1
    assert dialog.table.item(0, 13).text() == '4'
    assert dialog.table.item(0, 15).text() == '0.75'
    assert dialog.statistics.mae == 0.25
    assert dialog.statistics.excluded_count == 1

    path = tmp_path / 'report'
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', lambda *_args: (str(path), ''))
    headings = ['Вероятность', 'Приток', 'Приток (исходный текст)', 'Глубина']
    choose_export_columns(headings)
    dialog.excel_export_button.click()
    book = load_workbook(path.with_suffix('.xlsx'))
    try:
        assert book.sheetnames == ['Отчёт', 'Статистика']
        sheet = book['Отчёт']
        columns = {cell.value: cell.column for cell in sheet[1]}
        assert list(columns) == headings
        assert sheet.cell(2, columns['Вероятность']).value == 0.5
        assert sheet.cell(2, columns['Приток']).value == 4
        assert sheet.cell(2, columns['Глубина']).value == 0.75
        text_cell = sheet.cell(2, columns['Приток (исходный текст)'])
        assert text_cell.value == raw_text
        assert text_cell.data_type == 's'
        assert text_cell.alignment.wrap_text
        assert 'Исходный текст' in sheet.cell(1, text_cell.column).comment.text
        assert sheet.cell(3, columns['Вероятность']).value == 0.75
        assert sheet.cell(3, columns['Приток']).value is None
        stats = {row[0]: row[1:] for row in book['Статистика'].values}
        assert stats['Средняя |Δ| (MAE)'][0] == 0.25
        assert stats['Параметр сравнения'][0] == 'Глубина'
        assert stats['Макс. |Δ|'][2] == well.id
    finally:
        book.close()

    path = path.with_suffix('.csv')
    choose_export_columns(['Приток (исходный текст)'])
    dialog.export_button.click()
    with path.open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream, delimiter=';'))
    assert rows[0] == {'Приток (исходный текст)': raw_text}
    choose_export_columns([], checked=headings)
    dialog.excel_export_button.click()
    dialog.remove_column_button.click()
    dialog.remove_column_button.click()
    assert dialog.table.columnCount() == 14
    assert dialog.table.item(0, 13).text() == '4'
    assert dialog.statistics.mae == 0.25
    dialog.close()


def test_well_log_column_titles_follow_saved_aggregation(dataset, monkeypatch, tmp_path, choose_export_columns):
    import csv

    from PyQt5 import QtWidgets

    from models_db.model import WellLog
    from models_db.model_cluster import CanonicalWellLog
    import qt.regression_target_wizard as wizard
    from qt.prediction_well_check_dialog import PredictionWellCheckDialog

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(wizard, 'restore_form', lambda *_args: None)
    monkeypatch.setattr(wizard, 'save_form', lambda *_args: None)
    db, obj, _, well, selected, _ = dataset
    db.add_all([CanonicalWellLog(canonical_name='ГК'),
                WellLog(well_id=well.id, curve_name='ГК', curve_data='[1, 3, 9]', begin=0, end=2, step=1)])
    db.commit()
    report = collect_prediction_wells(db, obj.id, selected, lambda _: [(well, 1, 1)])
    dialog = PredictionWellCheckDialog(db, report.samples, 'Test')
    dialog.source_combo.setCurrentIndex(dialog.source_combo.findData('well_log'))
    dialog.depth_mode_combo.setCurrentIndex(dialog.depth_mode_combo.findData('fixed'))
    dialog.calculate_button.click()
    dialog.add_column_button.click()
    dialog.aggregation_combo.setCurrentIndex(dialog.aggregation_combo.findData('max'))
    dialog.add_column_button.click()
    assert dialog.table.horizontalHeaderItem(10).text() == 'Значение (медиана)'
    assert dialog.table.item(0, 7).text() == 'ГК (медиана)'
    headings = ['ГК (медиана)', 'ГК (максимум)']
    assert [dialog.table.horizontalHeaderItem(index).text() for index in (13, 14)] == headings
    assert [dialog.table.item(0, index).text() for index in (13, 14)] == ['3', '9']

    def unexpected_file_dialog(*_args):
        pytest.fail('The save dialog opened after canceling column selection')

    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', unexpected_file_dialog)
    choose_export_columns([])
    dialog.export_button.click()
    path = tmp_path / 'well_logs.csv'
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', lambda *_args: (str(path), ''))
    choose_export_columns(headings)
    dialog.export_button.click()
    with path.open(encoding='utf-8-sig') as stream:
        rows = list(csv.reader(stream, delimiter=';'))
    assert rows == [headings, ['3.0', '9.0'], ['3.0', '9.0']]
    dialog.close()


def test_export_restores_selection_after_reopening(dataset, monkeypatch, tmp_path, choose_export_columns):
    from PyQt5 import QtWidgets

    import qt.regression_target_wizard as wizard
    from qt.prediction_well_check_dialog import PredictionWellCheckDialog

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(wizard, 'restore_form', lambda *_args: None)
    monkeypatch.setattr(wizard, 'save_form', lambda *_args: None)
    db, obj, _, well, prediction, _ = dataset
    report = collect_prediction_wells(db, obj.id, prediction, lambda _: [(well, 1, 1)])
    path = tmp_path / 'report.csv'
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', lambda *_args: (str(path), ''))
    dialog = PredictionWellCheckDialog(db, report.samples, 'Test')
    choose_export_columns(['Скважина'])
    dialog.export_button.click()
    dialog.close()

    reopened = PredictionWellCheckDialog(db, report.samples, 'Test')
    choose_export_columns(['Профиль'], checked=['Скважина'])
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', lambda *_args: ('', ''))
    reopened.export_button.click()
    choose_export_columns([], checked=['Скважина'])
    reopened.export_button.click()
    choose_export_columns([])
    reopened.excel_export_button.click()
    reopened.close()


def test_named_column_preset_restores_parameters_and_recalculates_values(
        dataset, monkeypatch, tmp_path, choose_export_columns):
    from PyQt5 import QtWidgets

    from models_db.model import Boundary, CanonicalBoundary, CanonicalWellOption, WellLog, WellOptionally
    from models_db.model_cluster import CanonicalWellLog
    import qt.regression_target_wizard as wizard
    from qt.prediction_well_check_dialog import PredictionWellCheckDialog

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(wizard, 'restore_form', lambda *_args: None)
    monkeypatch.setattr(wizard, 'save_form', lambda *_args: None)
    db, obj, _, well, prediction, _ = dataset
    curve = WellLog(well_id=well.id, curve_name='ГК', curve_data='[1, 5, 12, 20]', begin=0, end=3, step=1)
    entry = WellOptionally(well_id=well.id, option='Приток', value='4 + 8 м')
    db.add_all([CanonicalBoundary(canonical_name='Кровля'),
                Boundary(well_id=well.id, title='Кровля', depth=1),
                CanonicalWellLog(canonical_name='ГК'), CanonicalWellOption(canonical_name='Приток'),
                curve, entry])
    db.commit()
    report = collect_prediction_wells(db, obj.id, prediction, lambda _: [(well, 1, 1)])
    dialog = PredictionWellCheckDialog(db, report.samples, 'Test')
    dialog.source_combo.setCurrentIndex(dialog.source_combo.findData('well_data'))
    dialog.add_text_column_button.click()
    dialog.source_combo.setCurrentIndex(dialog.source_combo.findData('well_log'))
    dialog.aggregation_combo.setCurrentIndex(dialog.aggregation_combo.findData('max'))
    dialog.interval.setValue(1)
    dialog.add_column_button.click()
    dialog.calculate_button.click()
    assert dialog.table.item(0, 14).text() == '12'
    monkeypatch.setattr(QtWidgets.QInputDialog, 'getText', lambda *_args, **_kw: ('  Каротаж  ', True))
    dialog.save_preset_button.click()
    assert dialog.presets_combo.currentData() == 'Каротаж'
    path = tmp_path / 'selected.csv'
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', lambda *_args: (str(path), ''))
    choose_export_columns(['Приток (исходный текст)'])
    dialog.export_button.click()
    dialog.close()

    curve.curve_data = '[1, 7, 9, 99]'
    entry.value = '4 + 9 м'
    db.commit()
    reopened = PredictionWellCheckDialog(db, report.samples, 'Test')
    assert reopened.table.columnCount() == 13
    index = reopened.presets_combo.findData('Каротаж')
    assert index > 0
    reopened.presets_combo.setCurrentIndex(index)
    reopened.presets_combo.activated[int].emit(index)
    assert reopened.table.horizontalHeaderItem(13).text() == 'Приток (исходный текст)'
    assert reopened.table.item(0, 13).text() == '4 + 9 м'
    assert reopened.table.horizontalHeaderItem(14).text() == 'ГК (максимум)'
    assert reopened.table.item(0, 14).text() == '9'
    assert reopened.table.item(0, 10).text() == '9'
    assert reopened.aggregation_combo.currentData() == 'max'
    assert reopened.interval.value() == 1
    assert reopened.statistics.count == 2
    choose_export_columns([], checked=['Приток (исходный текст)'])
    reopened.export_button.click()
    reopened.close()


def test_numeric_column_source_choice_applies_to_repeated_well_and_comparison(dataset, monkeypatch):
    from PyQt5 import QtWidgets

    from models_db.model import Boundary, CanonicalBoundary
    import qt.regression_target_wizard as wizard
    from qt.prediction_well_check_dialog import PredictionWellCheckDialog

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(wizard, 'restore_form', lambda *_args: None)
    monkeypatch.setattr(wizard, 'save_form', lambda *_args: None)
    db, obj, _, well, selected, _ = dataset
    db.add_all([CanonicalBoundary(canonical_name='Граница'),
                Boundary(well_id=well.id, title='Граница', depth=5),
                Boundary(well_id=well.id, title='Граница', depth=7)])
    db.commit()
    report = collect_prediction_wells(db, obj.id, selected, lambda _: [(well, 1, 1)])
    dialog = PredictionWellCheckDialog(db, report.samples, 'Test')
    dialog.calculate_button.click()
    dialog.add_column_button.click()
    assert dialog.statistics.count == 0
    assert dialog.table.item(0, 13).text() == 'выберите…'
    monkeypatch.setattr(QtWidgets.QInputDialog, 'getItem',
                        lambda _parent, _title, _prompt, items, *_args: (items[1], True))
    dialog.table.cellDoubleClicked.emit(0, 13)
    assert [dialog.table.item(row, 13).text() for row in range(2)] == ['7', '7']
    assert [candidate.resolution.value for candidate in dialog.candidates] == [7, 7]
    assert dialog.statistics.count == 2
    dialog.close()


def test_comparison_colors_follow_absolute_error_and_tolerance(dataset, monkeypatch):
    from dataclasses import replace
    from PyQt5 import QtWidgets
    import qt.regression_target_wizard as wizard
    from qt.prediction_well_check_dialog import PredictionWellCheckDialog
    from regression_target.service import Resolution

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(wizard, 'restore_form', lambda *_args: None)
    monkeypatch.setattr(wizard, 'save_form', lambda *_args: None)
    monkeypatch.setattr(wizard, 'list_canonical_targets', lambda *_args: [])
    db, obj, profiles, well, selected, second = dataset
    report = collect_prediction_wells(db, obj.id, selected, lambda _: [(well, 1, 1)])
    dialog = PredictionWellCheckDialog(db, [replace(report.samples[0]) for _ in range(4)], 'Test')
    for c, value in zip(dialog.candidates, [2, 3, -8, None]):
        c.resolution = Resolution('resolved', value=value) if value is not None else Resolution('missing')
    dialog.absolute_tolerance.setValue(0.1)
    dialog._render()
    colors = [dialog.table.item(row, 11).background().color() for row in range(4)]
    assert colors[0].name() == '#d9f2df'
    assert colors[2].name() == '#f4a6a6'  # largest magnitude, including negative delta
    assert colors[1].green() > colors[2].green()  # small error is closer to yellow
    assert colors[3].name() == '#eeeeee'
    assert dialog.table.item(2, 1).background().color() == colors[2]
    dialog.absolute_tolerance.setValue(1)
    assert dialog.table.item(1, 11).background().color().name() == '#d9f2df'
    dialog.absolute_tolerance.setValue(10)
    assert dialog.table.item(2, 11).background().color().name() == '#d9f2df'
    assert dialog.table.horizontalHeaderItem(11).text() == 'Δ'
    assert 'минус прогноз' in dialog.table.horizontalHeaderItem(11).toolTip()
    assert dialog.table.horizontalHeaderItem(7).text() == 'Параметр'
    assert dialog.table.columnWidth(3) > dialog.table.fontMetrics().horizontalAdvance(
        dialog.table.item(0, 3).text())
    assert dialog.table.horizontalHeader().sectionResizeMode(3) == QtWidgets.QHeaderView.Interactive
    dialog.close()


def test_statistics_refresh_navigation_and_export(dataset, monkeypatch, tmp_path):
    from dataclasses import replace
    from PyQt5 import QtWidgets
    import qt.regression_target_wizard as wizard
    from qt.prediction_well_check_dialog import PredictionWellCheckDialog
    from regression_target.service import Resolution
    import csv

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(wizard, 'restore_form', lambda *_args: None)
    monkeypatch.setattr(wizard, 'save_form', lambda *_args: None)
    monkeypatch.setattr(wizard, 'list_canonical_targets', lambda *_args: [])
    db, obj, profiles, well, selected, second = dataset
    report = collect_prediction_wells(db, obj.id, selected, lambda _: [(well, 1, 1)])
    samples = [report.samples[0], report.samples[1], replace(report.samples[0])]
    dialog = PredictionWellCheckDialog(db, samples, 'Test')
    assert dialog.statistics.count == 0
    assert dialog.statistics_labels['mae'].text() == '—'
    assert not dialog.maximum_button.isEnabled()
    for c, delta in zip(dialog.candidates, [2, -4, None]):
        c.resolution = (Resolution('resolved', value=c.stored_value + delta)
                        if delta is not None else Resolution('ambiguous'))
    dialog._render()
    assert dialog.statistics.count == 2 and dialog.statistics.well_count == 1
    assert dialog.statistics_labels['mae'].text() == '3'
    assert dialog.statistics_labels['bias'].text() == '-1'
    assert dialog.candidates[1].profile_name in dialog.statistics_labels['maximum'].text()
    dialog.maximum_button.click()
    assert dialog.table.currentRow() == 1
    dialog.minimum_button.click()
    assert dialog.table.currentRow() == 0
    dialog.absolute_tolerance.setValue(2)
    assert dialog.statistics_labels['match_percent'].text() == '1 из 2 (50.0%)'
    dialog.hide_missing_check.setChecked(True)
    assert dialog.statistics.count == 2
    path = tmp_path / 'statistics.csv'
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', lambda *_a: (str(path), ''))
    dialog.statistics_export_button.click()
    with path.open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f, delimiter=';'))
    maximum = next(r for r in rows if r['metric'] == 'maximum')
    assert maximum['value'] == '4.0'
    assert maximum['well_id'] == str(well.id)
    assert maximum['profile_id'] == str(profiles[1].id)
    assert next(r for r in rows if r['metric'] == 'count')['value'] == '2'
    dialog.close()
