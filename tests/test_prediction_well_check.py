import datetime
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models_db.model import (Base, GeoradarObject, Research, Profile,
                             ProfileModelPrediction, PredictionCorrect, Well)
import models_db.model_cluster  # register related mappings
import models_db.model_profile_features
from prediction_well_check import collect_prediction_wells


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


def test_comparison_dialog_reuses_sources_and_never_exposes_markup_actions(dataset, monkeypatch, tmp_path):
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
    dialog = PredictionWellCheckDialog(db, report.samples, 'Test')
    assert [dialog.source_combo.itemData(i) for i in range(3)] == ['boundary', 'well_data', 'well_log']
    assert not dialog.isModal()
    assert dialog.replace_button.isHidden()
    assert dialog.add_selected_button.isHidden()
    assert dialog.delete_selected_button.isHidden()
    assert dialog.table.isColumnHidden(0)
    assert dialog.table.item(0, 9).text() == '2'
    dialog.calculate_button.click()
    assert all(c.resolution.value == 5 for c in dialog.candidates)
    assert dialog.table.item(0, 11).text() == '+3'
    path = tmp_path / 'comparison.csv'
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', lambda *_a: (str(path), ''))
    dialog._export_csv()
    import csv
    with path.open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f, delimiter=';'))
    assert rows[0]['prediction'] == '2.0'
    assert rows[0]['well_minus_prediction'] == '3.0'
    assert json.loads(rows[0]['settings'])['source'] == 'boundary'
    dialog.source_combo.setCurrentIndex(2)
    assert dialog.candidates[0].resolution is None
    assert dialog.table.item(0, 10).text() == ''
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
