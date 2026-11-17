import ast
import json
from html import escape
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from models_db.model import (
    Base, BindingLayerPrediction, GeoradarObject, Layers, Profile,
    ProfileModelPrediction, Research,
)
import models_db.model_cluster  # Register related mappings.
import models_db.model_profile_features
from velocity_prediction_binding import bind_object_predictions


@pytest.fixture
def dataset():
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    obj, other = GeoradarObject(title='Object'), GeoradarObject(title='Other')
    first, second = Research(object=obj), Research(object=obj)
    profiles = [
        Profile(title='Source', research=first),
        Profile(title='Same research', research=first),
        Profile(title='Other research', research=second),
        Profile(title='Other object', research=Research(object=other)),
    ]
    for profile in profiles:
        # Deliberately create layers in the reverse order to the source bindings.
        profile.layers = [Layers(layer_title='bottom'), Layers(layer_title='top')]
        profile.predictions = [
            ProfileModelPrediction(model_id=7, type_model='reg', prediction='[1, 2]'),
            ProfileModelPrediction(model_id=8, type_model='reg', prediction='[3, 4]'),
            ProfileModelPrediction(model_id=7, type_model='cls', prediction='[0, 1]'),
        ]
    db.add_all(profiles)
    db.flush()
    for title, model_id in [('top', 7), ('bottom', 8)]:
        layer = next(layer for layer in profiles[0].layers if layer.layer_title == title)
        prediction = next(p for p in profiles[0].predictions if p.type_model == 'reg' and p.model_id == model_id)
        db.add(BindingLayerPrediction(layer_id=layer.id, prediction_id=prediction.id))
    db.commit()
    try:
        yield SimpleNamespace(db=db, obj=obj, profiles=profiles)
    finally:
        db.close()
        engine.dispose()


def snapshot(db):
    return db.query(
        BindingLayerPrediction.id, BindingLayerPrediction.layer_id,
        BindingLayerPrediction.prediction_id,
    ).order_by(BindingLayerPrediction.id).all()


def bindings(db, profile):
    return db.query(BindingLayerPrediction).join(Layers).filter(
        Layers.profile_id == profile.id,
    ).order_by(BindingLayerPrediction.id).all()


def apply(dataset):
    result = bind_object_predictions(dataset.db, dataset.obj.id, dataset.profiles[0].id)
    dataset.db.commit()
    return result


def test_matches_local_predictions_across_all_researches_in_source_order(dataset):
    source_before = snapshot(dataset.db)
    assert apply(dataset) == 2
    for profile in dataset.profiles[:3]:
        result = bindings(dataset.db, profile)
        assert [b.layer.layer_title for b in result] == ['top', 'bottom']
        assert [(b.prediction.type_model, b.prediction.model_id) for b in result] == [('reg', 7), ('reg', 8)]
        assert all(b.prediction.profile_id == profile.id for b in result)
    assert snapshot(dataset.db)[:2] == source_before
    assert bindings(dataset.db, dataset.profiles[3]) == []
    before_repeat = snapshot(dataset.db)
    assert apply(dataset) == 2
    assert snapshot(dataset.db) == before_repeat


def test_replaces_conflicting_and_duplicate_target_bindings(dataset):
    target = dataset.profiles[1]
    for _ in range(2):
        dataset.db.add(BindingLayerPrediction(
            layer_id=target.layers[0].id, prediction_id=target.predictions[2].id,
        ))
    dataset.db.commit()
    apply(dataset)
    result = bindings(dataset.db, target)
    assert [(b.layer.layer_title, b.prediction.model_id) for b in result] == [('top', 7), ('bottom', 8)]


@pytest.mark.parametrize('problem, expected', [
    ('extra_layer', 'лишние слои'),
    ('missing_layer', 'нет слоёв'),
    ('no_layers', 'нет слоёв'),
    ('duplicate_layer', 'повторяющиеся'),
    ('empty_title', 'пустые'),
    ('renamed_layer', 'состав слоёв отличается'),
    ('missing_prediction', 'нет нужного prediction'),
    ('duplicate_prediction', 'несколько prediction'),
    ('wrong_type', 'нет нужного prediction'),
])
def test_invalid_last_profile_aborts_every_profile_before_mutation(dataset, problem, expected):
    db = dataset.db
    target = dataset.profiles[2]
    if problem == 'extra_layer':
        db.add(Layers(profile_id=target.id, layer_title='middle'))
    elif problem == 'missing_layer':
        db.delete(target.layers[0])
    elif problem == 'no_layers':
        for layer in list(target.layers):
            db.delete(layer)
    elif problem == 'duplicate_layer':
        db.add(Layers(profile_id=target.id, layer_title='top'))
    elif problem == 'empty_title':
        target.layers[0].layer_title = ''
    elif problem == 'renamed_layer':
        target.layers[0].layer_title = 'different bottom'
    elif problem == 'missing_prediction':
        db.delete(target.predictions[0])
    elif problem == 'duplicate_prediction':
        db.add(ProfileModelPrediction(profile_id=target.id, model_id=7, type_model='reg'))
    elif problem == 'wrong_type':
        target.predictions[0].type_model = 'other'
    db.commit()
    before = snapshot(db)
    with pytest.raises(ValueError, match=expected) as error:
        apply(dataset)
    assert target.title in str(error.value)
    assert snapshot(db) == before
    assert not db.new and not db.deleted and not db.dirty


@pytest.mark.parametrize('problem', ['unbound', 'multiple_bindings', 'foreign_prediction', 'deleted_prediction'])
def test_requires_unambiguous_complete_source_bindings(dataset, problem):
    db, source = dataset.db, dataset.profiles[0]
    binding = bindings(db, source)[0]
    if problem == 'unbound':
        db.delete(binding)
    elif problem == 'multiple_bindings':
        db.add(BindingLayerPrediction(layer_id=binding.layer_id, prediction_id=binding.prediction_id))
    elif problem == 'foreign_prediction':
        binding.prediction_id = dataset.profiles[1].predictions[0].id
    else:
        binding.prediction_id = 999999
    db.commit()
    before = snapshot(db)
    with pytest.raises(ValueError, match='Текущий профиль'):
        apply(dataset)
    assert snapshot(db) == before


def test_source_must_belong_to_selected_object(dataset):
    with pytest.raises(ValueError, match='Выберите текущий профиль'):
        bind_object_predictions(dataset.db, dataset.obj.id, dataset.profiles[3].id)


def test_same_prediction_can_be_used_for_both_boundaries(dataset):
    source = dataset.profiles[0]
    source_bindings = bindings(dataset.db, source)
    source_bindings[1].prediction_id = source_bindings[0].prediction_id
    dataset.db.commit()
    apply(dataset)
    for profile in dataset.profiles[:3]:
        assert [b.prediction.model_id for b in bindings(dataset.db, profile)] == [7, 7]


def test_all_profiles_reported_when_predictions_missing(dataset):
    for target in dataset.profiles[1:3]:
        dataset.db.delete(target.predictions[0])
    dataset.db.commit()
    with pytest.raises(ValueError) as error:
        apply(dataset)
    assert all(target.title in str(error.value) for target in dataset.profiles[1:3])


def load_handler(dataset, messages, refreshed):
    path = Path(__file__).resolve().parents[1] / 'velocity_prediction.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    handler = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == 'add_bind_vel_prediction_all_profiles')
    namespace = dict(
        get_object_id=lambda: dataset.obj.id,
        get_profile_id=lambda: dataset.profiles[0].id,
        bind_object_predictions=bind_object_predictions, session=dataset.db,
        set_info=lambda text, color: messages.append((text, color)),
        update_list_bind_vel_prediction=lambda: refreshed.append(True),
        SQLAlchemyError=SQLAlchemyError, escape=escape,
    )
    exec(compile(ast.Module(body=[handler], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def test_button_handler_commits_and_refreshes(dataset):
    messages, refreshed = [], []
    namespace = load_handler(dataset, messages, refreshed)
    namespace['add_bind_vel_prediction_all_profiles']()
    assert messages[-1][1] == 'green'
    assert refreshed == [True]
    assert dataset.db.query(BindingLayerPrediction).count() == 6
    assert not dataset.db.new and not dataset.db.deleted


def test_button_handler_rolls_back_failed_save(dataset, monkeypatch):
    before = snapshot(dataset.db)
    messages, refreshed = [], []
    namespace = load_handler(dataset, messages, refreshed)
    def fail_commit():
        # Ensure actual SQL has been issued before simulating a failed save.
        dataset.db.flush()
        raise SQLAlchemyError('save failed')
    monkeypatch.setattr(dataset.db, 'commit', fail_commit)
    namespace['add_bind_vel_prediction_all_profiles']()
    assert messages[-1][1] == 'red'
    assert 'отменены' in messages[-1][0]
    assert refreshed == []
    assert snapshot(dataset.db) == before


def test_bulk_bound_profiles_render_thin_intervals_with_production_calculation(dataset):
    from scipy.signal import savgol_filter
    from models_db.model import CurrentProfile
    from velocity_depth_signal import build_velocity_depth_signal

    db = dataset.db
    for profile in dataset.profiles[:3]:
        for layer in profile.layers:
            layer.layer_line = json.dumps([2 if layer.layer_title == 'top' else 3] * 7)
        for prediction in profile.predictions:
            prediction.prediction = json.dumps([1.5 if prediction.model_id == 7 else 1.9] * 7)
    db.commit()
    apply(dataset)
    before = snapshot(db)
    target = dataset.profiles[2]
    db.add(CurrentProfile(profile_id=target.id, signal=json.dumps([list(range(512))] * 7)))
    db.commit()
    messages = []
    namespace = dict(
        session=db, json=json, escape=escape, savgol_filter=savgol_filter,
        CurrentProfile=CurrentProfile, Layers=Layers, BindingLayerPrediction=BindingLayerPrediction,
        get_profile_id=lambda: target.id, build_velocity_depth_signal=build_velocity_depth_signal,
        set_info=lambda text, color: messages.append(text),
        ui=SimpleNamespace(checkBox_minmax=SimpleNamespace(isChecked=lambda: False),
                           checkBox_corr_pred=SimpleNamespace(isChecked=lambda: False)),
    )
    root = Path(__file__).resolve().parents[1]
    for filename, names in [('func.py', {'interpolate_list', 'savgol_line'}),
                            ('velocity_prediction.py', {'calc_deep_predict_current_profile'})]:
        path = root / filename
        functions = [node for node in ast.parse(path.read_text(encoding='utf-8')).body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    result = namespace['calc_deep_predict_current_profile']()
    assert len(result) == 7
    assert all(len(trace) == 205 for trace in result)
    assert messages == []
    assert snapshot(db) == before
