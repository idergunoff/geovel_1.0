"""Exercise production table builders without starting the global Qt window."""

import ast
import json
import pickle
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models_db import model as db_models
from models_db import model_cluster, model_profile_features  # Register ORM relationships.


ROOT = Path(__file__).resolve().parents[1]


class CountingModel:
    calls = []

    def predict(self, sample):
        self.calls.append(list(sample))
        return np.asarray(sample)[:, 0] * 2

    def predict_proba(self, sample):
        self.calls.append(list(sample))
        first = np.asarray(sample)[:, 0] / 100
        return np.column_stack([first, 1 - first])


def load_functions(path, namespace, names=None):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and (names is None or node.name in names)
    ]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)


@pytest.fixture
def runtime():
    engine = create_engine('sqlite:///:memory:')
    db_models.Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    messages = []
    namespace = dict(vars(db_models))
    namespace.update(
        session=session, json=json, pd=pd, np=np, pickle=pickle,
        set_info=lambda text, color: messages.append(text),
        tqdm=lambda values: values,
        parse_range_exception=lambda value: [],
        list_param_geovel=[],
        ui=SimpleNamespace(progressBar=SimpleNamespace(
            setMaximum=lambda value: None, setValue=lambda value: None,
        )),
    )
    load_functions(ROOT / 'func.py', namespace, {
        'get_list_param_numerical', 'calc_atrib_measure', 'get_interpolate_list',
    })
    load_functions(ROOT / 'build_table.py', namespace)
    CountingModel.calls = []
    try:
        yield SimpleNamespace(db=session, functions=namespace, messages=messages)
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def formation(runtime):
    profile = db_models.Profile(
        id=7, title='Test profile', x_pulc=json.dumps([10, 20, 30]),
        y_pulc=json.dumps([1, 2, 3]),
    )
    upper = db_models.Layers(layer_line=json.dumps([0, 0, 0]), profile=profile)
    lower = db_models.Layers(layer_line=json.dumps([2, 2, 2]), profile=profile)
    formation = db_models.Formation(id=11, profile=profile, layer_up=upper, layer_down=lower)
    runtime.db.add(formation)
    runtime.db.commit()
    return formation


def add_model(runtime, tmp_path, type_model, model_id, params, mask=None):
    path = tmp_path / f'{type_model}_{model_id}.pkl'
    path.write_bytes(pickle.dumps(CountingModel()))
    model_class = db_models.TrainedModelClass if type_model == 'cls' else db_models.TrainedModelReg
    model = model_class(
        id=model_id, title=f'{type_model} {model_id}', path_model=str(path),
        list_params=json.dumps(params), except_signal='', except_crl='',
    )
    runtime.db.add(model)
    if mask is not None:
        parameter_mask = db_models.ParameterMask(mask=json.dumps(mask))
        runtime.db.add(parameter_mask)
        runtime.db.flush()
        mask_class = db_models.TrainedModelClassMask if type_model == 'cls' else db_models.TrainedModelRegMask
        runtime.db.add(mask_class(model_id=model_id, mask_id=parameter_mask.id))
    runtime.db.commit()
    return model


def save_prediction(runtime, formation, type_model, model_id, values, legacy_stale=False):
    runtime.db.add(db_models.ProfileModelPrediction(
        profile_id=formation.profile_id, type_model=type_model,
        model_id=model_id, prediction=json.dumps(values), is_stale=legacy_stale,
    ))
    runtime.db.commit()


@pytest.mark.parametrize('type_model,values', [('cls', [0.1, 0.2, 0.3]), ('reg', [10, 20, 30])])
@pytest.mark.parametrize('legacy_stale', [False, True])
def test_saved_predictions_skip_model_loading_and_parameter_collection(runtime, formation, type_model, values, legacy_stale):
    # The trained model and its file deliberately do not exist.
    save_prediction(runtime, formation, 'cls', 4, [0.1, 0.2, 0.3], legacy_stale)
    save_prediction(runtime, formation, 'reg', 4, [10, 20, 30], legacy_stale)
    predict = runtime.functions['calc_profile_model_predict']

    assert predict(f'model_{type_model}_test_id4', formation) == values
    assert predict(f'model_{type_model}_test_id4', formation, measure_indices=[2, 0]) == {
        2: values[2], 0: values[0],
    }
    assert len(CountingModel.calls) == 0
    assert all('Используется сохранённый прогноз' in message for message in runtime.messages)


@pytest.mark.parametrize('legacy_stale', [False, True])
def test_training_prefetch_distinguishes_model_types(runtime, formation, legacy_stale):
    save_prediction(runtime, formation, 'cls', 4, [0.1, 0.2, 0.3], legacy_stale)
    save_prediction(runtime, formation, 'reg', 4, [10, 20, 30], legacy_stale)
    cache = runtime.functions['_prefetch_profile_data'](
        [SimpleNamespace(profile_id=formation.profile_id)], ['model_cls_test_id4', 'model_reg_test_id4'],
    )

    assert cache[(formation.profile_id, 'model_cls_test_id4')] == [0.1, 0.2, 0.3]
    assert cache[(formation.profile_id, 'model_reg_test_id4')] == [10, 20, 30]


@pytest.mark.parametrize('type_model,analysis', [('cls', 'mlp'), ('reg', 'regmod')])
def test_object_table_skips_nested_models_excluded_by_mask(runtime, formation, tmp_path, type_model, analysis):
    model = add_model(runtime, tmp_path, type_model, 1, ['X', 'model_reg_unused_id99'], mask=['X'])
    table, _ = runtime.functions['build_table_test'](
        analysis, model, formation, use_model_mask=True,
    )

    assert table['X'].tolist() == [10, 20, 30]
    assert 'model_reg_unused_id99' not in table.columns
    assert not CountingModel.calls
    assert runtime.db.query(db_models.ProfileModelPrediction).count() == 0


@pytest.mark.parametrize('type_model,expected', [('reg', [20, 40, 60]), ('cls', [0.1, 0.2, 0.3])])
def test_full_nested_prediction_is_saved_and_reused(runtime, formation, tmp_path, type_model, expected):
    nested = add_model(runtime, tmp_path, type_model, 2, ['X'])
    parent = add_model(runtime, tmp_path, 'cls', 1, [f'model_{type_model}_nested_id2'])
    build = runtime.functions['build_table_test']

    first, _ = build('mlp', parent, formation, use_model_mask=True)
    assert first[f'model_{type_model}_nested_id2'].tolist() == expected
    assert runtime.db.query(db_models.ProfileModelPrediction).count() == 1
    assert len(CountingModel.calls) == 1

    Path(nested.path_model).unlink()
    second, _ = build('mlp', parent, formation, use_model_mask=True)
    pd.testing.assert_frame_equal(first, second)
    assert len(CountingModel.calls) == 1
    assert runtime.db.query(db_models.ProfileModelPrediction).count() == 1


def test_partial_nested_prediction_keeps_trace_indices_and_does_not_save_partial_profile(runtime, formation, tmp_path):
    nested = add_model(runtime, tmp_path, 'reg', 2, ['X'])
    parent = add_model(runtime, tmp_path, 'cls', 1, ['model_reg_nested_id2'])
    build = runtime.functions['build_table_test']

    partial, _ = build('mlp', parent, formation, measure_indices=[2, 0], use_model_mask=True)
    assert partial['prof_index'].tolist() == ['7_0', '7_2']
    assert partial['model_reg_nested_id2'].tolist() == [20, 60]
    assert runtime.db.query(db_models.ProfileModelPrediction).count() == 0
    assert CountingModel.calls == [[[10], [30]]]

    full, _ = build('mlp', parent, formation, use_model_mask=True)
    assert runtime.db.query(db_models.ProfileModelPrediction).count() == 1
    Path(nested.path_model).unlink()
    repeated, _ = build('mlp', parent, formation, measure_indices=[2, 0], use_model_mask=True)
    pd.testing.assert_frame_equal(repeated, full.iloc[[0, 2]].reset_index(drop=True))
    assert len(CountingModel.calls) == 2


def test_shared_table_grows_for_different_masks_of_same_sources(runtime, formation, tmp_path):
    x_model = add_model(runtime, tmp_path, 'reg', 1, ['X', 'Y'], mask=['X'])
    y_model = add_model(runtime, tmp_path, 'reg', 2, ['X', 'Y'], mask=['Y'])
    equivalent = add_model(runtime, tmp_path, 'reg', 3, ['X', 'Y'], mask=['X'])
    build = runtime.functions['build_table_test']
    cache = {}

    x_table, _ = build('regmod', x_model, formation, model_table_cache=cache, use_model_mask=True)
    y_table, _ = build('regmod', y_model, formation, model_table_cache=cache, use_model_mask=True)
    same_table, _ = build('regmod', equivalent, formation, model_table_cache=cache, use_model_mask=True)

    assert x_table['X'].tolist() == [10, 20, 30]
    assert y_table['Y'].tolist() == [1, 2, 3]
    assert same_table is y_table
    assert len(cache) == 1


def test_masks_of_expanded_features_share_collection_and_keep_required_column(runtime, formation, tmp_path):
    formation.profile.signal = json.dumps([[1, 3], [2, 6], [3, 9]])
    runtime.db.commit()
    first_model = add_model(runtime, tmp_path, 'reg', 1, ['sep_Abase_2'], mask=['sep_Abase_1'])
    second_model = add_model(runtime, tmp_path, 'reg', 2, ['sep_Abase_2'], mask=['sep_Abase_2'])
    build = runtime.functions['build_table_test']
    cache = {}

    first, _ = build('regmod', first_model, formation, model_table_cache=cache, use_model_mask=True)
    second, _ = build('regmod', second_model, formation, model_table_cache=cache, use_model_mask=True)

    assert first['sep_Abase_1'].tolist() == [1, 2, 3]
    assert second['sep_Abase_2'].tolist() == [3, 6, 9]
    assert first is second
    assert len(cache) == 1


def test_nested_models_collect_common_features_once_and_apply_their_own_masks(runtime, formation, tmp_path):
    formation.profile.signal = json.dumps([[1, 3], [2, 6], [3, 9]])
    runtime.db.commit()
    sources = ['sep_Abase_2', 'X', 'Y', 'model_reg_unused_id99']
    add_model(runtime, tmp_path, 'reg', 2, sources, mask=['X', 'sep_Abase_1'])
    add_model(runtime, tmp_path, 'reg', 3, sources, mask=['Y', 'sep_Abase_2'])
    parent = add_model(runtime, tmp_path, 'cls', 1, ['model_reg_first_id2', 'model_reg_second_id3'])
    transformations = []
    transform = runtime.functions['calc_atrib_measure']

    def counted_transform(signal, attribute):
        transformations.append((signal, attribute))
        return transform(signal, attribute)

    runtime.functions['calc_atrib_measure'] = counted_transform
    cache = {}
    table, _ = runtime.functions['build_table_test'](
        'mlp', parent, formation, model_table_cache=cache, use_model_mask=True,
    )

    assert len(transformations) == 3  # One transformation per trace for both models.
    assert CountingModel.calls == [
        [[10, 1], [20, 2], [30, 3]],
        [[1, 3], [2, 6], [3, 9]],
    ]
    assert table['model_reg_first_id2'].tolist() == [20, 40, 60]
    assert table['model_reg_second_id3'].tolist() == [2, 4, 6]
    assert runtime.db.query(db_models.ProfileModelPrediction).count() == 2
    assert len(cache) == 2  # Parent table and one shared regression source table.


@pytest.mark.parametrize('second_type,second_analysis', [('reg', 'regmod'), ('cls', 'mlp')])
def test_source_order_and_model_type_do_not_split_shared_table(runtime, formation, tmp_path, second_type, second_analysis):
    first = add_model(runtime, tmp_path, 'reg', 1, ['X', 'Y'], mask=['X'])
    second = add_model(runtime, tmp_path, second_type, 2, ['Y', 'X'], mask=['Y'])
    cache = {}
    build = runtime.functions['build_table_test']

    first_table, _ = build('regmod', first, formation, model_table_cache=cache, use_model_mask=True)
    second_table, _ = build(second_analysis, second, formation, model_table_cache=cache, use_model_mask=True)

    assert second_table['X'].tolist() == [10, 20, 30]
    assert second_table['Y'].tolist() == [1, 2, 3]
    assert len(cache) == 1


@pytest.mark.parametrize('measure_indices', [None, [2, 0]])
def test_wide_shared_table_extends_without_fragmentation_or_recalculating_features(runtime, formation, tmp_path, measure_indices):
    formation.profile.signal = json.dumps([
        (np.arange(512) + offset).tolist() for offset in (0, 1000, 2000)
    ])
    runtime.db.commit()
    sources = ['X', 'Signal_Abase', 'sep_Abase_128']
    first = add_model(runtime, tmp_path, 'reg', 1, sources, mask=['X'])
    second = add_model(runtime, tmp_path, 'reg', 2, sources, mask=['Signal_Abase_1'])
    third = add_model(runtime, tmp_path, 'reg', 3, sources, mask=['sep_Abase_128'])
    equivalent = add_model(runtime, tmp_path, 'reg', 4, sources, mask=['Signal_Abase_512', 'X'])
    cache = {}
    build = runtime.functions['build_table_test']
    transformations = []
    transform = runtime.functions['calc_atrib_measure']

    def counted_transform(signal, attribute):
        transformations.append((signal, attribute))
        return transform(signal, attribute)

    runtime.functions['calc_atrib_measure'] = counted_transform
    with warnings.catch_warnings():
        warnings.simplefilter('error', pd.errors.PerformanceWarning)
        initial, _ = build('regmod', first, formation, model_table_cache=cache,
                           measure_indices=measure_indices, use_model_mask=True)
        snapshot = initial.copy()
        wide, _ = build('regmod', second, formation, model_table_cache=cache,
                        measure_indices=measure_indices, use_model_mask=True)
        extended, _ = build('regmod', third, formation, model_table_cache=cache,
                            measure_indices=measure_indices, use_model_mask=True)
        reused, _ = build('regmod', equivalent, formation, model_table_cache=cache,
                          measure_indices=measure_indices, use_model_mask=True)

    indices = [0, 1, 2] if measure_indices is None else [0, 2]
    assert wide['Signal_Abase_512'].tolist() == [index * 1000 + 511 for index in indices]
    assert extended['sep_Abase_128'].tolist() == [index * 1000 + 1 for index in indices]
    pd.testing.assert_frame_equal(extended[snapshot.columns], snapshot)
    assert extended.columns.is_unique
    assert len(extended.columns) == 3 + 1 + 512 + 128
    assert len(transformations) == 2 * len(indices)
    assert reused is extended
    assert len(cache) == 1


def test_trace_selection_and_exceptions_keep_separate_source_tables(runtime, formation, tmp_path):
    model = add_model(runtime, tmp_path, 'reg', 1, ['X'])
    cache = {}
    build = runtime.functions['build_table_test']
    full, _ = build('regmod', model, formation, model_table_cache=cache, use_model_mask=True)
    partial, _ = build('regmod', model, formation, model_table_cache=cache, measure_indices=[2, 0], use_model_mask=True)
    model.except_signal = '1'
    changed, _ = build('regmod', model, formation, model_table_cache=cache, use_model_mask=True)

    assert full['X'].tolist() == [10, 20, 30]
    assert partial['X'].tolist() == [10, 30]
    assert full is not partial and full is not changed
    assert len(cache) == 3


def test_multilevel_models_are_saved_once_at_every_level(runtime, formation, tmp_path):
    leaf = add_model(runtime, tmp_path, 'reg', 3, ['X'])
    middle = add_model(runtime, tmp_path, 'reg', 2, ['model_reg_leaf_id3'])
    parent = add_model(runtime, tmp_path, 'cls', 1, ['model_reg_middle_id2'])
    build = runtime.functions['build_table_test']

    first, _ = build('mlp', parent, formation, use_model_mask=True)
    assert first['model_reg_middle_id2'].tolist() == [40, 80, 120]
    assert runtime.db.query(db_models.ProfileModelPrediction).count() == 2
    assert len(CountingModel.calls) == 2
    Path(leaf.path_model).unlink()
    Path(middle.path_model).unlink()
    repeated, _ = build('mlp', parent, formation, use_model_mask=True)
    pd.testing.assert_frame_equal(first, repeated)
    assert len(CountingModel.calls) == 2


@pytest.mark.parametrize('invalid_values', [None, [99], {'0': 99}])
def test_incomplete_saved_prediction_is_replaced_by_complete_result(runtime, formation, tmp_path, invalid_values):
    add_model(runtime, tmp_path, 'reg', 2, ['X'])
    save_prediction(runtime, formation, 'reg', 2, invalid_values)
    predict = runtime.functions['calc_profile_model_predict']

    assert predict('model_reg_test_id2', formation) == [20, 40, 60]
    assert predict('model_reg_test_id2', formation) == [20, 40, 60]
    assert len(CountingModel.calls) == 1


def _terrain_grids(runtime, formation):
    runtime.functions.update({name: cls for name, cls in vars(model_profile_features).items()
                              if name.endswith('Profile')})
    for name in ('wavelet', 'fractal', 'entropy', 'nonlinear', 'morphology', 'frequency',
                 'envelope', 'autocorr', 'emd', 'hht'):
        runtime.functions[f'list_{name}_features'] = []
    for name in ('morphology', 'frequency', 'envelope', 'autocorr', 'emd', 'hht'):
        runtime.functions[f'list_{name}_feature'] = []
    obj = db_models.GeoradarObject(title='Terrain test')
    formation.profile.research = db_models.Research(object=obj)
    common = db_models.CommonGrid(title='Common', type='r', is_local=False,
        grid_table=json.dumps([[x, y, 100] for x in range(0, 41, 10) for y in range(5)]))
    local = db_models.CommonGrid(title='Local', type='r', is_local=True,
        grid_table=json.dumps([[x, y, 200] for x in range(0, 41, 10) for y in range(5)]))
    runtime.db.add_all([obj, common, local])
    runtime.db.commit()
    return obj, common, local


def test_prediction_table_refreshes_land_after_binding(runtime, formation, tmp_path):
    from terrain_service import bind_grid
    obj, common, local = _terrain_grids(runtime, formation)
    model = add_model(runtime, tmp_path, 'reg', 1, ['land'])
    build = runtime.functions['build_table_test']
    cache = {}
    first, _ = build('regmod', model, formation, model_table_cache=cache)
    assert first['land'].tolist() == [100]*3
    bind_grid(runtime.db, obj.id, local.id)
    runtime.db.commit()
    second, _ = build('regmod', model, formation, model_table_cache=cache)
    assert second['land'].tolist() == [200]*3
    assert json.loads(formation.profile.abs_relief) == [200]*3


def test_saved_nested_prediction_is_preserved_and_reused_after_rebinding(runtime, formation, tmp_path):
    from terrain_service import bind_grid
    obj, common, local = _terrain_grids(runtime, formation)
    nested = add_model(runtime, tmp_path, 'reg', 2, ['land'])
    parent = add_model(runtime, tmp_path, 'reg', 1, ['land', 'model_reg_nested_id2'])
    build = runtime.functions['build_table_test']
    first, _ = build('regmod', parent, formation)
    assert first['land'].tolist() == [100]*3
    assert first['model_reg_nested_id2'].tolist() == [200]*3
    saved = runtime.db.query(db_models.ProfileModelPrediction).one()
    saved.is_stale = True  # Legacy marks must not block an existing result.
    runtime.db.commit()
    payload = saved.prediction
    Path(nested.path_model).unlink()
    bind_grid(runtime.db, obj.id, local.id)
    runtime.db.commit()
    cache = runtime.functions['_prefetch_profile_data'](
        [SimpleNamespace(profile_id=formation.profile_id)], ['model_reg_nested_id2'])
    assert cache[(formation.profile_id, 'model_reg_nested_id2')] == [200]*3
    second, _ = build('regmod', parent, formation)
    assert second['land'].tolist() == [200]*3
    assert second['model_reg_nested_id2'].tolist() == [200]*3
    third, _ = build('regmod', parent, formation)
    assert third['model_reg_nested_id2'].tolist() == [200]*3
    assert saved.prediction == payload
    assert runtime.db.query(db_models.ProfileModelPrediction).count() == 1
    assert len(CountingModel.calls) == 1


def test_saved_training_snapshot_is_unchanged_but_new_build_uses_current_land(runtime, formation, tmp_path, monkeypatch):
    import os
    from collections import Counter
    from terrain_service import bind_grid
    obj, common, local = _terrain_grids(runtime, formation)
    snapshot = tmp_path/'old.parquet'
    pd.DataFrame({'prof_well_index': ['7_1_0'], 'target_value': [42], 'land': [100]}).to_parquet(snapshot)
    saved_bytes = snapshot.read_bytes()
    analysis = db_models.AnalysisReg(title='Terrain analysis', data=str(snapshot), up_data=True)
    runtime.db.add(analysis); runtime.db.flush()
    runtime.db.add(db_models.MarkupReg(analysis=analysis, profile=formation.profile, formation=formation,
                                       target_value=42, list_measure='[0, 1, 2]', type_markup='test'))
    runtime.db.commit()
    runtime.functions.update(get_regmod_id=lambda: analysis.id, get_list_param_regmod=lambda: ['land'],
                             os=os, Counter=Counter, TRAIN_TABLE_ROW_CHUNK_SIZE=1000)
    bind_grid(runtime.db, obj.id, local.id)
    runtime.db.commit()
    old, _ = runtime.functions['build_table_train'](True, 'regmod')
    assert old.land.tolist() == [100]
    assert snapshot.read_bytes() == saved_bytes
    monkeypatch.chdir(tmp_path)
    (tmp_path/'data_tables'/'reg').mkdir(parents=True)
    new, _ = runtime.functions['build_table_train_no_db']('regmod', analysis.id, ['land'])
    assert new.land.tolist() == [200]*3
    assert snapshot.read_bytes() == saved_bytes


@pytest.mark.parametrize('analysis_kind', ['mlp', 'regmod'])
def test_training_streams_profiles_preserving_order_and_feature_values(
        runtime, formation, tmp_path, monkeypatch, analysis_kind):
    import os
    _terrain_grids(runtime, formation)  # Register feature tables and parameter families.
    runtime.functions.update(os=os, TRAIN_TABLE_ROW_CHUNK_SIZE=2)
    runtime.functions['list_wavelet_features'] = ['wvt_energ_D1']
    other_profile = db_models.Profile(id=8, x_pulc='[40, 50, 60]', signal='[[7, 8], [9, 10], [11, 12]]')
    other = db_models.Formation(id=12, profile=other_profile,
        layer_up=db_models.Layers(profile=other_profile, layer_line='[0, 0, 0]'),
        layer_down=db_models.Layers(profile=other_profile, layer_line='[2, 2, 2]'),
        A_mean='[400, 500, 600]')
    same_profile = db_models.Formation(id=13, profile=formation.profile,
        layer_up=formation.layer_up, layer_down=formation.layer_down, A_mean='[70, 80, 90]')
    formation.profile.signal = '[[1, 2], [3, 4], [5, 6]]'
    formation.A_mean = '[100, 200, 300]'
    runtime.db.add_all([other, same_profile,
        model_profile_features.WaveletFeatureProfile(profile_id=7, wvt_energ_D1='[10, 20, 30]'),
        model_profile_features.WaveletFeatureProfile(profile_id=8, wvt_energ_D1='[40, 50, 60]')])
    if analysis_kind == 'mlp':
        analysis = db_models.AnalysisMLP(title='Streaming')
        marker = db_models.MarkerMLP(analysis=analysis, title='Rock')
        markup_class = db_models.MarkupMLP
        target = dict(marker=marker)
    else:
        analysis = db_models.AnalysisReg(title='Streaming')
        markup_class = db_models.MarkupReg
        target = dict(target_value=42)
    runtime.db.add(analysis)
    runtime.db.flush()
    analysis_id = analysis.id
    for well_id, form, measures, fake in (
            (1, formation, '[0, 2, 0]', '[2]'), (2, formation, '[2]', None),
            (3, same_profile, '[1]', None), (4, other, '[1]', None), (5, formation, '[1]', None)):
        runtime.db.add(markup_class(analysis=analysis, formation=form, profile=form.profile,
            well_id=well_id, list_measure=measures, list_fake=fake, **target))
    runtime.db.commit()
    for form in (formation, other):
        save_prediction(runtime, form, 'reg', 99, [11, 22, 33])

    progress, loads = [], []
    runtime.functions['ui'].progressBar.setValue = progress.append
    prefetch = runtime.functions['_prefetch_profile_data']
    def record_prefetch(markups, params):
        assert len(markups) == 1  # Raw arrays are never loaded for future profiles.
        loads.append((markups[0].profile_id, progress[-1]))
        return prefetch(markups, params)
    runtime.functions['_prefetch_profile_data'] = record_prefetch
    monkeypatch.chdir(tmp_path)
    (tmp_path/'data_tables'/'cls').mkdir(parents=True)
    (tmp_path/'data_tables'/'reg').mkdir(parents=True)
    params = ['Signal_Abase', 'X', 'A_mean', 'prof_wvt_energ_D1', 'model_reg_saved_id99']
    table, _ = runtime.functions['build_table_train_no_db'](analysis_kind, analysis_id, params)
    target_name = 'mark' if analysis_kind == 'mlp' else 'target_value'
    expected = pd.DataFrame({
        'prof_well_index': ['7_1_0', '7_1_0', '7_2_2', '7_3_1', '8_4_1', '7_5_1'],
        target_name: ['Rock' if analysis_kind == 'mlp' else 42]*6,
        'Signal_Abase_1': [1, 1, 5, 3, 9, 3], 'Signal_Abase_2': [2, 2, 6, 4, 10, 4],
        'X': [10, 10, 30, 20, 50, 20], 'A_mean': [100, 100, 300, 80, 500, 200],
        'prof_wvt_energ_D1': [10, 10, 30, 20, 50, 20],
        'model_reg_saved_id99': [11, 11, 33, 22, 22, 22],
    })
    pd.testing.assert_frame_equal(table, expected, check_dtype=False)
    assert loads == [(7, 0), (8, 3), (7, 4)]
    assert progress == [0, 1, 2, 3, 4, 5]
    stored_analysis = runtime.db.get(type(analysis), analysis_id)
    pd.testing.assert_frame_equal(pd.read_parquet(stored_analysis.data), table)
    assert stored_analysis.up_data


def test_training_terrain_does_not_reload_markups_or_read_raw_signal(
        runtime, formation, tmp_path, monkeypatch):
    import os
    from sqlalchemy import event
    from terrain_service import ensure_formation_terrain
    _terrain_grids(runtime, formation)
    runtime.functions.update(os=os, TRAIN_TABLE_ROW_CHUNK_SIZE=1000)
    formation.profile.signal = 'unneeded raw signal'
    formation.top = '[90, 90, 90]'
    formation.layer_up.layer_line = '[2, 2, 2]'
    analysis = db_models.AnalysisReg(title='Terrain streaming')
    runtime.db.add(analysis)
    runtime.db.flush()
    analysis_id = analysis.id
    for i in range(3):
        runtime.db.add(db_models.MarkupReg(analysis=analysis, formation=formation,
            profile=formation.profile, well_id=i+1, target_value=42, list_measure=f'[{i}]'))
    ensure_formation_terrain(runtime.db, formation)
    runtime.db.commit()
    # Expire everything to exercise real database loads, rather than warm ORM objects.
    runtime.db.expire_all()
    queries = []
    def record_query(conn, cursor, statement, parameters, context, many):
        if statement.startswith('SELECT'):
            queries.append(statement)
    event.listen(runtime.db.bind, 'before_cursor_execute', record_query)
    monkeypatch.chdir(tmp_path)
    (tmp_path/'data_tables'/'reg').mkdir(parents=True)
    try:
        table, _ = runtime.functions['build_table_train_no_db'](
            'regmod', analysis_id, ['land', 'speed_cover'])
    finally:
        event.remove(runtime.db.bind, 'before_cursor_execute', record_query)
    assert table.land.tolist() == [100]*3
    assert table.speed_cover.tolist() == [62.5]*3
    assert sum('FROM markup_reg' in query for query in queries) == 1
    assert not any('profile.signal' in query for query in queries)


@pytest.mark.parametrize('use_land', [False, True])
@pytest.mark.parametrize('use_correction', [False, True])
def test_structural_map_reuses_old_prediction_with_current_land(runtime, formation, use_land, use_correction):
    from terrain_service import bind_grid, ensure_formation_terrain
    obj, common, local = _terrain_grids(runtime, formation)
    save_prediction(runtime, formation, 'reg', 1, [10, 20, 30], legacy_stale=True)
    saved = runtime.db.query(db_models.ProfileModelPrediction).one()
    correction = db_models.PredictionCorrect(prediction_id=saved.id, correct='[11, 21, 31]')
    runtime.db.add(correction); runtime.db.commit()
    payload = saved.prediction
    bind_grid(runtime.db, obj.id, local.id)
    runtime.db.commit()
    runtime.functions['ui'].listWidget_model_pred = SimpleNamespace(
        currentItem=lambda: SimpleNamespace(text=lambda: f'reg model id{saved.id}'))
    runtime.functions['ui'].checkBox_corr_pred = SimpleNamespace(isChecked=lambda: use_correction)
    runtime.functions['ui'].checkBox_use_land = SimpleNamespace(isChecked=lambda: use_land)
    runtime.functions['ensure_formation_terrain'] = ensure_formation_terrain
    load_functions(ROOT / 'draw.py', runtime.functions, {'build_table_profile_model_predict'})
    table = runtime.functions['build_table_profile_model_predict']()
    expected = [11, 21, 31] if use_correction else [10, 20, 30]
    assert table['prediction'].tolist() == expected
    if use_land:
        assert table['land'].tolist() == [200]*3
        assert table['abs_uf'].tolist() == [200 - value for value in expected]
    else:
        assert 'land' not in table and 'abs_uf' not in table
    assert saved.prediction == payload and correction.correct == '[11, 21, 31]'


def test_legacy_mark_does_not_block_velocity_from_saved_prediction(runtime, formation):
    formation.layer_up.layer_line = '[2, 2, 2]'
    save_prediction(runtime, formation, 'reg', 1, [10, 20, 30], legacy_stale=True)
    saved = runtime.db.query(db_models.ProfileModelPrediction).one()
    binding = db_models.BindingLayerPrediction(prediction=saved, layer=formation.layer_up)
    runtime.db.add(binding); runtime.db.commit()
    runtime.functions.update(get_profile_id=lambda: formation.profile_id,
                             savgol_line=lambda values, window: values)
    load_functions(ROOT / 'velocity_prediction.py', runtime.functions, {'calc_list_velocity'})
    assert runtime.functions['calc_list_velocity']() == [[62.5, 125, 187.5]]
    assert saved.prediction == '[10, 20, 30]' and binding.prediction_id == saved.id
