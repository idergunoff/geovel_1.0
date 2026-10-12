"""Exercise production table builders without starting the global Qt window."""

import ast
import json
import pickle
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


def save_prediction(runtime, formation, type_model, model_id, values):
    runtime.db.add(db_models.ProfileModelPrediction(
        profile_id=formation.profile_id, type_model=type_model,
        model_id=model_id, prediction=json.dumps(values),
    ))
    runtime.db.commit()


@pytest.mark.parametrize('type_model,values', [('cls', [0.1, 0.2, 0.3]), ('reg', [10, 20, 30])])
def test_saved_predictions_skip_model_loading_and_parameter_collection(runtime, formation, type_model, values):
    # The trained model and its file deliberately do not exist.
    save_prediction(runtime, formation, 'cls', 4, [0.1, 0.2, 0.3])
    save_prediction(runtime, formation, 'reg', 4, [10, 20, 30])
    predict = runtime.functions['calc_profile_model_predict']

    assert predict(f'model_{type_model}_test_id4', formation) == values
    assert predict(f'model_{type_model}_test_id4', formation, measure_indices=[2, 0]) == {
        2: values[2], 0: values[0],
    }
    assert len(CountingModel.calls) == 0
    assert all('Используется сохранённый прогноз' in message for message in runtime.messages)


def test_training_prefetch_distinguishes_model_types(runtime, formation):
    save_prediction(runtime, formation, 'cls', 4, [0.1, 0.2, 0.3])
    save_prediction(runtime, formation, 'reg', 4, [10, 20, 30])
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


def test_shared_table_cache_separates_sources_selected_by_different_masks(runtime, formation, tmp_path):
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
    assert same_table is x_table
    assert len(cache) == 2


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
