"""Exercise L1 controls and the production classifier pipeline without app globals."""
import ast
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, PowerTransformer, StandardScaler

from geochem_preprocessing import ComponentSelector
from qt.l1_controls import configure_l1_controls

ROOT = Path(__file__).resolve().parents[1]


class Control:
    def __init__(self, checked=False):
        self.checked = checked

    def isChecked(self):
        return self.checked


class Controls:
    def __getattr__(self, _):
        return Control()


def load_functions(file, names, scope):
    tree = ast.parse((ROOT / file).read_text())
    nodes = [node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / file), 'exec'), scope)
    return scope


def classifier(mode, power=False):
    ui = Controls()
    ui.checkBox_norm_l1 = Control(mode == 'selected')
    ui.checkBox_norm_l1_all = Control(mode == 'all')
    ui.checkBox_power_trans = Control(power)
    ui.checkBox_stdscaler = Control(True)
    ui.buttonGroup = SimpleNamespace(checkedButton=lambda: SimpleNamespace(text=lambda: 'LR'))
    selected = ['Ag', 'Au']
    scope = dict(ui_cls=ui, type_case='geochem', training_mask=SimpleNamespace(resolve=lambda: selected),
                 all_components=['Au', 'Ag', 'Cu'], np=np, Normalizer=Normalizer,
                 PowerTransformer=PowerTransformer, StandardScaler=StandardScaler,
                 ComponentSelector=ComponentSelector, Pipeline=Pipeline,
                 choice_model_classifier=lambda *_: (LogisticRegression(), 'LR'))
    load_functions('classification_func.py', {'input_features', 'build_pipeline'}, scope)
    data = pd.DataFrame({'Au': [2., 4., 3., 8., 6., 7., 4., 10.],
                         'Ag': [3., 8., 2., 3., 7., 9., 12., 14.],
                         'Cu': [95., 12., 15., 10., 20., 9., 8., 3.]})
    required = scope['input_features'](selected)
    x = data[required].to_numpy()
    y = np.array([0] * 4 + [1] * 4)
    _, _, _, pipe, text, _ = scope['build_pipeline'](y, x)
    return pipe, text, x, y, required


@pytest.mark.parametrize('mode', ['all', 'selected', 'off'])
@pytest.mark.parametrize('power', [False, True])
def test_pipeline_normalizes_correct_inputs_and_survives_save_and_cv(mode, power):
    pipe, text, x, y, required = classifier(mode, power)
    pipe.fit(x, y)
    expected = PowerTransformer().fit_transform(x) if power else x.copy()
    if mode != 'off':
        expected = Normalizer(norm='l1').fit_transform(expected)
    if mode == 'all':
        assert required == ['Au', 'Ag', 'Cu']
        expected = expected[:, [1, 0]]
        assert 'по всем компонентам' in text
    else:
        assert required == ['Ag', 'Au']
    expected = StandardScaler().fit_transform(expected)
    np.testing.assert_allclose(pipe[:-1].transform(x), expected)
    restored = pickle.loads(pickle.dumps(pipe))
    np.testing.assert_allclose(restored.predict_proba(x), pipe.predict_proba(x))
    assert restored.n_features_in_ == len(required)
    assert restored.named_steps['model'].n_features_in_ == 2
    result = cross_validate(clone(pipe), x, y, cv=StratifiedKFold(2), return_estimator=True)
    assert np.isfinite(result['test_score']).all()


@pytest.fixture(scope='module')
def qt_app():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize('form', ['tsne_form', 'point_param_graph_form', 'classifier_form'])
def test_l1_controls_switch_both_directions_and_allow_off(qt_app, form):
    from importlib import import_module
    from PyQt5.QtWidgets import QDialog
    module = import_module('qt.' + form)
    ui_type = next(value for name, value in vars(module).items() if name.startswith('Ui_'))
    ui, dialog = ui_type(), QDialog()
    ui.setupUi(dialog)
    configure_l1_controls(ui)
    selected, all_components = ui.checkBox_norm_l1, ui.checkBox_norm_l1_all
    assert not selected.isChecked() and not all_components.isChecked()
    selected.click()
    assert selected.isChecked() and not all_components.isChecked()
    all_components.click()
    assert all_components.isChecked() and not selected.isChecked()
    selected.click()
    assert selected.isChecked() and not all_components.isChecked()
    selected.click()
    assert not selected.isChecked() and not all_components.isChecked()
    all_components.setChecked(True)
    selected.setChecked(True)
    assert selected.isChecked() and not all_components.isChecked()
    dialog.close()


def test_training_table_includes_unselected_components_only_when_requested():
    class TrainParameter:
        pass
    class Parameter:
        pass
    class TrainPoint:
        pass
    class Category:
        pass
    class Maket:
        id = 1
    class FieldValue:
        pass
    class WellValue:
        pass

    components = [SimpleNamespace(id=i, title=name) for i, name in enumerate(['Au', 'Ag', 'Cu'], 1)]
    selected = [SimpleNamespace(param=p, param_id=p.id) for p in components[:2]]
    points = [SimpleNamespace(fake=False, title='sample', category=SimpleNamespace(title='A'),
                              type_point=kind, point_id=10, point_well_id=20) for kind in ['field', 'well']]

    class Query:
        def __init__(self, entity):
            self.entity = entity
        def filter_by(self, **kwargs):
            self.kwargs = kwargs
            return self
        def join(self, *_):
            return self
        def filter(self, *_):
            return self
        def all(self):
            return {TrainParameter: selected, Parameter: components, TrainPoint: points}[self.entity]
        def first(self):
            return SimpleNamespace(value={1: 2., 2: 3., 3: 95.}[self.kwargs['g_param_id']])

    scope = dict(pd=pd, np=np, session=SimpleNamespace(query=Query), get_maket_id=lambda: 1,
                 get_geochem_id=lambda: 1, GeochemTrainParameter=TrainParameter,
                 GeochemParameter=Parameter, GeochemTrainPoint=TrainPoint, GeochemCategory=Category,
                 GeochemMaket=Maket, GeochemPointValue=FieldValue, GeochemWellPointValue=WellValue)
    load_functions('geochem.py', {'build_geochem_table'}, scope)
    selected_data, names = scope['build_geochem_table']()
    full_data, full_names = scope['build_geochem_table'](include_all_components=True)
    assert names == full_names == ['Au', 'Ag']
    assert selected_data.columns.tolist() == ['title', 'category', 'Au', 'Ag']
    assert full_data.columns.tolist() == ['title', 'category', 'Au', 'Ag', 'Cu']
    assert full_data['Cu'].tolist() == [95., 95.]


@pytest.mark.parametrize('mode', ['all', 'selected', 'off'])
def test_training_split_receives_full_inputs_only_for_all_mode(mode):
    import datetime
    data = pd.DataFrame({'Au': [2., 4.], 'Ag': [3., 6.], 'Cu': [95., 90.]})
    ui = Controls()
    ui.checkBox_norm_l1_all = Control(mode == 'all')

    class ReachedSplit(Exception):
        pass

    def split(sample, *args, **kwargs):
        expected = data if mode == 'all' else data[['Ag', 'Au']]
        np.testing.assert_array_equal(sample, expected.to_numpy())
        raise ReachedSplit

    scope = dict(np=np, datetime=datetime, data_train=data, all_components=list(data.columns),
                 type_case='geochem', ui_cls=ui, markup=[0, 1],
                 training_mask=SimpleNamespace(resolve=lambda: ['Ag', 'Au']),
                 ui=SimpleNamespace(spinBox_seed=SimpleNamespace(value=lambda: 0)),
                 train_test_split=split)
    load_functions('classification_func.py', {'input_features', 'calc_model_class'}, scope)
    with pytest.raises(ReachedSplit):
        scope['calc_model_class']()


@pytest.mark.parametrize('all_components', [False, True])
def test_graph_means_and_confidence_intervals_share_normalization(all_components):
    from scipy import stats
    ui = Controls()
    ui.checkBox_norm_l1 = Control(not all_components)
    ui.checkBox_norm_l1_all = Control(all_components)
    ui.radioButton_mean = Control(True)
    data = pd.DataFrame({'well': ['W', 'W'], 'point': ['A', 'B'], 'color': ['red', 'red'],
                         'Au': [2., 4.], 'Ag': [3., 8.], 'Cu': [95., 88.]})
    from geochem_preprocessing import transform_components
    scope = dict(data_plot=data, ui_pg=ui, stats=stats, transform_components=transform_components)
    load_functions('geochem.py', {'calc_mean_well', 'calc_conf_interval_well'}, scope)
    mean = scope['calc_mean_well']('W', ['Au', 'Ag'], ['A', 'B'])
    expected = [.03, .055] if all_components else [(2/5 + 4/12)/2, (3/5 + 8/12)/2]
    np.testing.assert_allclose(mean, expected)
    lower, upper = scope['calc_conf_interval_well']('W', ['Au', 'Ag'])
    assert lower[0] < expected[0] < upper[0]
    assert lower[1] < expected[1] < upper[1]


@pytest.mark.parametrize('allow_all', [False, True])
def test_restored_controls_are_sanitized(qt_app, allow_all):
    from PyQt5.QtWidgets import QCheckBox
    ui = SimpleNamespace(checkBox_norm_l1=QCheckBox(), checkBox_norm_l1_all=QCheckBox())
    # QSettings restores buttons with their signals blocked.
    ui.checkBox_norm_l1.setChecked(True)
    ui.checkBox_norm_l1_all.setChecked(True)
    configure_l1_controls(ui, allow_all=allow_all)
    assert ui.checkBox_norm_l1.isChecked() == (not allow_all)
    assert ui.checkBox_norm_l1_all.isChecked() == allow_all
