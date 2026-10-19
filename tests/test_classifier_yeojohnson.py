"""Exercise the actual pipeline builder without importing the desktop application."""
import ast
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def build_classifier(enabled, after=False, standard=False):
    tree = ast.parse((ROOT / 'classification_func.py').read_text())
    builder = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == 'build_pipeline')

    class Control:
        def __init__(self, checked=False):
            self.checked = checked

        def isChecked(self):
            return self.checked

    class Controls:
        def __getattr__(self, name):
            return Control()

    ui = Controls()
    ui.checkBox_power_trans = Control(enabled)
    ui.checkBox_power_trans_after = Control(after)
    ui.checkBox_stdscaler = Control(standard)
    ui.buttonGroup = SimpleNamespace(checkedButton=lambda: SimpleNamespace(text=lambda: 'LR'))
    scope = dict(ui_cls=ui, PowerTransformer=PowerTransformer, StandardScaler=StandardScaler, Pipeline=Pipeline,
                 choice_model_classifier=lambda *_: (LogisticRegression(), 'LR'))
    exec(compile(ast.Module(body=[builder], type_ignores=[]), '<pipeline builder>', 'exec'), scope)
    x = np.array([[-8., 0.], [-5., 1.], [-3., 2.], [-1., 0.],
                  [0., 3.], [2., 4.], [4., 1.], [8., 7.], [12., 10.], [20., 5.]])
    y = np.array([0] * 5 + [1] * 5)
    _, _, _, pipe, text, _ = scope['build_pipeline'](y, x)
    return pipe, text, x, y


def test_yeojohnson_is_optional_and_saved_with_model():
    disabled, _, _, _ = build_classifier(False)
    assert list(disabled.named_steps) == ['model']
    pipe, text, x, y = build_classifier(True)
    assert 'Yeo–Johnson' in text
    assert not hasattr(pipe.named_steps['power_t'], 'lambdas_')
    pipe.fit(x, y)
    restored = pickle.loads(pickle.dumps(pipe))
    assert np.allclose(pipe.predict_proba(x), restored.predict_proba(x))
    assert np.isfinite(restored.named_steps['power_t'].transform(x)).all()


def test_cv_fits_transform_on_each_training_fold_only():
    pipe, _, x, y = build_classifier(True)
    cv = list(StratifiedKFold(2, shuffle=True, random_state=0).split(x, y))
    result = cross_validate(pipe, x, y, cv=cv, return_estimator=True)
    assert not hasattr(pipe.named_steps['power_t'], 'lambdas_')
    for (train, _), estimator in zip(cv, result['estimator']):
        expected = PowerTransformer(method='yeo-johnson', standardize=True).fit(x[train])
        assert np.allclose(estimator.named_steps['power_t'].lambdas_, expected.lambdas_)


def test_power_transform_order_and_serialization():
    for before in (False, True):
        for after in (False, True):
            for standard in (False, True):
                pipe, text, x, y = build_classifier(before, after, standard)
                expected = (['power_t'] if before else [])
                expected += ['std_scaler'] if standard else []
                expected += ['power_t_after'] if after else []
                assert list(pipe.named_steps) == expected + ['model']
                pipe.fit(x, y)
                restored = pickle.loads(pickle.dumps(pipe))
                assert np.allclose(pipe.predict_proba(x), restored.predict_proba(x))
                if after:
                    assert 'после StandardScaler' in text


def test_after_transform_fits_on_scaled_training_fold():
    pipe, _, x, y = build_classifier(False, after=True, standard=True)
    cv = list(StratifiedKFold(2, shuffle=True, random_state=0).split(x, y))
    result = cross_validate(pipe, x, y, cv=cv, return_estimator=True)
    for (train, _), estimator in zip(cv, result['estimator']):
        scaled = StandardScaler().fit_transform(x[train])
        expected = PowerTransformer().fit(scaled)
        assert np.allclose(estimator.named_steps['power_t_after'].lambdas_, expected.lambdas_)
