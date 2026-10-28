"""Mask compatibility and the actual training-dialog button handlers."""

import ast
import datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from training_feature_mask import TrainingFeatureMask, validate_mask_features


ROOT = Path(__file__).resolve().parents[1]


def test_mask_preserves_order_and_does_not_modify_inputs():
    features = ['model_reg_test_id3', 'X']
    result = validate_mask_features(features, ['X', 'Y', features[0]], ['X', 'Y', features[0]], 2)
    assert result == features
    assert result is not features


@pytest.mark.parametrize('missing', ['model_reg_test_id3', 'model_cls_test_id8', 'Signal_Abase_512'])
def test_missing_feature_has_actionable_warning(missing):
    with pytest.raises(ValueError) as error:
        validate_mask_features(['X', missing], ['X'], ['X'], 12)
    message = str(error.value)
    assert 'id12' in message
    assert missing in message
    assert 'обновите обучающую выборку' in message
    assert 'отключите применение маски' in message


@pytest.mark.parametrize('features', [[], None, {}, ['X', None], ['X', 'X']])
def test_invalid_masks_are_rejected(features):
    with pytest.raises(ValueError):
        validate_mask_features(features, ['X'], ['X'], 1)


def test_metadata_and_missing_table_columns_cannot_be_used_as_features():
    with pytest.raises(ValueError, match='target_value'):
        validate_mask_features(['target_value'], ['X'], ['X', 'target_value'], 1)
    with pytest.raises(ValueError, match='Y'):
        validate_mask_features(['Y'], ['X', 'Y'], ['X'], 1)


@pytest.fixture(scope='module')
def qt_app():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    qt = pytest.importorskip('PyQt5.QtWidgets')
    return qt.QApplication.instance() or qt.QApplication([])


@pytest.fixture
def controls(qt_app):
    from PyQt5 import QtWidgets

    checkbox = QtWidgets.QCheckBox()
    checkbox.setChecked(True)
    mask_list = QtWidgets.QListWidget()
    mask_list.addItems(['1 id1', '1 id2'])
    mask_list.setCurrentRow(0)
    masks = {'1': ['X'], '2': ['Y']}
    messages = []
    selector = TrainingFeatureMask(
        ['X', 'Y'], ['X', 'Y', 'mark'], checkbox, mask_list, masks.__getitem__, messages.append,
    )
    return SimpleNamespace(selector=selector, checkbox=checkbox, mask_list=mask_list,
                           masks=masks, messages=messages)


def test_switch_disable_and_reenable_mask(controls):
    c = controls
    assert c.selector.resolve() == ['X']
    c.mask_list.setCurrentRow(1)
    assert c.selector.resolve() == ['Y']
    c.masks['2'] = ['model_cls_missing_id7']
    assert c.selector.resolve() is None
    assert 'model_cls_missing_id7' in c.messages[-1]
    c.checkbox.setChecked(False)
    assert c.selector.resolve() == ['X', 'Y']
    c.checkbox.setChecked(True)
    c.mask_list.setCurrentRow(0)
    assert c.selector.resolve() == ['X']


def test_no_mask_selected_warns_instead_of_raising(controls):
    controls.mask_list.setCurrentRow(-1)
    assert controls.selector.resolve() is None
    assert 'не выбрана' in controls.messages[-1]


@pytest.mark.parametrize('failure', [AttributeError(), TypeError(), ValueError()])
def test_deleted_or_corrupt_mask_warns(controls, failure):
    def load(_):
        raise failure

    controls.selector.load_mask = load
    assert controls.selector.resolve() is None
    assert 'удалена или повреждена' in controls.messages[-1]


def test_incompatible_masks_visible_with_explanation(controls):
    item = controls.mask_list.item(0)
    record = SimpleNamespace(id=1, count_param=1, mask='["model_reg_missing_id9"]', mask_info='Описание')
    controls.selector.annotate_item(item, record)
    assert item.text() == 'Недоступна: 1 id1'
    assert 'Описание' in item.toolTip()
    assert 'model_reg_missing_id9' in item.toolTip()
    assert controls.selector.resolve() == ['X']  # The ID remains readable after annotation.


def load_function(file, name, scope):
    tree = ast.parse((ROOT / file).read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT / file), 'exec'), scope)
    return scope[name]


@pytest.mark.parametrize('kind', ['classification', 'regression'])
@pytest.mark.parametrize('features', [[], ['X', 'model_reg_missing_id9'], ['model_cls_missing_id7']])
def test_training_and_validation_buttons_warn_without_crashing(qt_app, monkeypatch, kind, features):
    """Open each production form and click its real buttons with an invalid mask.

    No model/session helpers are provided for the handlers: reaching training or
    coordinate collection would fail this test instead of displaying a warning.
    """
    from PyQt5 import QtCore, QtWidgets
    from qt.classifier_form import Ui_ClassifierForm
    from qt.regressor_form import Ui_RegressorForm
    import app_settings

    record = SimpleNamespace(id=4, count_param=len(features), mask=json.dumps(features), mask_info='Тест')
    messages = []
    data = pd.DataFrame({'prof_well_index': ['1_1', '1_2'],
                         'mark' if kind == 'classification' else 'target_value': [0, 1], 'X': [1., 2.]})
    monkeypatch.setattr(app_settings, 'bind_form', lambda *args: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning', lambda parent, title, text: messages.append(text))
    scope = dict(
        pd=pd, np=np, QtWidgets=QtWidgets, QtCore=QtCore,
        Ui_ClassifierForm=Ui_ClassifierForm, Ui_RegressorForm=Ui_RegressorForm,
        QListWidgetItem=QtWidgets.QListWidgetItem, TrainingFeatureMask=TrainingFeatureMask,
        ParameterMask=object(), session=SimpleNamespace(query=lambda _: SimpleNamespace(all=lambda: [record])),
        get_list_param_by_mask=lambda _: features, get_list_marker_mlp=lambda _: [0, 1],
        push_random_search=lambda: None, push_random_param=lambda: None, push_random_param_reg=lambda: None,
        build_table_train=lambda *args: (data, ['X']), get_list_param_numerical_for_train=lambda _: ['X'],
    )

    def exercise(dialog):
        check = dialog.findChild(QtWidgets.QCheckBox, 'checkBox_mask_param')
        mask_list = dialog.findChild(QtWidgets.QListWidget, 'listWidget_mask_param')
        assert mask_list.item(0).text().startswith('Недоступна:')
        mask_list.setCurrentRow(0)
        check.setChecked(True)
        buttons = ['pushButton_calc', 'pushButton_cov', 'pushButton_cvw']
        if kind == 'classification':
            buttons.append('pushButton_yellow_brick')
        for name in buttons:
            dialog.findChild(QtWidgets.QPushButton, name).click()
        assert len(messages) == len(buttons)
        assert all('id4' in message for message in messages)
        mask_list.setCurrentRow(-1)
        dialog.findChild(QtWidgets.QPushButton, 'pushButton_calc').click()
        assert 'не выбрана' in messages[-1]
        dialog.close()
        return 0

    monkeypatch.setattr(QtWidgets.QDialog, 'exec_', exercise)
    if kind == 'classification':
        train = load_function('classification_func.py', 'train_classifier', scope)
        train(data, ['X'], ['X'], {}, 'mark', 'prof_well_index', 'georadar')
    else:
        train = load_function('regression.py', 'train_regression_model', scope)
        train()


@pytest.mark.parametrize('kind', ['classification', 'regression'])
def test_training_uses_selected_features_then_restores_full_sample(controls, kind):
    """Run the production training callback up to its train/test split."""
    c = controls
    data = pd.DataFrame({'X': [1., 2.], 'Y': [3., 4.]})
    samples = []

    class ReachedSplit(Exception):
        pass

    def split(sample, *_args, **_kwargs):
        samples.append(np.asarray(sample).copy())
        raise ReachedSplit

    class UncheckedControls:
        def __getattr__(self, _):
            return SimpleNamespace(isChecked=lambda: False, value=lambda: 0)

    scope = dict(training_mask=c.selector, data_train=data, np=np, datetime=datetime,
                 set_marks=lambda: {0: 0, 1: 1}, markup=np.array([0, 1]), target=[0, 1],
                 ui_cls=UncheckedControls(), ui_r=UncheckedControls(), ui=UncheckedControls(),
                 train_test_split=split)
    file, name = (('classification_func.py', 'calc_model_class') if kind == 'classification'
                  else ('regression.py', 'calc_model_reg'))
    tree = ast.parse((ROOT / file).read_text(encoding='utf-8'))
    callback = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[callback], type_ignores=[]), file, 'exec'), scope)
    run = scope[name]
    with pytest.raises(ReachedSplit):
        run()
    np.testing.assert_array_equal(samples[-1], data[['X']].to_numpy())
    c.masks['1'] = ['model_reg_missing_id9']
    assert run() is None
    assert len(samples) == 1
    c.checkbox.setChecked(False)
    with pytest.raises(ReachedSplit):
        run()
    np.testing.assert_array_equal(samples[-1], data[['X', 'Y']].to_numpy())
    c.checkbox.setChecked(True)
    c.mask_list.setCurrentRow(1)
    with pytest.raises(ReachedSplit):
        run()
    np.testing.assert_array_equal(samples[-1], data[['Y']].to_numpy())


@pytest.mark.parametrize('ensemble', ['Voting', 'Stacking'])
def test_ensemble_pipeline_uses_current_feature_sample(ensemble):
    """Ensembles must receive the same selected columns as ordinary models."""
    from sklearn.ensemble import StackingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier

    class Controls:
        def __getattr__(self, name):
            return SimpleNamespace(isChecked=lambda: name in ('checkBox_stack_vote', 'checkBox_stv_dtc'))

    ui = Controls()
    ui.buttonGroup = SimpleNamespace(checkedButton=lambda: SimpleNamespace(text=lambda: 'LR'))
    ui.buttonGroup_stack_vote = SimpleNamespace(checkedButton=lambda: SimpleNamespace(text=lambda: ensemble))
    received = []

    def choose(_name, sample):
        received.append(sample)
        return LogisticRegression(), 'LR'

    scope = dict(ui_cls=ui, Pipeline=Pipeline, DecisionTreeClassifier=DecisionTreeClassifier,
                 VotingClassifier=VotingClassifier, StackingClassifier=StackingClassifier,
                 choice_model_classifier=choose)
    tree = ast.parse((ROOT / 'classification_func.py').read_text(encoding='utf-8'))
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                 and node.name in ('build_pipeline', 'build_stacking_voting_model')]
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'classification_func.py', 'exec'), scope)
    sample = np.arange(20, dtype=float).reshape(-1, 1)
    target = np.tile([0, 1], 10)
    _, _, _, pipe, _, _ = scope['build_pipeline'](target, sample)
    assert received[0] is sample
    pipe.set_params(model__n_jobs=1)
    pipe.fit(sample, target)
    assert pipe.predict_proba(sample).shape == (20, 2)
