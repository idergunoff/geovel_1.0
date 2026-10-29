"""Check class encoding at the production training callback's fit boundary."""

import ast
import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]


def load_callbacks(scope):
    tree = ast.parse((ROOT / 'classification_func.py').read_text(encoding='utf-8'))
    callbacks = [node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef)
                 and node.name in {'set_marks', 'calc_model_class', 'input_features'}]
    exec(compile(ast.Module(body=callbacks, type_ignores=[]),
                 str(ROOT / 'classification_func.py'), 'exec'), scope)
    return scope


@pytest.mark.parametrize('model_name', ['RFC', 'MLPC', 'TORCH'])
@pytest.mark.parametrize('radar_markers', [[], ['Георадарный класс 1', 'Георадарный класс 2']])
@pytest.mark.parametrize('markers, expected_codes', [
    (['Фон', 'Аномалия'], {'Фон': 0, 'Аномалия': 1}),
    (['Аномалия', 'Фон'], {'Аномалия': 0, 'Фон': 1}),
])
def test_geochem_training_uses_its_own_categories(model_name, radar_markers, markers, expected_codes):
    """Unrelated or missing radar analyses must not alter geochemical targets."""
    data = pd.DataFrame({'Cu': np.arange(20, dtype=float),
                         'category': ['Фон'] * 10 + ['Аномалия'] * 10})
    session = Mock()
    session.query.return_value.filter.return_value.all.return_value = [
        SimpleNamespace(title=title) for title in radar_markers
    ]

    class ReachedFit(Exception):
        pass

    pipe = SimpleNamespace(fit=Mock(side_effect=ReachedFit))
    scope = load_callbacks(dict(
        np=np, datetime=datetime, data_train=data, markup=data['category'].to_numpy(),
        list_marker=markers, type_case='geochem', mark='category',
        training_mask=SimpleNamespace(resolve=lambda: ['Cu']),
        ui_cls=SimpleNamespace(
            checkBox_cvw=SimpleNamespace(isChecked=lambda: False),
            checkBox_norm_l1_all=SimpleNamespace(isChecked=lambda: False),
            buttonGroup=SimpleNamespace(checkedButton=lambda: SimpleNamespace(text=lambda: model_name)),
        ),
        ui=SimpleNamespace(spinBox_seed=SimpleNamespace(value=lambda: 42)),
        train_test_split=train_test_split,
        build_pipeline=lambda y, x: (y, None, model_name, pipe, '', x),
        session=session, MarkerMLP=SimpleNamespace(analysis_id=None), get_MLP_id=lambda: '',
    ))
    if model_name != 'TORCH':
        scope['set_marks'] = Mock(side_effect=AssertionError('Only TORCH needs numeric class encoding'))

    with pytest.raises(ReachedFit):
        scope['calc_model_class']()

    sample, targets = pipe.fit.call_args.args
    original_targets = data.loc[sample[:, 0].astype(int), 'category'].to_numpy()
    if model_name == 'TORCH':
        expected = np.array([expected_codes[title] for title in original_targets], dtype=np.float32)
        np.testing.assert_array_equal(targets, expected)
        assert targets.dtype == np.float32
        assert scope['set_marks']() == expected_codes
    else:
        np.testing.assert_array_equal(targets, original_targets)
        scope['set_marks'].assert_not_called()
    session.query.assert_not_called()


@pytest.mark.parametrize('type_case', ['geochem', 'georadar'])
def test_class_encoding_preserves_all_selected_markers(type_case):
    """Encoding must not assign the third marker the first marker's code."""
    markers = ['Класс 2', 'Класс 1', 'Класс 3', 'Класс 4']
    session = Mock()
    session.query.return_value.filter.return_value.all.return_value = [
        SimpleNamespace(title=title) for title in markers
    ]
    scope = load_callbacks(dict(
        list_marker=markers, MarkerMLP=SimpleNamespace(analysis_id=None), get_MLP_id=lambda: '1',
        type_case=type_case, session=session,
    ))

    assert scope['set_marks']() == {'Класс 2': 0, 'Класс 1': 1, 'Класс 3': 2, 'Класс 4': 3}
    session.query.assert_not_called()
