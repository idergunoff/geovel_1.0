import ast
import json
from html import escape
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from scipy.signal import savgol_filter

from velocity_depth_signal import build_velocity_depth_signal


ROOT = Path(__file__).resolve().parents[1]


def load_functions(filename, names, namespace):
    path = ROOT / filename
    tree = ast.parse(path.read_text(encoding='utf-8'))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)


@pytest.fixture
def functions():
    namespace = {'savgol_filter': savgol_filter}
    load_functions('func.py', {'interpolate_list', 'savgol_line'}, namespace)
    return namespace


def convert(functions, top=2, bottom=6, top_depth=3, bottom_depth=7):
    return build_velocity_depth_signal(
        [list(range(12))], [('top', [top], [top_depth]), ('bottom', [bottom], [bottom_depth])],
        functions['interpolate_list'], functions['savgol_line'],
    )[0]


def test_ordinary_intervals_keep_legacy_interpolation(functions):
    interpolate = functions['interpolate_list']
    assert convert(functions) == (interpolate([0, 1], 3)
                                  + interpolate([2, 3, 4, 5], 4)
                                  + interpolate([6, 7, 8, 9, 10], 2))


def test_single_source_sample_is_constant_in_depth(functions):
    assert convert(functions, top=1)[:3] == [0, 0, 0]
    assert len(convert(functions, top=1)) == 9


def test_one_output_sample_and_subpixel_intervals_are_valid(functions):
    one = convert(functions, top_depth=1.8)
    assert one[0] == 0
    zero = convert(functions, top_depth=0.2)
    assert zero[0] == 2
    assert len(one) == len(zero) == 9


def test_thin_middle_interval_preserves_absolute_boundary_positions(functions):
    result = convert(functions, top_depth=3.1, bottom_depth=3.9)
    assert len(result) == 5
    assert result[3:] == [6, 10]


def test_no_signal_below_last_boundary_is_valid(functions):
    result = convert(functions, bottom=11)
    assert len(result) == 7


def test_thin_bottom_interval_does_not_raise(functions):
    assert len(convert(functions, bottom=10, bottom_depth=7.2)) == 7


def test_zero_surface_interval_with_zero_depth_is_valid(functions):
    assert len(convert(functions, top=0, top_depth=0)) == 9


@pytest.mark.parametrize('kwargs, message', [
    ({'top_depth': -1}, 'prediction нижней границы выше верхней'),
    ({'bottom_depth': 2}, 'prediction нижней границы выше верхней'),
    ({'bottom_depth': 3}, 'нулевая толщина'),
    ({'bottom': 1}, 'пересекаются'),
    ({'bottom': 2}, 'совпадают'),
    ({'top': 0}, 'пустой участок'),
    ({'top': -1}, 'граница вне сигнала'),
    ({'bottom': 12}, 'граница вне сигнала'),
    ({'top': 2.5}, 'нецелый отсчёт'),
    ({'top_depth': float('nan')}, 'некорректное значение'),
    ({'bottom_depth': float('inf')}, 'некорректное значение'),
])
def test_invalid_data_reports_interval_and_trace(functions, kwargs, message):
    with pytest.raises(ValueError, match=message) as error:
        convert(functions, **kwargs)
    assert 'трасса 1' in str(error.value)


@pytest.mark.parametrize('values', [[], None, [1, 2]])
def test_mismatched_prediction_length_cannot_silently_return_partial_signal(functions, values):
    with pytest.raises(ValueError, match='длина prediction'):
        build_velocity_depth_signal([list(range(12))], [('top', [2], values)],
                                    functions['interpolate_list'], functions['savgol_line'])


def test_smoothing_can_expose_crossing_predictions(functions):
    # A real cubic Savitzky-Golay filter overshoots this nonnegative raw thickness.
    traces = [list(range(12)) for _ in range(7)]
    with pytest.raises(ValueError, match='prediction нижней границы выше верхней'):
        build_velocity_depth_signal(traces, [
            ('top', [2] * 7, [3] * 7),
            ('bottom', [6] * 7, [3.01, 3.01, 3.01, 13, 3.01, 3.01, 3.01]),
        ], functions['interpolate_list'], functions['savgol_line'])


def test_unusable_total_depth_reports_units_instead_of_second_interpolation_error(functions):
    with pytest.raises(ValueError, match='меньше двух отсчётов'):
        build_velocity_depth_signal([[1, 2, 3]], [('bottom', [2], [0.5])],
                                    functions['interpolate_list'], functions['savgol_line'])


@pytest.fixture
def runtime(functions):
    db = Mock()
    current_type, minmax_type, bindings_type = object(), object(), SimpleNamespace(id=object())
    layers_type = SimpleNamespace(profile_id=object())
    current = SimpleNamespace(profile_id=7, signal=json.dumps([list(range(12))]))
    prediction = SimpleNamespace(profile_id=7, prediction='[3]', corrected=[])
    layer = SimpleNamespace(layer_title='top', layer_line='[1]')
    binding = SimpleNamespace(layer=layer, prediction=prediction)
    db.query.return_value.first.return_value = current
    db.query.return_value.join.return_value.filter.return_value.order_by.return_value.all.return_value = [binding]
    messages = []
    namespace = dict(functions, json=json, escape=escape, session=db,
                     CurrentProfile=current_type, CurrentProfileMinMax=minmax_type,
                     BindingLayerPrediction=bindings_type, Layers=layers_type,
                     get_profile_id=lambda: 7, build_velocity_depth_signal=build_velocity_depth_signal,
                     set_info=lambda text, color: messages.append((text, color)),
                     ui=SimpleNamespace(checkBox_minmax=Mock(), checkBox_corr_pred=Mock()))
    namespace['ui'].checkBox_minmax.isChecked.return_value = False
    namespace['ui'].checkBox_corr_pred.isChecked.return_value = False
    load_functions('velocity_prediction.py', {'calc_deep_predict_current_profile', 'draw_radar_vel_pred'}, namespace)
    return SimpleNamespace(namespace=namespace, current=current, binding=binding, messages=messages, db=db)


def test_calculation_handles_real_thin_interval_without_writing_database(runtime):
    assert runtime.namespace['calc_deep_predict_current_profile']()[0][:3] == [0, 0, 0]
    assert runtime.messages == []
    runtime.db.delete.assert_not_called()
    runtime.db.commit.assert_not_called()


def test_calculation_uses_corrected_prediction_when_enabled(runtime):
    runtime.binding.prediction.corrected = [SimpleNamespace(correct='[5]')]
    runtime.namespace['ui'].checkBox_corr_pred.isChecked.return_value = True
    assert runtime.namespace['calc_deep_predict_current_profile']()[0][:5] == [0] * 5


@pytest.mark.parametrize('problem', ['missing_prediction', 'foreign_prediction', 'bad_layer', 'mismatched_depths'])
def test_calculation_returns_diagnostic_without_deleting_bindings(runtime, problem):
    if problem == 'missing_prediction':
        runtime.binding.prediction = None
    elif problem == 'foreign_prediction':
        runtime.binding.prediction.profile_id = 8
    elif problem == 'bad_layer':
        runtime.binding.layer.layer_line = None
    else:
        runtime.binding.prediction.prediction = '[1, 2]'
    assert runtime.namespace['calc_deep_predict_current_profile']() is None
    assert 'Профиль id7' in runtime.messages[-1][0]
    assert 'top' in runtime.messages[-1][0]
    assert runtime.messages[-1][1] == 'red'
    runtime.db.delete.assert_not_called()
    runtime.db.commit.assert_not_called()


def test_draw_button_stops_before_mutating_current_profile_on_calculation_error(runtime):
    runtime.namespace['calc_deep_predict_current_profile'] = lambda: None
    runtime.namespace['clear_current_profile'] = Mock()
    runtime.namespace['draw_radar_vel_pred']()
    runtime.namespace['clear_current_profile'].assert_not_called()
    runtime.db.commit.assert_not_called()


@pytest.mark.parametrize('relief', [True, False])
def test_draw_relief_stops_on_invalid_velocity_signal(runtime, relief):
    ui = SimpleNamespace(
        spinBox_filter_model_nn=SimpleNamespace(value=lambda: 31),
        checkBox_relief=SimpleNamespace(isChecked=lambda: relief),
        checkBox_minmax=SimpleNamespace(isChecked=lambda: False),
        checkBox_vel=SimpleNamespace(isChecked=lambda: True),
    )
    namespace = dict(runtime.namespace, ui=ui,
                     remove_poly_item=lambda: None, remove_curve_fake=lambda: None,
                     remove_fill_form=lambda: None, calc_relief_profile=lambda profile: True,
                     calc_deep_predict_current_profile=lambda: None,
                     Profile=SimpleNamespace(id=object()), ProfileModelPrediction=SimpleNamespace(profile_id=object()))
    namespace['session'].query.return_value.join.return_value.filter.return_value.count.return_value = 1
    load_functions('draw.py', {'draw_relief'}, namespace)
    namespace['draw_relief']()
