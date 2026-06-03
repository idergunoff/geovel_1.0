import importlib.util
import math
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
_MODULE_PATH = Path(__file__).resolve().parents[1] / "cluster" / "well_feature_calculator.py"
_SPEC = importlib.util.spec_from_file_location("well_feature_calculator", _MODULE_PATH)
well_feature_calculator = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = well_feature_calculator
_SPEC.loader.exec_module(well_feature_calculator)

from well_feature_calculator import (
    apply_abs,
    apply_derivative,
    apply_log,
    apply_minmax,
    apply_robust,
    apply_rolling_mean,
    apply_rolling_median,
    apply_sqrt,
    apply_unary_operation,
    apply_zscore,
)


def assert_close_list(actual, expected, *, abs_tol=1e-9):
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected):
        assert math.isclose(actual_value, expected_value, abs_tol=abs_tol)


def error_codes(errors):
    return {error.code for error in errors}


def test_log_positive_values():
    values, errors = apply_log([1.0, math.e, math.e**2])

    assert errors == []
    assert_close_list(values, [0.0, 1.0, 2.0])


@pytest.mark.parametrize("source", [[0.0, 1.0], [-1.0, 1.0]])
def test_log_blocks_zero_and_negative_values(source):
    values, errors = apply_log(source)

    assert values == []
    assert error_codes(errors) == {"invalid_math_domain"}


def test_sqrt_positive_values():
    values, errors = apply_sqrt([0.0, 4.0, 9.0])

    assert errors == []
    assert_close_list(values, [0.0, 2.0, 3.0])


def test_sqrt_blocks_negative_values():
    values, errors = apply_sqrt([1.0, -4.0])

    assert values == []
    assert error_codes(errors) == {"invalid_math_domain"}


def test_abs_values():
    values, errors = apply_abs([-2.0, 0.0, 3.0])

    assert errors == []
    assert_close_list(values, [2.0, 0.0, 3.0])


def test_derivative_on_uniform_depth_grid():
    values, errors = apply_derivative([0.0, 2.0, 6.0, 12.0], step=2.0)

    assert errors == []
    assert_close_list(values, [1.0, 1.5, 2.5, 3.0])


def test_derivative_requires_two_points():
    values, errors = apply_derivative([1.0], step=1.0)

    assert values == []
    assert error_codes(errors) == {"not_enough_points"}


def test_rolling_mean_centered_window_with_edge_clamping():
    values, errors = apply_rolling_mean([1.0, 2.0, 3.0, 4.0, 5.0], window=3)

    assert errors == []
    assert_close_list(values, [2.0, 2.0, 3.0, 4.0, 4.0])


def test_rolling_median_centered_window_with_edge_clamping():
    values, errors = apply_rolling_median([5.0, 1.0, 4.0, 2.0, 3.0], window=3)

    assert errors == []
    assert_close_list(values, [4.0, 4.0, 2.0, 3.0, 3.0])


def test_zscore_uses_population_std():
    values, errors = apply_zscore([1.0, 2.0, 3.0])

    assert errors == []
    assert_close_list(values, [-math.sqrt(1.5), 0.0, math.sqrt(1.5)])


def test_zscore_can_use_separate_whole_well_stats():
    values, errors = apply_zscore([2.0, 3.0], stats_values=[1.0, 2.0, 3.0])

    assert errors == []
    assert_close_list(values, [0.0, math.sqrt(1.5)])


def test_zscore_ignores_non_finite_values_outside_interval_stats():
    values, errors = apply_zscore([2.0, 3.0], stats_values=[None, 1.0, 2.0, float("nan"), 3.0])

    assert errors == []
    assert_close_list(values, [0.0, math.sqrt(1.5)])


def test_robust_median_iqr_normalization():
    values, errors = apply_robust([1.0, 2.0, 3.0, 4.0])

    assert errors == []
    assert_close_list(values, [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0])


def test_robust_ignores_non_finite_values_outside_interval_stats():
    values, errors = apply_robust([2.0, 3.0], stats_values=[None, 1.0, 2.0, float("inf"), 3.0, 4.0])

    assert errors == []
    assert_close_list(values, [-1.0 / 3.0, 1.0 / 3.0])


def test_minmax_normalization():
    values, errors = apply_minmax([2.0, 4.0, 6.0])

    assert errors == []
    assert_close_list(values, [0.0, 0.5, 1.0])


@pytest.mark.parametrize(
    ("operation", "function", "expected_code"),
    [
        ("zscore", apply_zscore, "zero_std"),
        ("robust", apply_robust, "zero_iqr"),
        ("minmax", apply_minmax, "zero_range"),
    ],
)
def test_normalization_zero_stat_errors(operation, function, expected_code):
    values, errors = function([5.0, 5.0, 5.0])

    assert values == []
    assert error_codes(errors) == {expected_code}


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), None])
def test_nan_inf_and_none_block_calculation(bad_value):
    values, errors = apply_unary_operation("abs", [1.0, bad_value])

    assert values == []
    assert error_codes(errors) == {"non_finite_result"}


def test_unsupported_operation_error():
    values, errors = apply_unary_operation("divide", [1.0, 2.0])

    assert values == []
    assert error_codes(errors) == {"unsupported_operation"}
