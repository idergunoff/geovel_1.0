import math
from types import SimpleNamespace

import pytest

from prediction_comparison_statistics import comparison_statistics


def candidate(delta, well_id=1, status='resolved', prediction=10):
    return SimpleNamespace(well_id=well_id, stored_value=prediction,
                           resolution=SimpleNamespace(status=status, value=10 + delta))


def test_metrics_use_absolute_errors_but_preserve_bias_sign_and_tied_locations():
    rows = [candidate(0), candidate(2), candidate(-2, 2), candidate(4, 3), candidate(-4, 4)]
    stats = comparison_statistics(rows, 2)
    assert stats.count == 5 and stats.well_count == 4
    assert stats.excluded_count == 0
    assert stats.minimum == 0 and stats.maximum == 4
    assert stats.minimum_rows == (rows[0],)
    assert stats.maximum_rows == (rows[3], rows[4])
    assert stats.mae == pytest.approx(2.4)
    assert stats.median == 2
    assert stats.rmse == pytest.approx(math.sqrt(8))
    assert stats.bias == 0
    assert stats.p90 == 4
    assert stats.within_tolerance == 3 and stats.match_percent == 60


def test_p90_interpolates_and_negative_bias_means_overestimated_prediction():
    stats = comparison_statistics([candidate(-2), candidate(-4)], 0)
    assert stats.mae == 3 and stats.median == 3
    assert stats.bias == -3
    assert stats.p90 == pytest.approx(3.8)
    assert stats.match_percent == 0


def test_nonresolved_missing_and_nonfinite_values_are_excluded():
    good = candidate(2)
    rows = [good, candidate(5, status='ambiguous'), candidate(0, status='missing'),
            candidate(math.inf), candidate(math.nan), candidate(0, prediction=None),
            SimpleNamespace(well_id=2, stored_value=10, resolution=None),
            SimpleNamespace(well_id=2, stored_value=10,
                            resolution=SimpleNamespace(status='resolved', value=None))]
    stats = comparison_statistics(rows, 2)
    assert stats.count == stats.well_count == stats.within_tolerance == 1
    assert stats.excluded_count == 7
    assert stats.mae == stats.median == stats.rmse == stats.bias == stats.p90 == 2
    assert stats.minimum_rows == stats.maximum_rows == (good,)


def test_empty_results_have_no_metrics_or_false_zero_error():
    stats = comparison_statistics([candidate(0, status='invalid')], 0.1)
    assert stats.count == stats.well_count == stats.within_tolerance == 0
    assert stats.excluded_count == 1
    for name in ('minimum', 'maximum', 'mae', 'median', 'rmse', 'bias', 'p90', 'match_percent'):
        assert getattr(stats, name) is None
    assert not stats.minimum_rows and not stats.maximum_rows


def test_tolerance_only_changes_match_count_and_all_tied_extrema_are_retained():
    rows = [candidate(2), candidate(-2, 2)]
    low = comparison_statistics(rows, 1.9)
    inclusive = comparison_statistics(rows, 2)
    assert low.within_tolerance == 0 and inclusive.within_tolerance == 2
    assert low.mae == inclusive.mae == 2
    assert inclusive.minimum_rows == inclusive.maximum_rows == tuple(rows)
