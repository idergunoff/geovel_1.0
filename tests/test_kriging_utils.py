import numpy as np
import pytest

from kriging_utils import (
    inverse_distance_interpolation,
    prepare_variogram_data,
    savgol_parameters,
)


def test_prepare_variogram_data_averages_coincident_coordinates():
    coordinates, values, merged_count = prepare_variogram_data(
        [[0, 0], [1, 1], [0, 0], [2, 2]], [2, 4, 6, 8]
    )

    np.testing.assert_array_equal(coordinates, [[0, 0], [1, 1], [2, 2]])
    np.testing.assert_allclose(values, [4, 4, 8])
    assert merged_count == 1


def test_prepare_variogram_data_discards_non_finite_samples():
    coordinates, values, merged_count = prepare_variogram_data(
        [[0, 0], [np.nan, 1], [2, 2]], [1, 2, np.inf], min_unique_points=1
    )

    np.testing.assert_array_equal(coordinates, [[0, 0]])
    np.testing.assert_allclose(values, [1])
    assert merged_count == 0


def test_prepare_variogram_data_requires_distinct_coordinates():
    with pytest.raises(ValueError, match="distinct coordinates"):
        prepare_variogram_data([[0, 0], [0, 0]], [1, 2])


def test_inverse_distance_interpolation_preserves_source_values():
    grid_x = np.array([[0.0, 1.0]])
    grid_y = np.array([[0.0, 0.0]])

    result = inverse_distance_interpolation([[0, 0], [1, 0]], [2, 6], grid_x, grid_y)

    np.testing.assert_allclose(result, [[2, 6]])


def test_savgol_parameters_clamps_window_to_odd_data_length():
    assert savgol_parameters(13, 8) == (7, 3)


def test_savgol_parameters_skips_too_short_data():
    assert savgol_parameters(13, 2) is None
