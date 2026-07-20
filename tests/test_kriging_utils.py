import numpy as np
import pytest

from kriging_utils import prepare_variogram_data


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
