"""Input preparation shared by the continuous-map kriging workflow."""

import numpy as np


def prepare_variogram_data(coordinates, values, min_unique_points=2):
    """Validate variogram data and average values at coincident coordinates.

    A variogram cannot be fitted to NaN/inf data or to zero-distance pairs.
    Coincident samples are common for model results (several records can belong
    to one map point), so their values are averaged instead of failing the map.
    """
    coords = np.asarray(coordinates, dtype=float)
    data = np.asarray(values, dtype=float).reshape(-1)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coordinates must be an array with shape (n, 2)")
    if len(coords) != len(data):
        raise ValueError("coordinates and values must contain the same number of samples")

    finite_mask = np.isfinite(coords).all(axis=1) & np.isfinite(data)
    coords = coords[finite_mask]
    data = data[finite_mask]
    if len(coords) == 0:
        raise ValueError("no finite coordinate/value pairs are available")

    unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
    value_sums = np.bincount(inverse, weights=data)
    sample_counts = np.bincount(inverse)
    unique_values = value_sums / sample_counts

    if len(unique_coords) < min_unique_points:
        raise ValueError(
            f"at least {min_unique_points} distinct coordinates are required; "
            f"got {len(unique_coords)}"
        )

    return unique_coords, unique_values, int(len(data) - len(unique_coords))
