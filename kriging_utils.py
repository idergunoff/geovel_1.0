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


def inverse_distance_interpolation(coordinates, values, grid_x, grid_y, power=2.0):
    """Interpolate a grid with inverse-distance weighting.

    This is a deterministic fallback when a variogram cannot be fitted.  It
    also handles a grid cell that exactly matches a source coordinate without
    a division by zero.
    """
    coords = np.asarray(coordinates, dtype=float)
    data = np.asarray(values, dtype=float).reshape(-1)
    flat_grid = np.column_stack([np.ravel(grid_x), np.ravel(grid_y)])
    result = np.empty(len(flat_grid), dtype=float)
    exponent = max(float(power), np.finfo(float).eps)

    # Process chunks so a large map does not allocate a full grid-by-points
    # distance matrix at once.
    chunk_size = 10_000
    for start in range(0, len(flat_grid), chunk_size):
        stop = min(start + chunk_size, len(flat_grid))
        delta = flat_grid[start:stop, np.newaxis, :] - coords[np.newaxis, :, :]
        squared_distance = np.einsum("ijk,ijk->ij", delta, delta)
        exact_match = squared_distance == 0.0
        weights = 1.0 / np.maximum(squared_distance, np.finfo(float).eps) ** (exponent / 2.0)
        weighted_values = weights @ data
        interpolated = weighted_values / weights.sum(axis=1)
        if np.any(exact_match):
            exact_rows = np.any(exact_match, axis=1)
            interpolated[exact_rows] = (
                exact_match[exact_rows] @ data / exact_match[exact_rows].sum(axis=1)
            )
        result[start:stop] = interpolated

    return result.reshape(np.shape(grid_x))


def savgol_parameters(requested_window, data_length, polyorder=3):
    """Return valid Savitzky-Golay parameters or ``None`` when smoothing is impossible."""
    window = min(int(requested_window), int(data_length))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return None
    return window, min(int(polyorder), window - 1)
