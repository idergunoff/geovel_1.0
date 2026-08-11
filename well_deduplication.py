"""Fast candidate lookup helpers for well deduplication."""

from collections import defaultdict
import math


def build_spatial_index(wells, distance_threshold: float):
    """Return a grid index that limits duplicate checks to nearby wells.

    ``None`` coordinates retain the legacy meaning of zero.  Non-finite
    coordinates cannot satisfy the Euclidean distance check and are omitted.
    """
    if distance_threshold < 0:
        return {}, 0.0

    cell_size = distance_threshold
    index = defaultdict(list)
    for well in wells:
        x = well.x_coord or 0.0
        y = well.y_coord or 0.0
        if math.isfinite(x) and math.isfinite(y):
            cell = ((x, y) if cell_size == 0
                    else (math.floor(x / cell_size), math.floor(y / cell_size)))
            index[cell].append(well)
    return index, cell_size


def nearby_wells(well, spatial_index, cell_size: float, distance_threshold: float):
    """Yield later wells in the only grid cells that can be within the threshold."""
    if distance_threshold < 0:
        return

    x = well.x_coord or 0.0
    y = well.y_coord or 0.0
    if not math.isfinite(x) or not math.isfinite(y):
        return

    if cell_size == 0:
        yield from (candidate for candidate in spatial_index.get((x, y), ())
                    if candidate.id > well.id)
        return

    cell_x = math.floor(x / cell_size)
    cell_y = math.floor(y / cell_size)
    candidates = []
    for offset_x in range(-1, 2):
        for offset_y in range(-1, 2):
            candidates.extend(spatial_index.get((cell_x + offset_x, cell_y + offset_y), ()))

    yield from sorted((candidate for candidate in candidates if candidate.id > well.id),
                      key=lambda candidate: candidate.id)
