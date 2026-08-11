from dataclasses import dataclass

from well_deduplication import build_spatial_index, nearby_wells


@dataclass
class Well:
    id: int
    x_coord: float | None
    y_coord: float | None


def candidate_ids(well, wells, threshold):
    index, cell_size = build_spatial_index(wells, threshold)
    return [candidate.id for candidate in nearby_wells(well, index, cell_size, threshold)]


def test_nearby_wells_only_returns_later_wells_from_adjacent_cells():
    wells = [
        Well(1, 0.0, 0.0),
        Well(2, 4.9, 0.0),
        Well(3, 5.1, 5.1),
        Well(4, 100.0, 100.0),
    ]

    assert candidate_ids(wells[0], wells, 5.0) == [2, 3]


def test_zero_threshold_only_compares_identical_coordinate_cells():
    wells = [Well(1, None, None), Well(2, 0.0, 0.0), Well(3, 0.1, 0.0)]

    assert candidate_ids(wells[0], wells, 0.0) == [2]


def test_negative_threshold_has_no_candidates():
    wells = [Well(1, 0.0, 0.0), Well(2, 0.0, 0.0)]

    assert candidate_ids(wells[0], wells, -1.0) == []
