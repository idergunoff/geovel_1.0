import datetime
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from well_import import (
    WellLookupIndex,
    is_confident_match,
    parse_number,
    rank_well_candidates,
    requires_confirmation,
    text_similarity,
)


class Well:
    def __init__(self, ident, name, x, y):
        self.id, self.name, self.x_coord, self.y_coord = ident, name, x, y


def test_parse_number_rejects_excel_dates_with_clear_error():
    with pytest.raises(ValueError, match="вместо числа указана дата"):
        parse_number(datetime.datetime(2025, 1, 2), field="Альтитуда")


def test_parse_number_accepts_decimal_comma_and_rejects_non_finite():
    assert parse_number(" 1 234,50 ") == 1234.5
    with pytest.raises(ValueError, match="конечное"):
        parse_number(float("nan"))


def test_candidate_ranking_combines_name_area_and_coordinate_tolerance():
    wells = [Well(1, "Скв. 124", 1000, 2000), Well(2, "Другая 9", 5000, 6000)]
    ranked = rank_well_candidates(
        wells, "скв 124", 1003, 2004, "Северная площадь", {1: ["пл. Северная"]}
    )
    assert [item.well.id for item in ranked] == [1]
    assert ranked[0].distance == 5
    assert ranked[0].area_similarity > 0.7
    assert ranked[0].matched_area == "пл. Северная"
    assert is_confident_match(ranked[0], "Северная площадь")


def test_exact_nearby_well_is_confident_and_text_is_normalized():
    candidate = rank_well_candidates([Well(1, "Ёлка-12", 10, 20)], "елка 12", 10.2, 20.2)[0]
    assert text_similarity("Ёлка-12", "елка 12") == 1
    assert is_confident_match(candidate)


def test_numeric_excel_well_name_matches_database_integer_name():
    well = Well(1, "124", 10, 20)
    index = WellLookupIndex([well])

    candidates = rank_well_candidates(index.candidates("124.0", 10, 20), "124.0", 10, 20)

    assert [candidate.well.id for candidate in candidates] == [1]
    assert is_confident_match(candidates[0])


def test_uncertain_candidates_require_confirmation():
    near = rank_well_candidates([Well(1, "124", 0, 0)], "124", 30, 0)[0]
    far = rank_well_candidates([Well(1, "124", 0, 0)], "124", 70, 0)[0]
    wrong_area = rank_well_candidates(
        [Well(1, "124", 0, 0)], "124", 70, 0, "Южная", {1: ["Северная"]}
    )[0]
    assert is_confident_match(near)
    assert not requires_confirmation(near)
    assert requires_confirmation(far)
    assert requires_confirmation(wrong_area, "Южная")


def test_nearby_plausible_candidate_with_different_name_requires_confirmation():
    candidate = rank_well_candidates([Well(1, "124-A", 0, 0)], "124", 1, 1)[0]

    assert not is_confident_match(candidate)
    assert requires_confirmation(candidate)


def test_lookup_index_limits_search_and_keeps_far_exact_number():
    nearby = Well(1, "Соседняя", 10, 10)
    far_same_number = Well(2, "124", 5000, 5000)
    unrelated = Well(3, "Другая", 5000, 5000)
    index = WellLookupIndex([nearby, far_same_number, unrelated])
    assert {well.id for well in index.candidates("124", 0, 0)} == {1, 2}
