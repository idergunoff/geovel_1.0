import datetime
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from well_import import is_confident_match, parse_number, rank_well_candidates, text_similarity


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
    assert not is_confident_match(ranked[0])


def test_exact_nearby_well_is_confident_and_text_is_normalized():
    candidate = rank_well_candidates([Well(1, "Ёлка-12", 10, 20)], "елка 12", 10.2, 20.2)[0]
    assert text_similarity("Ёлка-12", "елка 12") == 1
    assert is_confident_match(candidate)
