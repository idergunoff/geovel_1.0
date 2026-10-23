"""Well identity evidence: date precision, elevations and arbitrary categories."""

import pytest

from core_description.identity import (
    WellMetadata, compare_identity, extract_identity, normalize_date, normalize_elevation,
)
from core_description.models import ParsedCoreDocument
from core_description.areas import area_name_similarity, area_names_match


def document(**fields):
    return ParsedCoreDocument("core.docx", "docx", "a" * 64, "12", "12", "Северная", "Северная",
                              None, None, **fields)


@pytest.mark.parametrize("raw, expected", [
    ("28.08.2019", "2019-08-28"), ("28.8.19", "2019-08-28"),
    ("31/12/99", "1999-12-31"), ("2019-08-28 00:00:00", "2019-08-28"),
    ("11.2019", "2019-11"), ("2019-11", "2019-11"),
    ("31.02.2019", None), ("13.2019", None), (".09.2019", None), ("", None),
])
def test_date_normalization_preserves_precision(raw, expected):
    assert normalize_date(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("132,432 м", 132.432), ("-4.5", -4.5), (0, 0.0),
    ("м", None), (None, None), ("NaN", None), ("132 / 133", None),
])
def test_elevation_normalization_does_not_invent_missing_values(raw, expected):
    assert normalize_elevation(raw) == expected


def test_extract_identity_from_paragraphs_and_adjacent_cells():
    values, warnings = extract_identity([
        "Альтитуда устья: 132,432 м\nАльтитуда ротора: 133,432 м",
        "Скважина начата 28.08.19",
    ], [["Скважина окончена", "", "02.09.2019"], ["Глубина фактическая", "175,5 м"]])
    assert values["drilling_start"] == "2019-08-28"
    assert values["drilling_end"] == "2019-09-02"
    assert values["wellhead_elevation"] == 132.432
    assert warnings == []


def test_extraction_preserves_invalid_ambiguous_and_partial_source_dates():
    values, warnings = extract_identity([
        "Скважина начата: 31.02.2019", "Скважина окончена: 11.2019",
        "Альтитуда устья: 132,4 м", "Альтитуда устья: 133,4 м",
    ], [])
    assert values["drilling_start"] is None
    assert values["drilling_start_raw"] == "31.02.2019"
    assert values["drilling_end"] == "2019-11"
    assert values["wellhead_elevation"] is None
    assert set(warnings) == {"drilling_start_invalid", "wellhead_elevation_ambiguous"}


def test_source_date_can_be_repeated_with_less_precision():
    values, warnings = extract_identity(["Скважина начата: 28.08.19", "Начало бурения: 08.2019"], [])
    assert values["drilling_start"] == "2019-08-28"
    assert warnings == []


@pytest.mark.parametrize("stored, status", [
    (("28.08.19",), "match"), (("2019-08-28", "08.2019"), "match"),
    (("08.2019",), "partial"), (("29.08.2019",), "different"),
    (("28.08.2019", "29.08.2019"), "conflict"),
    (("08.2019", "09.2019"), "conflict"), (("неизвестно",), "invalid_database"),
])
def test_date_comparison_explains_matches_and_conflicts(stored, status):
    source = document(drilling_start="2019-08-28", drilling_start_raw="28.08.19")
    metadata = WellMetadata(tuple(("Дата начала бурения", value) for value in stored))
    assert compare_identity(source, metadata, None)[1].status == status


@pytest.mark.parametrize("altitude, status", [
    (132.43, "match"), (133.9, "match"), (134.432, "match"), (130.432, "match"),
    (134.433, "different"), (130.431, "different"), (None, "missing"),
])
def test_elevation_comparison_allows_two_metres(altitude, status):
    source = document(wellhead_elevation=132.432, wellhead_elevation_raw="132,432")
    assert compare_identity(source, WellMetadata(), altitude)[3].status == status


@pytest.mark.parametrize('first, second, expected', [
    ('Мельничное поднятие', 'мельничный уч', True),
    ('Мельничное поднятие', 'Мелничный участок', True),
    ('abcd', 'abxy', False), ('abcde', 'abxde', True),
    ('Объект 58', 'Объект 580', False), ('Объект 58', 'Объект 59', False),
    ('Мельничный 1', 'Мельничный 2', False), ('', '', False),
])
def test_area_similarity_threshold_is_strictly_above_half(first, second, expected):
    assert area_names_match(first, second) is expected
    if first == 'abcd':
        assert area_name_similarity(first, second) == .5


def test_area_search_uses_arbitrary_categories_but_not_drilling_measurements():
    metadata = WellMetadata((("Залежь", "Северная"), ("Свободная категория", "Сарабикуловский"),
                             ("Альтитуда устья", "58"), ("Объект", "58")))
    assert ("Свободная категория", "Сарабикуловский") in metadata.area_entries("Сарабикуловское поднятие")
    assert ("Альтитуда устья", "58") not in metadata.area_entries("Объект 58")
    assert ("Объект", "58") in metadata.area_entries("Объект 58")


@pytest.mark.parametrize('source_area, attributes, expected, status', [
    ('Северная', (("Площадь", "Южная"), ("Участок", "Северная"),
                  ("Месторождение", "Западное")), 'Участок: Северная', 'match'),
    ('Сарабикуловское поднятие', (("Площадь", "Южная"),
                                ("Свободная категория", "Сарабикуловский")),
     'Свободная категория: Сарабикуловский', 'match'),
    ('Объект 58', (("Объект", "580"), ("Участок", "58")), 'Участок: 58', 'match'),
    ('abcd', (("Площадь", "wxyz"), ("Участок", "abxy")), 'Участок: abxy', 'different'),
    (None, (("Площадь", "Южная"), ("Участок", "Северная")), 'Площадь: Южная', 'missing'),
    ('Северная', (), '', 'missing'),
])
def test_area_comparison_displays_only_the_entry_used_for_similarity(source_area, attributes, expected, status):
    source = document()
    source.area_name_raw = source.area_name = source_area
    comparison = compare_identity(source, WellMetadata(attributes), None)[0]
    assert comparison.database == expected
    assert comparison.status == status


def test_area_comparison_chooses_first_entry_when_best_scores_are_equal():
    metadata = WellMetadata((("Участок", "Северная"), ("Площадь", "Северная")))
    assert compare_identity(document(), metadata, None)[0].database == 'Участок: Северная'


@pytest.mark.parametrize('kind, label, source, stored, altitude, expected, status', [
    ('drilling_start', 'Начало бурения', '2019-08-28', ('29.08.2019', '28.08.19'), None, '28.08.19', 'conflict'),
    ('drilling_end', 'Окончание бурения', '2019-09-02', ('03.09.2019', '02.09.19'), None, '02.09.19', 'conflict'),
    ('drilling_start', 'Начало бурения', '2019-08-28', ('08.2019', '28.08.19'), None, '28.08.19', 'match'),
    ('drilling_end', 'Окончание бурения', '2019-09-02', ('10.2019', '09.2019'), None, '09.2019', 'conflict'),
    ('drilling_start', 'Начало бурения', None, ('неизвестно', '28.08.19'), None, '28.08.19', 'missing'),
    ('drilling_end', 'Окончание бурения', '2019-09-02', ('неизвестно', 'ошибка'), None, 'неизвестно', 'invalid_database'),
    ('wellhead_elevation', 'Альтитуда устья', 132.432, ('150', '132,43 м'), None, '132,43 м', 'conflict'),
    ('wellhead_elevation', 'Альтитуда устья', 132.432, ('132',), 132.432, '132.432', 'match'),
    ('wellhead_elevation', 'Альтитуда устья', 132.432, ('неизвестно', '130.5'), None, '130.5', 'conflict'),
    ('wellhead_elevation', 'Альтитуда устья', 0., ('10', '0'), None, '0', 'conflict'),
    ('wellhead_elevation', 'Альтитуда устья', 132.432, (), None, '', 'missing'),
])
def test_identity_cells_show_one_value_without_losing_conflicts(kind, label, source, stored, altitude, expected, status):
    source_document = document(**{kind: source})
    metadata = WellMetadata(tuple((label, value) for value in stored))
    comparison = next(item for item in compare_identity(source_document, metadata, altitude) if item.field == kind)
    assert comparison.database == expected
    assert comparison.status == status
    values = tuple(dict.fromkeys((*stored, str(altitude)))) if altitude is not None else stored
    assert comparison.database_values == values


@pytest.mark.parametrize('source_number, db_number, status', [
    ('10 Вр', '10вр', 'match'), ('10вр', '10 ВР', 'match'),
    ('10\u00a0Вр', '10вр', 'match'), ('10 Вр', '10а', 'different'),
    ('10 Вр', '110 Вр', 'different'), (None, '10 Вр', 'missing'), ('', '', 'missing'),
])
def test_well_number_is_a_comparison_criterion(source_number, db_number, status):
    source = document()
    source.well_name_raw = source.well_name = source_number
    comparisons = compare_identity(source, WellMetadata(), None, well_name=db_number)
    number = next(item for item in comparisons if item.field == 'well_name')
    assert number.status == status
    assert number.database == (db_number or '')


@pytest.mark.parametrize('first, second, expected', [
    ('Верхнее поднятие', 'Верхнее 3', True), ('Верхнее 3', 'Верхнее поднятие', True),
    ('Верхнее 3', 'Верхнее 4', False), ('Верхнее 3', 'Верхнее 30', False),
    ('Объект 3', 'Объект 4', False), ('Объект 3', 'Верхнее 3', False),
    ('Верхнее поднятие', 'Нижнее 3', False),
])
def test_area_can_match_unnumbered_name_to_numbered_deposit(first, second, expected):
    assert area_names_match(first, second) is expected
