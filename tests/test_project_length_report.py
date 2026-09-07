import datetime

from project_length_report import (
    ProfileLength,
    build_year_report,
    calculate_measurement_length,
    calculate_profile_length,
    format_length_comparison,
)


def test_calculate_profile_length_follows_all_segments():
    assert calculate_profile_length("[0, 3, 3]", "[0, 4, 8]") == 9.0


def test_calculate_profile_length_rejects_bad_coordinate_data():
    assert calculate_profile_length("[0, 1]", "[0]") is None
    assert calculate_profile_length("not json", "[0]") is None
    assert calculate_profile_length(None, None) is None


def test_calculate_measurement_length_uses_two_and_a_half_metres_per_measurement():
    assert calculate_measurement_length("[[1], [2], [3], [4]]") == 10.0
    assert calculate_measurement_length(None) is None


def test_length_comparison_prints_one_value_only_when_values_match():
    assert format_length_comparison(10.0, 10.0) == "10,00 м"
    assert format_length_comparison(9.0, 10.0) == (
        "по координатам — 9,00 м; по измерениям — 10,00 м"
    )


def test_build_year_report_contains_totals_objects_and_profiles():
    profiles = [
        ProfileLength("Объект Б", datetime.date(2025, 2, 1), "Профиль 2", 250.5, 250.0),
        ProfileLength("Объект А", datetime.date(2025, 1, 10), "Профиль 1", 1000.0, 1000.0),
        ProfileLength("Объект А", datetime.date(2025, 3, 10), "Профиль 3", None, 100.0),
    ]
    report = build_year_report(2025, profiles)
    assert "Общий метраж за год: по координатам — 1 250,50 м; по измерениям — 1 350,00 м" in report
    assert "Объект А — по координатам — 1 000,00 м; по измерениям — 1 100,00 м" in report
    assert "Профиль 1 (10.01.2025): 1 000,00 м" in report
    assert "Профиль 3 (10.03.2025): по измерениям — 100,00 м; координаты отсутствуют" in report
    assert "Без корректных координат: 1" in report
