"""Расчёт и форматирование годового отчёта по метражу профилей."""

from collections import defaultdict
from dataclasses import dataclass
import datetime
import json
import math


@dataclass(frozen=True)
class ProfileLength:
    object_title: str
    research_date: object
    profile_title: str
    coordinate_length: float | None
    measurement_length: float | None


def calculate_profile_length(x_json, y_json):
    """Вернуть длину ломаной профиля в метрах или ``None`` для плохих координат."""
    try:
        x_values = json.loads(x_json) if isinstance(x_json, str) else x_json
        y_values = json.loads(y_json) if isinstance(y_json, str) else y_json
        if not isinstance(x_values, list) or not isinstance(y_values, list):
            return None
        if len(x_values) != len(y_values) or not x_values:
            return None

        points = [(float(x), float(y)) for x, y in zip(x_values, y_values)]
        if not all(math.isfinite(value) for point in points for value in point):
            return None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None

    return sum(
        math.hypot(x2 - x1, y2 - y1)
        for (x1, y1), (x2, y2) in zip(points, points[1:])
    )


def calculate_measurement_length(signal_json, measurement_step=2.5):
    """Вернуть метраж по числу измерений с заданным шагом."""
    try:
        measurements = json.loads(signal_json) if isinstance(signal_json, str) else signal_json
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(measurements, list):
        return None
    return len(measurements) * measurement_step


def format_length(length):
    """Форматировать метраж с разделителями разрядов и двумя знаками."""
    return f"{length:,.2f}".replace(",", " ").replace(".", ",") + " м"


def format_length_comparison(coordinate_length, measurement_length):
    """Показать один итог при совпадении методов, иначе — оба значения."""
    if coordinate_length is None and measurement_length is None:
        return "нет данных для расчёта"
    if coordinate_length is None:
        return f"по измерениям — {format_length(measurement_length)}; координаты отсутствуют"
    if measurement_length is None:
        return f"по координатам — {format_length(coordinate_length)}; измерения отсутствуют"
    if math.isclose(coordinate_length, measurement_length, abs_tol=0.005):
        return format_length(coordinate_length)
    return (
        f"по координатам — {format_length(coordinate_length)}; "
        f"по измерениям — {format_length(measurement_length)}"
    )


def build_year_report(year, profiles):
    """Сформировать подробный текстовый отчёт из последовательности ProfileLength."""
    objects = defaultdict(list)
    for profile in profiles:
        objects[profile.object_title or "Без названия"].append(profile)

    coordinate_lengths = [p.coordinate_length for p in profiles if p.coordinate_length is not None]
    measurement_lengths = [p.measurement_length for p in profiles if p.measurement_length is not None]
    lines = [
        f"ОТЧЁТ ПО МЕТРАЖУ ПРОФИЛЕЙ ЗА {year} ГОД",
        "=" * 52,
        "Общий метраж за год: " + format_length_comparison(
            sum(coordinate_lengths) if coordinate_lengths else None,
            sum(measurement_lengths) if measurement_lengths else None,
        ),
        f"Объектов: {len(objects)}; профилей: {len(profiles)}",
    ]
    invalid_coordinates = sum(p.coordinate_length is None for p in profiles)
    invalid_measurements = sum(p.measurement_length is None for p in profiles)
    if invalid_coordinates:
        lines.append(f"Без корректных координат: {invalid_coordinates}")
    if invalid_measurements:
        lines.append(f"Без данных измерений: {invalid_measurements}")

    for object_title in sorted(objects, key=str.casefold):
        object_profiles = sorted(
            objects[object_title],
            key=lambda p: (
                p.research_date is None,
                p.research_date or datetime.date.min,
                (p.profile_title or "").casefold(),
            ),
        )
        object_coordinates = [p.coordinate_length for p in object_profiles if p.coordinate_length is not None]
        object_measurements = [p.measurement_length for p in object_profiles if p.measurement_length is not None]
        object_length = format_length_comparison(
            sum(object_coordinates) if object_coordinates else None,
            sum(object_measurements) if object_measurements else None,
        )
        lines.extend(("", f"{object_title} — {object_length}"))
        for profile in object_profiles:
            date_text = profile.research_date.strftime("%d.%m.%Y") if profile.research_date else "дата не указана"
            length_text = format_length_comparison(profile.coordinate_length, profile.measurement_length)
            lines.append(f"  • {profile.profile_title or 'Без названия'} ({date_text}): {length_text}")

    if not profiles:
        lines.extend(("", "За выбранный год профили не найдены."))
    return "\n".join(lines)
