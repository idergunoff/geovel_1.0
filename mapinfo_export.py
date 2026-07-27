"""Export georadar profile polylines to native MapInfo TAB datasets."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_ZONE = 9
WGS_MATCH_TOLERANCE_METERS = 100.0


class ProfileExportError(ValueError):
    """Raised when profile coordinates cannot be exported safely."""


@dataclass(frozen=True)
class ExportProfile:
    profile_id: int
    research_id: int
    title: str
    projected_points: tuple[tuple[float, float], ...]
    wgs_points: tuple[tuple[float, float], ...]


def gauss_kruger_zone_from_longitude(longitude: float) -> int:
    """Return the six-degree Gauss-Kruger zone for a WGS longitude."""
    if not math.isfinite(longitude) or not -180.0 <= longitude <= 180.0:
        raise ProfileExportError(f"Некорректная долгота WGS 84: {longitude!r}")
    # Longitudes on a zone boundary belong to the zone immediately east of it.
    return max(1, min(60, math.floor(longitude / 6.0) + 1))


def gauss_kruger_zone_from_easting(easting: float) -> int | None:
    """Read a zone prefix from a full Soviet Gauss-Kruger easting."""
    if not math.isfinite(easting):
        return None
    zone = int(abs(easting) // 1_000_000)
    return zone if 1 <= zone <= 60 else None


def _numeric_array(raw_value: str | None, label: str, profile_title: str) -> list[float]:
    if not raw_value:
        raise ProfileExportError(f'У профиля "{profile_title}" отсутствуют координаты {label}.')
    try:
        values = json.loads(raw_value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProfileExportError(
            f'У профиля "{profile_title}" повреждены координаты {label}.'
        ) from exc
    if not isinstance(values, list):
        raise ProfileExportError(f'Координаты {label} профиля "{profile_title}" не являются массивом.')
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ProfileExportError(
            f'У профиля "{profile_title}" есть нечисловые координаты {label}.'
        ) from exc
    if not all(math.isfinite(value) for value in result):
        raise ProfileExportError(f'У профиля "{profile_title}" есть пустые координаты {label}.')
    return result


def prepare_export_profile(profile) -> ExportProfile:
    """Parse and validate all projected and WGS coordinate arrays of a profile."""
    title = profile.title or f"id {profile.id}"
    x_pulc = _numeric_array(profile.x_pulc, "x_pulc", title)
    y_pulc = _numeric_array(profile.y_pulc, "y_pulc", title)
    longitudes = _numeric_array(profile.x_wgs, "x_wgs", title)
    latitudes = _numeric_array(profile.y_wgs, "y_wgs", title)
    sizes = {len(x_pulc), len(y_pulc), len(longitudes), len(latitudes)}
    if len(sizes) != 1:
        raise ProfileExportError(
            f'У профиля "{title}" не совпадает количество координат СК-42 и WGS 84.'
        )
    if len(x_pulc) < 2:
        raise ProfileExportError(f'Для профиля "{title}" требуется минимум две точки.')
    if not all(-180.0 <= lon <= 180.0 for lon in longitudes):
        raise ProfileExportError(f'У профиля "{title}" некорректная долгота WGS 84.')
    if not all(-90.0 <= lat <= 90.0 for lat in latitudes):
        raise ProfileExportError(f'У профиля "{title}" некорректная широта WGS 84.')
    return ExportProfile(
        profile_id=profile.id,
        research_id=profile.research_id,
        title=title,
        projected_points=tuple(zip(x_pulc, y_pulc)),
        wgs_points=tuple(zip(longitudes, latitudes)),
    )


def determine_export_zone(profiles: Sequence[ExportProfile]) -> int:
    """Determine one zone and require projected prefixes to agree with WGS."""
    if not profiles:
        raise ProfileExportError("В текущем объекте нет профилей для экспорта.")
    wgs_zones = {
        gauss_kruger_zone_from_longitude(longitude)
        for profile in profiles
        for longitude, _latitude in profile.wgs_points
    }
    if len(wgs_zones) != 1:
        raise ProfileExportError(
            "Профили объекта находятся в разных зонах Гаусса–Крюгера по координатам WGS 84. "
            "Их необходимо экспортировать в отдельные TAB-файлы."
        )
    zone = wgs_zones.pop()
    prefixed_zones = {
        detected
        for profile in profiles
        for easting, _northing in profile.projected_points
        if (detected := gauss_kruger_zone_from_easting(easting)) is not None
    }
    if prefixed_zones and prefixed_zones != {zone}:
        raise ProfileExportError(
            f"Зона по WGS 84 ({zone}) не совпадает с префиксом координат СК-42 "
            f"({', '.join(map(str, sorted(prefixed_zones)))})."
        )
    return zone


def profile_length(points: Iterable[tuple[float, float]]) -> float:
    point_list = list(points)
    return sum(math.dist(first, second) for first, second in zip(point_list, point_list[1:]))


def tab_sidecar_paths(tab_path: str | Path) -> tuple[Path, ...]:
    path = Path(tab_path).with_suffix(".tab")
    return tuple(path.with_suffix(suffix) for suffix in (".tab", ".dat", ".map", ".id", ".ind"))
