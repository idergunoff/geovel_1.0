"""Export georadar profile polylines to native MapInfo TAB datasets."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


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


def _mif_number(value: float) -> str:
    return format(value, ".15g")


def _mif_points(profile: ExportProfile, zone: int) -> tuple[tuple[float, float], ...]:
    """Return EPSG-style points, adding the zone prefix when it is absent."""
    result = []
    for easting, northing in profile.projected_points:
        if gauss_kruger_zone_from_easting(easting) is None:
            easting += math.copysign(zone * 1_000_000, easting)
        result.append((easting, northing))
    return tuple(result)


def _mid_text(value: object) -> str:
    text = str(value).replace('"', '""').replace("\r", " ").replace("\n", " ")
    return f'"{text}"'


def _mif_header(zone: int) -> str:
    central_meridian = zone * 6 - 3
    false_easting = zone * 1_000_000 + 500_000
    # MapInfo projection 8 is Transverse Mercator; datum 1001 is Pulkovo 1942.
    return (
        'Version 300\n'
        'Charset "UTF-8"\n'
        'Delimiter ";"\n'
        f'CoordSys Earth Projection 8, 1001, "m", {central_meridian}, 0, 1, '
        f'{false_easting}, 0\n'
        'Columns 7\n'
        '  PROFILE_ID Integer\n'
        '  RESEARCH_ID Integer\n'
        '  TITLE Char(254)\n'
        '  POINTS Integer\n'
        '  LENGTH_M Float\n'
        '  ZONE Integer\n'
        '  EPSG Integer\n'
        'Data\n'
    )


def write_mif_mid(
    mif_path: str | Path,
    profiles: Sequence[ExportProfile],
    zone: int,
) -> tuple[Path, Path]:
    """Write MapInfo Interchange geometry and attributes without GIS dependencies."""
    output_mif = Path(mif_path).with_suffix(".mif")
    output_mid = output_mif.with_suffix(".mid")
    output_mif.parent.mkdir(parents=True, exist_ok=True)
    temp_paths: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=output_mif.parent, delete=False
        ) as mif_file, tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=output_mid.parent, delete=False
        ) as mid_file:
            temp_mif = Path(mif_file.name)
            temp_mid = Path(mid_file.name)
            temp_paths.extend((temp_mif, temp_mid))
            mif_file.write(_mif_header(zone))
            for profile in profiles:
                points = _mif_points(profile, zone)
                mif_file.write(f"Pline {len(points)}\n")
                for easting, northing in points:
                    mif_file.write(f"{_mif_number(easting)} {_mif_number(northing)}\n")
                mid_file.write(
                    ";".join(
                        (
                            str(profile.profile_id),
                            str(profile.research_id),
                            _mid_text(profile.title),
                            str(len(points)),
                            _mif_number(profile_length(points)),
                            str(zone),
                            str(28400 + zone),
                        )
                    )
                    + "\n"
                )
        os.replace(temp_mif, output_mif)
        temp_paths.remove(temp_mif)
        os.replace(temp_mid, output_mid)
        temp_paths.remove(temp_mid)
    finally:
        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)
    return output_mif, output_mid
