"""Parsing and comparing well identity details without Qt or database imports."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Iterable

from .areas import area_names_match, area_name_similarity, AREA_SIMILARITY_THRESHOLD
from .models import ParsedCoreDocument

DATE_TOKEN = re.compile(r"(?<!\d)(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[./-]\d{1,2}[./-](?:\d{4}|\d{2})|\d{1,2}[./]\d{4}|\d{4}-\d{1,2})(?!\d)")
NUMBER_TOKEN = re.compile(r"[+-]?\d+(?:[.,]\d+)?")
ELEVATION_TOLERANCE = 2.0
MIN_IDENTITY_MATCHES = 2


def label_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    return re.sub(r"[^а-яa-z0-9]", "", text)


def identity_kind(label: str) -> str | None:
    key = label_key(label)
    if (re.search(r"начал.*(?:бур|проход)|скв(?:ажина)?начат|начат.*бур|бур.*(?:начал|начат)", key)
            or any(key.startswith(alias) for alias in ("датаначала", "drillingstart", "spuddate", "startdate"))):
        return "drilling_start"
    if (re.search(r"(?:оконч|конец|заверш).*бур|скв(?:ажина)?(?:оконч|закон)|оконч.*проход|бур.*(?:оконч|конец|заверш)", key)
            or any(key.startswith(alias) for alias in ("датаокончания", "drillingend", "completiondate", "enddate"))):
        return "drilling_end"
    if "ротор" in key or "rotor" in key:
        return None
    if (re.fullmatch(r"(?:альтитуда|alt|elevation|wellheadelevation|groundelevation)\d*", key)
            or re.search(r"(?:альтитуд|отметк).*(?:усть|земл)", key)):
        return "wellhead_elevation"
    return None


def normalize_date(value: str | None) -> str | None:
    text = (value or "").strip().removesuffix("г.").strip()
    # Excel exports may include a midnight time component.
    text = re.sub(r"[T ]00:00:00(?:[.]0+)?$", "", text)
    try:
        if match := re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", text):
            year, month, day = map(int, match.groups())
        elif match := re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4}|\d{2})", text):
            day, month, year = map(int, match.groups())
            if len(match.group(3)) == 2:
                year += 2000 if year <= 68 else 1900
        elif match := re.fullmatch(r"(\d{1,2})[./](\d{4})|(\d{4})-(\d{1,2})", text):
            month, year = (map(int, match.groups()[:2]) if match.group(1)
                           else (int(match.group(4)), int(match.group(3))))
            date(year, month, 1)
            return f"{year:04d}-{month:02d}"
        else:
            return None
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def normalize_elevation(value: str | float | None) -> float | None:
    text = str(value if value is not None else "").strip().replace("\u00a0", " ")
    text = re.sub(r"\s*(?:м[.]?|m|метр(?:а|ов)?)$", "", text, flags=re.I).strip()
    if not re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?", text):
        return None
    result = float(text.replace(",", "."))
    return result if math.isfinite(result) else None


def extract_identity(texts: list[str], rows: list[list[str]]) -> tuple[dict, list[str]]:
    """Read paragraphs and adjacent metadata cells, preserving partial dates."""
    values: dict[str, list[str]] = {}
    lines = [line.strip() for text in texts for line in text.splitlines() if line.strip()]
    entries = [(line, line) for line in lines]
    entries += [(line, lines[index + 1]) for index, line in enumerate(lines[:-1])
                if identity_kind(line) and not NUMBER_TOKEN.search(line)]
    for row in rows:
        for index, cell in enumerate(row):
            entries.append((cell, cell))
            if identity_kind(cell) and not NUMBER_TOKEN.search(cell):
                adjacent = next((value for value in row[index + 1:] if value.strip()), "")
                entries.append((cell, adjacent))
    for label, value in entries:
        kind = identity_kind(label)
        if kind is None:
            continue
        token = (NUMBER_TOKEN.search(value) if kind == "wellhead_elevation"
                 else DATE_TOKEN.search(value))
        if token:
            values.setdefault(kind, []).append(token.group())
    result: dict = {}
    warnings: list[str] = []
    for kind, raw_values in values.items():
        normalize = normalize_elevation if kind == "wellhead_elevation" else normalize_date
        normalized = {normalize(raw) for raw in raw_values}
        result[kind + "_raw"] = " / ".join(dict.fromkeys(raw_values))
        if kind != "wellhead_elevation" and None not in normalized:
            most_precise = max(normalized, key=len)
            if all(most_precise.startswith(value) for value in normalized):
                normalized = {most_precise}
        if len(normalized) != 1:
            result[kind] = None
            warnings.append(kind + "_ambiguous")
        else:
            result[kind] = normalized.pop()
            if result[kind] is None:
                warnings.append(kind + "_invalid")
    return result, warnings


@dataclass(frozen=True)
class WellMetadata:
    attributes: tuple[tuple[str, str], ...] = ()

    def values(self, kind: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys(value for label, value in self.attributes if identity_kind(label) == kind))

    def area_entries(self, wanted: str | None = None) -> tuple[tuple[str, str], ...]:
        return tuple((label, value) for label, value in self.attributes
                     if identity_kind(label) is None and (any(word in label_key(label) for word in (
                         "площад", "залеж", "месторожд", "подняти", "структур", "участ", "объект", "area", "field", "deposit"))
                         or label_key(label) in {"пл", "уч"} or area_names_match(wanted, value)))


@dataclass(frozen=True)
class IdentityComparison:
    field: str
    source: str
    database: str
    status: str
    similarity: float | None = None


def compare_identity(document: ParsedCoreDocument, metadata: WellMetadata,
                     altitude: float | None) -> tuple[IdentityComparison, ...]:
    result = []
    area_entries = metadata.area_entries(document.area_name)
    similarity = max((area_name_similarity(document.area_name, value) for _, value in area_entries), default=0.0)
    area_status = ("match" if similarity > AREA_SIMILARITY_THRESHOLD
                   else "different" if document.area_name and area_entries else "missing")
    result.append(IdentityComparison("area", document.area_name_raw or document.area_name or "",
                                    "; ".join(f"{label}: {value}" for label, value in area_entries), area_status, similarity))
    for kind in ("drilling_start", "drilling_end", "wellhead_elevation"):
        raw = getattr(document, kind + "_raw")
        source = getattr(document, kind)
        stored = metadata.values(kind)
        if kind == "wellhead_elevation" and normalize_elevation(altitude) is not None:
            stored = tuple(dict.fromkeys((*stored, str(altitude))))
        normalize = normalize_elevation if kind == "wellhead_elevation" else normalize_date
        normalized = [normalize(value) for value in stored]
        valid = [value for value in normalized if value is not None]
        status = "missing"
        if source is not None and valid:
            if kind == "wellhead_elevation":
                matches = [math.isclose(source, value, rel_tol=0, abs_tol=ELEVATION_TOLERANCE + 1e-9) for value in valid]
                compatible = matches
            else:
                matches = [source == value and len(source) == 10 for value in valid]
                compatible = [source.startswith(value) or value.startswith(source) for value in valid]
            if any(matches) and all(compatible) and len(valid) == len(stored):
                status = "match"
            elif any(compatible) and (not all(compatible) or len(valid) != len(stored)):
                status = "conflict"  # Several database values disagree with each other.
            elif kind != "wellhead_elevation" and all(compatible):
                status = "partial"
            else:
                status = "different"
        elif raw and source is None:
            status = "invalid_source"
        elif stored and not valid:
            status = "invalid_database"
        result.append(IdentityComparison(kind, raw or (str(source) if source is not None else ""),
                                         "; ".join(stored), status))
    return tuple(result)


def identity_match_count(comparisons: Iterable[IdentityComparison]) -> int:
    return len({item.field for item in comparisons if item.status == "match"})


def has_identity_conflicts(comparisons: Iterable[IdentityComparison]) -> bool:
    return any(item.field != "area" and item.status in {
        "different", "conflict", "invalid_source", "invalid_database"} for item in comparisons)
