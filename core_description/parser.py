"""Structural parser for core-description DOCX tables."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

from .converter import DocConversionError, docx_source
from .docx import DocxExtractionError, ExtractedTable, extract_docx
from .models import ParsedCoreDocument, ParsedCoreInterval

NUMBER = r"[+-]?\d+(?:[\s\u00a0]*[.,]\d+)?"
RANGE_RE = re.compile(rf"^\s*({NUMBER})\s*(?:-|–|—)\s*({NUMBER})\s*(?:м\b)?", re.I)
WELL_RE = re.compile(
    r"(?:№\s*скв(?:ажины)?\.?|скв(?:ажина|ажины|\.)?)\s*(?:№|:)?\s*([^,;\n]+)", re.I
)
AREA_RE = re.compile(r"(?:площадь|пл\.)\s*(?:№|:)?\s*(.+)", re.I)
AUTHOR_RE = re.compile(r"^\s*описал(?:а)?(?:\s+геолог)?\s*:?[\s\-–—]*(.+?)\s*$", re.I)


class CoreDescriptionParseError(ValueError):
    """Raised when a document has no unambiguous supported structure."""


def _space(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\u00a0", " ")).strip(" \t\r\n:;,-–—")


def _number(value: str) -> Optional[float]:
    cleaned = value.strip().lower().replace("\u00a0", " ").replace("м", "")
    cleaned = cleaned.replace(" ", "").replace(",", ".")
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", cleaned):
        return None
    try:
        result = float(cleaned)
    except ValueError:
        return None
    return result if result == result and abs(result) != float("inf") else None


def _normalized_header(value: str) -> str:
    normalized = _space(value).lower().replace("ё", "е")
    return re.sub(r"\s*,?\s*м(?:етр(?:а|ов)?)?\.?$", "", normalized).strip()


def _find_metadata(
    texts: list[str], table_rows: list[list[str]]
) -> tuple[tuple[Optional[str], Optional[str], Optional[str], list[str]], Optional[str]]:
    wells: list[str] = []
    areas: list[str] = []
    authors: list[tuple[str, str]] = []
    for text in texts:
        if match := WELL_RE.search(text):
            wells.append(_space(match.group(1)))
        if match := AREA_RE.search(text):
            areas.append(_space(match.group(1)))
        if match := AUTHOR_RE.match(text):
            authors.append((text.strip(), _space(match.group(1))))
    # Converted forms often place a label and its value into adjacent cells.
    for row in table_rows:
        for index, cell in enumerate(row[:-1]):
            label = _normalized_header(cell).rstrip(":")
            adjacent = _space(row[index + 1])
            if adjacent and re.fullmatch(r"(?:скважина|скв\.?|№\s*скв\.?)", label, re.I):
                wells.append(adjacent)
            elif adjacent and re.fullmatch(r"(?:площадь|пл\.?)", label, re.I):
                areas.append(adjacent)
    warnings: list[str] = []
    unique_wells = list(dict.fromkeys(filter(None, wells)))
    unique_areas = list(dict.fromkeys(filter(None, areas)))
    unique_authors = list(dict.fromkeys(authors))
    if not unique_wells:
        warnings.append("well_name_missing")
    elif len(unique_wells) > 1:
        warnings.append("well_name_ambiguous")
    if len(unique_areas) > 1:
        warnings.append("area_name_ambiguous")
    if not unique_authors:
        warnings.append("description_author_missing")
    elif len(unique_authors) > 1:
        warnings.append("description_author_ambiguous")
    author_raw, author = unique_authors[-1] if len(unique_authors) == 1 else (None, None)
    return (
        unique_wells[0] if len(unique_wells) == 1 else None,
        unique_areas[0] if len(unique_areas) == 1 else None,
        author_raw,
        warnings,
    ), author


def _column_map(rows: list[list[str]]) -> tuple[dict[str, int], int, int]:
    """Return logical columns, header end offset, and structural score."""

    aliases = {
        "top": {"от", "верх", "кровля", "начало интервала"},
        "bottom": {"до", "низ", "подошва", "конец интервала"},
        "range": {"интервал", "глубина", "интервал, м"},
        "description": {"описание", "описание керна", "описание породы", "литологическое описание"},
        "geological_index": {"геологический индекс", "индекс"},
        "manifestation": {"проявление", "нефтепроявление"},
    }
    best: tuple[dict[str, int], int, int] = ({}, 0, 0)
    combined = [""] * max((len(row) for row in rows[:4]), default=0)
    for row_index, row in enumerate(rows[:4]):
        for index in range(len(combined)):
            value = _normalized_header(row[index]) if index < len(row) else ""
            if value:
                combined[index] = _space(f"{combined[index]} {value}")
        mapping: dict[str, int] = {}
        for index, header in enumerate(combined):
            for name, variants in aliases.items():
                if any(header == item or header.endswith(" " + item) for item in variants):
                    mapping[name] = index
        depth_score = 2 if {"top", "bottom"} <= mapping.keys() else 2 if "range" in mapping else 0
        score = depth_score + (2 if "description" in mapping else 0)
        if score > best[2]:
            best = mapping, row_index + 1, score
    return best


def _value(row: list[str], index: Optional[int]) -> str:
    return row[index].strip() if index is not None and index < len(row) else ""


def _parse_rows(table: ExtractedTable, table_index: int, mapping: dict[str, int], start: int) -> list[ParsedCoreInterval]:
    intervals: list[ParsedCoreInterval] = []
    for row_index, row in enumerate(table.rows[start:], start=start):
        if not any(cell.strip() for cell in row):
            continue
        top: Optional[float]
        bottom: Optional[float]
        if "range" in mapping:
            match = RANGE_RE.match(_value(row, mapping["range"]))
            top, bottom = (_number(match.group(1)), _number(match.group(2))) if match else (None, None)
        else:
            top = _number(_value(row, mapping.get("top")))
            bottom = _number(_value(row, mapping.get("bottom")))
        description = _value(row, mapping.get("description"))
        errors: list[str] = []
        warnings: list[str] = []
        if top is None and bottom is None:
            errors.append("missing_depths")
        elif top is None or bottom is None:
            errors.append("missing_depth_boundary")
        elif top == bottom:
            errors.append("zero_length_interval")
            if re.search(r"\b(?:кровля|подошва|граница|горизонт)\b", description, re.I):
                errors.append("named_boundary_not_imported")
        elif top > bottom:
            errors.append("reversed_interval")
        if description.casefold().strip(" .") == "изолированный керн":
            warnings.append("description_is_placeholder")
        if not description:
            errors.append("description_missing")
        intervals.append(ParsedCoreInterval(
            top_depth=top,
            bottom_depth=bottom,
            raw_description=description,
            source_table=table_index,
            source_row=row_index,
            geological_index_raw=_value(row, mapping.get("geological_index")) or None,
            manifestation_raw=_value(row, mapping.get("manifestation")) or None,
            warnings=warnings,
            errors=errors,
            selected_by_default=not errors,
        ))
    return intervals


def parse_core_document(path: str | Path) -> ParsedCoreDocument:
    """Parse one DOC/DOCX into a pure intermediate representation."""

    source = Path(path)
    if not source.is_file():
        raise CoreDescriptionParseError(f"File does not exist: {source}")
    source_format = source.suffix.lower().lstrip(".")
    if source_format not in {"doc", "docx"}:
        raise CoreDescriptionParseError(f"Unsupported file extension: {source.suffix}")
    file_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        with docx_source(source) as converted:
            extracted = extract_docx(converted)
    except (DocConversionError, DocxExtractionError) as error:
        raise CoreDescriptionParseError(str(error)) from error

    all_text = [block for block in extracted.blocks if isinstance(block, str) and block.strip()]
    all_rows = [row for block in extracted.blocks if isinstance(block, ExtractedTable) for row in block.rows]
    metadata, described_by = _find_metadata(all_text, all_rows)
    well, area, author_raw, warnings = metadata
    candidates: list[tuple[int, ExtractedTable, dict[str, int], int, int]] = []
    table_index = 0
    for block in extracted.blocks:
        if isinstance(block, ExtractedTable):
            mapping, start, score = _column_map(block.rows)
            if score:
                candidates.append((table_index, block, mapping, start, score))
            table_index += 1
    if not candidates:
        raise CoreDescriptionParseError("No supported core-description table found")
    best_score = max(candidate[4] for candidate in candidates)
    best = [candidate for candidate in candidates if candidate[4] == best_score]
    if best_score < 4:
        raise CoreDescriptionParseError("Table has depth or description columns, but not both")
    if len(best) != 1:
        raise CoreDescriptionParseError("Several equally suitable core-description tables found")
    index, table, mapping, start, _ = best[0]
    intervals = _parse_rows(table, index, mapping, start)
    if not intervals:
        raise CoreDescriptionParseError("Core-description table contains no data rows")
    return ParsedCoreDocument(
        source_path=str(source), source_format=source_format, file_hash=file_hash,
        well_name_raw=well, well_name=well, area_name_raw=area, area_name=area,
        described_by_raw=author_raw, described_by=described_by,
        intervals=intervals, warnings=warnings,
    )
