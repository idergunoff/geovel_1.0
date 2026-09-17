"""Database matching, duplicate detection and persistence for core descriptions."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping

from models_db.model import (
    CoreDescriptionDocument, CoreDescriptionInterval, CoreDescriptionRock,
    Well, WellOptionally,
)

from .models import ParsedCoreDocument, ParsedCoreInterval


class CoreDescriptionSaveError(ValueError):
    """Raised when a confirmed import command cannot safely be persisted."""


class DuplicateCoreDocumentError(CoreDescriptionSaveError):
    """Raised when an already imported source hash is rejected."""


@dataclass(frozen=True)
class WellMatchCandidate:
    well_id: int
    well_name: str
    confidence: float
    method: str
    area_name: str | None = None


@dataclass(frozen=True)
class WellMatchResult:
    status: str
    selected: WellMatchCandidate | None = None
    candidates: tuple[WellMatchCandidate, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DuplicateCheckResult:
    exact_document_id: int | None = None
    overlapping_interval_ids: tuple[int, ...] = ()


@dataclass
class SaveCoreImportResult:
    document_id: int
    intervals_saved: int
    rocks_saved: int
    replaced_document_id: int | None = None
    warnings: list[str] = field(default_factory=list)


def normalize_lookup_value(value: str | None) -> str:
    """Normalize document/database labels without inventing missing content."""

    value = unicodedata.normalize('NFKC', value or '').casefold().replace('ё', 'е')
    return re.sub(r'[^\w]+', '', value, flags=re.UNICODE)


def _area_rows(session: Any, well_ids: Iterable[int]) -> dict[int, list[str]]:
    ids = tuple(well_ids)
    if not ids:
        return {}
    rows = session.query(WellOptionally).filter(WellOptionally.well_id.in_(ids)).all()
    result: dict[int, list[str]] = {}
    for row in rows:
        option = normalize_lookup_value(row.option)
        if option in {'площадь', 'пл', 'area'} and row.value:
            result.setdefault(row.well_id, []).append(row.value)
    return result


def match_well(session: Any, well_name: str | None, area_name: str | None = None) -> WellMatchResult:
    """Resolve only unambiguous exact matches; return fuzzy alternatives for review."""

    wanted = normalize_lookup_value(well_name)
    if not wanted:
        return WellMatchResult('manual_required', warnings=('well_name_missing',))
    wells = session.query(Well).order_by(Well.id).all()
    areas = _area_rows(session, (well.id for well in wells))
    exact = [well for well in wells if normalize_lookup_value(well.name) == wanted]
    wanted_area = normalize_lookup_value(area_name)
    if len(exact) > 1 and wanted_area:
        area_exact = [well for well in exact if any(
            normalize_lookup_value(value) == wanted_area for value in areas.get(well.id, ())) ]
        if area_exact:
            exact = area_exact
    if len(exact) == 1:
        well = exact[0]
        values = areas.get(well.id, ())
        area_matches = not wanted_area or any(normalize_lookup_value(v) == wanted_area for v in values)
        method = 'exact_name_area' if wanted_area and area_matches else 'exact_name'
        confidence = 1.0 if area_matches else 0.9
        warnings = () if area_matches else ('area_not_confirmed',)
        return WellMatchResult('matched', WellMatchCandidate(
            well.id, well.name, confidence, method, values[0] if values else None), warnings=warnings)
    if len(exact) > 1:
        candidates = tuple(WellMatchCandidate(
            well.id, well.name, 1.0, 'exact_name_ambiguous',
            areas.get(well.id, [None])[0]) for well in exact)
        return WellMatchResult('ambiguous', candidates=candidates, warnings=('multiple_exact_wells',))

    candidates = []
    for well in wells:
        score = SequenceMatcher(None, wanted, normalize_lookup_value(well.name)).ratio()
        if score >= 0.6:
            candidates.append(WellMatchCandidate(
                well.id, well.name, round(score, 4), 'fuzzy_suggestion',
                areas.get(well.id, [None])[0]))
    candidates.sort(key=lambda item: (-item.confidence, item.well_name, item.well_id))
    return WellMatchResult('manual_required', candidates=tuple(candidates[:10]),
                           warnings=('no_exact_well',))


def check_duplicates(session: Any, document: ParsedCoreDocument, well_id: int) -> DuplicateCheckResult:
    """Find the exact source and approximate depth overlaps without making a decision."""

    exact = session.query(CoreDescriptionDocument.id).filter_by(
        source_file_hash=document.file_hash).first()
    overlaps: set[int] = set()
    for item in document.intervals:
        if item.top_depth is None or item.bottom_depth is None:
            continue
        rows = session.query(CoreDescriptionInterval.id).filter(
            CoreDescriptionInterval.well_id == well_id,
            CoreDescriptionInterval.top_depth < item.bottom_depth,
            CoreDescriptionInterval.bottom_depth > item.top_depth,
        ).all()
        overlaps.update(row[0] for row in rows)
    return DuplicateCheckResult(exact[0] if exact else None, tuple(sorted(overlaps)))


def _key(interval: ParsedCoreInterval) -> str:
    return f'{interval.source_table}:{interval.source_row}'


def _selected(interval: ParsedCoreInterval, selected: set[Any]) -> bool:
    return (_key(interval) in selected or interval.source_row in selected
            or (interval.source_table, interval.source_row) in selected)


def _edit_for(interval: ParsedCoreInterval, edits: Mapping[Any, Mapping[str, Any]]) -> Mapping[str, Any]:
    return (edits.get(_key(interval)) or edits.get((interval.source_table, interval.source_row))
            or edits.get(interval.source_row) or {})


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def save_core_import(
    session: Any,
    parsed_document: ParsedCoreDocument,
    *,
    well_id: int,
    selected_interval_ids: Iterable[Any],
    edits: Mapping[Any, Mapping[str, Any]] | None = None,
    duplicate_policy: str = 'reject',
    match_method: str = 'manual',
    match_confidence: float = 1.0,
) -> SaveCoreImportResult:
    """Atomically save selected rows; exact duplicates require an explicit policy."""

    if duplicate_policy not in {'reject', 'replace'}:
        raise CoreDescriptionSaveError('duplicate_policy must be reject or replace')
    selected = set(selected_interval_ids)
    edits = edits or {}
    intervals = [item for item in parsed_document.intervals if _selected(item, selected)]
    if not intervals:
        raise CoreDescriptionSaveError('No intervals selected')
    replaced_id = None
    rocks_saved = 0
    try:
        well = session.query(Well).filter_by(id=well_id).first()
        if well is None:
            raise CoreDescriptionSaveError(f'Well {well_id} does not exist')
        duplicate = session.query(CoreDescriptionDocument).filter_by(
            source_file_hash=parsed_document.file_hash).first()
        if duplicate is not None:
            if duplicate_policy == 'reject':
                raise DuplicateCoreDocumentError(
                    f'Document hash is already imported as document {duplicate.id}')
            replaced_id = duplicate.id
            session.delete(duplicate)
            session.flush()
        document = CoreDescriptionDocument(
            well_id=well_id, source_file_name=Path(parsed_document.source_path).name,
            source_file_hash=parsed_document.file_hash, source_format=parsed_document.source_format,
            well_name_raw=parsed_document.well_name_raw, area_name_raw=parsed_document.area_name_raw,
            described_by_raw=parsed_document.described_by_raw, described_by=parsed_document.described_by,
            parser_version=parsed_document.parser_version,
            dictionary_version=parsed_document.dictionary_version,
            match_method=match_method, match_confidence=match_confidence, status='imported',
            diagnostics=_json({'warnings': parsed_document.warnings,
                               'intervals_parsed': len(parsed_document.intervals),
                               'intervals_selected': len(intervals)}),
        )
        session.add(document)
        session.flush()
        for parsed in intervals:
            edit = dict(_edit_for(parsed, edits))
            top = edit.get('top_depth', parsed.top_depth)
            bottom = edit.get('bottom_depth', parsed.bottom_depth)
            raw = edit.get('raw_description', parsed.raw_description)
            oil = edit.get('oil_saturation', parsed.oil_saturation)
            confidence = edit.get('confidence', parsed.confidence)
            if (top is None or bottom is None or not math.isfinite(float(top))
                    or not math.isfinite(float(bottom)) or float(top) >= float(bottom)):
                raise CoreDescriptionSaveError(f'Invalid depths for interval {_key(parsed)}')
            if not str(raw).strip():
                raise CoreDescriptionSaveError(f'Empty description for interval {_key(parsed)}')
            row = CoreDescriptionInterval(
                document_id=document.id, well_id=well_id, top_depth=float(top), bottom_depth=float(bottom),
                raw_description=str(raw), normalized_description=' '.join(str(raw).casefold().split()),
                oil_saturation=str(oil), confidence=float(confidence),
                needs_review=bool(edit.get('needs_review', parsed.errors or parsed.confidence < 0.8)),
                source_table_index=parsed.source_table, source_row_index=parsed.source_row,
                manually_edited=bool(edit), recognition_details=_json({
                    'matches': [asdict(match) for match in parsed.semantic_matches],
                    'warnings': parsed.warnings, 'errors': parsed.errors,
                }),
            )
            session.add(row)
            session.flush()
            rocks = edit.get('rocks', parsed.rocks)
            for rock in rocks:
                data = asdict(rock) if not isinstance(rock, Mapping) else dict(rock)
                match = next((item for item in parsed.semantic_matches
                              if item.category == 'rock' and item.matched_text == data.get('matched_text')), None)
                session.add(CoreDescriptionRock(
                    interval_id=row.id, rock_name=data['canonical_value'],
                    role=data.get('relation', 'unknown'), oil_saturation=data.get('oil_saturation'),
                    source_text=data.get('matched_text', ''), confidence=float(data.get('confidence', 1.0)),
                    match_start=match.start if match else None, match_end=match.end if match else None,
                ))
                rocks_saved += 1
        session.commit()
        return SaveCoreImportResult(document.id, len(intervals), rocks_saved, replaced_id)
    except Exception:
        session.rollback()
        raise
