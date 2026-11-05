"""Preview, apply and undo semantic parsing of stored intervals without reimporting."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from uuid import uuid4

from sqlalchemy.orm import joinedload, selectinload

from models_db.model import CoreDescriptionInterval, CoreDescriptionRock
from .models import PARSER_VERSION
from .semantics import SEMANTIC_PARSER_VERSION, analyze_description, load_dictionary_payload, validate_dictionary


class ReparseError(ValueError):
    """A preview is stale or a requested operation cannot be applied safely."""


class ReparseCancelled(ReparseError):
    """The user cancelled a read-only preview."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def recognition_details(interval):
    try:
        value = json.loads(interval.recognition_details)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def processing_metadata(interval):
    details = recognition_details(interval)
    return details.get("processing") or {
        "parser_version": interval.document.parser_version,
        "dictionary_version": interval.document.dictionary_version,
    }


ROCK_FIELDS = ("rock_name", "role", "oil_saturation", "source_text", "confidence", "match_start", "match_end")


def _state(interval, *, history=False):
    details = recognition_details(interval)
    if not history:
        details.pop("history", None)
    return {
        "oil_saturation": interval.oil_saturation, "confidence": interval.confidence,
        "needs_review": interval.needs_review, "recognition_details": details,
        "rocks": [{key: getattr(rock, key) for key in ROCK_FIELDS}
                  for rock in sorted(interval.rocks, key=lambda item: item.id or 0)],
    }


def _token(interval):
    return _json({
        "id": interval.id, "document_id": interval.document_id, "well_id": interval.well_id,
        "top": interval.top_depth, "bottom": interval.bottom_depth,
        "text": interval.raw_description, "normalized": interval.normalized_description,
        "manual": interval.manually_edited, "thickness": interval.thickness,
        "index": interval.geological_index_raw, "state": _state(interval, history=True),
    })


@dataclass(frozen=True)
class ReparseProposal:
    interval_id: int
    well_id: int
    well_name: str
    top_depth: float
    bottom_depth: float
    text: str
    protected: bool
    token: str
    before: str
    after: str

    @property
    def changed(self):
        before, after = json.loads(self.before), json.loads(self.after)
        before["recognition_details"].pop("processing", None)
        after["recognition_details"].pop("processing", None)
        return before != after


@dataclass(frozen=True)
class ReparsePreview:
    proposals: tuple[ReparseProposal, ...]
    dictionary_json: str
    dictionary_hash: str


def _query(session):
    return session.query(CoreDescriptionInterval).options(
        selectinload(CoreDescriptionInterval.rocks), joinedload(CoreDescriptionInterval.document),
        joinedload(CoreDescriptionInterval.well))


def preview_reparse(session, *, interval_ids=None, well_ids=None, dictionary_payload=None, progress=None):
    """No writes. The caller must explicitly specify an interval or well scope."""
    if interval_ids is None and well_ids is None:
        raise ReparseError("Выберите интервалы или скважины для переразбора.")
    payload = deepcopy(dictionary_payload if dictionary_payload is not None else load_dictionary_payload())
    dictionary = validate_dictionary(payload)
    encoded = _json(payload)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    query = _query(session)
    # Filter in Python to avoid SQLite's parameter limit on large selections.
    wanted_intervals = set(interval_ids) if interval_ids is not None else None
    wanted_wells = set(well_ids) if well_ids is not None else None
    if wanted_wells is not None and len(wanted_wells) < 500:
        query = query.filter(CoreDescriptionInterval.well_id.in_(wanted_wells))
    if wanted_intervals is not None and len(wanted_intervals) < 500:
        query = query.filter(CoreDescriptionInterval.id.in_(wanted_intervals))
    proposals = []
    with session.no_autoflush:
        rows = [row for row in query.order_by(CoreDescriptionInterval.well_id, CoreDescriptionInterval.top_depth,
                                             CoreDescriptionInterval.id)
                if (wanted_intervals is None or row.id in wanted_intervals)
                and (wanted_wells is None or row.well_id in wanted_wells)]
    for index, interval in enumerate(rows):
        if progress is not None and progress(index, len(rows)) is False:
            raise ReparseCancelled("Расчёт отменён. Сохранённые результаты не изменены.")
        before = _state(interval)
        rocks, oil, confidence, matches, warnings = analyze_description(interval.raw_description, dictionary)
        details = deepcopy(before["recognition_details"])
        # Structural warnings survive; semantic warnings are replaced on every run.
        old_semantic = {"uncertain_wording", "locality_marker"}
        structural = [w for w in details.get("warnings", []) if w not in old_semantic]
        details.update(matches=[asdict(match) for match in matches],
                       warnings=list(dict.fromkeys(structural + warnings)))
        details["processing"] = {
            "parser_version": PARSER_VERSION, "semantic_parser_version": SEMANTIC_PARSER_VERSION,
            "dictionary_version": dictionary.version, "dictionary_hash": digest,
        }
        rock_matches = [match for match in matches if match.category == "rock"]
        new_rocks = []
        for rock, match in zip(rocks, rock_matches):
            new_rocks.append({"rock_name": rock.canonical_value, "role": rock.relation,
                              "oil_saturation": None, "source_text": rock.matched_text,
                              "confidence": rock.confidence, "match_start": match.start,
                              "match_end": match.end})
        after = {"oil_saturation": oil, "confidence": confidence,
                 "needs_review": bool(details.get("errors") or confidence < .8),
                 "rocks": new_rocks, "recognition_details": details}
        proposals.append(ReparseProposal(
            interval.id, interval.well_id, interval.well.name, interval.top_depth,
            interval.bottom_depth, interval.raw_description, interval.manually_edited,
            _token(interval), _json(before), _json(after)))
    if progress is not None and progress(len(rows), len(rows)) is False:
        raise ReparseCancelled("Расчёт отменён. Сохранённые результаты не изменены.")
    return ReparsePreview(tuple(proposals), encoded, digest)


def _write_state(interval, state, history):
    interval.oil_saturation = state["oil_saturation"]
    interval.confidence = state["confidence"]
    interval.needs_review = state["needs_review"]
    details = deepcopy(state["recognition_details"])
    details["history"] = history
    interval.recognition_details = _json(details)
    interval.rocks[:] = [CoreDescriptionRock(**rock) for rock in state["rocks"]]


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _locked_rows(session, selected):
    """Acquire SQLite's write lock before checking the snapshot; no values change."""
    session.flush()
    table = CoreDescriptionInterval.__table__
    ids = sorted(selected)
    for start in range(0, len(ids), 400):
        session.execute(table.update().where(table.c.id.in_(ids[start:start + 400])).values(
            recognition_details=table.c.recognition_details))
    query = _query(session).populate_existing()
    return query.filter(CoreDescriptionInterval.id.in_(ids)).all() if len(ids) < 500 else [
        row for row in query if row.id in selected]


def apply_reparse(session, preview, selected_interval_ids, *, allow_manual=False):
    """Apply exactly the reviewed rows atomically; reject stale or protected rows."""
    selected = set(selected_interval_ids)
    proposals = {p.interval_id: p for p in preview.proposals}
    if not selected or not selected <= proposals.keys():
        raise ReparseError("Выберите строки из предпросмотра.")
    try:
        # Fetch fresh rows even when another window/session has edited them.
        rows = _locked_rows(session, selected)
        if {row.id for row in rows} != selected:
            raise ReparseError("Интервал удалён. Повторите предпросмотр.")
        for row in rows:
            if _token(row) != proposals[row.id].token:
                raise ReparseError("Описание изменилось после предпросмотра. Повторите разбор.")
            if row.manually_edited and not allow_manual:
                raise ReparseError("Интервал содержит ручные правки и защищён от замены.")
        now = datetime.now(timezone.utc).isoformat()
        for row in rows:
            proposal = proposals[row.id]
            history = recognition_details(row).get("history", [])
            state = json.loads(proposal.after)
            state["recognition_details"]["processing"]["processed_at"] = now
            history.append({"id": uuid4().hex, "action": "reparse", "at": now,
                            "before": json.loads(proposal.before),
                            "source_hash": _digest(row.raw_description), "result_hash": _digest(_json(state))})
            _write_state(row, state, history)
            diagnostic = json.loads(row.document.diagnostics or "{}")
            diagnostic.setdefault("semantic_dictionaries", {})[preview.dictionary_hash] = json.loads(preview.dictionary_json)
            row.document.diagnostics = _json(diagnostic)
        session.commit()
        return len(rows)
    except Exception:
        session.rollback()
        raise


def undo_reparse(session, interval_id, *, expected_token=None):
    """Undo the latest active reparse; retain the audit trail and all source data."""
    try:
        rows = _locked_rows(session, {interval_id})
        row = rows[0] if rows else None
        if row is None or expected_token is not None and _token(row) != expected_token:
            raise ReparseError("Описание изменилось. Обновите окно перед откатом.")
        history = recognition_details(row).get("history", [])
        undone = {entry.get("undoes") for entry in history if entry.get("action") == "undo"}
        target = next((entry for entry in reversed(history)
                       if entry.get("action") == "reparse" and entry["id"] not in undone), None)
        if target is None:
            raise ReparseError("Нет переразбора, который можно отменить.")
        if (target.get("source_hash", _digest(row.raw_description)) != _digest(row.raw_description)
                or target.get("result_hash", _digest(_json(_state(row)))) != _digest(_json(_state(row)))):
            raise ReparseError("Текст или распознавание изменены после переразбора. Откат заблокирован.")
        history.append({"id": uuid4().hex, "action": "undo", "undoes": target["id"],
                        "at": datetime.now(timezone.utc).isoformat(), "before": _state(row)})
        _write_state(row, target["before"], history)
        session.commit()
    except Exception:
        session.rollback()
        raise


def interval_token(interval):
    """Optimistic guard for the viewer's undo action."""
    return _token(interval)
