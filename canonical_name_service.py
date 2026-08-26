"""Canonical-name aliases shared by boundaries and well-data feature names."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from models_db.model import (
    AliasBoundary, AliasWellOption, Boundary, CanonicalBoundary,
    CanonicalWellOption, Well, WellOptionally, session,
)


class CanonicalNameError(ValueError):
    pass


@dataclass(frozen=True)
class CanonicalNameKind:
    canonical_model: object
    alias_model: object
    source_model: object
    source_name: object
    well_id: object


BOUNDARIES = CanonicalNameKind(CanonicalBoundary, AliasBoundary, Boundary, Boundary.title, Boundary.well_id)
WELL_DATA = CanonicalNameKind(
    CanonicalWellOption, AliasWellOption, WellOptionally, WellOptionally.option, WellOptionally.well_id
)


def _clean(value: str) -> str:
    value = (value or '').strip()
    if not value:
        raise CanonicalNameError('Название не может быть пустым')
    return value


def create_canonical(kind: CanonicalNameKind, name: str):
    row = kind.canonical_model(canonical_name=_clean(name))
    session.add(row)
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise CanonicalNameError(f'Каноническое название "{name}" уже существует') from exc
    return row


def delete_canonical(kind: CanonicalNameKind, canonical_id: int) -> None:
    row = session.query(kind.canonical_model).filter_by(id=canonical_id).first()
    if row is None:
        raise CanonicalNameError('Каноническое название не найдено')
    session.delete(row)
    session.commit()


def list_canonicals(kind: CanonicalNameKind):
    alias_counts = dict(session.query(kind.alias_model.canonical_id, func.count(kind.alias_model.id))
                        .group_by(kind.alias_model.canonical_id).all())
    return [{'id': row.id, 'canonical_name': row.canonical_name,
             'alias_count': int(alias_counts.get(row.id, 0))}
            for row in session.query(kind.canonical_model).order_by(kind.canonical_model.canonical_name).all()]


def add_alias(kind: CanonicalNameKind, canonical_id: int, name: str):
    if session.query(kind.canonical_model).filter_by(id=canonical_id).first() is None:
        raise CanonicalNameError('Каноническое название не найдено')
    row = kind.alias_model(alias_name=_clean(name), canonical_id=canonical_id)
    session.add(row)
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise CanonicalNameError(f'Алиас "{name}" уже назначен') from exc
    return row


def remove_alias(kind: CanonicalNameKind, canonical_id: int, name: str) -> None:
    normalized = _clean(name).casefold()
    row = session.query(kind.alias_model).filter(
        kind.alias_model.canonical_id == canonical_id,
        kind.alias_model.alias_name_norm == normalized,
    ).first()
    if row is None:
        raise CanonicalNameError('Алиас не найден')
    session.delete(row)
    session.commit()


def list_aliases(kind: CanonicalNameKind, canonical_id: int):
    return [row[0] for row in session.query(kind.alias_model.alias_name).filter_by(
        canonical_id=canonical_id).order_by(kind.alias_model.alias_name).all()]


def list_source_names(kind: CanonicalNameKind):
    rows = session.query(kind.source_name, func.count(kind.source_model.id)).filter(
        kind.source_name.isnot(None), func.trim(kind.source_name) != '').group_by(kind.source_name).order_by(
        func.count(kind.source_model.id).desc(), kind.source_name).all()
    return [{'name': name, 'usage_count': int(count)} for name, count in rows]


def wells_for_name(kind: CanonicalNameKind, name: str):
    return session.query(Well.id, Well.name).join(kind.source_model, kind.well_id == Well.id).filter(
        kind.source_name == name).distinct().order_by(Well.name, Well.id).all()


def resolve_name(kind: CanonicalNameKind, name: Optional[str]) -> Optional[str]:
    normalized = (name or '').strip().casefold()
    if not normalized:
        return None
    row = session.query(kind.canonical_model.canonical_name).join(
        kind.alias_model, kind.alias_model.canonical_id == kind.canonical_model.id).filter(
        kind.alias_model.alias_name_norm == normalized).first()
    return row[0] if row else None


def resolve_boundary_name(name: Optional[str]) -> Optional[str]:
    """Return the universal boundary name used by depth-binding consumers."""
    return resolve_name(BOUNDARIES, name)


def resolve_well_data_name(name: Optional[str]) -> Optional[str]:
    """Return the universal feature name used by well-data/model consumers."""
    return resolve_name(WELL_DATA, name)
