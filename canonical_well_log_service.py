from __future__ import annotations

from typing import Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from models_db.model import session, WellLog
from models_db.model_cluster import CanonicalWellLog, AliasWellLog


class CanonicalWellLogServiceError(ValueError):
    """Base service error for canonical well log CRUD operations."""


class CanonicalWellLogNotFoundError(CanonicalWellLogServiceError):
    """Raised when canonical well log was not found."""


class CanonicalWellLogConflictError(CanonicalWellLogServiceError):
    """Raised on uniqueness conflicts during create/rename."""


def _clean_name(value: str) -> str:
    if value is None:
        raise CanonicalWellLogServiceError('Name cannot be empty')
    name = value.strip()
    if not name:
        raise CanonicalWellLogServiceError('Name cannot be empty')
    return name


def create_canonical_well_log(canonical_name: str, description: Optional[str] = None) -> CanonicalWellLog:
    canonical_name = _clean_name(canonical_name)

    canonical = CanonicalWellLog(canonical_name=canonical_name, description=description)
    session.add(canonical)

    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise CanonicalWellLogConflictError(
            f'Canonical name "{canonical_name}" already exists'
        ) from exc

    return canonical


def rename_canonical_well_log(canonical_id: int, canonical_name: str) -> CanonicalWellLog:
    canonical_name = _clean_name(canonical_name)

    canonical = session.query(CanonicalWellLog).filter(CanonicalWellLog.id == canonical_id).first()
    if canonical is None:
        raise CanonicalWellLogNotFoundError(f'Canonical well log with id={canonical_id} not found')

    canonical.canonical_name = canonical_name

    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise CanonicalWellLogConflictError(
            f'Canonical name "{canonical_name}" already exists'
        ) from exc

    return canonical


def delete_canonical_well_log(canonical_id: int) -> None:
    canonical = session.query(CanonicalWellLog).filter(CanonicalWellLog.id == canonical_id).first()
    if canonical is None:
        raise CanonicalWellLogNotFoundError(f'Canonical well log with id={canonical_id} not found')

    session.delete(canonical)
    session.commit()


def get_canonical_well_logs_stats() -> List[Dict[str, object]]:
    total_distinct_curve_names = session.query(func.count(func.distinct(WellLog.curve_name))).filter(
        WellLog.curve_name.isnot(None),
        func.trim(WellLog.curve_name) != ''
    ).scalar() or 0

    alias_count_by_canonical = dict(
        session.query(AliasWellLog.canonical_id, func.count(AliasWellLog.id))
        .group_by(AliasWellLog.canonical_id)
        .all()
    )

    coverage_count_by_canonical = dict(
        session.query(
            AliasWellLog.canonical_id,
            func.count(func.distinct(WellLog.curve_name))
        )
        .join(WellLog, AliasWellLog.alias_name_norm == func.lower(func.trim(WellLog.curve_name)))
        .filter(WellLog.curve_name.isnot(None), func.trim(WellLog.curve_name) != '')
        .group_by(AliasWellLog.canonical_id)
        .all()
    )

    canonical_rows = session.query(CanonicalWellLog).order_by(CanonicalWellLog.canonical_name).all()

    result: List[Dict[str, object]] = []
    for canonical in canonical_rows:
        alias_count = int(alias_count_by_canonical.get(canonical.id, 0))
        coverage_count = int(coverage_count_by_canonical.get(canonical.id, 0))
        coverage_pct = 0.0
        if total_distinct_curve_names > 0:
            coverage_pct = round((coverage_count / total_distinct_curve_names) * 100, 2)

        result.append({
            'id': canonical.id,
            'canonical_name': canonical.canonical_name,
            'description': canonical.description,
            'alias_count': alias_count,
            'coverage_count': coverage_count,
            'coverage_pct': coverage_pct,
            'total_distinct_curve_names': int(total_distinct_curve_names),
        })

    return result
