"""Pure-ish target resolvers shared by the regression wizard and tests.

The module deliberately contains no Qt code.  A resolution always retains the
source row IDs and calculation details so an automatically generated training
value can be audited and reproduced later.
"""
from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from sqlalchemy import func

from models_db.model import (
    AliasBoundary,
    AliasWellOption,
    Boundary,
    CanonicalBoundary,
    CanonicalWellOption,
    WellLog,
    WellOptionally,
)
from models_db.model_cluster import AliasWellLog, CanonicalWellLog

ResolutionStatus = Literal["resolved", "ambiguous", "missing", "invalid"]


@dataclass(frozen=True)
class ResolutionCandidate:
    value: float
    source_id: int
    source_name: str
    raw_value: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Resolution:
    status: ResolutionStatus
    value: float | None = None
    candidates: list[ResolutionCandidate] = field(default_factory=list)
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def select(self, index: int) -> None:
        candidate = self.candidates[index]
        self.value = candidate.value
        self.status = "resolved"
        self.message = f'Выбрано вручную: {candidate.source_name}'
        self.details = {**self.details, "selected": asdict(candidate), "manual_override": True}


@dataclass(frozen=True)
class TargetSettings:
    source: Literal["boundary", "well_data", "well_log"]
    canonical_id: int
    strict_numeric: bool = False
    allow_explicit_sum: bool = True
    aggregation: Literal["median", "mean"] = "median"
    operation: Literal["single", "upper_lower_ratio", "lower_upper_ratio", "difference"] = "single"
    depth_mode: Literal["fixed", "boundary"] = "boundary"
    fixed_depth: float = 0.0
    boundary_canonical_id: int | None = None
    interval: float = 5.0
    interval_position: Literal["below", "above", "centered"] = "below"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_NUMBER = r"[+-]?(?:\d{1,3}(?:[ \u00a0]\d{3})+|\d+)(?:[.,]\d+)?"
_SINGLE_RE = re.compile(rf"(?<![\d.,])({_NUMBER})(?![\d.,])")
_SUM_RE = re.compile(rf"^\s*({_NUMBER})(?:\s*\+\s*({_NUMBER}))+\s*(?:[^\d+].*)?$", re.IGNORECASE)


def _to_float(token: str) -> float:
    return float(token.replace(" ", "").replace("\u00a0", "").replace(",", "."))


def parse_numeric_value(raw: Any, *, allow_explicit_sum: bool = True, strict: bool = False) -> Resolution:
    """Parse a numeric well attribute without silently interpreting ranges.

    Explicit ``a + b`` expressions may be summed.  Dates, ranges, inequalities,
    slash-separated values and strings containing multiple unrelated numbers
    remain ambiguous and require a user decision.
    """
    text = "" if raw is None else str(raw).strip()
    if not text:
        return Resolution("missing", message="Пустое значение")
    if allow_explicit_sum and "+" in text:
        match = _SUM_RE.match(text)
        if match:
            expression = text
            # Units are allowed only after the final operand; extract the arithmetic prefix.
            prefix = re.match(rf"^\s*({_NUMBER}(?:\s*\+\s*{_NUMBER})+)", expression)
            if prefix:
                values = [_to_float(item) for item in _SINGLE_RE.findall(prefix.group(1))]
                return Resolution("resolved", sum(values), message=f"Сумма {values}", details={"parts": values})
    tokens = _SINGLE_RE.findall(text)
    if len(tokens) != 1:
        status = "missing" if not tokens else "ambiguous"
        return Resolution(status, candidates=[
            ResolutionCandidate(_to_float(token), index, "числовая часть", token)
            for index, token in enumerate(tokens)
        ], message="Не найдено число" if not tokens else "Найдено несколько числовых частей")
    token = tokens[0]
    # A lone number surrounded by range/date/comparison syntax is not unambiguous.
    if any(symbol in text for symbol in ("<", ">")) or re.search(r"\d\s*[/–—]\s*\d", text):
        return Resolution("ambiguous", candidates=[ResolutionCandidate(_to_float(token), 0, "числовая часть", token)],
                          message="Диапазон, отношение или ограничение требует ручного решения")
    if strict and text != token:
        return Resolution("ambiguous", candidates=[ResolutionCandidate(_to_float(token), 0, "числовая часть", token)],
                          message="Строгий режим не допускает текст вокруг числа")
    return Resolution("resolved", _to_float(token), message=f'Распознано из "{text}"')


def _alias_norms(session, alias_model, canonical_id: int) -> list[str]:
    names = [row[0] for row in session.query(alias_model.alias_name_norm).filter(
        alias_model.canonical_id == canonical_id).all()]
    canonical_model = {AliasBoundary: CanonicalBoundary, AliasWellOption: CanonicalWellOption,
                       AliasWellLog: CanonicalWellLog}[alias_model]
    canonical = session.query(canonical_model.canonical_name_norm).filter(canonical_model.id == canonical_id).first()
    if canonical and canonical[0] not in names:
        names.append(canonical[0])
    return names


def resolve_boundary(session, well_id: int, canonical_id: int) -> Resolution:
    aliases = _alias_norms(session, AliasBoundary, canonical_id)
    rows = session.query(Boundary).filter(
        Boundary.well_id == well_id,
        func.lower(func.trim(Boundary.title)).in_(aliases),
    ).order_by(Boundary.depth, Boundary.id).all() if aliases else []
    candidates = [ResolutionCandidate(float(row.depth), row.id, row.title or "", str(row.depth),
                                      {"boundary_id": row.id}) for row in rows if row.depth is not None]
    if not candidates:
        return Resolution("missing", message="Граница выбранного типа отсутствует")
    if len(candidates) > 1:
        return Resolution("ambiguous", candidates=candidates, message="Найдено несколько границ")
    return Resolution("resolved", candidates[0].value, candidates, "Найдена одна граница",
                      {"selected": asdict(candidates[0])})


def resolve_well_data(session, well_id: int, settings: TargetSettings) -> Resolution:
    aliases = _alias_norms(session, AliasWellOption, settings.canonical_id)
    rows = session.query(WellOptionally).filter(
        WellOptionally.well_id == well_id,
        func.lower(func.trim(WellOptionally.option)).in_(aliases),
    ).order_by(WellOptionally.id).all() if aliases else []
    candidates: list[ResolutionCandidate] = []
    errors: list[str] = []
    for row in rows:
        parsed = parse_numeric_value(row.value, allow_explicit_sum=settings.allow_explicit_sum,
                                     strict=settings.strict_numeric)
        if parsed.status == "resolved" and parsed.value is not None:
            candidates.append(ResolutionCandidate(parsed.value, row.id, row.option or "", row.value or "",
                                                  {"well_option_id": row.id, **parsed.details}))
        else:
            errors.append(f'{row.option}: {parsed.message}')
    if not rows:
        return Resolution("missing", message="Данные выбранного типа отсутствуют")
    if not candidates:
        return Resolution("invalid", message="; ".join(errors) or "Нет числовых значений")
    if len(candidates) > 1:
        return Resolution("ambiguous", candidates=candidates,
                          message="Найдено несколько значений; " + "; ".join(errors))
    return Resolution("resolved", candidates[0].value, candidates, "Значение распознано",
                      {"selected": asdict(candidates[0]), "warnings": errors})


def _curve_values(row: WellLog, top: float, bottom: float) -> list[float]:
    if row.step is None or row.begin is None or float(row.step) <= 0 or not row.curve_data:
        return []
    try:
        raw = json.loads(row.curve_data)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    values = []
    for index, value in enumerate(raw if isinstance(raw, list) else []):
        depth = float(row.begin) + index * float(row.step)
        if top <= depth <= bottom:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                values.append(numeric)
    return values


def _aggregate(values: list[float], method: str) -> float:
    return float(statistics.median(values) if method == "median" else statistics.fmean(values))


def _interval(depth: float, length: float, position: str) -> tuple[float, float]:
    if position == "above":
        return depth - length, depth
    if position == "centered":
        return depth - length / 2.0, depth + length / 2.0
    return depth, depth + length


def resolve_well_log(session, well_id: int, settings: TargetSettings,
                     boundary_candidate: ResolutionCandidate | None = None) -> Resolution:
    if settings.interval <= 0:
        return Resolution("invalid", message="Интервал должен быть больше нуля")
    if settings.depth_mode == "boundary":
        if settings.boundary_canonical_id is None:
            return Resolution("invalid", message="Не выбрана опорная граница")
        boundary = resolve_boundary(session, well_id, settings.boundary_canonical_id)
        if boundary_candidate is not None:
            matching_candidate = next(
                (candidate for candidate in boundary.candidates
                 if candidate.source_id == boundary_candidate.source_id), None)
            if matching_candidate is None:
                return Resolution("invalid", message="Выбранная опорная глубина больше не существует")
            boundary.select(boundary.candidates.index(matching_candidate))
        if boundary.status != "resolved":
            boundary.message = "Опорная глубина: " + boundary.message
            boundary.details["pending_selection"] = "boundary_depth"
            return boundary
        depth = float(boundary.value)
        boundary_details = boundary.details
    else:
        depth = float(settings.fixed_depth)
        boundary_details = {"fixed_depth": depth}

    aliases = _alias_norms(session, AliasWellLog, settings.canonical_id)
    curves = session.query(WellLog).filter(
        WellLog.well_id == well_id,
        func.lower(func.trim(WellLog.curve_name)).in_(aliases),
    ).order_by(WellLog.id).all() if aliases else []
    candidates: list[ResolutionCandidate] = []
    for curve in curves:
        lower_interval = (_interval(depth, settings.interval, settings.interval_position)
                          if settings.operation == "single" else (depth, depth + settings.interval))
        lower_values = _curve_values(curve, *lower_interval)
        if not lower_values:
            continue
        lower = _aggregate(lower_values, settings.aggregation)
        upper_interval = (depth - settings.interval, depth)
        upper_values = _curve_values(curve, *upper_interval)
        if settings.operation == "single":
            value = lower
        elif not upper_values:
            continue
        else:
            upper = _aggregate(upper_values, settings.aggregation)
            if settings.operation == "upper_lower_ratio":
                if lower == 0:
                    continue
                value = upper / lower
            elif settings.operation == "lower_upper_ratio":
                if upper == 0:
                    continue
                value = lower / upper
            else:
                value = lower - upper
        details = {
            "well_log_id": curve.id,
            "depth": depth,
            "interval": list(lower_interval),
            "upper_interval": list(upper_interval) if settings.operation != "single" else None,
            "valid_point_count": len(lower_values),
            "aggregation": settings.aggregation,
            "operation": settings.operation,
            **boundary_details,
        }
        candidates.append(ResolutionCandidate(float(value), curve.id, curve.curve_name or "", str(value), details))
    if not curves:
        return Resolution("missing", message="Кривая выбранного типа отсутствует")
    if not candidates:
        return Resolution("invalid", message="В выбранном интервале нет достаточных валидных отсчётов")
    if len(candidates) > 1:
        return Resolution("ambiguous", candidates=candidates, message="Найдено несколько подходящих кривых")
    details = {"selected": asdict(candidates[0])}
    if boundary_details.get("manual_override"):
        details["manual_override"] = True
    return Resolution("resolved", candidates[0].value, candidates, "Каротажное значение рассчитано", details)


def resolve_target(session, well_id: int, settings: TargetSettings,
                   boundary_candidate: ResolutionCandidate | None = None) -> Resolution:
    if settings.source == "boundary":
        return resolve_boundary(session, well_id, settings.canonical_id)
    if settings.source == "well_data":
        return resolve_well_data(session, well_id, settings)
    if settings.source == "well_log":
        return resolve_well_log(session, well_id, settings, boundary_candidate)
    return Resolution("invalid", message=f"Неизвестный источник: {settings.source}")


def list_canonical_targets(session, source: str):
    model = {"boundary": CanonicalBoundary, "well_data": CanonicalWellOption,
             "well_log": CanonicalWellLog}[source]
    return session.query(model).order_by(model.canonical_name, model.id).all()
