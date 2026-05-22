from __future__ import annotations

from typing import Callable, Iterable, TypedDict


class CanonicalPreparationRow(TypedDict):
    requested_name: str
    canonical_name: str | None
    normalized_name: str
    status: str


class CanonicalPreparationSummary(TypedDict):
    total: int
    resolved: int
    unresolved: int


def normalize_curve_name(curve_name: str | None) -> str:
    """Normalize incoming curve name the same way as canonical resolver does."""
    if curve_name is None:
        return ""
    return str(curve_name).strip().lower()


def prepare_canonical_resolution_preview(
    curve_names: Iterable[str | None],
    resolver: Callable[[str | None], str | None],
) -> tuple[list[CanonicalPreparationRow], CanonicalPreparationSummary]:
    """
    Build a dry-run preview for future cluster module canonical integration.

    The function is intentionally side-effect free and can be used before
    the main clustering integration block is implemented.
    """
    rows: list[CanonicalPreparationRow] = []
    resolved_count = 0

    for raw_name in curve_names:
        normalized_name = normalize_curve_name(raw_name)
        canonical_name = resolver(raw_name)
        status = 'resolved' if canonical_name else 'unresolved'
        if canonical_name:
            resolved_count += 1
        rows.append({
            'requested_name': '' if raw_name is None else str(raw_name),
            'canonical_name': canonical_name,
            'normalized_name': normalized_name,
            'status': status,
        })

    total_count = len(rows)
    summary: CanonicalPreparationSummary = {
        'total': total_count,
        'resolved': resolved_count,
        'unresolved': total_count - resolved_count,
    }
    return rows, summary
