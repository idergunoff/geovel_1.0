"""Automatic target-value resolution for regression well markups."""

from .service import (
    Resolution,
    ResolutionCandidate,
    TargetSettings,
    parse_numeric_value,
    resolve_target,
)

__all__ = [
    "Resolution",
    "ResolutionCandidate",
    "TargetSettings",
    "parse_numeric_value",
    "resolve_target",
]
