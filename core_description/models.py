"""Intermediate, database- and GUI-independent core description models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class ParsedCoreInterval:
    """One logical row from a core-description table."""

    top_depth: Optional[float]
    bottom_depth: Optional[float]
    raw_description: str
    source_table: int
    source_row: int
    geological_index_raw: Optional[str] = None
    manifestation_raw: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    selected_by_default: bool = True
    rocks: list[RockMention] = field(default_factory=list)
    oil_saturation: str = "unknown"
    confidence: float = 0.0
    semantic_matches: list[SemanticMatch] = field(default_factory=list)


@dataclass
class SemanticMatch:
    """Explanation of one dictionary rule matched in source text."""

    rule_id: str
    category: str
    canonical_value: str
    matched_text: str
    start: int
    end: int


@dataclass
class RockMention:
    """A rock recognized in an interval and its contextual role."""

    canonical_value: str
    matched_text: str
    relation: str = "primary"
    confidence: float = 1.0
    rule_ids: list[str] = field(default_factory=list)


@dataclass
class ParsedCoreDocument:
    """Structural parsing result; semantic recognition is intentionally separate."""

    source_path: str
    source_format: str
    file_hash: str
    well_name_raw: Optional[str]
    well_name: Optional[str]
    area_name_raw: Optional[str]
    area_name: Optional[str]
    described_by_raw: Optional[str]
    described_by: Optional[str]
    intervals: list[ParsedCoreInterval] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    parser_version: str = "3.0"
    dictionary_version: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this result."""

        return asdict(self)
