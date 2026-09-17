"""Data-driven semantic recognition for core descriptions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .models import ParsedCoreInterval, RockMention, SemanticMatch

ALLOWED_CATEGORIES = {
    "rock", "rock_relation", "oil_indicator", "oil_saturation",
    "negation", "uncertainty", "ignore",
}
ALLOWED_PATTERN_TYPES = {"word", "phrase", "regex"}


class DictionaryValidationError(ValueError):
    """Raised when a semantic dictionary is malformed or unsafe."""


@dataclass(frozen=True)
class DictionaryRule:
    id: str
    category: str
    canonical_value: str
    patterns: tuple[str, ...]
    pattern_type: str
    priority: int


@dataclass(frozen=True)
class SemanticDictionary:
    version: str
    rules: tuple[DictionaryRule, ...]


def load_dictionary(path: str | Path | None = None) -> SemanticDictionary:
    """Load and strictly validate a versioned JSON dictionary."""

    source = Path(path) if path else Path(__file__).with_name("dictionary.json")
    try:
        payload: Any = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DictionaryValidationError(f"Cannot load semantic dictionary: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise DictionaryValidationError("Unsupported dictionary schema_version")
    if not isinstance(payload.get("version"), str) or not payload["version"].strip():
        raise DictionaryValidationError("Dictionary version must be a non-empty string")
    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, list):
        raise DictionaryValidationError("Dictionary rules must be a list")
    rules: list[DictionaryRule] = []
    ids: set[str] = set()
    required = {"id", "category", "canonical_value", "patterns", "pattern_type", "priority", "enabled"}
    for number, item in enumerate(raw_rules):
        if not isinstance(item, dict) or not required <= item.keys():
            raise DictionaryValidationError(f"Rule {number} has missing fields")
        if item["id"] in ids:
            raise DictionaryValidationError(f"Duplicate rule id: {item['id']}")
        ids.add(item["id"])
        if item["category"] not in ALLOWED_CATEGORIES:
            raise DictionaryValidationError(f"Unknown category: {item['category']}")
        if item["pattern_type"] not in ALLOWED_PATTERN_TYPES:
            raise DictionaryValidationError(f"Unknown pattern_type: {item['pattern_type']}")
        if not isinstance(item["patterns"], list) or not item["patterns"] or not all(
            isinstance(value, str) and value for value in item["patterns"]
        ):
            raise DictionaryValidationError(f"Rule {item['id']} has invalid patterns")
        if not isinstance(item["priority"], int) or not isinstance(item["enabled"], bool):
            raise DictionaryValidationError(f"Rule {item['id']} has invalid priority/enabled")
        if item["pattern_type"] == "regex":
            for pattern in item["patterns"]:
                try:
                    re.compile(pattern, re.I)
                except re.error as error:
                    raise DictionaryValidationError(f"Invalid regex in {item['id']}: {error}") from error
        if item["enabled"]:
            rules.append(DictionaryRule(
                item["id"], item["category"], item["canonical_value"],
                tuple(item["patterns"]), item["pattern_type"], item["priority"],
            ))
    return SemanticDictionary(payload["version"], tuple(rules))


def _matches(text: str, rules: Iterable[DictionaryRule]) -> list[SemanticMatch]:
    found: list[tuple[int, SemanticMatch]] = []
    for rule in rules:
        for pattern in rule.patterns:
            expression = pattern if rule.pattern_type == "regex" else re.escape(pattern)
            if rule.pattern_type in {"word", "phrase"}:
                expression = rf"(?<!\w){expression}(?!\w)"
            for match in re.finditer(expression, text, re.I):
                found.append((rule.priority, SemanticMatch(
                    rule.id, rule.category, rule.canonical_value,
                    match.group(0), match.start(), match.end(),
                )))
    return [item for _, item in sorted(found, key=lambda pair: (pair[1].start, -pair[0], pair[1].rule_id))]


def analyze_description(text: str, dictionary: SemanticDictionary | None = None) -> tuple[
    list[RockMention], str, float, list[SemanticMatch], list[str]
]:
    """Recognize rocks, their relations, saturation and uncertainty."""

    dictionary = dictionary or load_dictionary()
    matches = _matches(text, dictionary.rules)
    rock_matches = [match for match in matches if match.category == "rock"]
    relations = [match for match in matches if match.category == "rock_relation"]
    rocks: list[RockMention] = []
    for index, match in enumerate(rock_matches):
        relation = "primary" if index == 0 else "secondary"
        preceding = [item for item in relations if item.end <= match.start]
        if preceding:
            nearest = preceding[-1]
            previous_rock_end = rock_matches[index - 1].end if index else 0
            if nearest.start >= previous_rock_end:
                relation = nearest.canonical_value
        if any(item.canonical_value == "interbedding" for item in relations):
            relation = "interbedding"
        rocks.append(RockMention(match.canonical_value, match.matched_text, relation, 1.0, [match.rule_id]))

    negations = [match for match in matches if match.category == "negation"]
    explicit = [match for match in matches if match.category == "oil_saturation"]
    indicators = [match for match in matches if match.category == "oil_indicator"]
    uncertain = [match for match in matches if match.category == "uncertainty"]
    if negations:
        saturation = "none"
    elif explicit:
        saturation = explicit[0].canonical_value
    elif indicators:
        saturation = "present"
    else:
        saturation = "unknown"
    warnings: list[str] = []
    if uncertain and saturation != "none":
        if any(match.canonical_value == "locality" for match in uncertain):
            warnings.append("locality_marker")
        if any(match.canonical_value == "uncertain" for match in uncertain):
            warnings.append("uncertain_wording")
            if saturation != "unknown":
                saturation = "uncertain"
    recognized = bool(rocks or saturation != "unknown")
    confidence = 1.0 if recognized else 0.0
    if warnings:
        confidence = 0.65 if "uncertain_wording" in warnings else 0.8
    return rocks, saturation, confidence, matches, warnings


def enrich_intervals(
    intervals: Iterable[ParsedCoreInterval], dictionary: SemanticDictionary | None = None
) -> str:
    """Add semantic results to intervals in place and return dictionary version."""

    dictionary = dictionary or load_dictionary()
    for interval in intervals:
        rocks, oil, confidence, matches, warnings = analyze_description(
            interval.raw_description, dictionary
        )
        interval.rocks = rocks
        interval.oil_saturation = oil
        interval.confidence = confidence
        interval.semantic_matches = matches
        interval.warnings.extend(item for item in warnings if item not in interval.warnings)
    return dictionary.version
