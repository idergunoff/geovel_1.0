"""Normalized names and fuzzy similarity for area labels in core headers."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache


OBJECT_RE = re.compile(r"\bобъект(?:а)?\s*(?:№|номер|n[оo]?[.]?)?\s*(\d+(?:[-/]\d+)?[а-яa-z]?)\b", re.I)
AREA_KIND_RE = re.compile(r"\b(?:площадь|пл|поднятие|месторождение|структура|залежь|участок|уч|area|field|deposit)\b[.]?", re.I)
AREA_SIMILARITY_THRESHOLD = 0.5


@lru_cache(maxsize=8192)
def normalize_area_name(value: str | None) -> str:
    """Compare grammatical variants, preserving proper names and object numbers.

    These keys are for suggestions, not permission to save a fuzzy match.
    """
    text = unicodedata.normalize("NFKC", value or "").casefold().replace("ё", "е")
    # NFKC changes the numero sign to 'No'.
    if match := OBJECT_RE.search(text):
        return "объект" + match.group(1)
    text = AREA_KIND_RE.sub(" ", text)
    text = re.sub(r"^\s*пл\b[.]?\s*", "", text)
    words = re.findall(r"[а-яa-z0-9]+", text)
    if len(words) == 1 and words[0].isdigit():
        return "объект" + words[0]
    return "".join(re.sub(r"(ск|цк|н|в)(?:ый|ий|ая|ое|ые|ие|ого|ой|ому|ым|им|ых|их|ую)$", r"\1", word)
                   for word in words)


@lru_cache(maxsize=8192)
def area_name_similarity(first: str | None, second: str | None) -> float:
    first_key, second_key = normalize_area_name(first), normalize_area_name(second)
    if not first_key or not second_key:
        return 0.0
    # Object numbers must remain distinct even when their text is very similar.
    if (first_key.startswith("объект") or second_key.startswith("объект")
            or re.findall(r"\d+", first_key) != re.findall(r"\d+", second_key)):
        return float(first_key == second_key)
    return SequenceMatcher(None, first_key, second_key, autojunk=False).ratio()


def area_names_match(first: str | None, second: str | None) -> bool:
    return area_name_similarity(first, second) > AREA_SIMILARITY_THRESHOLD
