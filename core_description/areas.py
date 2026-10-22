"""Conservative lookup keys for area labels used in core-document headers."""

from __future__ import annotations

import re
import unicodedata


OBJECT_RE = re.compile(r"\bобъект(?:а)?\s*(?:№|номер|n[оo]?[.]?)?\s*(\d+(?:[-/]\d+)?[а-яa-z]?)\b", re.I)
AREA_KIND_RE = re.compile(r"\b(?:площадь|пл\.|поднятие|месторождение|структура|залежь|area|field|deposit)\b", re.I)


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


def area_names_match(first: str | None, second: str | None) -> bool:
    key = normalize_area_name(first)
    return bool(key and key == normalize_area_name(second))
