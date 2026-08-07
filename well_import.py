"""Pure helpers for validating and matching wells imported from spreadsheets."""

from dataclasses import dataclass
from datetime import date, datetime
from difflib import SequenceMatcher
import math
import re
from collections import defaultdict


def parse_number(value, *, field="значение"):
    """Return a finite float or raise a user-facing ValueError.

    Excel dates must not silently turn into numbers: this was the source of a
    ``round(datetime)`` crash in the old importer.
    """
    if isinstance(value, (date, datetime)):
        raise ValueError(f'{field}: вместо числа указана дата «{value}»')
    if value is None:
        raise ValueError(f'{field}: пустое значение')
    if isinstance(value, str):
        value = value.strip().replace("\u00a0", "").replace(" ", "").replace(",", ".")
        if not value:
            raise ValueError(f'{field}: пустое значение')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{field}: «{value}» не является числом') from error
    if not math.isfinite(result):
        raise ValueError(f'{field}: требуется конечное число')
    return result


def normalize_text(value):
    text = "" if value is None else str(value).casefold().replace("ё", "е")
    tokens = re.findall(r"[a-zа-я0-9]+", text)
    # Common area prefixes carry no identifying information and often differ
    # between sources ("пл. Северная" / "Северная площадь").
    tokens = [token for token in tokens if token not in {"пл", "площадь", "area"}]
    return " ".join(tokens)


def text_similarity(left, right):
    left, right = normalize_text(left), normalize_text(right)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


@dataclass(frozen=True)
class WellCandidate:
    well: object
    distance: float
    name_similarity: float
    area_similarity: float
    score: float


class WellLookupIndex:
    """Small in-memory index that avoids comparing every imported row to every well."""

    def __init__(self, wells=(), *, cell_size=100.0):
        self.cell_size = cell_size
        self._by_cell = defaultdict(list)
        self._by_name = defaultdict(list)
        for well in wells:
            self.add(well)

    def _cell(self, x, y):
        return math.floor(x / self.cell_size), math.floor(y / self.cell_size)

    def add(self, well):
        self._by_name[normalize_text(well.name)].append(well)
        if well.x_coord is not None and well.y_coord is not None:
            self._by_cell[self._cell(well.x_coord, well.y_coord)].append(well)

    def candidates(self, name, x, y):
        """Return exact-name wells plus wells in neighbouring coordinate cells."""
        result = {well.id: well for well in self._by_name.get(normalize_text(name), ())}
        cell_x, cell_y = self._cell(x, y)
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                for well in self._by_cell.get((cell_x + offset_x, cell_y + offset_y), ()):
                    result[well.id] = well
        return result.values()


def rank_well_candidates(wells, name, x, y, area="", areas_by_well=None, *, max_distance=100.0):
    """Rank plausible duplicates using number/name, area and nearby coordinates."""
    areas_by_well = areas_by_well or {}
    ranked = []
    for well in wells:
        if well.x_coord is None or well.y_coord is None:
            continue
        distance = math.hypot(well.x_coord - x, well.y_coord - y)
        name_similarity = text_similarity(well.name, name)
        area_similarity = max(
            (text_similarity(candidate_area, area) for candidate_area in areas_by_well.get(well.id, ())),
            default=0.0,
        )
        # Do not show unrelated records. A number match may compensate for
        # shifted coordinates; close coordinates may compensate for spelling.
        if distance > max_distance and name_similarity < 0.9:
            continue
        coordinate_score = max(0.0, 1.0 - distance / max_distance)
        available = [(name_similarity, 0.5), (coordinate_score, 0.35)]
        if normalize_text(area):
            available.append((area_similarity, 0.15))
        score = sum(value * weight for value, weight in available) / sum(weight for _, weight in available)
        if score >= 0.45:
            ranked.append(WellCandidate(well, distance, name_similarity, area_similarity, score))
    return sorted(ranked, key=lambda item: (-item.score, item.distance))


def other_fields_match(candidate, area=""):
    """Check the well number/name and, when supplied, the area."""
    return candidate.name_similarity >= 0.9 and (
        not normalize_text(area) or candidate.area_similarity >= 0.8
    )


def is_confident_match(candidate, area=""):
    """Matching fields and coordinates within 50 m are safe to merge automatically."""
    return candidate.distance <= 50.0 and other_fields_match(candidate, area)


def requires_confirmation(candidate, area=""):
    """Ask the user only when other fields match but coordinates differ by over 50 m."""
    return candidate.distance > 50.0 and other_fields_match(candidate, area)
