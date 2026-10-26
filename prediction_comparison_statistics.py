"""Error statistics for resolved well/prediction pairs, independent of Qt."""
import math
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class ComparisonStatistics:
    count: int
    well_count: int
    excluded_count: int
    within_tolerance: int
    tolerance: float
    minimum: float | None = None
    maximum: float | None = None
    mae: float | None = None
    median: float | None = None
    rmse: float | None = None
    bias: float | None = None
    p90: float | None = None
    minimum_rows: tuple = ()
    maximum_rows: tuple = ()

    @property
    def match_percent(self):
        return self.within_tolerance * 100 / self.count if self.count else None


def comparison_statistics(candidates, tolerance):
    """Each well/profile/prediction pair has equal weight, including repeat wells.

    Error sign follows the comparison table: well minus prediction. Unresolved,
    absent and non-finite values cannot contribute to the statistics. Extrema
    retain all tied pairs so users can locate the wells rather than just a number.
    """
    candidates = list(candidates)
    pairs = []
    for candidate in candidates:
        result = candidate.resolution
        if result is None or result.status != 'resolved':
            continue
        try:
            delta = float(result.value) - float(candidate.stored_value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(delta):
            pairs.append((candidate, delta))
    base = dict(count=len(pairs), well_count=len({c.well_id for c, _ in pairs}),
                excluded_count=len(candidates) - len(pairs), tolerance=tolerance,
                within_tolerance=sum(abs(delta) <= tolerance for _, delta in pairs))
    if not pairs:
        return ComparisonStatistics(**base)
    errors = sorted(abs(delta) for _, delta in pairs)
    minimum, maximum = errors[0], errors[-1]
    rank = (len(errors) - 1) * 0.9
    lower, upper = math.floor(rank), math.ceil(rank)
    p90 = errors[lower] + (errors[upper] - errors[lower]) * (rank - lower)
    return ComparisonStatistics(
        **base, minimum=minimum, maximum=maximum,
        mae=statistics.fmean(errors), median=statistics.median(errors),
        rmse=math.hypot(*(delta for _, delta in pairs)) / math.sqrt(len(pairs)),
        bias=statistics.fmean(delta for _, delta in pairs), p90=p90,
        minimum_rows=tuple(c for c, delta in pairs if abs(delta) == minimum),
        maximum_rows=tuple(c for c, delta in pairs if abs(delta) == maximum))
