from __future__ import annotations

"""QC metric calculations for per-profile signal quality assessment."""

from dataclasses import dataclass
import json
from typing import Iterable, Mapping, Sequence

import numpy as np

from models_db.model import Profile, session


@dataclass(frozen=True)
class IntersectionConsistency:
    """Summary of correlation consistency for intersecting profiles."""

    mean_correlation: float | None
    min_correlation: float | None
    correlations: list[float]


def _safe_mean(values: Sequence[float]) -> float | None:
    """Return mean value or None for an empty sequence."""
    if not values:
        return None
    return float(np.mean(values))


def _safe_min(values: Sequence[float]) -> float | None:
    """Return min value or None for an empty sequence."""
    if not values:
        return None
    return float(np.min(values))


def _flatten_signals(signals: Sequence[Sequence[float]]) -> np.ndarray:
    """Flatten a list of traces into a 1D array of samples."""
    if not signals:
        return np.array([], dtype=float)
    return np.concatenate([np.asarray(trace, dtype=float) for trace in signals])


def _snr_for_trace(trace: np.ndarray, noise_window_ratio: float) -> tuple[float, float]:
    """Compute trace SNR and noise RMS using the leading noise window."""
    if trace.size == 0:
        return 0.0, 0.0
    noise_count = max(1, int(trace.size * noise_window_ratio))
    noise_segment = trace[:noise_count]
    signal_rms = float(np.sqrt(np.mean(trace ** 2)))
    noise_rms = float(np.sqrt(np.mean(noise_segment ** 2)))
    if noise_rms == 0:
        return float("inf"), 0.0
    return signal_rms / noise_rms, noise_rms


def _jump_ratio(trace: np.ndarray, z_threshold: float) -> float:
    """Return share of adjacent-sample jumps exceeding mean + z*std."""
    if trace.size < 2:
        return 0.0
    diffs = np.abs(np.diff(trace))
    if diffs.size == 0:
        return 0.0
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs))
    threshold = mean_diff + z_threshold * std_diff
    if threshold == 0:
        return 0.0
    return float(np.mean(diffs > threshold))


def _amplitude_outlier_ratio(flat_signal: np.ndarray, z_threshold: float) -> float:
    """Return ratio of samples exceeding z-score threshold in amplitude."""
    if flat_signal.size == 0:
        return 0.0
    mean_val = float(np.mean(flat_signal))
    std_val = float(np.std(flat_signal))
    if std_val == 0:
        return 0.0
    outlier_mask = np.abs(flat_signal - mean_val) > (z_threshold * std_val)
    return float(np.mean(outlier_mask))


def _intersection_consistency(
    intersections: Iterable[Mapping[str, Sequence[float]]],
) -> IntersectionConsistency:
    """Summarize similarity of signals at profile intersections via correlation."""
    correlations: list[float] = []
    for item in intersections:
        trace_a = np.asarray(item.get("signal_a", []), dtype=float)
        trace_b = np.asarray(item.get("signal_b", []), dtype=float)
        if trace_a.size == 0 or trace_b.size == 0:
            continue
        min_len = min(trace_a.size, trace_b.size)
        if min_len < 2:
            continue
        trace_a = trace_a[:min_len]
        trace_b = trace_b[:min_len]
        if np.std(trace_a) == 0 or np.std(trace_b) == 0:
            continue
        corr = float(np.corrcoef(trace_a, trace_b)[0, 1])
        correlations.append(corr)
    return IntersectionConsistency(
        mean_correlation=_safe_mean(correlations),
        min_correlation=_safe_min(correlations),
        correlations=correlations,
    )


def _point_density_metrics(
    x_coords: Sequence[float] | None,
    y_coords: Sequence[float] | None,
    gap_factor: float,
) -> dict[str, float | None]:
    """Calculate spacing statistics to detect non-uniform sampling or gaps."""
    if not x_coords or not y_coords:
        return {
            "mean_spacing": None,
            "spacing_cv": None,
            "gap_ratio": None,
            "max_spacing": None,
        }
    x = np.asarray(x_coords, dtype=float)
    y = np.asarray(y_coords, dtype=float)
    if x.size < 2 or y.size < 2:
        return {
            "mean_spacing": None,
            "spacing_cv": None,
            "gap_ratio": None,
            "max_spacing": None,
        }
    distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    mean_spacing = float(np.mean(distances))
    std_spacing = float(np.std(distances))
    if mean_spacing == 0:
        spacing_cv = 0.0
    else:
        spacing_cv = float(std_spacing / mean_spacing)
    gap_threshold = mean_spacing * gap_factor
    gap_ratio = float(np.mean(distances > gap_threshold)) if mean_spacing > 0 else 0.0
    return {
        "mean_spacing": mean_spacing,
        "spacing_cv": spacing_cv,
        "gap_ratio": gap_ratio,
        "max_spacing": float(np.max(distances)) if distances.size else None,
    }


def calculate_profile_qc_metrics(
    signals: Sequence[Sequence[float]],
    x_coords: Sequence[float] | None = None,
    y_coords: Sequence[float] | None = None,
    intersections: Iterable[Mapping[str, Sequence[float]]] | None = None,
    noise_window_ratio: float = 0.1,
    outlier_z_threshold: float = 3.0,
    jump_z_threshold: float = 3.0,
    gap_factor: float = 2.5,
) -> dict[str, object]:
    """
    Calculate QC metrics for a profile.

    Args:
        signals: List of traces (each trace is a sequence of amplitudes).
        x_coords: X coordinates for profile points.
        y_coords: Y coordinates for profile points.
        intersections: Iterable of mappings with keys "signal_a" and "signal_b" for
            comparing signals at profile intersections.
        noise_window_ratio: Fraction of the trace used as noise window from the start.
        outlier_z_threshold: Z-score threshold for amplitude outliers.
        jump_z_threshold: Z-score threshold for sharp amplitude jumps.
        gap_factor: Multiplier to define gaps relative to mean spacing.

    Returns:
        Dictionary with computed QC metrics:
            - snr_mean, snr_min: aggregate signal-to-noise ratio per trace.
            - noise_level_mean: mean noise RMS from leading noise windows.
            - amplitude_outlier_ratio: share of samples exceeding z-threshold.
            - amplitude_jump_ratio_mean: average share of sharp jumps in traces.
            - intersection_consistency: correlations for intersecting profiles.
            - point_density: spacing stats and gap ratios based on coordinates.
    """
    if intersections is None:
        intersections = []
    traces = [np.asarray(trace, dtype=float) for trace in signals]
    snr_values: list[float] = []
    noise_levels: list[float] = []
    jump_ratios: list[float] = []

    for trace in traces:
        snr, noise_level = _snr_for_trace(trace, noise_window_ratio)
        if np.isfinite(snr):
            snr_values.append(float(snr))
        noise_levels.append(noise_level)
        jump_ratios.append(_jump_ratio(trace, jump_z_threshold))

    flat_signal = _flatten_signals(signals)
    outlier_ratio = _amplitude_outlier_ratio(flat_signal, outlier_z_threshold)
    consistency = _intersection_consistency(intersections)
    density_metrics = _point_density_metrics(x_coords, y_coords, gap_factor)

    return {
        "snr_mean": _safe_mean(snr_values),
        "snr_min": _safe_min(snr_values),
        "noise_level_mean": _safe_mean(noise_levels),
        "amplitude_outlier_ratio": outlier_ratio,
        "amplitude_jump_ratio_mean": _safe_mean(jump_ratios),
        "intersection_consistency": {
            "mean_correlation": consistency.mean_correlation,
            "min_correlation": consistency.min_correlation,
            "correlations": consistency.correlations,
        },
        "point_density": density_metrics,
    }


def _find_intersection_points(
    x1: Sequence[float],
    y1: Sequence[float],
    x2: Sequence[float],
    y2: Sequence[float],
) -> list[tuple[int, int]]:
    """Return index pairs (i, j) for intersecting segments in two polylines."""
    intersection_pairs: list[tuple[int, int]] = []
    len_x1 = len(x1)
    len_x2 = len(x2)

    for i in range(len_x1 - 1):
        xa1, ya1 = x1[i], y1[i]
        xa2, ya2 = x1[i + 1], y1[i + 1]

        for j in range(len_x2 - 1):
            xb1, yb1 = x2[j], y2[j]
            xb2, yb2 = x2[j + 1], y2[j + 1]

            if (
                max(xa1, xa2) < min(xb1, xb2)
                or max(xb1, xb2) < min(xa1, xa2)
                or max(ya1, ya2) < min(yb1, yb2)
                or max(yb1, yb2) < min(ya1, ya2)
            ):
                continue

            den = (yb2 - yb1) * (xa2 - xa1) - (xb2 - xb1) * (ya2 - ya1)
            if den == 0:
                continue

            ua = ((xb2 - xb1) * (ya1 - yb1) - (yb2 - yb1) * (xa1 - xb1)) / den
            ub = ((xa2 - xa1) * (ya1 - yb1) - (ya2 - ya1) * (xa1 - xb1)) / den

            if 0 <= ua <= 1 and 0 <= ub <= 1:
                intersection_pairs.append((i, j))

    return intersection_pairs


def collect_profile_qc_inputs(profile_id: int) -> dict[str, object]:
    """
    Gather signals, coordinates, and intersection signal pairs for a profile.

    Intersections are searched only among profiles that belong to the same
    research as the provided profile_id.
    """
    profile = session.query(Profile).filter(Profile.id == profile_id).first()
    if not profile:
        raise ValueError(f"Profile id={profile_id} not found.")

    signals = json.loads(profile.signal) if profile.signal else []
    x_coords = json.loads(profile.x_pulc) if profile.x_pulc else []
    y_coords = json.loads(profile.y_pulc) if profile.y_pulc else []

    intersections: list[dict[str, Sequence[float]]] = []
    if x_coords and y_coords and profile.research_id:
        profiles = (
            session.query(Profile)
            .filter(Profile.research_id == profile.research_id)
            .filter(Profile.id != profile_id)
            .all()
        )
        for other in profiles:
            if not other.x_pulc or not other.y_pulc or not other.signal:
                continue
            other_x = json.loads(other.x_pulc)
            other_y = json.loads(other.y_pulc)
            other_signals = json.loads(other.signal)
            if len(other_x) < 2 or len(other_y) < 2:
                continue

            for i_idx, j_idx in _find_intersection_points(
                x_coords,
                y_coords,
                other_x,
                other_y,
            ):
                if i_idx < len(signals) and j_idx < len(other_signals):
                    intersections.append(
                        {
                            "signal_a": signals[i_idx],
                            "signal_b": other_signals[j_idx],
                        }
                    )

    return {
        "signals": signals,
        "x_coords": x_coords,
        "y_coords": y_coords,
        "intersections": intersections,
    }
