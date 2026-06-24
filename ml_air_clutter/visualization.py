"""Unified visualization helpers for the ML air-clutter experiment.

The UI still delegates the actual drawing to ``MainWindow``; this module owns the
small, deterministic transformation API used to decide which arrays, traces,
metric curves and log lines belong to a preview.  Keeping this logic outside the
Qt dialog makes the paired ``noisy -> clean`` pipeline easy to test and reuse.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np


@dataclass
class VisualizationBundle:
    """Container with all artifacts needed by the UI visualization layer."""

    radarograms: Dict[str, np.ndarray] = field(default_factory=dict)
    traces: Dict[str, np.ndarray] = field(default_factory=dict)
    metric_curves: Dict[str, List[float]] = field(default_factory=dict)
    log_lines: List[str] = field(default_factory=list)

    def log_text(self) -> str:
        return "\n".join(self.log_lines)


def _as_2d(name: str, data) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D radarogram, got ndim={arr.ndim}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _same_shape(reference_name: str, reference: np.ndarray, name: str, data) -> np.ndarray:
    arr = _as_2d(name, data)
    if arr.shape != reference.shape:
        raise ValueError(f"{name} shape {arr.shape} does not match {reference_name} shape {reference.shape}.")
    return arr


def build_paired_visualization(
    *,
    noisy,
    clean=None,
    clean_pred=None,
    alpha: float = 1.0,
    trace_index: Optional[int] = None,
    title: str = "ML Clutter paired visualization",
    include_pair_residual: bool = True,
) -> VisualizationBundle:
    """Build mandatory Stage-12 paired-pipeline visual layers.

    Required mode is ``Noisy``.  If ``clean`` is supplied the bundle includes
    ``Clean`` and, when prediction is available, ``Error map predicted-clean``.
    If ``clean_pred`` is supplied it also includes ``Predicted clean``,
    ``Cleaned with alpha`` and ``Residual noisy-predicted``. For dataset-pair
    previews, ``include_pair_residual=False`` keeps the bundle to the actual
    supervised pair only: clean target and noisy input.
    """

    noisy_arr = _as_2d("noisy", noisy)
    clean_arr = _same_shape("noisy", noisy_arr, "clean", clean) if clean is not None else None
    pred_arr = _same_shape("noisy", noisy_arr, "clean_pred", clean_pred) if clean_pred is not None else None
    alpha = float(np.clip(alpha, 0.0, 1.0))

    if clean_arr is not None and not include_pair_residual and pred_arr is None:
        radarograms: Dict[str, np.ndarray] = {"Clean": clean_arr.copy(), "Noisy": noisy_arr.copy()}
    else:
        radarograms = {"Noisy": noisy_arr.copy()}
        if clean_arr is not None:
            radarograms["Clean"] = clean_arr.copy()
    if pred_arr is not None:
        cleaned = (1.0 - alpha) * noisy_arr + alpha * pred_arr
        radarograms["Predicted clean"] = pred_arr.copy()
        radarograms["Cleaned with alpha"] = cleaned
        radarograms["Residual noisy-predicted"] = noisy_arr - pred_arr
        if clean_arr is not None:
            radarograms["Error map predicted-clean"] = pred_arr - clean_arr
    elif clean_arr is not None and include_pair_residual:
        radarograms["Residual noisy-clean"] = noisy_arr - clean_arr

    if trace_index is None:
        trace_index = noisy_arr.shape[0] // 2
    trace_index = int(np.clip(trace_index, 0, noisy_arr.shape[0] - 1))
    traces = {name: array[trace_index].copy() for name, array in radarograms.items()}
    log_lines = [
        title,
        f"Shape: {tuple(noisy_arr.shape)}",
        f"Trace comparison index: {trace_index}",
        f"Alpha: {alpha:.3f}",
        "Visible radarogram modes: " + ", ".join(radarograms.keys()),
    ]
    return VisualizationBundle(radarograms=radarograms, traces=traces, log_lines=log_lines)


def build_training_metric_curves(history: Sequence[Mapping[str, object]]) -> Dict[str, List[float]]:
    """Extract finite training curves from epoch history dictionaries."""

    curves: Dict[str, List[float]] = {}
    for key in ("train_loss", "val_loss", "train_clean_l1", "val_clean_l1"):
        values: List[float] = []
        for row in history or []:
            try:
                value = float(row.get(key))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        if values:
            curves[key] = values
    return curves


def build_experiment_log(*sections: object) -> str:
    """Join non-empty UI log sections into a readable experiment log."""

    return "\n\n".join(str(section).strip() for section in sections if str(section).strip())
