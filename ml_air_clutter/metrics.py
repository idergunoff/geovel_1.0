"""Quality metrics for paired ML air-clutter cleaning experiments."""

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


def mae(prediction, target) -> float:
    prediction, target = _paired_arrays(prediction, target)
    return float(np.mean(np.abs(prediction - target)))


def rmse(prediction, target) -> float:
    prediction, target = _paired_arrays(prediction, target)
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def snr_db(reference, error) -> float:
    reference = np.asarray(reference, dtype=float)
    error = np.asarray(error, dtype=float)
    signal_power = float(np.mean(reference ** 2))
    noise_power = float(np.mean(error ** 2))
    if noise_power <= 0:
        return float("inf")
    if signal_power <= 0:
        return float("-inf")
    return float(10.0 * math.log10(signal_power / noise_power))


def psnr_db(prediction, target, data_range: Optional[float] = 256.0) -> float:
    value = rmse(prediction, target)
    if value <= 0:
        return float("inf")
    if data_range is None:
        target = np.asarray(target, dtype=float)
        data_range = float(np.max(target) - np.min(target)) if target.size else 0.0
    if data_range <= 0:
        return float("nan")
    return float(20.0 * math.log10(float(data_range) / value))


def trace_correlation(prediction, target) -> float:
    prediction, target = _paired_arrays(prediction, target)
    a = prediction.reshape(-1)
    b = target.reshape(-1)
    if a.size < 2 or np.std(a) <= 0 or np.std(b) <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def structural_correlation(prediction, target) -> float:
    """Lightweight SSIM-like structural correlation for visual-quality tracking."""

    prediction, target = _paired_arrays(prediction, target)
    a = prediction - np.mean(prediction)
    b = target - np.mean(target)
    denom = float(np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def high_energy_error(prediction, target, quantile: float = 0.75) -> Dict[str, float]:
    prediction, target = _paired_arrays(prediction, target)
    threshold = float(np.quantile(np.abs(target), quantile)) if target.size else 0.0
    mask = np.abs(target) >= threshold
    if not np.any(mask):
        return {"threshold": threshold, "mae": float("nan"), "rmse": float("nan"), "coverage": 0.0}
    return {
        "threshold": threshold,
        "mae": mae(prediction[mask], target[mask]),
        "rmse": rmse(prediction[mask], target[mask]),
        "coverage": float(np.mean(mask)),
    }


def gradient_rmse(prediction, target) -> float:
    prediction, target = _paired_arrays(prediction, target)
    gx_pred, gz_pred = np.gradient(prediction, axis=0), np.gradient(prediction, axis=1)
    gx_true, gz_true = np.gradient(target, axis=0), np.gradient(target, axis=1)
    return rmse(np.stack([gx_pred, gz_pred]), np.stack([gx_true, gz_true]))


def paired_cleaning_metrics(clean, noisy, clean_pred, data_range: Optional[float] = 256.0) -> Dict[str, object]:
    """Compare noisy-before and predicted-after arrays against paired clean ground truth."""

    clean, noisy = _paired_arrays(clean, noisy)
    clean_pred, clean = _paired_arrays(clean_pred, clean)
    before_error = noisy - clean
    after_error = clean_pred - clean
    residual = clean_pred - noisy
    before_snr = snr_db(clean, before_error)
    after_snr = snr_db(clean, after_error)
    return {
        "mae_before": mae(noisy, clean),
        "mae_after": mae(clean_pred, clean),
        "mae_improvement": mae(noisy, clean) - mae(clean_pred, clean),
        "rmse_before": rmse(noisy, clean),
        "rmse_after": rmse(clean_pred, clean),
        "rmse_improvement": rmse(noisy, clean) - rmse(clean_pred, clean),
        "snr_before_db": before_snr,
        "snr_after_db": after_snr,
        "snr_gain_db": after_snr - before_snr,
        "psnr_before_db": psnr_db(noisy, clean, data_range),
        "psnr_after_db": psnr_db(clean_pred, clean, data_range),
        "structural_correlation_before": structural_correlation(noisy, clean),
        "structural_correlation_after": structural_correlation(clean_pred, clean),
        "trace_correlation_after": trace_correlation(clean_pred, clean),
        "gradient_rmse_before": gradient_rmse(noisy, clean),
        "gradient_rmse_after": gradient_rmse(clean_pred, clean),
        "high_energy_error_after": high_energy_error(clean_pred, clean),
        "residual_energy": float(np.mean(residual ** 2)),
        "residual_mean_abs": float(np.mean(np.abs(residual))),
        "changed_energy_ratio": _energy(residual) / max(_energy(noisy), 1e-12),
    }


def summarize_metric_rows(rows: Iterable[Dict[str, object]]) -> Dict[str, object]:
    rows = list(rows)
    summary: Dict[str, object] = {"num_samples": len(rows)}
    if not rows:
        return summary
    numeric_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float, np.floating))]
    for key in numeric_keys:
        values = np.asarray([row[key] for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        summary[key] = float(np.mean(finite)) if finite.size else float("nan")
    return summary


def write_metrics_report(path, report: Dict[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(report), fh, indent=2, ensure_ascii=False)
    return path


def _paired_arrays(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Metric arrays must have matching shapes, got {a.shape} and {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Metric arrays must contain only finite values")
    return a, b


def _energy(data) -> float:
    data = np.asarray(data, dtype=float)
    return float(np.mean(data ** 2))


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = float(value)
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return str(value)
    return value
