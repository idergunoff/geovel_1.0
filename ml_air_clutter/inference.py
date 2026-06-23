"""Full-profile inference utilities for ML air-clutter cleaning."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .train import _require_torch, make_input_channels, torch


@dataclass(frozen=True)
class InferenceConfig:
    """Serializable sliding-window inference configuration."""

    patch_width: int = 64
    stride: int = 32
    alpha: float = 1.0
    device: str = "auto"
    clip_min: float = 0.0
    clip_max: float = 256.0
    window: str = "hann"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _resolve_device(requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _patch_starts(num_traces: int, patch_width: int, stride: int) -> List[int]:
    if patch_width <= 0 or stride <= 0:
        raise ValueError("patch_width and stride must be positive.")
    if num_traces < patch_width:
        raise ValueError(f"profile has {num_traces} traces, fewer than patch_width={patch_width}.")
    starts = list(range(0, num_traces - patch_width + 1, stride))
    last = num_traces - patch_width
    if starts[-1] != last:
        starts.append(last)
    return starts


def _blend_window(patch_width: int, kind: str) -> np.ndarray:
    if kind == "uniform" or patch_width == 1:
        values = np.ones(patch_width, dtype=np.float32)
    elif kind == "hann":
        values = np.hanning(patch_width).astype(np.float32)
        if not np.any(values > 0):
            values = np.ones(patch_width, dtype=np.float32)
        values = np.maximum(values, 1e-3)
    else:
        raise ValueError("window must be 'hann' or 'uniform'.")
    return values[:, None]


def run_full_profile_inference(model, model_config, noisy_profile, config: InferenceConfig = None) -> Dict[str, object]:
    """Predict a clean full profile using overlapping model patches.

    The model is expected to be a direct-clean predictor trained on patches in
    the 0..256 amplitude range. Overlapping patch predictions are accumulated
    with a 1D x-axis blend window and normalized by accumulated weights.
    """

    _require_torch()
    config = config or InferenceConfig()
    noisy = np.asarray(noisy_profile, dtype=np.float32)
    if noisy.ndim != 2:
        raise ValueError(f"noisy_profile must be 2D, got ndim={noisy.ndim}.")
    if not np.isfinite(noisy).all():
        raise ValueError("noisy_profile contains NaN or infinite values.")
    starts = _patch_starts(noisy.shape[0], int(config.patch_width), int(config.stride))
    device = _resolve_device(config.device)
    original_device = next(model.parameters()).device
    model.to(device)
    model.eval()
    output_sum = np.zeros_like(noisy, dtype=np.float32)
    weight_sum = np.zeros_like(noisy, dtype=np.float32)
    window = _blend_window(int(config.patch_width), config.window).astype(np.float32)
    with torch.no_grad():
        for start in starts:
            end = start + int(config.patch_width)
            patch = noisy[start:end]
            x = make_input_channels(patch, model_config.input_channels)[None, ...] / 256.0
            tensor = torch.from_numpy(x.astype(np.float32)).to(device)
            pred = model(tensor).detach().cpu().numpy()[0, 0].astype(np.float32) * 256.0
            output_sum[start:end] += pred * window
            weight_sum[start:end] += window
    model.to(original_device)
    if np.any(weight_sum <= 0):
        raise RuntimeError("sliding-window inference left uncovered traces; check patch_width/stride.")
    clean_pred = output_sum / weight_sum
    clean_pred = np.clip(clean_pred, float(config.clip_min), float(config.clip_max))
    alpha = float(np.clip(config.alpha, 0.0, 1.0))
    cleaned = (1.0 - alpha) * noisy + alpha * clean_pred
    residual = noisy - clean_pred
    return {
        "noisy": noisy.copy(),
        "clean_pred": clean_pred.astype(np.float32),
        "cleaned": cleaned.astype(np.float32),
        "residual": residual.astype(np.float32),
        "meta": {
            "inference_schema": "ml_air_clutter_full_profile_inference_v1",
            "config": config.to_dict(),
            "effective_alpha": alpha,
            "num_windows": len(starts),
            "window_starts": [int(s) for s in starts],
            "profile_shape": list(noisy.shape),
            "input_channels": list(model_config.input_channels),
        },
    }


def blend_inference_result(noisy, clean_pred, alpha: float):
    """Recompute cleaned and residual arrays for an existing prediction."""

    noisy = np.asarray(noisy, dtype=np.float32)
    clean_pred = np.asarray(clean_pred, dtype=np.float32)
    if noisy.shape != clean_pred.shape:
        raise ValueError("noisy and clean_pred shapes must match.")
    alpha = float(np.clip(alpha, 0.0, 1.0))
    cleaned = (1.0 - alpha) * noisy + alpha * clean_pred
    return {"cleaned": cleaned.astype(np.float32), "residual": (noisy - clean_pred).astype(np.float32), "alpha": alpha}


def save_inference_result(path, result: Dict[str, object]) -> Path:
    """Save full-profile inference arrays and metadata as a compressed NPZ."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        noisy=result["noisy"],
        clean_pred=result["clean_pred"],
        cleaned=result["cleaned"],
        residual=result["residual"],
        meta=json.dumps(result.get("meta", {}), ensure_ascii=False),
    )
    return path
