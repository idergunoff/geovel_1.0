"""Real-pattern based air-clutter generator for ML Clutter experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np

from .pattern_library import NoisePattern, PatternLibrary
from .synthetic_clutter import SyntheticClutterConfig, compute_snr_db, generate_synthetic_clutter


@dataclass(frozen=True)
class PatternClutterConfig:
    """Serializable configuration for real-pattern clutter placement and augmentation."""

    seed: Optional[int] = 42
    mode: str = "pattern"  # pattern | mixed
    pattern_ids: Optional[Sequence[str]] = None
    num_patterns: int = 1
    amplitude_scale_min: float = 0.8
    amplitude_scale_max: float = 1.2
    pattern_strength: float = 1.0
    synthetic_strength: float = 1.0
    target_snr_db: Optional[float] = 6.0
    random_crop: bool = True
    min_crop_fraction: float = 0.65
    horizontal_flip_probability: float = 0.5
    polarity_flip_probability: float = 0.5
    trace_stretch_min: float = 0.9
    trace_stretch_max: float = 1.1
    sample_stretch_min: float = 0.9
    sample_stretch_max: float = 1.1
    fade_probability: float = 0.7
    jitter_std: float = 0.01
    smooth_mask: bool = True

    def to_dict(self):
        data = asdict(self)
        if data["pattern_ids"] is not None:
            data["pattern_ids"] = list(data["pattern_ids"])
        return data


def generate_pattern_clutter(clean_profile, pattern_library, config=None, synthetic_config=None):
    """Return ``noisy, clutter, mask, meta`` using real extracted noise patterns.

    ``pattern_library`` may be a :class:`PatternLibrary` or an iterable of
    :class:`NoisePattern` instances. In ``mixed`` mode an analytical synthetic
    component is generated and saved separately in the metadata.
    """

    clean = np.asarray(clean_profile, dtype=float)
    _validate_clean(clean)
    config = config or PatternClutterConfig()
    rng = np.random.default_rng(config.seed)
    patterns = _select_patterns(pattern_library, config, rng)

    pattern_clutter = np.zeros_like(clean, dtype=float)
    pattern_mask = np.zeros_like(clean, dtype=float)
    placements = []
    for pattern in patterns:
        transformed, transformed_mask, transform_meta = transform_pattern(pattern, config, rng)
        placed, placed_mask, place_meta = place_pattern(clean.shape, transformed, transformed_mask, rng, config.smooth_mask)
        pattern_clutter += placed
        pattern_mask = np.maximum(pattern_mask, placed_mask)
        placements.append({
            "pattern_id": pattern.pattern_id,
            "source_profile": pattern.source_profile,
            "transform": transform_meta,
            "placement": place_meta,
        })

    pattern_clutter *= float(config.pattern_strength)
    synthetic_clutter = np.zeros_like(clean, dtype=float)
    synthetic_mask = np.zeros_like(clean, dtype=float)
    synthetic_meta = None
    if config.mode == "mixed":
        syn_cfg = synthetic_config or SyntheticClutterConfig(seed=config.seed, target_snr_db=None)
        _, synthetic_clutter, synthetic_mask, synthetic_meta = generate_synthetic_clutter(clean, syn_cfg)
        synthetic_clutter *= float(config.synthetic_strength)

    total_clutter = pattern_clutter + synthetic_clutter
    total_clutter, beta = scale_clutter_to_target_snr(clean, total_clutter, config.target_snr_db)
    pattern_clutter *= beta
    synthetic_clutter *= beta
    total_mask = np.maximum(pattern_mask, synthetic_mask)
    noisy = clean + total_clutter
    meta = {
        "config": config.to_dict(),
        "placements": placements,
        "target_snr_scale": beta,
        "actual_snr_db": compute_snr_db(clean, total_clutter),
        "pattern_clutter": {"rms": _rms(pattern_clutter), "mask_coverage": float(np.mean(pattern_mask > 0))},
        "synthetic_clutter": synthetic_meta,
        "total_clutter": {"rms": _rms(total_clutter), "mask_coverage": float(np.mean(total_mask > 0))},
    }
    return noisy, total_clutter, (total_mask > 0).astype(float), meta


def transform_pattern(pattern: NoisePattern, config: PatternClutterConfig, rng):
    arr = np.asarray(pattern.array, dtype=float).copy()
    mask = np.asarray(pattern.mask, dtype=float).copy()
    meta = {"original_shape": list(arr.shape)}

    if config.random_crop and arr.shape[0] > 1 and arr.shape[1] > 1:
        min_frac = min(max(float(config.min_crop_fraction), 0.05), 1.0)
        h = int(rng.integers(max(1, math.ceil(arr.shape[0] * min_frac)), arr.shape[0] + 1))
        w = int(rng.integers(max(1, math.ceil(arr.shape[1] * min_frac)), arr.shape[1] + 1))
        x0 = int(rng.integers(0, arr.shape[0] - h + 1))
        z0 = int(rng.integers(0, arr.shape[1] - w + 1))
        arr, mask = arr[x0:x0 + h, z0:z0 + w], mask[x0:x0 + h, z0:z0 + w]
        meta["crop"] = [x0, x0 + h, z0, z0 + w]

    if rng.random() < config.horizontal_flip_probability:
        arr, mask = arr[::-1].copy(), mask[::-1].copy()
        meta["horizontal_flip"] = True
    if rng.random() < config.polarity_flip_probability:
        arr = -arr
        meta["polarity_flip"] = True

    trace_scale = float(rng.uniform(config.trace_stretch_min, config.trace_stretch_max))
    sample_scale = float(rng.uniform(config.sample_stretch_min, config.sample_stretch_max))
    new_shape = (max(1, int(round(arr.shape[0] * trace_scale))), max(1, int(round(arr.shape[1] * sample_scale))))
    arr = _resize_2d(arr, new_shape)
    mask = np.clip(_resize_2d(mask, new_shape), 0.0, 1.0)
    meta["stretch"] = {"trace_scale": trace_scale, "sample_scale": sample_scale, "shape": list(new_shape)}

    amp = float(rng.uniform(config.amplitude_scale_min, config.amplitude_scale_max))
    arr *= amp
    meta["amplitude_scale"] = amp
    if rng.random() < config.fade_probability:
        fade = np.outer(np.hanning(arr.shape[0] + 2)[1:-1], np.hanning(arr.shape[1] + 2)[1:-1])
        arr *= fade
        mask *= fade
        meta["fade"] = True
    if config.jitter_std > 0:
        sigma = float(config.jitter_std) * (_rms(arr) or 1.0)
        arr += rng.normal(scale=sigma, size=arr.shape) * (mask > 0)
        meta["jitter_std"] = sigma
    return arr, mask, meta


def place_pattern(clean_shape, pattern, mask, rng, smooth_mask=True):
    traces, samples = clean_shape
    h, w = pattern.shape
    h, w = min(h, traces), min(w, samples)
    pattern, mask = pattern[:h, :w], mask[:h, :w]
    x0 = int(rng.integers(0, traces - h + 1))
    z0 = int(rng.integers(0, samples - w + 1))
    out = np.zeros(clean_shape, dtype=float)
    out_mask = np.zeros(clean_shape, dtype=float)
    effective_mask = np.clip(mask, 0.0, 1.0) if smooth_mask else (mask > 0).astype(float)
    out[x0:x0 + h, z0:z0 + w] = pattern * effective_mask
    out_mask[x0:x0 + h, z0:z0 + w] = effective_mask
    return out, out_mask, {"x_start": x0, "x_end": x0 + h, "z_start": z0, "z_end": z0 + w, "placed_shape": [h, w]}


def scale_clutter_to_target_snr(clean, clutter, target_snr_db):
    if target_snr_db is None or not np.any(clutter):
        return clutter, 1.0
    signal_power = float(np.mean(np.asarray(clean, dtype=float) ** 2))
    clutter_power = float(np.mean(np.asarray(clutter, dtype=float) ** 2))
    if clutter_power <= 0:
        return clutter, 1.0
    target_noise_power = signal_power / (10.0 ** (float(target_snr_db) / 10.0))
    beta = math.sqrt(target_noise_power / clutter_power)
    return clutter * beta, float(beta)


def _select_patterns(pattern_library, config, rng):
    patterns = pattern_library.patterns if isinstance(pattern_library, PatternLibrary) else list(pattern_library)
    if config.pattern_ids:
        allowed = set(config.pattern_ids)
        patterns = [p for p in patterns if p.pattern_id in allowed]
    if not patterns:
        raise ValueError("Pattern library does not contain selectable real-noise patterns.")
    count = max(1, int(config.num_patterns))
    indices = rng.choice(len(patterns), size=count, replace=len(patterns) < count)
    return [patterns[int(i)] for i in np.atleast_1d(indices)]


def _resize_2d(array, new_shape):
    arr = np.asarray(array, dtype=float)
    new_x = np.linspace(0, arr.shape[0] - 1, new_shape[0])
    tmp = np.vstack([np.interp(new_x, np.arange(arr.shape[0]), arr[:, j]) for j in range(arr.shape[1])]).T
    new_z = np.linspace(0, arr.shape[1] - 1, new_shape[1])
    return np.vstack([np.interp(new_z, np.arange(arr.shape[1]), tmp[i]) for i in range(tmp.shape[0])])


def _validate_clean(clean):
    if clean.ndim != 2:
        raise ValueError(f"Clean profile must be a 2D array, got ndim={clean.ndim}.")
    if not np.isfinite(clean).all():
        raise ValueError("Clean profile contains non-finite amplitudes.")


def _rms(array):
    arr = np.asarray(array, dtype=float)
    return float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0
