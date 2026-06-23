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
    overlay_mode: str = "dominant_amplitude"  # raw_dominant_amplitude | dominant_amplitude | soft_dominance | additive
    preserve_pattern_depth: bool = True
    overlay_midpoint: float = 128.0
    soft_dominance_temperature: float = 12.0

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
    raw_dominant_mode = config.overlay_mode == "raw_dominant_amplitude"
    for pattern in patterns:
        if raw_dominant_mode:
            transformed = np.asarray(pattern.array, dtype=float).copy()
            transformed_mask = np.asarray(pattern.mask, dtype=float).copy()
            transform_meta = {"mode": "raw", "original_shape": list(transformed.shape)}
        else:
            transformed, transformed_mask, transform_meta = transform_pattern(pattern, config, rng)
        target_z_start = _pattern_target_z_start(pattern, transform_meta) if config.preserve_pattern_depth else None
        placed, placed_mask, place_meta = place_pattern(
            clean.shape,
            transformed,
            transformed_mask,
            rng,
            config.smooth_mask,
            target_z_start=target_z_start,
        )
        pattern_clutter += placed
        pattern_mask = np.maximum(pattern_mask, placed_mask)
        placements.append({
            "pattern_id": pattern.pattern_id,
            "source_profile": pattern.source_profile,
            "transform": transform_meta,
            "placement": place_meta,
        })

    if not raw_dominant_mode:
        pattern_clutter *= float(config.pattern_strength)
    synthetic_clutter = np.zeros_like(clean, dtype=float)
    synthetic_mask = np.zeros_like(clean, dtype=float)
    synthetic_meta = None
    if config.mode == "mixed":
        syn_cfg = synthetic_config or SyntheticClutterConfig(seed=config.seed, target_snr_db=None)
        _, synthetic_clutter, synthetic_mask, synthetic_meta = generate_synthetic_clutter(clean, syn_cfg)
        if not raw_dominant_mode:
            synthetic_clutter *= float(config.synthetic_strength)

    total_noise = pattern_clutter + synthetic_clutter
    pre_overlay_noise_rms = _rms(total_noise)
    total_mask = np.maximum(pattern_mask, synthetic_mask)
    total_noise, beta = scale_clutter_to_target_snr_for_overlay(clean, total_noise, total_mask, config)
    scaled_noise_rms = _rms(total_noise)
    pattern_clutter *= beta
    synthetic_clutter *= beta
    noisy, effective_mask = overlay_noise(
        clean,
        total_noise,
        total_mask,
        config.overlay_mode,
        config.overlay_midpoint,
        soft_dominance_temperature=config.soft_dominance_temperature,
    )
    effective_clutter = noisy - clean
    effective_pixels = effective_mask > 0
    meta = {
        "config": config.to_dict(),
        "placements": placements,
        "target_snr_scale": beta,
        "actual_snr_db": compute_snr_db(clean, effective_clutter),
        "overlay_mode": config.overlay_mode,
        "overlay_midpoint": float(config.overlay_midpoint),
        "pattern_clutter": {"rms": _rms(pattern_clutter), "mask_coverage": float(np.mean(pattern_mask > 0))},
        "synthetic_clutter": synthetic_meta,
        "total_clutter": {"rms": _rms(effective_clutter), "mask_coverage": float(np.mean(effective_pixels))},
        "diagnostics": {
            "target_snr_db": config.target_snr_db,
            "pre_overlay_noise_rms": pre_overlay_noise_rms,
            "scaled_noise_rms": scaled_noise_rms,
            "effective_clutter_rms": _rms(effective_clutter),
            "input_mask_coverage": float(np.mean(total_mask > 0)),
            "effective_mask_coverage": float(np.mean(effective_pixels)),
            "effective_pixel_count": int(np.count_nonzero(effective_pixels)),
            "effective_pixel_fraction": float(np.mean(effective_pixels)),
            "mean_abs_effective_clutter": float(np.mean(np.abs(effective_clutter))),
        },
    }
    return noisy, effective_clutter, effective_mask, meta


def overlay_noise(clean_profile, noise_profile, mask=None, mode="dominant_amplitude", midpoint=128.0, soft_dominance_temperature=12.0):
    """Overlay a noise image using the selected ML Clutter overlay mode."""

    if mode in {"raw_dominant_amplitude", "dominant_amplitude"}:
        return overlay_noise_by_dominant_amplitude(clean_profile, noise_profile, mask, midpoint)
    if mode == "soft_dominance":
        return overlay_noise_by_soft_dominance(clean_profile, noise_profile, mask, midpoint, soft_dominance_temperature)
    if mode == "additive":
        return overlay_noise_additive(clean_profile, noise_profile, mask, midpoint)
    raise ValueError(f"Unsupported pattern overlay mode: {mode}")


def overlay_noise_additive(clean_profile, noise_profile, mask=None, midpoint=128.0):
    """Legacy additive overlay kept as an explicit selectable mode."""

    clean = np.asarray(clean_profile, dtype=float)
    noise = np.asarray(noise_profile, dtype=float)
    if clean.shape != noise.shape:
        raise ValueError(f"Noise profile shape {noise.shape} must match clean profile shape {clean.shape}.")
    if mask is None:
        active_mask = np.ones_like(clean, dtype=bool)
    else:
        mask_array = np.asarray(mask, dtype=float)
        if mask_array.shape != clean.shape:
            raise ValueError(f"Noise mask shape {mask_array.shape} must match clean profile shape {clean.shape}.")
        active_mask = mask_array > 0

    midpoint = float(midpoint)
    min_amplitude = 0.0
    max_amplitude = midpoint * 2.0
    additive = np.zeros_like(clean, dtype=float)
    additive[active_mask] = noise[active_mask] - midpoint
    noisy = np.clip(clean + additive, min_amplitude, max_amplitude)
    changed_mask = active_mask & (np.abs(noisy - clean) > 0)
    return noisy, changed_mask.astype(float)


def overlay_noise_by_dominant_amplitude(clean_profile, noise_profile, mask=None, midpoint=128.0):
    """Overlay noise where its centered absolute amplitude dominates clean signal.

    Clean and noise are interpreted as amplitude images with the same scale
    (for the ML Clutter workflow this is typically 0..256). Values are shifted
    around ``midpoint`` and the larger absolute centered amplitude wins.
    """

    clean = np.asarray(clean_profile, dtype=float)
    noise = np.asarray(noise_profile, dtype=float)
    if clean.shape != noise.shape:
        raise ValueError(f"Noise profile shape {noise.shape} must match clean profile shape {clean.shape}.")
    if mask is None:
        active_mask = np.ones_like(clean, dtype=bool)
    else:
        mask_array = np.asarray(mask, dtype=float)
        if mask_array.shape != clean.shape:
            raise ValueError(f"Noise mask shape {mask_array.shape} must match clean profile shape {clean.shape}.")
        active_mask = mask_array > 0

    midpoint = float(midpoint)
    min_amplitude = 0.0
    max_amplitude = midpoint * 2.0
    clipped_noise = np.clip(noise, min_amplitude, max_amplitude)
    clean_centered = clean - midpoint
    noise_centered = clipped_noise - midpoint
    noise_dominates = active_mask & (np.abs(noise_centered) > np.abs(clean_centered))
    noisy = np.clip(clean.copy(), min_amplitude, max_amplitude)
    noisy[noise_dominates] = clipped_noise[noise_dominates]
    return noisy, noise_dominates.astype(float)


def overlay_noise_by_soft_dominance(clean_profile, noise_profile, mask=None, midpoint=128.0, temperature=12.0):
    """Blend noise with clean data using a smooth dominance weight.

    The hard ``dominant_amplitude`` mode switches pixels abruptly. This mode
    uses the same centered-amplitude comparison but turns it into a sigmoid
    blend, so strong noise gradually overtakes the clean signal.
    """

    clean = np.asarray(clean_profile, dtype=float)
    noise = np.asarray(noise_profile, dtype=float)
    if clean.shape != noise.shape:
        raise ValueError(f"Noise profile shape {noise.shape} must match clean profile shape {clean.shape}.")
    if mask is None:
        active_mask = np.ones_like(clean, dtype=bool)
    else:
        mask_array = np.asarray(mask, dtype=float)
        if mask_array.shape != clean.shape:
            raise ValueError(f"Noise mask shape {mask_array.shape} must match clean profile shape {clean.shape}.")
        active_mask = mask_array > 0

    midpoint = float(midpoint)
    min_amplitude = 0.0
    max_amplitude = midpoint * 2.0
    clipped_noise = np.clip(noise, min_amplitude, max_amplitude)
    clean_centered = clean - midpoint
    noise_centered = clipped_noise - midpoint
    temp = max(float(temperature), 1e-6)
    dominance = (np.abs(noise_centered) - np.abs(clean_centered)) / temp
    dominance = np.clip(dominance, -60.0, 60.0)
    weights = 1.0 / (1.0 + np.exp(-dominance))
    weights = np.where(active_mask, weights, 0.0)
    noisy = np.clip(clean * (1.0 - weights) + clipped_noise * weights, min_amplitude, max_amplitude)
    changed_mask = active_mask & (np.abs(noisy - clean) > 1e-9)
    return noisy, changed_mask.astype(float)


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


def place_pattern(clean_shape, pattern, mask, rng, smooth_mask=True, target_z_start=None):
    traces, samples = clean_shape
    h, w = pattern.shape
    h, w = min(h, traces), min(w, samples)
    pattern, mask = pattern[:h, :w], mask[:h, :w]
    x0 = int(rng.integers(0, traces - h + 1))
    if target_z_start is None:
        z0 = int(rng.integers(0, samples - w + 1))
    else:
        z0 = min(max(0, int(target_z_start)), samples - w)
    out = np.zeros(clean_shape, dtype=float)
    out_mask = np.zeros(clean_shape, dtype=float)
    effective_mask = np.clip(mask, 0.0, 1.0) if smooth_mask else (mask > 0).astype(float)
    out[x0:x0 + h, z0:z0 + w] = pattern * effective_mask
    out_mask[x0:x0 + h, z0:z0 + w] = effective_mask
    return out, out_mask, {"x_start": x0, "x_end": x0 + h, "z_start": z0, "z_end": z0 + w, "placed_shape": [h, w]}


def _pattern_target_z_start(pattern, transform_meta):
    del transform_meta  # Depth is anchored to the extracted pattern bbox, not to random augmentations.
    bbox = getattr(pattern, "bbox", None) or [0, 0, 0, 0]
    return int(bbox[2]) if len(bbox) >= 3 else 0


def scale_clutter_to_target_snr_for_overlay(clean, clutter, mask, config):
    """Scale clutter so the final overlaid residual is closest to target SNR.

    Dominant/soft-dominance overlays are nonlinear: the pre-overlay pattern RMS
    is not the same as the residual left in ``noisy - clean``.  In particular,
    changing ``num_patterns`` changes mask coverage and overlap, so a single
    pre-overlay RMS normalization produces visibly different intensities.  This
    helper chooses a global multiplier against the actual overlay residual,
    keeping the requested SNR stable for one or many mixed real patterns.
    """

    target_snr_db = config.target_snr_db
    if config.overlay_mode == "raw_dominant_amplitude" or target_snr_db is None or not np.any(clutter):
        return clutter, 1.0

    clean = np.asarray(clean, dtype=float)
    clutter = np.asarray(clutter, dtype=float)
    signal_power = float(np.mean(clean ** 2))
    target_noise_power = signal_power / (10.0 ** (float(target_snr_db) / 10.0))
    if target_noise_power <= 0:
        return clutter, 1.0

    def objective(beta):
        noisy, _ = overlay_noise(
            clean,
            clutter * beta,
            mask,
            config.overlay_mode,
            config.overlay_midpoint,
            soft_dominance_temperature=config.soft_dominance_temperature,
        )
        residual = noisy - clean
        return abs(float(np.mean(residual ** 2)) - target_noise_power)

    # Hard dominance can be non-monotonic around the overlay midpoint.  Use a
    # bounded log-grid search followed by local golden-section refinement rather
    # than assuming residual power grows monotonically with beta.
    candidates = np.concatenate(([0.0], np.geomspace(1e-4, 1e6, num=241)))
    errors = np.array([objective(float(beta)) for beta in candidates])
    best_index = int(np.argmin(errors))
    best = float(candidates[best_index])

    left = float(candidates[max(0, best_index - 1)])
    right = float(candidates[min(len(candidates) - 1, best_index + 1)])
    if right > left:
        inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
        c = right - inv_phi * (right - left)
        d = left + inv_phi * (right - left)
        fc = objective(c)
        fd = objective(d)
        for _ in range(48):
            if fc < fd:
                right = d
                d = c
                fd = fc
                c = right - inv_phi * (right - left)
                fc = objective(c)
            else:
                left = c
                c = d
                fc = fd
                d = left + inv_phi * (right - left)
                fd = objective(d)
        refined = (left + right) / 2.0
        if objective(refined) < objective(best):
            best = refined

    return clutter * best, float(best)


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
