"""Analytical synthetic air-clutter generator for ML Clutter experiments."""

from __future__ import annotations

import math

import numpy as np

from .config import SyntheticClutterConfig


def generate_synthetic_clutter(clean_profile, config=None):
    """Return ``noisy, clutter, mask, meta`` for a clean ``[traces, samples]`` profile."""

    clean = np.asarray(clean_profile, dtype=float)
    _validate_clean(clean)
    config = config or SyntheticClutterConfig()
    rng = np.random.default_rng(config.seed)
    clutter = np.zeros_like(clean, dtype=float)
    mask = np.zeros_like(clean, dtype=float)
    meta = {"config": config.to_dict(), "objects": []}

    if config.hyperbolas:
        _add_hyperbolas(clutter, mask, meta, rng, config)
    if config.sloped_events:
        _add_sloped_events(clutter, mask, meta, rng, config)
    if config.ringing:
        _add_ringing(clutter, mask, meta, rng, config)
    if config.vertical_spikes:
        _add_vertical_spikes(clutter, mask, meta, rng, config)
    if config.noise_zones:
        _add_noise_zones(clutter, mask, meta, rng, config)

    clutter, scale = _scale_to_target_snr(clean, clutter, config.target_snr_db)
    noisy = clean + clutter
    mask = (mask > 0).astype(float)
    meta["target_snr_scale"] = scale
    meta["actual_snr_db"] = compute_snr_db(clean, clutter)
    return noisy, clutter, mask, meta


def compute_snr_db(signal, noise, eps=1e-12):
    signal_power = float(np.mean(np.asarray(signal, dtype=float) ** 2))
    noise_power = float(np.mean(np.asarray(noise, dtype=float) ** 2))
    return 10.0 * math.log10((signal_power + eps) / (noise_power + eps))


def ricker_wavelet(offset, frequency):
    x = np.asarray(offset, dtype=float) * float(frequency)
    return (1.0 - 2.0 * np.pi**2 * x**2) * np.exp(-(np.pi**2) * x**2)


def _validate_clean(clean):
    if clean.ndim != 2:
        raise ValueError(f"Clean profile must be a 2D array, got ndim={clean.ndim}.")
    if not np.isfinite(clean).all():
        raise ValueError("Clean profile contains non-finite amplitudes.")


def _add_hyperbolas(clutter, mask, meta, rng, config):
    traces, samples = clutter.shape
    xs = np.arange(traces)
    zs = np.arange(samples)
    for _ in range(max(0, config.num_hyperbolas)):
        x0 = float(rng.uniform(0, max(1, traces - 1)))
        z0 = float(rng.uniform(8, max(9, samples * 0.35)))
        curvature = float(rng.uniform(0.002, 0.03))
        width_x = float(rng.uniform(max(4, traces * 0.05), max(5, traces * 0.35)))
        amplitude = _signed_amplitude(rng, config.base_amplitude, 0.5, 1.4)
        frequency = float(rng.uniform(0.035, 0.09))
        phase = float(rng.uniform(-np.pi, np.pi))
        decay = float(rng.uniform(0.7, 2.2))
        z_curve = z0 + curvature * (xs - x0) ** 2
        horizontal_weight = np.exp(-0.5 * ((xs - x0) / max(width_x, 1.0)) ** 2)
        for ix, zc in enumerate(z_curve):
            pulse = ricker_wavelet(zs - zc + phase / (2 * np.pi * frequency), frequency)
            contribution = amplitude * horizontal_weight[ix] * pulse * np.exp(-abs(ix - x0) / max(width_x * decay, 1.0))
            clutter[ix] += contribution
            mask[ix] = np.maximum(mask[ix], (np.abs(zs - zc) <= max(4, 1.5 / frequency)) * horizontal_weight[ix])
        meta["objects"].append({"type": "hyperbola", "x0": x0, "z0": z0, "curvature": curvature, "width_x": width_x, "amplitude": amplitude, "frequency": frequency, "phase": phase, "decay": decay})


def _add_sloped_events(clutter, mask, meta, rng, config):
    traces, samples = clutter.shape
    xs = np.arange(traces)
    zs = np.arange(samples)
    for _ in range(max(0, config.num_sloped_events)):
        x_start = int(rng.integers(0, max(1, traces)))
        x_end = int(rng.integers(x_start + 1, traces + 1)) if x_start < traces - 1 else traces
        z_start = float(rng.uniform(5, samples * 0.5))
        slope = float(rng.uniform(-0.4, 0.4))
        amplitude = _signed_amplitude(rng, config.base_amplitude, 0.25, 1.1)
        thickness = float(rng.uniform(2.0, 8.0))
        frequency = float(rng.uniform(0.025, 0.08))
        for ix in range(x_start, x_end):
            zc = z_start + slope * (ix - x_start)
            if -thickness <= zc < samples + thickness:
                pulse = ricker_wavelet(zs - zc, frequency)
                clutter[ix] += amplitude * pulse
                mask[ix] = np.maximum(mask[ix], (np.abs(zs - zc) <= thickness).astype(float))
        meta["objects"].append({"type": "sloped_event", "x_start": x_start, "x_end": x_end, "z_start": z_start, "slope": slope, "amplitude": amplitude, "thickness": thickness, "frequency": frequency})


def _add_ringing(clutter, mask, meta, rng, config):
    traces, samples = clutter.shape
    zs = np.arange(samples)
    for _ in range(max(0, config.num_ringing_events)):
        x0 = int(rng.integers(0, max(1, traces)))
        x1 = int(rng.integers(x0 + 1, traces + 1)) if x0 < traces - 1 else traces
        z0 = float(rng.uniform(0, samples * 0.25))
        frequency = float(rng.uniform(0.045, 0.14))
        decay = float(rng.uniform(25.0, 110.0))
        phase = float(rng.uniform(-np.pi, np.pi))
        amplitude = _signed_amplitude(rng, config.base_amplitude, 0.15, 0.9)
        wave = amplitude * np.sin(2 * np.pi * frequency * np.maximum(zs - z0, 0) + phase) * np.exp(-np.maximum(zs - z0, 0) / decay)
        wave[zs < z0] = 0.0
        clutter[x0:x1] += wave
        mask[x0:x1] = np.maximum(mask[x0:x1], (zs >= z0).astype(float) * np.exp(-np.maximum(zs - z0, 0) / decay))
        meta["objects"].append({"type": "ringing", "x_start": x0, "x_end": x1, "z_start": z0, "frequency": frequency, "decay": decay, "phase": phase, "amplitude": amplitude})


def _add_vertical_spikes(clutter, mask, meta, rng, config):
    traces, samples = clutter.shape
    zs = np.arange(samples)
    for _ in range(max(0, config.num_vertical_spikes)):
        x0 = int(rng.integers(0, max(1, traces)))
        width = int(rng.integers(1, max(2, min(8, traces + 1))))
        amplitude = _signed_amplitude(rng, config.base_amplitude, 0.2, 1.0)
        profile = amplitude * (0.7 * rng.normal(size=samples) + 0.3 * np.sin(2 * np.pi * rng.uniform(0.01, 0.06) * zs))
        profile *= np.exp(-zs / float(rng.uniform(samples * 0.8, samples * 2.0)))
        lo, hi = max(0, x0 - width), min(traces, x0 + width + 1)
        weights = np.hanning(max(3, hi - lo + 2))[1:-1]
        for j, ix in enumerate(range(lo, hi)):
            clutter[ix] += weights[j] * profile
            mask[ix] = np.maximum(mask[ix], weights[j])
        meta["objects"].append({"type": "vertical_spike", "x0": x0, "width": width, "amplitude": amplitude})


def _add_noise_zones(clutter, mask, meta, rng, config):
    traces, samples = clutter.shape
    for _ in range(max(0, config.num_noise_zones)):
        x0 = int(rng.integers(0, max(1, traces)))
        width = int(rng.integers(max(2, traces // 20), max(3, max(4, traces // 3))))
        z0 = int(rng.integers(0, max(1, samples // 2)))
        height = int(rng.integers(max(8, samples // 20), max(9, samples // 3)))
        amp = float(rng.uniform(0.05, 0.4) * config.base_amplitude)
        x1, z1 = min(traces, x0 + width), min(samples, z0 + height)
        patch = amp * rng.normal(size=(x1 - x0, z1 - z0))
        clutter[x0:x1, z0:z1] += patch
        mask[x0:x1, z0:z1] = 1.0
        meta["objects"].append({"type": "noise_zone", "x_start": x0, "x_end": x1, "z_start": z0, "z_end": z1, "amplitude": amp})


def _signed_amplitude(rng, base, low, high):
    return float(base * rng.uniform(low, high) * rng.choice([-1.0, 1.0]))


def _scale_to_target_snr(clean, clutter, target_snr_db):
    if target_snr_db is None or not np.any(clutter):
        return clutter, 1.0
    signal_power = float(np.mean(clean ** 2))
    clutter_power = float(np.mean(clutter ** 2))
    if clutter_power <= 0:
        return clutter, 1.0
    target_noise_power = signal_power / (10.0 ** (float(target_snr_db) / 10.0))
    scale = math.sqrt(target_noise_power / clutter_power)
    return clutter * scale, float(scale)
