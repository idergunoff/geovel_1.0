from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .preprocessing import Normalizer
from .config import NormalizationConfig

DEFAULT_PATTERN_TAGS = ("pole", "powerline", "metal_fence", "building", "ringing", "vertical_spike", "unknown")


@dataclass(frozen=True)
class PatternExtractionConfig:
    """Configuration for extracting real air-clutter patches from noisy radarograms."""

    patch_width: int = 64
    patch_height: int = 128
    stride_x: int = 32
    stride_z: int = 64
    energy_percentile: float = 95.0
    min_mask_coverage: float = 0.05
    max_patterns: Optional[int] = 32
    normalization_mode: str = "none"

    def to_dict(self):
        return asdict(self)


def validate_bbox(bbox: Sequence[int], shape: Tuple[int, int]):
    if len(bbox) != 4:
        raise ValueError("Pattern bbox must contain [x_start, x_end, z_start, z_end].")
    x_start, x_end, z_start, z_end = [int(value) for value in bbox]
    num_traces, num_samples = shape
    if x_start < 0 or z_start < 0 or x_end > num_traces or z_end > num_samples:
        raise ValueError(f"Pattern bbox {bbox} is outside profile shape {shape}.")
    if x_end <= x_start or z_end <= z_start:
        raise ValueError(f"Pattern bbox {bbox} must have positive width and height.")
    return x_start, x_end, z_start, z_end


def extract_pattern_from_bbox(profile, bbox, mask=None, config=None):
    """Extract one real-noise pattern from a manually selected bbox.

    By default the raw amplitude scale is preserved so extracted patterns stay
    comparable with clean radarograms in the 0..256 ML Clutter workflow.
    """

    array = np.asarray(profile, dtype=float)
    _validate_profile_array(array)
    x_start, x_end, z_start, z_end = validate_bbox(bbox, array.shape)
    patch = array[x_start:x_end, z_start:z_end].copy()
    if mask is None:
        patch_mask = np.ones_like(patch, dtype=float)
    else:
        mask_array = np.asarray(mask, dtype=float)
        if mask_array.shape != array.shape:
            raise ValueError(f"Mask shape {mask_array.shape} must match profile shape {array.shape}.")
        patch_mask = (mask_array[x_start:x_end, z_start:z_end] > 0).astype(float)
    return normalize_pattern_patch(patch, patch_mask, config or PatternExtractionConfig())


def extract_energy_patterns(profile, config=None):
    """Semi-automatic extraction of high-energy patch patterns using moving-window RMS."""

    config = config or PatternExtractionConfig()
    array = np.asarray(profile, dtype=float)
    _validate_profile_array(array)
    _validate_config(config, array.shape)
    energy = np.abs(array)
    threshold = float(np.percentile(energy, config.energy_percentile))
    candidate_mask = (energy >= threshold).astype(float)
    candidates = []
    for x_start in range(0, array.shape[0] - config.patch_width + 1, config.stride_x):
        for z_start in range(0, array.shape[1] - config.patch_height + 1, config.stride_z):
            x_end = x_start + config.patch_width
            z_end = z_start + config.patch_height
            mask_patch = candidate_mask[x_start:x_end, z_start:z_end]
            coverage = float(np.mean(mask_patch > 0))
            if coverage < config.min_mask_coverage:
                continue
            patch = array[x_start:x_end, z_start:z_end].copy()
            normalized = normalize_pattern_patch(patch, mask_patch, config)
            normalized["bbox"] = [x_start, x_end, z_start, z_end]
            normalized["energy_score"] = float(np.sqrt(np.mean(patch ** 2)))
            normalized["mask_coverage"] = coverage
            candidates.append(normalized)
    candidates.sort(key=lambda item: item["energy_score"], reverse=True)
    if config.max_patterns is not None:
        candidates = candidates[: int(config.max_patterns)]
    return candidates


def normalize_pattern_patch(patch, mask, config):
    patch = np.asarray(patch, dtype=float)
    mask = np.asarray(mask, dtype=float)
    if patch.ndim != 2:
        raise ValueError("Pattern patch must be a 2D array.")
    if mask.shape != patch.shape:
        raise ValueError(f"Pattern mask shape {mask.shape} must match patch shape {patch.shape}.")
    result = Normalizer.fit_transform(patch, NormalizationConfig(mode=config.normalization_mode))
    stats = pattern_statistics(result.data, mask)
    return {
        "array": result.data,
        "mask": (mask > 0).astype(float),
        "normalization": {"config": result.config.to_dict(), "params": result.params},
        "stats": stats,
    }


def pattern_statistics(array, mask=None):
    arr = np.asarray(array, dtype=float)
    stats = {
        "shape": list(arr.shape),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "rms": float(np.sqrt(np.mean(arr ** 2))),
    }
    if mask is not None:
        stats["mask_coverage"] = float(np.mean(np.asarray(mask) > 0))
    return stats


def _validate_profile_array(array):
    if array.ndim != 2:
        raise ValueError(f"Noisy radarogram must be a 2D array, got ndim={array.ndim}.")
    if not np.isfinite(array).all():
        raise ValueError("Noisy radarogram contains non-finite amplitudes.")


def _validate_config(config, shape):
    if config.patch_width <= 0 or config.patch_height <= 0:
        raise ValueError("Patch width and height must be positive.")
    if config.stride_x <= 0 or config.stride_z <= 0:
        raise ValueError("Extraction strides must be positive.")
    if config.patch_width > shape[0] or config.patch_height > shape[1]:
        raise ValueError(f"Patch size {(config.patch_width, config.patch_height)} exceeds profile shape {shape}.")
    if not 0.0 <= config.energy_percentile <= 100.0:
        raise ValueError("Energy percentile must be in [0, 100].")
