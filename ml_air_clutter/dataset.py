"""Paired clean/noisy patch dataset utilities for ML air-clutter experiments."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


@dataclass(frozen=True)
class PatchDatasetConfig:
    """Serializable patching and split configuration for paired datasets."""

    patch_width: int = 64
    stride: int = 32
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    min_num_traces: int = 1
    gap_width: Optional[int] = None

    def to_dict(self):
        return asdict(self)


class PairValidationError(ValueError):
    """Raised when a clean/noisy pair cannot be used as supervised data."""


def _stats(data: np.ndarray) -> Dict[str, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        finite = np.array([np.nan])
    return {
        "min": float(np.nanmin(finite)),
        "max": float(np.nanmax(finite)),
        "mean": float(np.nanmean(finite)),
        "std": float(np.nanstd(finite)),
        "p01": float(np.nanpercentile(finite, 1)),
        "p99": float(np.nanpercentile(finite, 99)),
    }


def convert_profile_to_amplitude_0256(data) -> np.ndarray:
    """Return a profile in the unsigned 0..256 amplitude display range.

    ML Clutter stores paired clean/noisy supervised data as amplitude images in
    ``0..256``. Some source profiles and synthetic generator outputs can still
    be centered around zero (approximately ``-128..+128``). Those profiles must
    be shifted by +128 before clipping so previews show the clean signal and the
    signal with overlaid noise, not the centered residual/noise layer.
    """

    arr = np.asarray(data, dtype=float)
    if arr.size and np.nanmin(arr) < 0.0:
        arr = arr + 128.0
    return np.clip(arr, 0.0, 256.0).copy()


def prepare_amplitude_pair_0256(clean, noisy):
    """Normalize a clean/noisy pair to the 0..256 ML Clutter amplitude range."""

    clean_arr = np.asarray(clean, dtype=float)
    noisy_arr = np.asarray(noisy, dtype=float)
    if clean_arr.shape != noisy_arr.shape:
        return clean_arr.copy(), noisy_arr.copy()
    return convert_profile_to_amplitude_0256(clean_arr), convert_profile_to_amplitude_0256(noisy_arr)


def validate_clean_noisy_pair(clean, noisy, min_num_traces: int = 1) -> Dict[str, object]:
    """Validate paired arrays and return errors plus diagnostic warnings."""

    clean = np.asarray(clean, dtype=float)
    noisy = np.asarray(noisy, dtype=float)
    errors: List[str] = []
    warnings: List[str] = []
    if clean.ndim != 2:
        errors.append(f"clean must be 2D, got ndim={clean.ndim}")
    if noisy.ndim != 2:
        errors.append(f"noisy must be 2D, got ndim={noisy.ndim}")
    if clean.ndim == 2 and noisy.ndim == 2 and clean.shape != noisy.shape:
        errors.append(f"clean and noisy shapes must match, got {clean.shape} and {noisy.shape}")
    shape = clean.shape if clean.ndim == 2 else ()
    if clean.ndim == 2:
        if clean.shape[1] != 512:
            errors.append(f"pair must have 512 samples per trace, got clean shape={clean.shape}")
            if clean.shape[0] == 512:
                warnings.append("clean looks transposed as [512, num_traces]")
        if clean.shape[0] < min_num_traces:
            errors.append(f"pair must contain at least {min_num_traces} traces, got {clean.shape[0]}")
    for name, data in (("clean", clean), ("noisy", noisy)):
        if not np.isfinite(data).all():
            errors.append(f"{name} contains NaN or infinite values")
    clean_stats = _stats(clean) if clean.size else {}
    noisy_stats = _stats(noisy) if noisy.size else {}
    difference = noisy - clean if clean.shape == noisy.shape and clean.ndim == 2 else np.array([])
    difference_stats = _stats(difference) if difference.size else {}
    if not errors and difference.size:
        amp = max(clean_stats["p99"] - clean_stats["p01"], 1e-8)
        diff_std = difference_stats["std"]
        if diff_std < amp * 1e-3:
            warnings.append("noisy-clean difference is very small; pair may contain almost no clutter")
        if diff_std > amp * 2.0:
            warnings.append("noisy-clean difference is very large; profiles may be mismatched")
        for key in ("min", "max", "p01", "p99"):
            if abs(clean_stats[key] - noisy_stats[key]) > amp * 2.0:
                warnings.append(f"clean/noisy {key} ranges differ strongly")
                break
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "shape": list(shape),
        "clean_stats": clean_stats,
        "noisy_stats": noisy_stats,
        "difference_stats": difference_stats,
    }


def build_paired_patch_dataset(pairs: Iterable[Dict[str, object]], config: PatchDatasetConfig):
    """Build leak-resistant patch samples and a JSON-serializable summary."""

    pairs = list(pairs)
    if not pairs:
        raise PairValidationError("Add at least one clean/noisy pair before building the dataset.")
    if config.patch_width <= 0 or config.stride <= 0:
        raise PairValidationError("patch_width and stride must be positive.")
    if not np.isclose(config.train_fraction + config.validation_fraction + config.test_fraction, 1.0):
        raise PairValidationError("train/validation/test fractions must sum to 1.0.")

    samples = {"train": [], "validation": [], "test": []}
    pair_summaries = []
    if len(pairs) >= 3:
        ordered = sorted(pairs, key=lambda p: str(p["pair_id"]))
        n = len(ordered)
        train_end = max(1, int(round(n * config.train_fraction)))
        val_end = min(n - 1, train_end + max(1, int(round(n * config.validation_fraction))))
        split_by_pair = {p["pair_id"]: "train" for p in ordered[:train_end]}
        split_by_pair.update({p["pair_id"]: "validation" for p in ordered[train_end:val_end]})
        split_by_pair.update({p["pair_id"]: "test" for p in ordered[val_end:]})
    else:
        split_by_pair = {}

    for pair in pairs:
        clean = np.asarray(pair["clean"], dtype=float)
        noisy = np.asarray(pair["noisy"], dtype=float)
        report = validate_clean_noisy_pair(clean, noisy, config.min_num_traces)
        if not report["valid"]:
            raise PairValidationError(f"Pair {pair['pair_id']} is invalid: {'; '.join(report['errors'])}")
        pair_summaries.append({
            "pair_id": pair["pair_id"],
            "clean_path": pair.get("clean_path", ""),
            "noisy_path": pair.get("noisy_path", ""),
            "shape": list(clean.shape),
            "clean_stats": report["clean_stats"],
            "noisy_stats": report["noisy_stats"],
            "difference_stats": report["difference_stats"],
            "warnings": report["warnings"],
        })
        for split, x_start, x_end in _iter_split_windows(clean.shape[0], config, split_by_pair.get(pair["pair_id"])):
            sample = {
                "noisy": noisy[x_start:x_end].copy(),
                "clean": clean[x_start:x_end].copy(),
                "pair_id": pair["pair_id"],
                "source_clean_profile": pair.get("clean_path", pair.get("clean_name", "")),
                "source_noisy_profile": pair.get("noisy_path", pair.get("noisy_name", "")),
                "x_start": int(x_start),
                "x_end": int(x_end),
                "normalization": pair.get("normalization", {}),
                "sample_meta": {"split_strategy": "by_pair_or_trace_blocks"},
                "residual": noisy[x_start:x_end] - clean[x_start:x_end],
            }
            samples[split].append(sample)
    summary = {
        "dataset_schema": "paired_clean_noisy_v1",
        "pairs": pair_summaries,
        "patch_width": config.patch_width,
        "stride": config.stride,
        "split_strategy": "by_pair_or_trace_blocks",
        "num_train_patches": len(samples["train"]),
        "num_validation_patches": len(samples["validation"]),
        "num_test_patches": len(samples["test"]),
        "normalization": {},
        "config": config.to_dict(),
    }
    return samples, summary


def _iter_split_windows(num_traces: int, config: PatchDatasetConfig, fixed_split: Optional[str]):
    if num_traces < config.patch_width:
        return
    starts = list(range(0, num_traces - config.patch_width + 1, config.stride))
    if starts[-1] != num_traces - config.patch_width:
        starts.append(num_traces - config.patch_width)
    if fixed_split:
        for start in starts:
            yield fixed_split, start, start + config.patch_width
        return
    gap = config.gap_width if config.gap_width is not None else config.patch_width
    train_limit = int(num_traces * config.train_fraction)
    val_limit = int(num_traces * (config.train_fraction + config.validation_fraction))
    for start in starts:
        end = start + config.patch_width
        center = (start + end) / 2.0
        if end <= max(0, train_limit - gap):
            split = "train"
        elif start >= train_limit + gap and end <= max(train_limit + gap, val_limit - gap):
            split = "validation"
        elif start >= val_limit + gap:
            split = "test"
        else:
            continue
        yield split, start, end


def save_dataset(directory, samples, summary) -> Path:
    """Persist samples as compressed npz files plus dataset_summary.json."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "dataset_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    for split, split_samples in samples.items():
        split_dir = directory / split
        split_dir.mkdir(exist_ok=True)
        for index, sample in enumerate(split_samples):
            np.savez_compressed(
                split_dir / f"{sample['pair_id']}_{index:06d}.npz",
                noisy=sample["noisy"], clean=sample["clean"], residual=sample["residual"],
                meta=json.dumps({k: v for k, v in sample.items() if k not in {"noisy", "clean", "residual"}}, ensure_ascii=False),
            )
    return directory / "dataset_summary.json"
