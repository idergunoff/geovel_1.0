"""Persistence helpers for reproducible ML air-clutter experiment runs."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def make_experiment_run_dir(root="experiments/ml_clutter", experiment_name="experiment") -> Path:
    """Create a timestamped run directory for a reproducible experiment."""

    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(experiment_name)).strip("_")
    if not safe_name:
        safe_name = "experiment"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    root = Path(root)
    candidate = root / f"{stamp}_{safe_name}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stamp}_{safe_name}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def build_split_manifest(samples: Dict[str, List[Dict[str, object]]]) -> Dict[str, object]:
    """Return a JSON-serializable manifest of train/validation/test patch membership."""

    manifest = {"split_schema": "ml_air_clutter_split_v1", "splits": {}}
    for split_name in ("train", "validation", "test"):
        rows = []
        for index, sample in enumerate(samples.get(split_name, [])):
            rows.append({
                "sample_index": index,
                "pair_id": sample.get("pair_id", ""),
                "source_clean_profile": sample.get("source_clean_profile", ""),
                "source_noisy_profile": sample.get("source_noisy_profile", ""),
                "x_start": int(sample.get("x_start", 0)),
                "x_end": int(sample.get("x_end", 0)),
                "normalization": sample.get("normalization", {}),
                "sample_meta": sample.get("sample_meta", {}),
            })
        manifest["splits"][split_name] = rows
    manifest["counts"] = {name: len(rows) for name, rows in manifest["splits"].items()}
    return manifest


def save_preview_arrays(preview_dir: Path, samples: Dict[str, List[Dict[str, object]]], training_summary: Optional[Dict[str, object]] = None) -> Dict[str, str]:
    """Save deterministic preview arrays for later inspection."""

    preview_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    sample = None
    split_name = ""
    for name in ("validation", "test", "train"):
        if samples.get(name):
            split_name = name
            sample = samples[name][0]
            break
    if sample is not None:
        for key in ("clean", "noisy", "residual"):
            if key in sample:
                path = preview_dir / f"{split_name}_{key}.npy"
                np.save(path, np.asarray(sample[key]))
                saved[f"{split_name}_{key}"] = str(path)
    metrics = (training_summary or {}).get("metrics", {}) if isinstance(training_summary, dict) else {}
    for split, payload in metrics.get("splits", {}).items():
        path = preview_dir / f"{split}_metrics_preview.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(_json_safe(payload.get("samples_preview", [])), fh, indent=2, ensure_ascii=False)
        saved[f"{split}_metrics_preview"] = str(path)
    return saved


def write_markdown_report(path: Path, config: Dict[str, object], dataset_summary: Optional[Dict[str, object]], split_manifest: Dict[str, object], training_summary: Optional[Dict[str, object]], metrics: Optional[Dict[str, object]], preview_files: Dict[str, str]) -> Path:
    """Write a compact reproducibility report for the experiment run."""

    lines = [
        "# ML Clutter experiment report",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Reproducibility inputs",
        f"- Seed: `{config.get('seed', config.get('generator_seed', 'unknown'))}`",
        f"- Dataset schema: `{(dataset_summary or {}).get('dataset_schema', 'not built')}`",
        f"- Split counts: `{split_manifest.get('counts', {})}`",
        "",
        "## Required artifacts",
    ]
    for filename in ("config.json", "dataset_summary.json", "split.json", "train_log.csv", "metrics.json", "model_best.pt"):
        lines.append(f"- `{filename}`")
    lines.extend(["", "## Training summary"])
    if training_summary:
        lines.extend([
            f"- Best epoch: `{training_summary.get('best_epoch')}`",
            f"- Best validation loss: `{training_summary.get('best_validation_loss')}`",
            f"- Epochs completed: `{training_summary.get('epochs_completed')}`",
            f"- Model best: `{training_summary.get('model_best')}`",
            f"- Model last: `{training_summary.get('model_last')}`",
        ])
    else:
        lines.append("Training has not been run yet.")
    lines.extend(["", "## Metrics"])
    for split, payload in (metrics or {}).get("splits", {}).items():
        summary = payload.get("summary", {})
        lines.append(
            f"- **{split}**: samples={summary.get('num_samples', 0)}, "
            f"MAE {summary.get('mae_before')} -> {summary.get('mae_after')}, "
            f"SNR gain={summary.get('snr_gain_db')} dB"
        )
    lines.extend(["", "## Preview files"])
    if preview_files:
        lines.extend(f"- `{name}`: `{file_path}`" for name, file_path in sorted(preview_files.items()))
    else:
        lines.append("No preview arrays were available.")
    lines.extend(["", "## Full configuration", "", "```json", json.dumps(_json_safe(config), indent=2, ensure_ascii=False), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _copy_if_exists(source, destination: Path) -> Optional[Path]:
    if not source:
        return None
    source = Path(source)
    if not source.exists() or source.resolve() == destination.resolve():
        return source if source.exists() else None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def save_experiment_artifacts(directory, config: Dict[str, object], dataset_summary: Optional[Dict[str, object]], samples: Optional[Dict[str, List[Dict[str, object]]]], training_summary: Optional[Dict[str, object]] = None, metrics: Optional[Dict[str, object]] = None) -> Dict[str, str]:
    """Persist the minimum reproducibility bundle for an ML Clutter run."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    config = dict(config or {})
    config.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
    split_manifest = build_split_manifest(samples or {})
    paths = {}
    for filename, payload in (
        ("config.json", config),
        ("dataset_summary.json", dataset_summary or {}),
        ("split.json", split_manifest),
    ):
        path = root / filename
        with path.open("w", encoding="utf-8") as fh:
            json.dump(_json_safe(payload), fh, indent=2, ensure_ascii=False)
        paths[filename] = str(path)
    training_summary = training_summary or {}
    metrics = metrics or training_summary.get("metrics") or {}
    if metrics:
        metrics_path = root / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(_json_safe(metrics), fh, indent=2, ensure_ascii=False)
        paths["metrics.json"] = str(metrics_path)
    for source_key, filename in (("train_log", "train_log.csv"), ("model_best", "model_best.pt"), ("model_last", "model_last.pt")):
        copied = _copy_if_exists(training_summary.get(source_key), root / filename)
        if copied is not None:
            paths[filename] = str(copied)
    preview_files = save_preview_arrays(root / "previews", samples or {}, training_summary)
    paths.update({f"previews/{name}": path for name, path in preview_files.items()})
    report_path = write_markdown_report(root / "report.md", config, dataset_summary, split_manifest, training_summary, metrics, preview_files)
    paths["report.md"] = str(report_path)
    return paths
