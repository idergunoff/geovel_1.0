"""Training utilities for supervised ML air-clutter cleaning.

The MVP training path learns direct restoration from paired patches:
``noisy -> clean``.  Clutter residuals and masks remain diagnostic artifacts and
are not required by the loss.
"""

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .model import ModelConfig, checkpoint_payload, count_parameters

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - exercised only without PyTorch.
    torch = None
    nn = None
    DataLoader = None
    Dataset = object
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


@dataclass(frozen=True)
class TrainingConfig:
    """Serializable configuration for the paired clean/noisy train loop."""

    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_loss_weight: float = 0.1
    early_stopping_patience: int = 5
    min_delta: float = 1e-5
    device: str = "auto"
    seed: int = 42
    output_dir: str = "experiments/ml_clutter/training_run"

    def to_dict(self):
        return asdict(self)


class TrainingCancelled(RuntimeError):
    """Raised by callbacks to stop training gracefully."""


def _require_torch():
    if torch is None or nn is None or DataLoader is None:
        raise ImportError("PyTorch is required for ML Clutter training.") from _TORCH_IMPORT_ERROR


def make_input_channels(noisy: np.ndarray, channels: Iterable[str]) -> np.ndarray:
    """Build a [C, width, 512] input tensor from a noisy patch."""

    noisy = np.asarray(noisy, dtype=np.float32)
    built: List[np.ndarray] = []
    for channel in channels:
        if channel == "raw":
            built.append(noisy)
        elif channel == "envelope":
            built.append(np.abs(noisy))
        elif channel == "grad_x":
            built.append(np.gradient(noisy, axis=0).astype(np.float32))
        elif channel == "grad_z":
            built.append(np.gradient(noisy, axis=1).astype(np.float32))
        else:
            raise ValueError(f"Unsupported input channel: {channel}")
    if not built:
        raise ValueError("At least one input channel is required for training.")
    return np.stack(built, axis=0).astype(np.float32)


class PairedPatchTorchDataset(Dataset):
    """Torch dataset wrapper for in-memory paired patch samples."""

    def __init__(self, samples: List[Dict[str, object]], input_channels: Iterable[str]):
        _require_torch()
        self.samples = list(samples)
        self.input_channels = tuple(input_channels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        x = make_input_channels(sample["noisy"], self.input_channels) / 256.0
        y = np.asarray(sample["clean"], dtype=np.float32)[None, ...] / 256.0
        return torch.from_numpy(x), torch.from_numpy(y)


def gradient_loss(prediction, target):
    """L1 loss between x/z finite differences of prediction and target."""

    loss = torch.mean(torch.abs(prediction[:, :, 1:, :] - prediction[:, :, :-1, :] - (target[:, :, 1:, :] - target[:, :, :-1, :])))
    loss = loss + torch.mean(torch.abs(prediction[:, :, :, 1:] - prediction[:, :, :, :-1] - (target[:, :, :, 1:] - target[:, :, :, :-1])))
    return loss


def _resolve_device(requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _run_epoch(model, loader, optimizer, config: TrainingConfig, device, train: bool):
    criterion = nn.L1Loss()
    model.train(train)
    total_loss = total_clean = total_grad = 0.0
    total_items = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            predictions = model(inputs)
            clean_loss = criterion(predictions, targets)
            grad = gradient_loss(predictions, targets)
            loss = clean_loss + float(config.grad_loss_weight) * grad
            if train:
                loss.backward()
                optimizer.step()
        batch_size = inputs.shape[0]
        total_items += batch_size
        total_loss += float(loss.detach().cpu()) * batch_size
        total_clean += float(clean_loss.detach().cpu()) * batch_size
        total_grad += float(grad.detach().cpu()) * batch_size
    denom = max(total_items, 1)
    return {"loss": total_loss / denom, "clean_l1": total_clean / denom, "grad_l1": total_grad / denom}


def _validation_preview(model, sample, channels, device):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(make_input_channels(sample["noisy"], channels)[None, ...] / 256.0).to(device)
        pred = model(x).detach().cpu().numpy()[0, 0] * 256.0
    clean = np.asarray(sample["clean"], dtype=float)
    noisy = np.asarray(sample["noisy"], dtype=float)
    return {"clean": clean, "noisy": noisy, "clean_pred": pred, "error": pred - clean}


def train_model(
    model,
    model_config: ModelConfig,
    samples: Dict[str, List[Dict[str, object]]],
    config: TrainingConfig,
    progress_callback: Optional[Callable[[str, object], None]] = None,
) -> Dict[str, object]:
    """Train a direct-clean model and persist best/last checkpoints."""

    _require_torch()
    train_samples = list(samples.get("train", []))
    val_samples = list(samples.get("validation", []))
    if not train_samples:
        raise ValueError("Training split is empty; build a dataset with train patches first.")
    if not val_samples:
        val_samples = train_samples
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    device = _resolve_device(config.device)
    model.to(device)
    train_loader = DataLoader(PairedPatchTorchDataset(train_samples, model_config.input_channels), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(PairedPatchTorchDataset(val_samples, model_config.input_channels), batch_size=config.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.csv"
    best_path = output_dir / "model_best.pt"
    last_path = output_dir / "model_last.pt"
    best_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    start_time = time.time()

    with log_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "train_clean_l1", "train_grad_l1", "val_loss", "val_clean_l1", "val_grad_l1"])
        writer.writeheader()
        for epoch in range(1, int(config.epochs) + 1):
            train_metrics = _run_epoch(model, train_loader, optimizer, config, device, train=True)
            val_metrics = _run_epoch(model, val_loader, optimizer, config, device, train=False)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_clean_l1": train_metrics["clean_l1"],
                "train_grad_l1": train_metrics["grad_l1"],
                "val_loss": val_metrics["loss"],
                "val_clean_l1": val_metrics["clean_l1"],
                "val_grad_l1": val_metrics["grad_l1"],
            }
            writer.writerow(row)
            fh.flush()
            history.append(row)
            improved = val_metrics["loss"] < best_loss - float(config.min_delta)
            if improved:
                best_loss = val_metrics["loss"]
                best_epoch = epoch
                bad_epochs = 0
                _save_training_checkpoint(best_path, model, model_config, config, row, best_epoch, best_loss)
            else:
                bad_epochs += 1
            _save_training_checkpoint(last_path, model, model_config, config, row, best_epoch, best_loss)
            preview = _validation_preview(model, val_samples[0], model_config.input_channels, device)
            if progress_callback:
                progress_callback("epoch_finished", row)
                progress_callback("preview_ready", preview)
            if bad_epochs >= int(config.early_stopping_patience):
                if progress_callback:
                    progress_callback("log_message", f"Early stopping at epoch {epoch}; best epoch={best_epoch}.")
                break
    summary = {
        "training_schema": "ml_air_clutter_training_v1",
        "config": config.to_dict(),
        "model_config": model_config.to_dict(),
        "num_parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_completed": len(history),
        "duration_seconds": time.time() - start_time,
        "train_log": str(log_path),
        "model_best": str(best_path),
        "model_last": str(last_path),
        "history": history,
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    if progress_callback:
        progress_callback("training_finished", summary)
    return summary


def _save_training_checkpoint(path, model, model_config, training_config, metrics, best_epoch, best_loss):
    payload = checkpoint_payload(model, model_config)
    payload.update({
        "training_config": training_config.to_dict(),
        "metrics": metrics,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
    })
    torch.save(payload, path)
