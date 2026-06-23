"""Baseline PyTorch models for paired ML air-clutter cleaning.

The MVP models operate in supervised direct-clean mode: input patches contain a
noisy radarogram and the network predicts the clean radarogram. Residual/clutter
prediction is intentionally left out of the mandatory baseline workflow.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without PyTorch.
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


SUPPORTED_MODEL_TYPES = ("baseline_cnn", "small_unet")
SUPPORTED_INPUT_CHANNELS = ("raw", "envelope", "grad_x", "grad_z")
SUPPORTED_OUTPUT_MODES = ("direct_clean",)


@dataclass(frozen=True)
class ModelConfig:
    """Serializable configuration for the baseline cleaning model."""

    model_type: str = "baseline_cnn"
    input_channels: Tuple[str, ...] = ("raw",)
    output_mode: str = "direct_clean"
    base_channels: int = 16
    num_layers: int = 5
    kernel_size: int = 3

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["input_channels"] = list(self.input_channels)
        return data


def _require_torch():
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for ML Clutter baseline models.") from _TORCH_IMPORT_ERROR


def validate_model_config(config: ModelConfig) -> None:
    """Validate the constrained MVP model configuration."""

    if config.model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model_type={config.model_type!r}; expected one of {SUPPORTED_MODEL_TYPES}.")
    if config.output_mode not in SUPPORTED_OUTPUT_MODES:
        raise ValueError("Only direct_clean output mode is supported in the MVP baseline.")
    if not config.input_channels:
        raise ValueError("At least one input channel is required.")
    unknown = [channel for channel in config.input_channels if channel not in SUPPORTED_INPUT_CHANNELS]
    if unknown:
        raise ValueError(f"Unsupported input channels: {unknown}; expected subset of {SUPPORTED_INPUT_CHANNELS}.")
    if config.base_channels <= 0:
        raise ValueError("base_channels must be positive.")
    if config.num_layers < 3:
        raise ValueError("num_layers must be at least 3 for baseline_cnn.")
    if config.kernel_size <= 0 or config.kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number.")


if nn is not None:
    class BaselineCNN(nn.Module):
        """Small fully-convolutional direct-clean baseline."""

        def __init__(self, in_channels: int, base_channels: int = 16, num_layers: int = 5, kernel_size: int = 3):
            super().__init__()
            padding = kernel_size // 2
            layers = [nn.Conv2d(in_channels, base_channels, kernel_size, padding=padding), nn.ReLU(inplace=True)]
            for _ in range(num_layers - 2):
                layers.extend([
                    nn.Conv2d(base_channels, base_channels, kernel_size, padding=padding),
                    nn.BatchNorm2d(base_channels),
                    nn.ReLU(inplace=True),
                ])
            layers.append(nn.Conv2d(base_channels, 1, kernel_size, padding=padding))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)


    class _DoubleConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)


    class SmallUNet(nn.Module):
        """Compact U-Net that preserves patch shape [B, C, width, 512]."""

        def __init__(self, in_channels: int, base_channels: int = 16):
            super().__init__()
            self.enc1 = _DoubleConv(in_channels, base_channels)
            self.pool1 = nn.MaxPool2d(2)
            self.enc2 = _DoubleConv(base_channels, base_channels * 2)
            self.pool2 = nn.MaxPool2d(2)
            self.bottleneck = _DoubleConv(base_channels * 2, base_channels * 4)
            self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
            self.dec2 = _DoubleConv(base_channels * 4, base_channels * 2)
            self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
            self.dec1 = _DoubleConv(base_channels * 2, base_channels)
            self.out = nn.Conv2d(base_channels, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            b = self.bottleneck(self.pool2(e2))
            d2 = self.up2(b)
            d2 = torch.cat([d2, e2[..., :d2.shape[-2], :d2.shape[-1]]], dim=1)
            d2 = self.dec2(d2)
            d1 = self.up1(d2)
            d1 = torch.cat([d1, e1[..., :d1.shape[-2], :d1.shape[-1]]], dim=1)
            d1 = self.dec1(d1)
            return self.out(d1)
else:
    BaselineCNN = None
    SmallUNet = None


def create_model(config: ModelConfig):
    """Create an untrained baseline model for noisy -> clean prediction."""

    _require_torch()
    validate_model_config(config)
    in_channels = len(config.input_channels)
    if config.model_type == "baseline_cnn":
        return BaselineCNN(in_channels, config.base_channels, config.num_layers, config.kernel_size)
    if config.model_type == "small_unet":
        return SmallUNet(in_channels, config.base_channels)
    raise ValueError(f"Unsupported model_type={config.model_type!r}")


def count_parameters(model) -> int:
    """Return the number of trainable parameters."""

    _require_torch()
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def checkpoint_payload(model, config: ModelConfig) -> Dict[str, object]:
    _require_torch()
    return {
        "schema": "ml_air_clutter_model_checkpoint_v1",
        "config": config.to_dict(),
        "output_mode": "direct_clean",
        "state_dict": model.state_dict(),
        "num_parameters": count_parameters(model),
        "notes": "MVP checkpoint stores an untrained or trained direct clean predictor; residual/clutter targets are diagnostic only.",
    }


def save_model_checkpoint(path, model, config: ModelConfig) -> Path:
    """Save a model checkpoint plus JSON sidecar metadata."""

    _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(model, config)
    torch.save(payload, path)
    meta_path = path.with_suffix(path.suffix + ".json")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in payload.items() if k != "state_dict"}, fh, indent=2, ensure_ascii=False)
    return path


def load_model_checkpoint(path, map_location="cpu"):
    """Load a checkpoint and recreate its configured baseline model."""

    _require_torch()
    payload = torch.load(path, map_location=map_location)
    config_dict = payload["config"].copy()
    config_dict["input_channels"] = tuple(config_dict.get("input_channels", ("raw",)))
    config = ModelConfig(**config_dict)
    model = create_model(config)
    model.load_state_dict(payload["state_dict"])
    return model, config, payload
