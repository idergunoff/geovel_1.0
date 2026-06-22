from dataclasses import dataclass

import numpy as np

from .config import NormalizationConfig

STANDARD_NORMALIZATION = "standard"
ROBUST_NORMALIZATION = "robust"
PERCENTILE_STANDARD_NORMALIZATION = "percentile_standard"
NO_NORMALIZATION = "none"
NORMALIZATION_MODES = {
    NO_NORMALIZATION,
    STANDARD_NORMALIZATION,
    ROBUST_NORMALIZATION,
    PERCENTILE_STANDARD_NORMALIZATION,
}


@dataclass(frozen=True)
class NormalizationResult:
    data: np.ndarray
    config: NormalizationConfig
    params: dict

    def inverse_transform(self):
        return Normalizer.inverse_transform(self.data, self.params)


class Normalizer:
    """Forward and inverse transforms for ML Clutter profile preprocessing."""

    @staticmethod
    def fit_transform(data, config=None):
        config = config or NormalizationConfig()
        array = np.asarray(data, dtype=float)
        Normalizer._validate_input(array)
        if config.mode not in NORMALIZATION_MODES:
            raise ValueError(f"Unsupported normalization mode: {config.mode}")
        if config.eps <= 0:
            raise ValueError("Normalization eps must be positive.")

        params = {"mode": config.mode, "eps": float(config.eps), "input_shape": tuple(array.shape)}
        if config.mode == NO_NORMALIZATION:
            normalized = array.copy()
            params.update({"center": 0.0, "scale": 1.0})
        elif config.mode == STANDARD_NORMALIZATION:
            center = float(np.mean(array))
            scale = Normalizer._safe_scale(float(np.std(array)), config.eps)
            normalized = (array - center) / scale
            params.update({"center": center, "scale": scale})
        elif config.mode == ROBUST_NORMALIZATION:
            center = float(np.median(array))
            mad = float(np.median(np.abs(array - center)))
            scale = Normalizer._safe_scale(mad, config.eps)
            normalized = (array - center) / scale
            params.update({"center": center, "scale": scale, "mad": mad})
        else:
            lower = float(np.percentile(array, config.clip_lower_percentile))
            upper = float(np.percentile(array, config.clip_upper_percentile))
            clipped = np.clip(array, lower, upper)
            center = float(np.mean(clipped))
            scale = Normalizer._safe_scale(float(np.std(clipped)), config.eps)
            normalized = (clipped - center) / scale
            params.update({
                "center": center,
                "scale": scale,
                "clip_lower": lower,
                "clip_upper": upper,
                "clip_lower_percentile": float(config.clip_lower_percentile),
                "clip_upper_percentile": float(config.clip_upper_percentile),
            })
        return NormalizationResult(normalized, config, params)

    @staticmethod
    def inverse_transform(data, params):
        array = np.asarray(data, dtype=float)
        return array * float(params["scale"]) + float(params["center"])

    @staticmethod
    def _validate_input(array):
        if array.ndim != 2:
            raise ValueError(f"Profile must be a 2D array, got ndim={array.ndim}.")
        if not np.isfinite(array).all():
            raise ValueError("Profile contains non-finite amplitudes.")

    @staticmethod
    def _safe_scale(scale, eps):
        return scale if abs(scale) > eps else eps


def build_preprocessing_report(before_stats, after_stats, normalization_result):
    config = normalization_result.config
    params = normalization_result.params
    lines = [
        "Preprocessing normalization report",
        f"Mode: {config.mode}",
        f"Config: {config.to_dict()}",
        f"Parameters: {params}",
        "Before normalization:",
        _format_stats(before_stats),
        "After normalization:",
        _format_stats(after_stats),
    ]
    return "\n".join(lines)


def _format_stats(stats):
    return (
        f"  min/max={stats['min']:.6g}/{stats['max']:.6g}; "
        f"mean/std={stats['mean']:.6g}/{stats['std']:.6g}; "
        f"median={stats['median']:.6g}; p01/p99={stats['p01']:.6g}/{stats['p99']:.6g}"
    )
