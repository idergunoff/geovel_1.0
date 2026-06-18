from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml_air_clutter.config import NormalizationConfig
from ml_air_clutter.preprocessing import Normalizer


def test_standard_normalization_roundtrip():
    data = np.arange(1024, dtype=float).reshape(2, 512)
    result = Normalizer.fit_transform(data, NormalizationConfig(mode="standard"))

    assert abs(float(np.mean(result.data))) < 1e-12
    assert abs(float(np.std(result.data)) - 1.0) < 1e-12
    np.testing.assert_allclose(result.inverse_transform(), data)
    assert result.params["mode"] == "standard"


def test_robust_normalization_uses_median_and_mad():
    data = np.tile(np.arange(512, dtype=float), (3, 1))
    result = Normalizer.fit_transform(data, NormalizationConfig(mode="robust"))

    assert result.params["center"] == float(np.median(data))
    assert result.params["mad"] == float(np.median(np.abs(data - np.median(data))))
    np.testing.assert_allclose(result.inverse_transform(), data)


def test_percentile_standard_clips_before_scaling_and_stores_config():
    data = np.arange(1024, dtype=float).reshape(2, 512)
    data[0, 0] = -1000.0
    data[-1, -1] = 5000.0
    config = NormalizationConfig(mode="percentile_standard", clip_lower_percentile=1.0, clip_upper_percentile=99.0)
    result = Normalizer.fit_transform(data, config)

    assert result.params["clip_lower"] == float(np.percentile(data, 1.0))
    assert result.params["clip_upper"] == float(np.percentile(data, 99.0))
    assert result.config.to_dict()["mode"] == "percentile_standard"
    restored = result.inverse_transform()
    assert restored.min() >= result.params["clip_lower"]
    assert restored.max() <= result.params["clip_upper"]


def test_normalization_rejects_non_finite_values():
    data = np.zeros((2, 512), dtype=float)
    data[0, 0] = np.nan

    try:
        Normalizer.fit_transform(data)
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("Expected non-finite profile rejection")
