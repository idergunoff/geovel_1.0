from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml_air_clutter.config import SyntheticClutterConfig
from ml_air_clutter.synthetic_clutter import compute_snr_db, generate_synthetic_clutter


def test_synthetic_clutter_returns_expected_shapes_and_meta():
    clean = np.ones((32, 512), dtype=float)
    config = SyntheticClutterConfig(seed=7, target_snr_db=3.0)

    noisy, clutter, mask, meta = generate_synthetic_clutter(clean, config)

    assert noisy.shape == clean.shape
    assert clutter.shape == clean.shape
    assert mask.shape == clean.shape
    assert len(meta["objects"]) > 0
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    np.testing.assert_allclose(noisy, clean + clutter)
    assert abs(compute_snr_db(clean, clutter) - 3.0) < 1e-6


def test_synthetic_clutter_is_reproducible_for_same_seed():
    clean = np.linspace(-1.0, 1.0, 16 * 512).reshape(16, 512)
    config = SyntheticClutterConfig(seed=123, num_hyperbolas=1, num_sloped_events=1, num_ringing_events=1)

    first = generate_synthetic_clutter(clean, config)
    second = generate_synthetic_clutter(clean, config)

    for left, right in zip(first[:3], second[:3]):
        np.testing.assert_allclose(left, right)
    assert first[3] == second[3]


def test_synthetic_clutter_can_disable_all_objects():
    clean = np.ones((8, 512), dtype=float)
    config = SyntheticClutterConfig(
        hyperbolas=False,
        sloped_events=False,
        ringing=False,
        vertical_spikes=False,
        noise_zones=False,
    )

    noisy, clutter, mask, meta = generate_synthetic_clutter(clean, config)

    np.testing.assert_allclose(noisy, clean)
    np.testing.assert_allclose(clutter, 0.0)
    np.testing.assert_allclose(mask, 0.0)
    assert meta["objects"] == []
