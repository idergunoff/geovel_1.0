from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml_air_clutter.pattern_generator import (
    PatternClutterConfig,
    generate_pattern_clutter,
    overlay_noise_by_dominant_amplitude,
    overlay_noise_by_soft_dominance,
    transform_pattern,
)
from ml_air_clutter.pattern_library import NoisePattern, PatternLibrary
from ml_air_clutter.synthetic_clutter import SyntheticClutterConfig


def _library():
    arr = np.zeros((12, 64), dtype=float)
    arr[:, 20:32] = 240.0
    mask = (arr != 0).astype(float)
    pattern = NoisePattern.create("src", arr, mask, [0, 12, 100, 164], pattern_id="p1", tags=["ringing"])
    return PatternLibrary([pattern])


def test_pattern_generator_places_real_pattern_with_dominant_amplitude_overlay():
    clean = np.full((32, 512), 128.0, dtype=float)
    cfg = PatternClutterConfig(
        seed=7,
        target_snr_db=None,
        random_crop=False,
        num_patterns=1,
        jitter_std=0.0,
        fade_probability=0.0,
        polarity_flip_probability=0.0,
        amplitude_scale_min=1.0,
        amplitude_scale_max=1.0,
    )

    noisy, clutter, mask, meta = generate_pattern_clutter(clean, _library(), cfg)

    assert noisy.shape == clean.shape
    assert clutter.shape == clean.shape
    assert mask.shape == clean.shape
    assert np.any(clutter)
    assert np.any(mask)
    assert meta["placements"][0]["pattern_id"] == "p1"
    assert meta["overlay_mode"] == "dominant_amplitude"
    assert meta["placements"][0]["placement"]["z_start"] == 100
    np.testing.assert_allclose(noisy, clean + clutter)
    assert np.all((noisy >= 0.0) & (noisy <= 256.0))


def test_pattern_transform_is_reproducible_with_seed_and_records_augmentations():
    pattern = _library().get("p1")
    cfg = PatternClutterConfig(seed=3, random_crop=True, jitter_std=0.0, horizontal_flip_probability=1.0)
    rng1 = np.random.default_rng(cfg.seed)
    rng2 = np.random.default_rng(cfg.seed)

    arr1, mask1, meta1 = transform_pattern(pattern, cfg, rng1)
    arr2, mask2, meta2 = transform_pattern(pattern, cfg, rng2)

    np.testing.assert_allclose(arr1, arr2)
    np.testing.assert_allclose(mask1, mask2)
    assert meta1 == meta2
    assert meta1["horizontal_flip"] is True
    assert "stretch" in meta1


def test_mixed_mode_keeps_pattern_and_synthetic_components_in_meta():
    clean = np.full((40, 512), 128.0, dtype=float)
    cfg = PatternClutterConfig(seed=5, mode="mixed", target_snr_db=None, random_crop=False, jitter_std=0.0)
    syn_cfg = SyntheticClutterConfig(seed=5, target_snr_db=None, hyperbolas=False, sloped_events=False, ringing=False, vertical_spikes=True, noise_zones=False)

    noisy, clutter, mask, meta = generate_pattern_clutter(clean, _library(), cfg, syn_cfg)

    assert np.any(noisy - clean)
    assert np.any(clutter)
    assert np.any(mask)
    assert meta["synthetic_clutter"] is not None
    assert meta["pattern_clutter"]["rms"] > 0
    assert meta["total_clutter"]["rms"] > 0


def test_overlay_noise_by_dominant_amplitude_preserves_range_and_selects_stronger_signal():
    clean = np.array([[128.0, 200.0, 20.0]])
    noise = np.array([[300.0, 150.0, 250.0]])
    mask = np.array([[1.0, 1.0, 0.0]])

    noisy, dominance_mask = overlay_noise_by_dominant_amplitude(clean, noise, mask, midpoint=128.0)

    np.testing.assert_allclose(noisy, [[256.0, 200.0, 20.0]])
    np.testing.assert_allclose(dominance_mask, [[1.0, 0.0, 0.0]])
    assert np.all((noisy >= 0.0) & (noisy <= 256.0))


def test_pattern_generator_can_use_legacy_additive_overlay_mode():
    clean = np.full((32, 512), 128.0, dtype=float)
    cfg = PatternClutterConfig(
        seed=7,
        overlay_mode="additive",
        target_snr_db=None,
        random_crop=False,
        num_patterns=1,
        jitter_std=0.0,
        fade_probability=0.0,
        polarity_flip_probability=0.0,
        amplitude_scale_min=1.0,
        amplitude_scale_max=1.0,
    )

    noisy, clutter, mask, meta = generate_pattern_clutter(clean, _library(), cfg)

    assert meta["overlay_mode"] == "additive"
    assert np.any(mask)
    np.testing.assert_allclose(noisy, clean + clutter)
    assert np.max(noisy) == 240.0


def test_pattern_generator_can_use_soft_dominance_overlay_mode():
    clean = np.full((32, 512), 128.0, dtype=float)
    cfg = PatternClutterConfig(
        seed=7,
        overlay_mode="soft_dominance",
        soft_dominance_temperature=8.0,
        target_snr_db=None,
        random_crop=False,
        num_patterns=1,
        jitter_std=0.0,
        fade_probability=0.0,
        polarity_flip_probability=0.0,
        amplitude_scale_min=1.0,
        amplitude_scale_max=1.0,
    )

    noisy, clutter, mask, meta = generate_pattern_clutter(clean, _library(), cfg)

    assert meta["overlay_mode"] == "soft_dominance"
    assert meta["diagnostics"]["effective_pixel_count"] > 0
    assert np.any(mask)
    np.testing.assert_allclose(noisy, clean + clutter)
    assert np.max(noisy) < 240.0
    assert np.max(noisy) > 128.0


def test_overlay_noise_by_soft_dominance_blends_instead_of_hard_replace():
    clean = np.array([[128.0, 200.0, 20.0]])
    noise = np.array([[240.0, 150.0, 250.0]])
    mask = np.array([[1.0, 1.0, 0.0]])

    noisy, effective_mask = overlay_noise_by_soft_dominance(clean, noise, mask, midpoint=128.0, temperature=12.0)

    assert 128.0 < noisy[0, 0] < 240.0
    assert 150.0 < noisy[0, 1] < 200.0
    assert noisy[0, 2] == 20.0
    np.testing.assert_allclose(effective_mask, [[1.0, 1.0, 0.0]])



def test_pattern_generator_keeps_target_snr_stable_when_mixing_more_real_patterns():
    clean = np.full((64, 512), 128.0, dtype=float)
    common = dict(
        seed=11,
        target_snr_db=30.0,
        random_crop=False,
        jitter_std=0.0,
        fade_probability=0.0,
        polarity_flip_probability=0.0,
        amplitude_scale_min=1.0,
        amplitude_scale_max=1.0,
    )

    _, _, _, one_meta = generate_pattern_clutter(clean, _library(), PatternClutterConfig(num_patterns=1, **common))
    _, _, _, many_meta = generate_pattern_clutter(clean, _library(), PatternClutterConfig(num_patterns=4, **common))

    assert abs(one_meta["actual_snr_db"] - 30.0) < 0.05
    assert abs(many_meta["actual_snr_db"] - 30.0) < 0.05
    assert abs(one_meta["actual_snr_db"] - many_meta["actual_snr_db"]) < 0.05


def test_random_crop_does_not_move_preserved_pattern_depth():
    pattern = _library().get("p1")
    clean = np.full((32, 512), 128.0, dtype=float)
    cfg = PatternClutterConfig(
        seed=2,
        target_snr_db=None,
        random_crop=True,
        min_crop_fraction=0.5,
        num_patterns=1,
        jitter_std=0.0,
        fade_probability=0.0,
        polarity_flip_probability=0.0,
        amplitude_scale_min=1.0,
        amplitude_scale_max=1.0,
    )

    _, _, _, meta = generate_pattern_clutter(clean, PatternLibrary([pattern]), cfg)

    assert "crop" in meta["placements"][0]["transform"]
    assert meta["placements"][0]["placement"]["z_start"] == pattern.bbox[2]


def test_place_pattern_clamps_preserved_depth_to_profile_bounds():
    arr = np.full((12, 64), 240.0, dtype=float)
    pattern = NoisePattern.create("src", arr, np.ones_like(arr), [0, 12, 500, 564], pattern_id="deep", tags=["ringing"])
    clean = np.full((32, 512), 128.0, dtype=float)
    cfg = PatternClutterConfig(
        seed=7,
        target_snr_db=None,
        random_crop=False,
        num_patterns=1,
        jitter_std=0.0,
        fade_probability=0.0,
        polarity_flip_probability=0.0,
        amplitude_scale_min=1.0,
        amplitude_scale_max=1.0,
    )

    _, _, _, meta = generate_pattern_clutter(clean, PatternLibrary([pattern]), cfg)

    assert meta["placements"][0]["placement"]["z_start"] == 448
