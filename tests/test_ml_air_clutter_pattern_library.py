from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml_air_clutter.noise_patterns import PatternExtractionConfig, extract_energy_patterns, extract_frequency_band_patterns, extract_pattern_from_bbox
from ml_air_clutter.pattern_library import NoisePattern, PatternLibrary


def test_extract_pattern_from_manual_bbox_preserves_raw_amplitude_and_masks():
    profile = np.arange(16 * 512, dtype=float).reshape(16, 512)
    mask = np.zeros_like(profile)
    mask[2:6, 10:30] = 1

    extracted = extract_pattern_from_bbox(profile, [2, 6, 10, 30], mask=mask)

    assert extracted["array"].shape == (4, 20)
    assert extracted["mask"].shape == (4, 20)
    assert extracted["stats"]["shape"] == [4, 20]
    np.testing.assert_allclose(extracted["array"], profile[2:6, 10:30])
    assert extracted["normalization"]["config"]["mode"] == "none"
    assert extracted["stats"]["mask_coverage"] == 1.0


def test_extract_pattern_from_manual_bbox_can_still_standardize_when_requested():
    profile = np.arange(16 * 512, dtype=float).reshape(16, 512)
    config = PatternExtractionConfig(normalization_mode="standard")

    extracted = extract_pattern_from_bbox(profile, [2, 6, 10, 30], config=config)

    assert abs(float(np.mean(extracted["array"]))) < 1e-12
    assert extracted["normalization"]["config"]["mode"] == "standard"


def test_extract_energy_patterns_returns_high_energy_candidates():
    profile = np.zeros((32, 512), dtype=float)
    profile[8:16, 100:140] = 10.0
    config = PatternExtractionConfig(
        patch_width=8,
        patch_height=64,
        stride_x=8,
        stride_z=32,
        energy_percentile=90.0,
        min_mask_coverage=0.05,
        max_patterns=3,
    )

    patterns = extract_energy_patterns(profile, config)

    assert 1 <= len(patterns) <= 3
    assert patterns[0]["bbox"][0] == 8
    assert patterns[0]["energy_score"] > 0


def test_extract_frequency_band_patterns_keeps_dominant_band_metadata():
    z = np.arange(128)
    sinusoid = 8.0 * np.sin(2.0 * np.pi * 9.0 * z / z.size)
    profile = np.zeros((16, 256), dtype=float)
    profile[4:12, 64:192] = sinusoid
    config = PatternExtractionConfig(
        patch_width=8,
        patch_height=128,
        stride_x=4,
        stride_z=64,
        energy_percentile=85.0,
        min_mask_coverage=0.05,
        max_patterns=2,
        frequency_bandwidth_bins=1,
    )

    patterns = extract_frequency_band_patterns(profile, config)

    assert patterns
    assert "frequency_band" in patterns[0]
    assert patterns[0]["frequency_band"]["peak_bin"] == 9
    assert patterns[0]["energy_score"] > 0


def test_pattern_library_save_load_roundtrip(tmp_path):
    array = np.ones((4, 8), dtype=float)
    mask = np.ones_like(array)
    pattern = NoisePattern.create(
        source_profile="noisy_profile",
        array=array,
        mask=mask,
        bbox=[1, 5, 2, 10],
        tags=["ringing"],
        comment="test pattern",
        pattern_id="pattern-1",
    )
    library = PatternLibrary()
    library.add_pattern(pattern)

    index_path = library.save(tmp_path)
    loaded = PatternLibrary.load(tmp_path)

    assert index_path.name == "pattern_library_index.json"
    assert loaded.summary() == {"num_patterns": 1, "tags": {"ringing": 1}}
    np.testing.assert_allclose(loaded.get("pattern-1").array, array)
    assert loaded.get("pattern-1").comment == "test pattern"


def test_pattern_library_remove_and_clear_patterns():
    pattern = NoisePattern.create(
        "src",
        np.ones((2, 3)),
        np.ones((2, 3)),
        [0, 2, 0, 3],
        pattern_id="pattern-to-delete",
    )
    library = PatternLibrary([pattern])

    assert library.remove_pattern("missing") is None
    assert library.remove_pattern("pattern-to-delete") is pattern
    assert library.summary() == {"num_patterns": 0, "tags": {}}

    library.add_pattern(pattern)
    library.clear()

    assert library.patterns == []
    assert library.summary() == {"num_patterns": 0, "tags": {}}
