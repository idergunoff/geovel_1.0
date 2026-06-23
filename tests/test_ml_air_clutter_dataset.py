import json

import numpy as np
import pytest

from ml_air_clutter.dataset import (
    PairValidationError,
    PatchDatasetConfig,
    build_paired_patch_dataset,
    prepare_amplitude_pair_0256,
    save_dataset,
    validate_clean_noisy_pair,
)


def test_validate_clean_noisy_pair_accepts_matching_512_sample_arrays():
    clean = np.zeros((96, 512), dtype=float)
    noisy = clean + 0.1

    report = validate_clean_noisy_pair(clean, noisy)

    assert report["valid"] is True
    assert report["shape"] == [96, 512]
    assert report["errors"] == []
    assert "difference_stats" in report


def test_validate_clean_noisy_pair_rejects_mismatched_shapes():
    report = validate_clean_noisy_pair(np.zeros((8, 512)), np.zeros((9, 512)))

    assert report["valid"] is False
    assert any("shapes must match" in error for error in report["errors"])


def test_build_paired_patch_dataset_uses_trace_blocks_with_gap_for_single_pair():
    clean = np.arange(160 * 512, dtype=float).reshape(160, 512)
    noisy = clean + 5.0
    config = PatchDatasetConfig(patch_width=16, stride=8, train_fraction=0.6, validation_fraction=0.2, test_fraction=0.2)

    samples, summary = build_paired_patch_dataset([
        {"pair_id": "pair_a", "clean": clean, "noisy": noisy, "clean_path": "clean.npy", "noisy_path": "noisy.npy"}
    ], config)

    assert summary["dataset_schema"] == "paired_clean_noisy_v1"
    assert summary["num_train_patches"] == len(samples["train"])
    assert summary["num_validation_patches"] == len(samples["validation"])
    assert summary["num_test_patches"] == len(samples["test"])
    assert samples["train"]
    assert samples["validation"]
    assert samples["test"]
    train_max = max(sample["x_end"] for sample in samples["train"])
    val_min = min(sample["x_start"] for sample in samples["validation"])
    assert val_min - train_max >= config.patch_width
    first = samples["train"][0]
    assert np.array_equal(first["noisy"], noisy[first["x_start"]:first["x_end"]])
    assert np.array_equal(first["clean"], clean[first["x_start"]:first["x_end"]])
    assert np.array_equal(first["residual"], first["noisy"] - first["clean"])


def test_build_paired_patch_dataset_rejects_invalid_pair():
    config = PatchDatasetConfig(patch_width=8, stride=4)

    with pytest.raises(PairValidationError):
        build_paired_patch_dataset([{"pair_id": "bad", "clean": np.zeros((8, 256)), "noisy": np.zeros((8, 256))}], config)


def test_save_dataset_writes_summary_and_npz_samples(tmp_path):
    clean = np.zeros((48, 512), dtype=float)
    noisy = clean + 1.0
    samples, summary = build_paired_patch_dataset(
        [{"pair_id": "pair_a", "clean": clean, "noisy": noisy}],
        PatchDatasetConfig(patch_width=8, stride=8),
    )

    summary_path = save_dataset(tmp_path, samples, summary)

    assert summary_path.exists()
    loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded_summary["dataset_schema"] == "paired_clean_noisy_v1"
    assert list((tmp_path / "train").glob("*.npz"))


def test_prepare_amplitude_pair_shifts_centered_profiles_to_0256():
    clean = np.array([[-128.0, 0.0, 127.0, 128.0]])
    noisy = np.array([[-100.0, 20.0, 128.0, 140.0]])

    clean_0256, noisy_0256 = prepare_amplitude_pair_0256(clean, noisy)

    np.testing.assert_allclose(clean_0256, [[0.0, 128.0, 255.0, 256.0]])
    np.testing.assert_allclose(noisy_0256, [[28.0, 148.0, 256.0, 256.0]])


def test_prepare_amplitude_pair_keeps_existing_0256_profiles():
    clean = np.array([[0.0, 64.0, 128.0, 256.0]])
    noisy = np.array([[10.0, 80.0, 140.0, 260.0]])

    clean_0256, noisy_0256 = prepare_amplitude_pair_0256(clean, noisy)

    np.testing.assert_allclose(clean_0256, clean)
    np.testing.assert_allclose(noisy_0256, [[10.0, 80.0, 140.0, 256.0]])
