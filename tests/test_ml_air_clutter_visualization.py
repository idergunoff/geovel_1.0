import numpy as np
import pytest

from ml_air_clutter.visualization import (
    build_experiment_log,
    build_paired_visualization,
    build_training_metric_curves,
)


def test_build_paired_visualization_contains_stage_12_modes():
    clean = np.full((3, 512), 100.0)
    noisy = clean + 10.0
    pred = clean + 2.0

    bundle = build_paired_visualization(clean=clean, noisy=noisy, clean_pred=pred, alpha=0.5, trace_index=1)

    assert list(bundle.radarograms) == [
        "Noisy",
        "Clean",
        "Predicted clean",
        "Cleaned with alpha",
        "Residual noisy-predicted",
        "Error map predicted-clean",
    ]
    np.testing.assert_allclose(bundle.radarograms["Cleaned with alpha"], 106.0)
    np.testing.assert_allclose(bundle.radarograms["Residual noisy-predicted"], 8.0)
    np.testing.assert_allclose(bundle.radarograms["Error map predicted-clean"], 2.0)
    assert bundle.traces["Noisy"].shape == (512,)
    assert "Trace comparison index: 1" in bundle.log_text()


def test_build_paired_visualization_supports_pair_without_prediction():
    clean = np.zeros((2, 512))
    noisy = np.ones((2, 512))

    bundle = build_paired_visualization(clean=clean, noisy=noisy)

    assert "Residual noisy-clean" in bundle.radarograms
    np.testing.assert_allclose(bundle.radarograms["Residual noisy-clean"], 1.0)


def test_build_paired_visualization_can_show_dataset_pair_without_residual():
    clean = np.full((2, 512), 128.0)
    noisy = clean + 20.0

    bundle = build_paired_visualization(clean=clean, noisy=noisy, include_pair_residual=False)

    assert list(bundle.radarograms) == ["Clean", "Noisy"]
    np.testing.assert_allclose(bundle.radarograms["Clean"], 128.0)
    np.testing.assert_allclose(bundle.radarograms["Noisy"], 148.0)
    assert "Residual noisy-clean" not in bundle.radarograms


def test_build_paired_visualization_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        build_paired_visualization(noisy=np.zeros((2, 512)), clean=np.zeros((3, 512)))


def test_build_training_metric_curves_filters_non_finite_values():
    curves = build_training_metric_curves([
        {"train_loss": 2.0, "val_loss": 3.0},
        {"train_loss": float("nan"), "val_loss": 1.0},
    ])

    assert curves["train_loss"] == [2.0]
    assert curves["val_loss"] == [3.0, 1.0]


def test_build_experiment_log_joins_non_empty_sections():
    assert build_experiment_log(" first ", "", {"alpha": 1}) == "first\n\n{'alpha': 1}"
