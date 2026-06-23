import json

import numpy as np

from ml_air_clutter.metrics import paired_cleaning_metrics, summarize_metric_rows, write_metrics_report


def test_paired_cleaning_metrics_compare_before_and_after():
    clean = np.full((4, 512), 100.0)
    noisy = clean + 10.0
    clean_pred = clean + 2.0

    metrics = paired_cleaning_metrics(clean, noisy, clean_pred)

    assert metrics["mae_before"] == 10.0
    assert metrics["mae_after"] == 2.0
    assert metrics["rmse_before"] == 10.0
    assert metrics["rmse_after"] == 2.0
    assert metrics["snr_gain_db"] > 0
    assert metrics["psnr_after_db"] > metrics["psnr_before_db"]
    assert metrics["residual_energy"] > 0


def test_summarize_metric_rows_averages_numeric_metrics():
    summary = summarize_metric_rows([
        {"mae_before": 4.0, "mae_after": 2.0, "nested": {"ignored": True}},
        {"mae_before": 2.0, "mae_after": 1.0, "nested": {"ignored": True}},
    ])

    assert summary["num_samples"] == 2
    assert summary["mae_before"] == 3.0
    assert summary["mae_after"] == 1.5


def test_write_metrics_report_serializes_non_finite_values(tmp_path):
    path = write_metrics_report(tmp_path / "metrics.json", {"value": float("inf")})

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["value"] == "inf"
