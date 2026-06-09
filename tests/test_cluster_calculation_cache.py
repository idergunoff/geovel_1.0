import importlib.util
import json
import sys
import types
from pathlib import Path


def load_result_cache_module():
    common = types.ModuleType("cluster.common")
    common.__dict__.update(
        {
            "Any": object,
            "json": json,
            "hashlib": __import__("hashlib"),
            "base64": __import__("base64"),
            "gzip": __import__("gzip"),
        }
    )
    package = types.ModuleType("cluster")
    package.__path__ = [str(Path("cluster").resolve())]
    sys.modules["cluster"] = package
    sys.modules["cluster.common"] = common

    spec = importlib.util.spec_from_file_location(
        "cluster.result_cache",
        Path("cluster/result_cache.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["cluster.result_cache"] = module
    spec.loader.exec_module(module)
    return module


def test_calculation_cache_key_is_stable_for_equivalent_config_order():
    module = load_result_cache_module()

    first = module.build_cluster_calculation_cache_key(
        source_type="gpr",
        dataset_id=7,
        data_hash="data-v1",
        config={"method": "kmeans", "params": {"n_init": 10, "clusters": 4}},
    )
    second = module.build_cluster_calculation_cache_key(
        source_type="GPR",
        dataset_id=7,
        data_hash="data-v1",
        config={"params": {"clusters": 4, "n_init": 10}, "method": "kmeans"},
    )

    assert first == second


def test_calculation_cache_key_changes_with_data_or_settings():
    module = load_result_cache_module()
    base = module.build_cluster_calculation_cache_key(
        source_type="well_log",
        dataset_id=3,
        data_hash="data-v1",
        config={"method": "kmeans", "method_params": {"kmeans_n_clusters": 3}},
    )

    changed_data = module.build_cluster_calculation_cache_key(
        source_type="well_log",
        dataset_id=3,
        data_hash="data-v2",
        config={"method": "kmeans", "method_params": {"kmeans_n_clusters": 3}},
    )
    changed_settings = module.build_cluster_calculation_cache_key(
        source_type="well_log",
        dataset_id=3,
        data_hash="data-v1",
        config={"method": "kmeans", "method_params": {"kmeans_n_clusters": 4}},
    )

    assert base != changed_data
    assert base != changed_settings


def test_calculation_payload_round_trip_and_invalid_value():
    module = load_result_cache_module()
    payload = {
        "result_type": "well_log",
        "labels": [0, 1, -1],
        "visualization_data": {"dataset_title": "Каротаж №1"},
    }

    encoded = module.encode_cluster_calculation_payload(payload)

    assert encoded.startswith(module.CLUSTER_CALC_CACHE_GZIP_PREFIX)
    assert module.decode_cluster_calculation_payload(encoded) == payload
    assert module.decode_cluster_calculation_payload("not-json") is None


def test_cached_labels_accept_hits_for_gpr_and_well_log():
    module = load_result_cache_module()

    assert module.get_cached_cluster_labels(
        {"result_type": "gpr", "labels": [0, 1, 1]},
        result_type="gpr",
        expected_count=3,
    ) == [0, 1, 1]
    assert module.get_cached_cluster_labels(
        {"result_type": "well_log", "labels": [2, -1]},
        result_type="well_log",
    ) == [2, -1]


def test_cached_labels_reject_wrong_source_stale_size_and_invalid_payload():
    module = load_result_cache_module()
    payload = {"result_type": "gpr", "labels": [0, 1]}

    assert module.get_cached_cluster_labels(payload, result_type="well_log") is None
    assert module.get_cached_cluster_labels(payload, result_type="gpr", expected_count=3) is None
    assert module.get_cached_cluster_labels(
        {"result_type": "gpr", "labels": ["bad"]},
        result_type="gpr",
    ) is None


def _full_manual_config(*, method="kmeans", pca_enabled=False, smoothing_enabled=False):
    return {
        "clean": {
            "use_non_finite": False,
            "non_finite_mode": "impute",
            "use_variance_threshold": True,
            "use_correlation_filter": False,
        },
        "preprocess_mode": "standard",
        "pca": {
            "enabled": pca_enabled,
            "mode": "fixed_components" if pca_enabled else None,
            "value": 3 if pca_enabled else None,
            "fixed_components": 3,
            "variance_ratio": 0.91,
        },
        "method": method,
        "method_params": {
            "kmeans_n_clusters": 4,
            "kmeans_n_init": 10,
            "hdbscan_min_cluster_size": 20,
            "hdbscan_min_samples": 7,
            "hdbscan_metric": "euclidean",
            "gmm_n_components": 6,
            "gmm_covariance_type": "diag",
        },
        "metrics": {"use_silhouette": True, "use_db": True, "use_ch": False},
        "smoothing": {"enabled": smoothing_enabled, "method": "maj", "window": 5},
    }


def test_cache_key_ignores_inactive_controls_and_all_postprocessing_settings():
    module = load_result_cache_module()
    first = _full_manual_config()
    second = _full_manual_config()
    second["clean"]["non_finite_mode"] = "drop"
    second["pca"]["fixed_components"] = 12
    second["pca"]["variance_ratio"] = 0.55
    second["method_params"]["gmm_n_components"] = 11
    second["method_params"]["hdbscan_min_samples"] = 99
    second["smoothing"]["enabled"] = True
    second["smoothing"]["method"] = "med"
    second["smoothing"]["window"] = 21
    second["metrics"] = {"use_silhouette": False, "use_db": False, "use_ch": True}

    first_key = module.build_cluster_calculation_cache_key(
        source_type="gpr", dataset_id=1, data_hash="same", config=first
    )
    second_key = module.build_cluster_calculation_cache_key(
        source_type="gpr", dataset_id=1, data_hash="same", config=second
    )

    assert first_key == second_key


def test_cache_key_changes_when_active_auto_parameter_changes():
    module = load_result_cache_module()
    first = _full_manual_config(method="gmm", pca_enabled=True, smoothing_enabled=True)
    second = _full_manual_config(method="gmm", pca_enabled=True, smoothing_enabled=True)
    second["method_params"]["gmm_n_components"] = 7

    assert module.build_cluster_calculation_cache_key(
        source_type="well_log", dataset_id=2, data_hash="same", config=first
    ) != module.build_cluster_calculation_cache_key(
        source_type="well_log", dataset_id=2, data_hash="same", config=second
    )


def test_gpr_cache_requires_complete_preparation_payload():
    module = load_result_cache_module()
    complete = {
        "result_type": "gpr",
        "labels": [0, 1],
        "kept_row_indices": [3, 5],
        "data_for_diagnostics": [[0.1], [0.2]],
        "cluster_info": {"n_clusters": 2},
        "pca_info_report": {"components_after_pca": 1},
    }

    assert module.get_cached_gpr_calculation(complete) == {
        "labels": [0, 1],
        "kept_row_indices": [3, 5],
        "data_for_diagnostics": [[0.1], [0.2]],
        "cluster_info": {"n_clusters": 2},
        "pca_info_report": {"components_after_pca": 1},
    }
    incomplete = dict(complete, data_for_diagnostics=[[0.1]])
    assert module.get_cached_gpr_calculation(incomplete) is None


def test_well_log_cache_contains_pre_smoothing_row_assignments():
    module = load_result_cache_module()
    payload = {
        "result_type": "well_log",
        "labels": [2, 2, 0, 1],
        "kept_row_indices": [0, 1, 3, 4],
        "data_for_diagnostics": [[1.0], [1.1], [3.0], [5.0]],
        "cluster_info": {"n_clusters": 3},
        "pca_info_report": {},
    }

    cached = module.get_cached_base_calculation(payload, result_type="well_log")

    assert cached["labels"] == [2, 2, 0, 1]
    assert cached["kept_row_indices"] == [0, 1, 3, 4]
    smoothed_for_output = list(cached["labels"])
    smoothed_for_output[1] = 0
    assert cached["labels"] == [2, 2, 0, 1]
