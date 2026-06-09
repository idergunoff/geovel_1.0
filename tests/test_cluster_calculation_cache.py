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
        config={"method": "kmeans", "clusters": 3},
    )

    changed_data = module.build_cluster_calculation_cache_key(
        source_type="well_log",
        dataset_id=3,
        data_hash="data-v2",
        config={"method": "kmeans", "clusters": 3},
    )
    changed_settings = module.build_cluster_calculation_cache_key(
        source_type="well_log",
        dataset_id=3,
        data_hash="data-v1",
        config={"method": "kmeans", "clusters": 4},
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
