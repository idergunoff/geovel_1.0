import importlib.util
import sys
import types
from pathlib import Path

import pytest


class ArrayStub(list):
    def __ne__(self, other):
        return ArrayStub([item != other for item in self])

    def __getitem__(self, key):
        if isinstance(key, list):
            return ArrayStub([item for item, keep in zip(self, key) if keep])
        return super().__getitem__(key)


class NumpyStub:
    @staticmethod
    def asarray(value, dtype=None):
        return ArrayStub(value)

    @staticmethod
    def unique(value):
        return ArrayStub(sorted(set(value)))

    @staticmethod
    def count_nonzero(value):
        return sum(1 for item in value if item)

    @staticmethod
    def isfinite(value):
        return value not in {float("inf"), float("-inf")} and value == value

    class linalg:
        @staticmethod
        def matrix_rank(value):
            return 1


def load_auto_candidates_module():
    common = types.ModuleType("cluster.common")
    common.__dict__.update(
        {
            "__future__": None,
            "Any": object,
            "Dict": dict,
            "Literal": lambda *args, **kwargs: object,
            "Optional": object,
            "TypedDict": dict,
            "Counter": dict,
            "OrderedDict": dict,
            "product": __import__("itertools").product,
        }
    )
    models = types.ModuleType("cluster.models")
    models.__dict__.update(
        {
            "CandidateConfig": dict,
            "CandidateMetrics": dict,
            "CandidateResult": dict,
            "CandidateStats": dict,
            "GMM_COVARIANCE_TYPES": ("full", "diag", "tied", "spherical"),
            "AUTO_CANDIDATE_HARD_TIMEOUT_SEC": 1,
            "AUTO_CANDIDATE_WATCHDOG_TIMEOUT_SEC": 300,
            "AUTO_PCA_PILOT_ENABLED": False,
            "AUTO_PCA_PILOT_MAX_ROWS": 10,
            "AUTO_SILHOUETTE_MAX_SAMPLES": 100,
            "AUTO_TUNING_MAX_FEATURES": 10,
            "AUTO_TUNING_MAX_ROWS": 100,
        }
    )
    package = types.ModuleType("cluster")
    package.__path__ = [str(Path("cluster").resolve())]
    sys.modules["cluster"] = package
    sys.modules["cluster.common"] = common
    sys.modules["cluster.models"] = models

    spec = importlib.util.spec_from_file_location(
        "cluster.auto_candidates",
        Path("cluster/auto_candidates.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["cluster.auto_candidates"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def auto_candidates(monkeypatch):
    module = load_auto_candidates_module()
    monkeypatch.setattr(module, "_sample_rows_for_auto_tuning", lambda data, max_rows: data, raising=False)
    monkeypatch.setattr(module, "_reduce_feature_space_for_auto_tuning", lambda data, max_features: data, raising=False)
    monkeypatch.setattr(module, "clean_features", lambda data, **kwargs: (data, {}), raising=False)
    monkeypatch.setattr(module, "preprocess_features", lambda data, mode: data, raising=False)
    monkeypatch.setattr(module, "_build_partition_hash", lambda labels: "hash", raising=False)
    monkeypatch.setattr(module, "np", NumpyStub, raising=False)
    return module


def test_run_cluster_candidate_rejects_result_above_max_clusters(auto_candidates, monkeypatch):
    monkeypatch.setattr(
        auto_candidates,
        "cluster_data",
        lambda data, method, **kwargs: ([0, 1, 2, 3, 4], {"n_clusters": 5, "noise_fraction": 0.0}),
        raising=False,
    )
    evaluate_called = False

    def evaluate_clustering(*args, **kwargs):
        nonlocal evaluate_called
        evaluate_called = True
        return {"metrics": {"silhouette": 0.5, "davies_bouldin": 1.0, "calinski_harabasz": 10.0}}

    monkeypatch.setattr(auto_candidates, "evaluate_clustering", evaluate_clustering, raising=False)

    result = auto_candidates.run_cluster_candidate(
        [[float(i)] for i in range(5)],
        {"scaler_mode": "none", "pca_enabled": False, "pca_mode": None, "pca_value": None, "method": "hdbscan", "method_params": {}},
        max_clusters=4,
    )

    assert result["status"] == "invalid"
    assert result["stats"]["n_clusters"] == 5
    assert result["error_text"] == "n_clusters 5 > max_clusters 4"
    assert evaluate_called is False


def test_rank_candidates_removes_cached_result_above_max_clusters(auto_candidates):
    result = {
        "candidate_id": "C001",
        "candidate_config": {"scaler_mode": "none", "pca_enabled": False, "pca_mode": None, "pca_value": None, "method": "hdbscan", "method_params": {}},
        "metrics": {"silhouette": 0.9, "davies_bouldin": 0.1, "calinski_harabasz": 100.0},
        "stats": {"n_clusters": 6, "noise_fraction": 0.0},
        "score": None,
        "status": "ok",
        "error_text": "",
    }

    ranked = auto_candidates.rank_candidates([result], max_clusters=4)

    assert ranked == []


def test_build_fine_search_space_clamps_kmeans_and_gmm_to_max_clusters(auto_candidates):
    top_results = [
        {
            "status": "ok",
            "candidate_config": {
                "scaler_mode": "none",
                "pca_enabled": False,
                "pca_mode": None,
                "pca_value": None,
                "method": "kmeans",
                "method_params": {"kmeans_n_clusters": 4, "kmeans_n_init": 10},
            },
        },
        {
            "status": "ok",
            "candidate_config": {
                "scaler_mode": "none",
                "pca_enabled": False,
                "pca_mode": None,
                "pca_value": None,
                "method": "gmm",
                "method_params": {"gmm_n_components": 4, "gmm_covariance_type": "full"},
            },
        },
    ]

    candidates = auto_candidates.build_fine_search_space(
        top_results,
        top_k=2,
        max_candidates=None,
        max_clusters=4,
    )

    for candidate in candidates:
        params = candidate["method_params"]
        assert params.get("kmeans_n_clusters", 0) <= 4
        assert params.get("gmm_n_components", 0) <= 4


def test_candidate_worker_uses_only_fork_to_avoid_second_gui_window(auto_candidates, monkeypatch):
    class MpStub:
        @staticmethod
        def get_all_start_methods():
            return ["fork", "forkserver", "spawn"]

    monkeypatch.setattr(auto_candidates, "mp", MpStub, raising=False)

    assert auto_candidates._get_candidate_worker_start_methods() == ["fork"]
    assert auto_candidates._select_candidate_worker_start_method() == "fork"


def test_isolated_candidate_payload_drops_runtime_caches(auto_candidates):
    payload = auto_candidates._build_isolated_candidate_payload({
        "candidate_id": "C001",
        "transform_cache": {"old": "value"},
        "preprocess_cache": {"old": "value"},
        "preprocess_rank_cache": {"old": "value"},
        "metrics_cache": {"old": "value"},
        "base_data": [[1.0]],
    })

    assert payload["transform_cache"] is None
    assert payload["preprocess_cache"] is None
    assert payload["preprocess_rank_cache"] is None
    assert payload["metrics_cache"] is None
    assert payload["base_data"] == [[1.0]]


def test_candidate_without_ui_timeout_runs_inline(auto_candidates, monkeypatch):
    called = {}

    def run_cluster_candidate(**kwargs):
        called["kwargs"] = kwargs
        return {"status": "ok", "candidate_id": kwargs.get("candidate_id")}

    class MpStub:
        @staticmethod
        def get_all_start_methods():
            raise AssertionError("multiprocessing must not be used without explicit UI timeout")

    monkeypatch.setattr(auto_candidates, "run_cluster_candidate", run_cluster_candidate, raising=False)
    monkeypatch.setattr(auto_candidates, "mp", MpStub, raising=False)

    result = auto_candidates.run_cluster_candidate_isolated(candidate_id="C001", hard_timeout_sec=None)

    assert result == {"status": "ok", "candidate_id": "C001"}
    assert called["kwargs"]["candidate_id"] == "C001"
    assert auto_candidates._resolve_candidate_hard_timeout(None) is None
