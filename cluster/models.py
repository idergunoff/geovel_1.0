from __future__ import annotations

from .common import *

cluster_profile_cache = {}
is_cluster_redraw_in_progress = False


class ClusterContextError(ValueError):
    """Ошибка построения runtime-контекста кластеризации."""


class ClusterRunContext(TypedDict, total=False):
    """
    Единый runtime-контракт источника данных для расчета кластеризации.
    """
    source_type: Literal["gpr", "well_log"]
    dataset_id: int
    dataset_title: str
    raw_rows: list[list[Any]]
    feature_names: list[str]
    meta_columns: list[str]
    feature_columns: list[str]
    row_count: int
    ui_tab_key: str
    data_hash: str
    meta: list[dict[str, Any]]
    X: list[list[float]]
    diagnostics: dict[str, Any]





class WellLogClusterVisualizationData(TypedDict, total=False):
    """
    Runtime-данные последнего ручного CALC Well Log для будущего окна визуализации.
    """
    run_id: str
    source_type: Literal["well_log"]
    dataset_id: int
    dataset_title: str
    created_at: str
    data_hash: str
    feature_names: list[str]
    labels: list[int]
    rows: list[dict[str, Any]]
    cluster_summary: list[dict[str, Any]]
    metrics: dict[str, Any]
    config: dict[str, Any]
    diagnostics: dict[str, Any]
    summary: dict[str, Any]


class CandidateConfig(TypedDict):
    """
    Единый контракт кандидата для AUTO-подбора параметров кластеризации.
    """
    scaler_mode: str
    pca_enabled: bool
    pca_mode: Optional[str]
    pca_value: Optional[float]
    method: Literal["kmeans", "hdbscan", "gmm"]
    method_params: Dict[str, Any]


class CandidateMetrics(TypedDict, total=False):
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float


class CandidateStats(TypedDict, total=False):
    n_clusters: int
    noise_fraction: float
    n_samples_eval: int
    pca_components_after: int
    partition_hash: str


class CandidateResult(TypedDict):
    """
    Единый контракт результата прогона кандидата для AUTO-подбора.
    """
    candidate_id: str
    candidate_config: CandidateConfig
    metrics: CandidateMetrics
    stats: CandidateStats
    score: Optional[float]
    status: Literal["ok", "invalid", "error"]
    error_text: str


class AutoTuningClusterSizeLimits(TypedDict):
    min_cluster_samples: int
    recommended_default_value: int
    max_spinbox_value: int


cluster_auto_results_cache: list[CandidateResult] = []
cluster_auto_results_by_context: dict[tuple[str, int, str], list[CandidateResult]] = {}
well_log_cluster_result_cache: dict[int, dict[str, Any]] = {}
well_log_cluster_visualization_window = None


CLUSTER_DATA_GZIP_PREFIX = "gzjson:"
GMM_COVARIANCE_TYPES = ("full", "diag", "tied", "spherical")
AUTO_TRANSFORM_CACHE_MAX_BYTES = 512 * 1024 * 1024
AUTO_TRANSFORM_CACHE_MAX_ITEM_BYTES = 128 * 1024 * 1024
AUTO_SILHOUETTE_MAX_SAMPLES = 5000
AUTO_SILHOUETTE_COARSE_MAX_SAMPLES = 1500
AUTO_SILHOUETTE_ADAPTIVE_ALPHA = 24.0
AUTO_SILHOUETTE_MIN_SAMPLES = 400
AUTO_FINE_SEED_DELTA_FROM_BEST = 0.03
AUTO_PCA_PILOT_ENABLED = True
AUTO_PCA_PILOT_MAX_ROWS = 500
AUTO_TUNING_MAX_ROWS = 20000
AUTO_TRANSFORM_CACHE_MAX_ROWS = 12000
AUTO_TUNING_MAX_WORKING_SET_BYTES = 256 * 1024 * 1024
AUTO_TUNING_MIN_ROWS = 256
AUTO_TUNING_MAX_FEATURES = 512
AUTO_TUNING_REDUCE_FEATURES_THRESHOLD = 10000
AUTO_TUNING_FEATURE_REDUCTION_MODE = "auto"
AUTO_CANDIDATE_HARD_TIMEOUT_SEC: Optional[float] = None
# Аварийный watchdog применяется даже когда UI-лимит кандидата отключен.
# Он нужен не для штатного ограничения подбора, а чтобы зависший worker
# не блокировал повторный AUTO/retune-расчет бесконечно.
AUTO_CANDIDATE_WATCHDOG_TIMEOUT_SEC: Optional[float] = 300.0
AUTO_CHECKPOINT_SAVE_EVERY = 10




