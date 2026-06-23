"""Experimental ML air-clutter removal helpers."""

from .config import NormalizationConfig
from .noise_patterns import PatternExtractionConfig, extract_energy_patterns, extract_frequency_band_patterns, extract_pattern_from_bbox
from .pattern_library import NoisePattern, PatternLibrary
from .pattern_generator import PatternClutterConfig, generate_pattern_clutter
from .inference import InferenceConfig, blend_inference_result, run_full_profile_inference, save_inference_result
from .preprocessing import NormalizationResult, Normalizer, build_preprocessing_report
from .metrics import paired_cleaning_metrics, summarize_metric_rows
from .model import ModelConfig, count_parameters, create_model, load_model_checkpoint, save_model_checkpoint
from .visualization import VisualizationBundle, build_experiment_log, build_paired_visualization, build_training_metric_curves

__all__ = [
    "NormalizationConfig",
    "NoisePattern",
    "PatternExtractionConfig",
    "PatternLibrary",
    "PatternClutterConfig",
    "InferenceConfig",
    "blend_inference_result",
    "run_full_profile_inference",
    "save_inference_result",
    "generate_pattern_clutter",
    "paired_cleaning_metrics",
    "summarize_metric_rows",
    "ModelConfig",
    "count_parameters",
    "create_model",
    "load_model_checkpoint",
    "save_model_checkpoint",
    "NormalizationResult",
    "Normalizer",
    "build_preprocessing_report",
    "VisualizationBundle",
    "build_experiment_log",
    "build_paired_visualization",
    "build_training_metric_curves",
    "extract_energy_patterns",
    "extract_frequency_band_patterns",
    "extract_pattern_from_bbox",
]
