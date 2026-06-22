"""Experimental ML air-clutter removal helpers."""

from .config import NormalizationConfig
from .noise_patterns import PatternExtractionConfig, extract_energy_patterns, extract_frequency_band_patterns, extract_pattern_from_bbox
from .pattern_library import NoisePattern, PatternLibrary
from .pattern_generator import PatternClutterConfig, generate_pattern_clutter
from .preprocessing import NormalizationResult, Normalizer, build_preprocessing_report

__all__ = [
    "NormalizationConfig",
    "NoisePattern",
    "PatternExtractionConfig",
    "PatternLibrary",
    "PatternClutterConfig",
    "generate_pattern_clutter",
    "NormalizationResult",
    "Normalizer",
    "build_preprocessing_report",
    "extract_energy_patterns",
    "extract_frequency_band_patterns",
    "extract_pattern_from_bbox",
]
