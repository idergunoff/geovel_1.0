from __future__ import annotations

import json
from typing import Any

SYSTEM_FEATURE_NAMES = {
    "well_id_depth",
    "sample_index",
    "sample index",
    "sample",
    "well_id",
    "depth",
}


def normalized_feature_name(value: Any) -> str:
    return str(value or "").strip().casefold()


def feature_name_conflict_message(
    feature_name: str,
    *,
    existing_feature_names: set[str],
    canonical_names: set[str],
) -> str | None:
    normalized = normalized_feature_name(feature_name)
    if not normalized:
        return "Feature name is required."
    if normalized in existing_feature_names:
        return f"Feature name '{feature_name}' already exists in feature_calculator."
    if normalized in canonical_names:
        return f"Feature name '{feature_name}' matches a canonical curve name."
    if normalized in SYSTEM_FEATURE_NAMES:
        return f"Feature name '{feature_name}' conflicts with a system dataset column."
    return None


def feature_library_entry_mode(feature: dict[str, Any]) -> str:
    params = feature.get("params_json")
    try:
        params = json.loads(params) if isinstance(params, str) else params
    except Exception:
        params = None
    if isinstance(params, dict) and params.get("mode") in {"operation", "formula"}:
        return str(params["mode"])
    transform_type = str(feature.get("transform_type") or "").strip().casefold()
    return "formula" if transform_type == "formula" else "operation"


def filter_feature_library_features(
    features: list[dict[str, Any]],
    *,
    search_text: str = "",
    mode_filter: str = "all",
    canonical_filter: str = "all",
    used_in_current_dataset: bool = False,
) -> list[dict[str, Any]]:
    search = normalized_feature_name(search_text)
    mode_filter = str(mode_filter or "all")
    canonical_filter_norm = normalized_feature_name(canonical_filter if canonical_filter != "all" else "")
    filtered: list[dict[str, Any]] = []
    for feature in features:
        name = str(feature.get("feature_name") or "")
        expression = str(feature.get("expression") or "")
        inputs_text = str(feature.get("inputs_text") or "") or str(feature.get("used_canonical_well_log") or "")
        if search and search not in normalized_feature_name(name) and search not in normalized_feature_name(expression):
            continue
        if mode_filter != "all" and feature_library_entry_mode(feature) != mode_filter:
            continue
        if canonical_filter_norm and canonical_filter_norm not in normalized_feature_name(inputs_text):
            continue
        if used_in_current_dataset and not bool(feature.get("in_dataset")):
            continue
        filtered.append(feature)
    return filtered
