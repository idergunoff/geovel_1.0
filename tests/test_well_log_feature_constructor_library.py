import importlib.util
import json
from pathlib import Path
import sys

_MODULE_PATH = Path(__file__).resolve().parents[1] / "cluster" / "well_feature_library.py"
_SPEC = importlib.util.spec_from_file_location("well_feature_library", _MODULE_PATH)
well_feature_library = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = well_feature_library
_SPEC.loader.exec_module(well_feature_library)

from well_feature_library import (
    feature_library_entry_mode,
    feature_name_conflict_message,
    filter_feature_library_features,
)


def _feature(name, *, mode="operation", expression="", inputs=None, in_dataset=False):
    params = {"version": 1, "mode": mode, "inputs": inputs or []}
    if mode == "formula":
        params["expression"] = expression
    else:
        params["operation"] = "log"
    return {
        "feature_name": name,
        "transform_type": "formula" if mode == "formula" else "log",
        "expression": expression,
        "used_canonical_well_log": json.dumps(inputs or []),
        "inputs_text": ", ".join(entry.get("canonical_name", "") for entry in inputs or []),
        "params_json": json.dumps(params),
        "in_dataset": in_dataset,
    }


def test_feature_library_search_by_name_and_expression():
    features = [
        _feature("GR_LOG", expression="log(GR)", inputs=[{"canonical_name": "GR"}]),
        _feature("RHOB_RATIO", mode="formula", expression="RHOB / GR", inputs=[{"canonical_name": "RHOB"}, {"canonical_name": "GR"}]),
    ]

    assert [feature["feature_name"] for feature in filter_feature_library_features(features, search_text="ratio")] == ["RHOB_RATIO"]
    assert [feature["feature_name"] for feature in filter_feature_library_features(features, search_text="log(GR)")] == ["GR_LOG"]


def test_feature_library_filters_by_operation_formula_curve_and_current_dataset_usage():
    features = [
        _feature("GR_LOG", inputs=[{"canonical_name": "GR"}], in_dataset=True),
        _feature("RHOB_RATIO", mode="formula", expression="RHOB / GR", inputs=[{"canonical_name": "RHOB"}, {"canonical_name": "GR"}]),
        _feature("NPHI_LOG", inputs=[{"canonical_name": "NPHI"}]),
    ]

    assert [feature["feature_name"] for feature in filter_feature_library_features(features, mode_filter="formula")] == ["RHOB_RATIO"]
    assert [feature["feature_name"] for feature in filter_feature_library_features(features, mode_filter="operation")] == ["GR_LOG", "NPHI_LOG"]
    assert [feature["feature_name"] for feature in filter_feature_library_features(features, canonical_filter="RHOB")] == ["RHOB_RATIO"]
    assert [feature["feature_name"] for feature in filter_feature_library_features(features, used_in_current_dataset=True)] == ["GR_LOG"]


def test_feature_library_entry_mode_falls_back_to_transform_type():
    assert feature_library_entry_mode({"transform_type": "formula", "params_json": "not json"}) == "formula"
    assert feature_library_entry_mode({"transform_type": "zscore", "params_json": "not json"}) == "operation"


def test_feature_name_conflicts_with_system_column():
    assert "system dataset column" in feature_name_conflict_message(
        "well_id_depth",
        existing_feature_names=set(),
        canonical_names=set(),
    )


def test_feature_name_conflicts_with_canonical_curve():
    assert "canonical curve" in feature_name_conflict_message(
        "GR",
        existing_feature_names=set(),
        canonical_names={"gr"},
    )
