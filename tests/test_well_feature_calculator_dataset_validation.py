import importlib.util
import json
from pathlib import Path
import sys
import types

sys.path.append(str(Path(__file__).resolve().parents[1]))
_MODULE_PATH = Path(__file__).resolve().parents[1] / "cluster" / "well_feature_calculator.py"
_SPEC = importlib.util.spec_from_file_location("well_feature_calculator_dataset", _MODULE_PATH)
well_feature_calculator = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = well_feature_calculator
_SPEC.loader.exec_module(well_feature_calculator)

from well_feature_calculator_dataset import validate_calculators_for_dataset


class Field:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return ("eq", self.name, other)

    def in_(self, values):
        return ("in", self.name, values)


class Query:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, *args, **kwargs):
        return self

    def join(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def all(self):
        return list(self.rows)

    def first(self):
        return self.rows[0] if self.rows else None


class FakeSession:
    def __init__(self, *, values):
        self.canonical = types.SimpleNamespace(id=1, canonical_name="GR", canonical_name_norm="gr")
        self.well = types.SimpleNamespace(id=7, name="Well-1")
        self.well_for_cluster = types.SimpleNamespace(
            dataset_id=3,
            well_id=self.well.id,
            top_md=100.0,
            bottom_md=100.2,
            well=self.well,
        )
        self.well_log = types.SimpleNamespace(
            well_id=self.well.id,
            curve_name="GR",
            curve_data=json.dumps(values),
            begin=100.0,
            end=100.2,
            step=0.1,
        )
        self.calculator = types.SimpleNamespace(
            id=11,
            feature_name="LOG_GR",
            expression=None,
            transform_type="log",
            params_json=json.dumps(
                {
                    "version": 1,
                    "mode": "operation",
                    "operation": "log",
                    "inputs": [{"canonical_id": 1, "canonical_name": "GR"}],
                    "normalization_scope": "whole_well",
                    "outlier_policy": "none",
                }
            ),
        )

    def query(self, *entities):
        first = entities[0]
        name = getattr(first, "__name__", None)
        if name == "WellForCluster":
            return Query([self.well_for_cluster])
        if name == "FeatureCalculator":
            return Query([self.calculator])
        if name == "CanonicalWellLog":
            return Query([self.canonical])
        if name == "WellLog":
            return Query([self.well_log])
        # AliasWellLog.alias_name_norm projection.
        if getattr(first, "name", None) == "alias_name_norm":
            return Query([("gr",)])
        return Query([])


def install_fake_model_modules(fake_session, monkeypatch):
    model = types.ModuleType("models_db.model")
    model.session = fake_session

    class WellLog:
        well_id = Field("well_id")

    model.WellLog = WellLog

    model_cluster = types.ModuleType("models_db.model_cluster")

    class WellForCluster:
        dataset_id = Field("dataset_id")
        well_id = Field("well_id")

    class ClusterWellLogParameterFromCalculator:
        dataset_id = Field("dataset_id")
        calculator_id = Field("calculator_id")

    class FeatureCalculator:
        id = Field("id")
        feature_name = Field("feature_name")

    class CanonicalWellLog:
        id = Field("id")
        canonical_name_norm = Field("canonical_name_norm")

    class AliasWellLog:
        alias_name_norm = Field("alias_name_norm")
        canonical_id = Field("canonical_id")

    model_cluster.WellForCluster = WellForCluster
    model_cluster.ClusterWellLogParameterFromCalculator = ClusterWellLogParameterFromCalculator
    model_cluster.FeatureCalculator = FeatureCalculator
    model_cluster.CanonicalWellLog = CanonicalWellLog
    model_cluster.AliasWellLog = AliasWellLog

    package = types.ModuleType("models_db")
    monkeypatch.setitem(sys.modules, "models_db", package)
    monkeypatch.setitem(sys.modules, "models_db.model", model)
    monkeypatch.setitem(sys.modules, "models_db.model_cluster", model_cluster)


def test_validate_calculators_for_dataset_accepts_valid_selected_calculator(monkeypatch):
    fake_session = FakeSession(values=[1.0, 2.0, 3.0])
    install_fake_model_modules(fake_session, monkeypatch)

    errors = validate_calculators_for_dataset(3, db_session=fake_session)

    assert errors == []


def test_validate_calculators_for_dataset_reports_preflight_math_error(monkeypatch):
    fake_session = FakeSession(values=[1.0, 0.0, 3.0])
    install_fake_model_modules(fake_session, monkeypatch)

    errors = validate_calculators_for_dataset(3, db_session=fake_session)

    assert {error.code for error in errors} == {"invalid_math_domain"}
    assert errors[0].feature_name == "LOG_GR"
    assert errors[0].well_name == "Well-1"


def test_build_feature_calculator_coverage_report_summarizes_ok_and_blocks_errors(monkeypatch):
    fake_session = FakeSession(values=[1.0, 2.0, 3.0])
    install_fake_model_modules(fake_session, monkeypatch)
    config, errors = well_feature_calculator.parse_feature_calculator_config(fake_session.calculator)

    assert errors == []
    report = well_feature_calculator.build_feature_calculator_coverage_report(3, config, db_session=fake_session)

    assert report.summary.wells_total == 1
    assert report.summary.wells_ok == 1
    assert report.summary.error_count == 0
    assert report.summary.min_points == 3
    assert report.summary.max_points == 3
    assert report.summary.input_curves == ["GR"]
    assert report.summary.can_be_added is True
    assert report.rows[0].status == "OK"
    assert report.rows[0].points == 3

    failing_session = FakeSession(values=[1.0, 0.0, 3.0])
    install_fake_model_modules(failing_session, monkeypatch)
    failing_config, errors = well_feature_calculator.parse_feature_calculator_config(failing_session.calculator)

    assert errors == []
    failing_report = well_feature_calculator.build_feature_calculator_coverage_report(3, failing_config, db_session=failing_session)

    assert failing_report.summary.wells_total == 1
    assert failing_report.summary.wells_ok == 0
    assert failing_report.summary.error_count == 1
    assert failing_report.summary.can_be_added is False
    assert failing_report.rows[0].status == "Invalid math domain"
    assert failing_report.rows[0].points == 0
    assert failing_report.rows[0].errors[0].well_name == "Well-1"
