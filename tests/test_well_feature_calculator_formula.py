import importlib.util
import math
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
_MODULE_PATH = Path(__file__).resolve().parents[1] / "cluster" / "well_feature_calculator.py"
_SPEC = importlib.util.spec_from_file_location("well_feature_calculator_formula", _MODULE_PATH)
well_feature_calculator = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = well_feature_calculator
_SPEC.loader.exec_module(well_feature_calculator)

from well_feature_calculator_formula import (
    FeatureCalculatorConfig,
    FeatureCalculatorInput,
    WellLogCurveSeries,
    evaluate_feature_calculator_for_well,
    evaluate_formula_series,
    extract_formula_input_names,
    parse_safe_formula,
    validate_depth_grid_compatibility,
)


def series(name, values, *, step=0.1, depths=None):
    depths = depths or [100.0 + index * step for index in range(len(values))]
    return WellLogCurveSeries(
        well_id=7,
        canonical_id=1,
        canonical_name=name,
        depths=list(depths),
        values=list(values),
        step=step,
        begin=depths[0],
        end=depths[-1],
        source_curve_name=name,
    )


def error_codes(errors):
    return {error.code for error in errors}


def assert_close_list(actual, expected, *, abs_tol=1e-9):
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected):
        assert math.isclose(actual_value, expected_value, abs_tol=abs_tol)


def test_extract_formula_input_names_ignores_allowed_functions():
    names, errors = extract_formula_input_names("log(GR) + sqrt(RHOB) - abs(GR)")

    assert errors == []
    assert names == ["GR", "RHOB"]

def test_formula_adds_two_curves():
    values, errors = evaluate_formula_series("A + B", [series("A", [1.0, 2.0]), series("B", [3.0, 4.0])])

    assert errors == []
    assert_close_list(values, [4.0, 6.0])


def test_formula_normalized_difference():
    values, errors = evaluate_formula_series(
        "(A - B) / (A + B)",
        [series("A", [3.0, 6.0]), series("B", [1.0, 2.0])],
    )

    assert errors == []
    assert_close_list(values, [0.5, 0.5])


def test_formula_blocks_division_by_zero():
    values, errors = evaluate_formula_series("A / B", [series("A", [1.0]), series("B", [0.0])])

    assert values == []
    assert error_codes(errors) == {"division_by_zero"}


def test_formula_blocks_log_negative_value():
    values, errors = evaluate_formula_series("log(A)", [series("A", [-1.0])])

    assert values == []
    assert error_codes(errors) == {"invalid_log_domain"}


def test_formula_blocks_unknown_curve_name():
    formula, errors = parse_safe_formula("A + C", {"A", "B"})

    assert formula is None
    assert error_codes(errors) == {"formula_validation_error"}


def test_formula_blocks_unknown_function():
    formula, errors = parse_safe_formula("sin(A)", {"A"})

    assert formula is None
    assert error_codes(errors) == {"formula_validation_error"}


def test_formula_blocks_attribute_access():
    formula, errors = parse_safe_formula("A.real", {"A"})

    assert formula is None
    assert error_codes(errors) == {"formula_validation_error"}


def test_depth_grid_mismatch_on_different_steps():
    config = FeatureCalculatorConfig(
        mode="formula",
        expression="A + B",
        feature_name="sum_ab",
        inputs=[FeatureCalculatorInput(canonical_name="A"), FeatureCalculatorInput(canonical_name="B")],
    )

    errors = validate_depth_grid_compatibility(
        [series("A", [1.0, 2.0], step=0.1), series("B", [1.0, 2.0], step=0.2)],
        config=config,
    )

    assert error_codes(errors) == {"depth_grid_mismatch"}
    assert "Интерполяция" in errors[0].message


def test_formula_evaluation_uses_loaded_series_and_strict_grid(monkeypatch):
    config = FeatureCalculatorConfig(
        mode="formula",
        expression="A + B",
        inputs=[FeatureCalculatorInput(canonical_name="A"), FeatureCalculatorInput(canonical_name="B")],
    )

    def fake_loader(config, well_id, top_md, bottom_md, *, db_session=None):
        return [series("A", [1.0, 2.0]), series("B", [3.0, 4.0])], []

    monkeypatch.setattr(well_feature_calculator, "load_input_series_for_calculator", fake_loader)
    result = evaluate_feature_calculator_for_well(config, well_id=7, top_md=100.0, bottom_md=100.1)

    assert result.ok
    assert_close_list(result.values, [4.0, 6.0])
    assert_close_list(result.depths, [100.0, 100.1])


def test_complete_calculator_depths_uses_calculator_intersection_not_canonical_union():
    complete = well_feature_calculator.complete_calculator_depths(
        {
            100.0: {1: 10.0},
            100.1: {1: 11.0, 2: 21.0},
            100.2: {1: 12.0, 2: 22.0},
            100.3: {2: 23.0},
        },
        [1, 2],
    )

    assert complete == [100.1, 100.2]
