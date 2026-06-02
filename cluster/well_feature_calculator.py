from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

FEATURE_CALCULATOR_CONFIG_VERSION = 1
FEATURE_CALCULATOR_SUPPORTED_MODES = {"operation", "formula"}
FEATURE_CALCULATOR_DEPTH_GRID_POLICY = "strict_equal"
FEATURE_CALCULATOR_INVALID_MATH_POLICY = "block"
FEATURE_CALCULATOR_DEFAULT_OUTLIER_POLICY = "none"
FEATURE_CALCULATOR_DEFAULT_NORMALIZATION_SCOPE = "whole_well"


@dataclass(slots=True)
class FeatureCalculatorError:
    """Structured calculator error returned instead of raising unexpected exceptions."""

    code: str
    message: str
    calculator_id: int | None = None
    feature_name: str | None = None
    well_id: int | None = None
    well_name: str | None = None
    canonical_name: str | None = None
    severity: str = "error"


@dataclass(slots=True)
class FeatureCalculatorInput:
    """Canonical Well Log curve used as an input for a calculated feature."""

    canonical_id: int | None = None
    canonical_name: str | None = None


@dataclass(slots=True)
class FeatureCalculatorConfig:
    """
    Versioned params_json configuration for a global Well Log calculated feature.

    Operation mode example::

        {
          "version": 1,
          "mode": "operation",
          "operation": "zscore",
          "inputs": [{"canonical_id": 12, "canonical_name": "GR"}],
          "normalization_scope": "whole_well",
          "outlier_policy": "none"
        }

    Formula mode example::

        {
          "version": 1,
          "mode": "formula",
          "expression": "(GR - RHOB) / (GR + RHOB)",
          "inputs": [
            {"canonical_id": 12, "canonical_name": "GR"},
            {"canonical_id": 17, "canonical_name": "RHOB"}
          ],
          "depth_grid_policy": "strict_equal",
          "invalid_math_policy": "block",
          "outlier_policy": "none"
        }
    """

    version: int = FEATURE_CALCULATOR_CONFIG_VERSION
    mode: str = "operation"
    inputs: list[FeatureCalculatorInput] = field(default_factory=list)
    operation: str | None = None
    expression: str | None = None
    normalization_scope: str = FEATURE_CALCULATOR_DEFAULT_NORMALIZATION_SCOPE
    depth_grid_policy: str = FEATURE_CALCULATOR_DEPTH_GRID_POLICY
    invalid_math_policy: str = FEATURE_CALCULATOR_INVALID_MATH_POLICY
    outlier_policy: str = FEATURE_CALCULATOR_DEFAULT_OUTLIER_POLICY
    calculator_id: int | None = None
    feature_name: str | None = None
    raw_params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FeatureCalculatorEvaluationResult:
    """Placeholder evaluation result for future calculator stages."""

    values: list[float | None] = field(default_factory=list)
    errors: list[FeatureCalculatorError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(error.severity == "error" for error in self.errors)


def _calculator_attr(calculator: Any, name: str, default: Any = None) -> Any:
    if isinstance(calculator, dict):
        return calculator.get(name, default)
    return getattr(calculator, name, default)


def _error(
    code: str,
    message: str,
    calculator_id: int | None = None,
    feature_name: str | None = None,
    severity: str = "error",
) -> FeatureCalculatorError:
    return FeatureCalculatorError(
        code=code,
        message=message,
        calculator_id=calculator_id,
        feature_name=feature_name,
        severity=severity,
    )


def _parse_inputs(raw_inputs: Any) -> tuple[list[FeatureCalculatorInput], list[FeatureCalculatorError]]:
    errors: list[FeatureCalculatorError] = []
    inputs: list[FeatureCalculatorInput] = []
    if raw_inputs is None:
        return inputs, errors
    if not isinstance(raw_inputs, list):
        return inputs, [_error("invalid_params_json", "Поле inputs должно быть списком.")]

    for index, raw_input in enumerate(raw_inputs):
        if not isinstance(raw_input, dict):
            errors.append(_error("invalid_params_json", f"Элемент inputs[{index}] должен быть объектом."))
            continue
        canonical_id = raw_input.get("canonical_id")
        if canonical_id is not None:
            try:
                canonical_id = int(canonical_id)
            except (TypeError, ValueError):
                errors.append(_error("invalid_params_json", f"canonical_id в inputs[{index}] должен быть целым числом."))
                canonical_id = None
        canonical_name = raw_input.get("canonical_name")
        if canonical_name is not None:
            canonical_name = str(canonical_name).strip() or None
        inputs.append(FeatureCalculatorInput(canonical_id=canonical_id, canonical_name=canonical_name))
    return inputs, errors


def parse_feature_calculator_config(calculator: Any) -> tuple[FeatureCalculatorConfig | None, list[FeatureCalculatorError]]:
    """
    Parse FeatureCalculator.params_json into a versioned config without crashing.

    ``calculator`` may be an ORM object or a dict with FeatureCalculator-like keys.
    The function returns ``(config, errors)``; invalid JSON/schema issues are reported
    as ``FeatureCalculatorError`` entries.
    """
    calculator_id = _calculator_attr(calculator, "id")
    feature_name = _calculator_attr(calculator, "feature_name")
    params_json = _calculator_attr(calculator, "params_json")
    errors: list[FeatureCalculatorError] = []

    if not params_json:
        return None, [_error("empty_params_json", "params_json пустой.", calculator_id, feature_name)]

    try:
        raw_params = json.loads(params_json) if isinstance(params_json, str) else dict(params_json)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return None, [
            _error("invalid_params_json", f"params_json не является корректным JSON: {exc}", calculator_id, feature_name)
        ]

    if not isinstance(raw_params, dict):
        return None, [_error("invalid_params_json", "params_json должен содержать JSON-объект.", calculator_id, feature_name)]

    inputs, input_errors = _parse_inputs(raw_params.get("inputs", []))
    for input_error in input_errors:
        input_error.calculator_id = calculator_id
        input_error.feature_name = feature_name
    errors.extend(input_errors)

    raw_version = raw_params.get("version", FEATURE_CALCULATOR_CONFIG_VERSION) or FEATURE_CALCULATOR_CONFIG_VERSION
    try:
        version = int(raw_version)
    except (TypeError, ValueError):
        version = FEATURE_CALCULATOR_CONFIG_VERSION
        errors.append(_error("invalid_params_json", "Поле version должно быть целым числом.", calculator_id, feature_name))

    config = FeatureCalculatorConfig(
        version=version,
        mode=str(raw_params.get("mode", "operation") or "operation"),
        inputs=inputs,
        operation=raw_params.get("operation") or _calculator_attr(calculator, "transform_type"),
        expression=raw_params.get("expression") or _calculator_attr(calculator, "expression"),
        normalization_scope=str(raw_params.get("normalization_scope", FEATURE_CALCULATOR_DEFAULT_NORMALIZATION_SCOPE) or FEATURE_CALCULATOR_DEFAULT_NORMALIZATION_SCOPE),
        depth_grid_policy=str(raw_params.get("depth_grid_policy", FEATURE_CALCULATOR_DEPTH_GRID_POLICY) or FEATURE_CALCULATOR_DEPTH_GRID_POLICY),
        invalid_math_policy=str(raw_params.get("invalid_math_policy", FEATURE_CALCULATOR_INVALID_MATH_POLICY) or FEATURE_CALCULATOR_INVALID_MATH_POLICY),
        outlier_policy=str(raw_params.get("outlier_policy", FEATURE_CALCULATOR_DEFAULT_OUTLIER_POLICY) or FEATURE_CALCULATOR_DEFAULT_OUTLIER_POLICY),
        calculator_id=int(calculator_id) if calculator_id is not None else None,
        feature_name=str(feature_name) if feature_name else None,
        raw_params=raw_params,
    )
    return config, errors


def validate_feature_calculator_config(config: FeatureCalculatorConfig) -> list[FeatureCalculatorError]:
    """Stage-1 schema validation placeholder for a calculated-feature config."""
    errors: list[FeatureCalculatorError] = []
    if config.version != FEATURE_CALCULATOR_CONFIG_VERSION:
        errors.append(_error("unsupported_config_version", f"Версия конфигурации {config.version} не поддерживается.", config.calculator_id, config.feature_name))
    if config.mode not in FEATURE_CALCULATOR_SUPPORTED_MODES:
        errors.append(_error("unsupported_mode", f"Режим '{config.mode}' не поддерживается.", config.calculator_id, config.feature_name))
    if not config.inputs:
        errors.append(_error("missing_inputs", "Не задан список входных canonical-кривых.", config.calculator_id, config.feature_name))
    if config.mode == "operation" and not config.operation:
        errors.append(_error("missing_operation", "Для mode=operation необходимо поле operation.", config.calculator_id, config.feature_name))
    if config.mode == "formula" and not config.expression:
        errors.append(_error("missing_expression", "Для mode=formula необходимо поле expression.", config.calculator_id, config.feature_name))
    if config.depth_grid_policy != FEATURE_CALCULATOR_DEPTH_GRID_POLICY:
        errors.append(_error("unsupported_depth_grid_policy", "В MVP разрешён только depth_grid_policy=strict_equal.", config.calculator_id, config.feature_name))
    if config.invalid_math_policy != FEATURE_CALCULATOR_INVALID_MATH_POLICY:
        errors.append(_error("unsupported_invalid_math_policy", "В MVP разрешён только invalid_math_policy=block.", config.calculator_id, config.feature_name))
    if config.outlier_policy != FEATURE_CALCULATOR_DEFAULT_OUTLIER_POLICY:
        errors.append(_error("unsupported_outlier_policy", "В MVP разрешён только outlier_policy=none.", config.calculator_id, config.feature_name))
    return errors


def validate_feature_calculator_for_dataset(dataset_id: int, calculator_id: int) -> list[FeatureCalculatorError]:
    """Stage-1 dataset applicability placeholder; real checks are implemented later."""
    return [
        FeatureCalculatorError(
            code="not_implemented",
            message=(
                "Проверка применимости расчётного признака к Well Log dataset "
                f"id={dataset_id} ещё не реализована."
            ),
            calculator_id=int(calculator_id),
            severity="warning",
        )
    ]


def evaluate_feature_calculator_for_well(
    config: FeatureCalculatorConfig,
    well_id: int,
    top_md: float,
    bottom_md: float,
) -> FeatureCalculatorEvaluationResult:
    """Stage-1 calculation placeholder that returns a controlled NotImplemented error."""
    return FeatureCalculatorEvaluationResult(
        errors=[
            FeatureCalculatorError(
                code="not_implemented",
                message=(
                    "Расчёт Well Log признака ещё не реализован "
                    f"для well_id={well_id}, interval=[{top_md:g}, {bottom_md:g}]."
                ),
                calculator_id=config.calculator_id,
                feature_name=config.feature_name,
                well_id=int(well_id),
            )
        ]
    )
