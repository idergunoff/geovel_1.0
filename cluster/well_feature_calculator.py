from __future__ import annotations

import ast
import json
import math
import operator
import statistics
from dataclasses import dataclass, field
from typing import Any, Sequence

FEATURE_CALCULATOR_CONFIG_VERSION = 1
FEATURE_CALCULATOR_SUPPORTED_MODES = {"operation", "formula"}
FEATURE_CALCULATOR_DEPTH_GRID_POLICY = "strict_equal"
FEATURE_CALCULATOR_INVALID_MATH_POLICY = "block"
FEATURE_CALCULATOR_DEFAULT_OUTLIER_POLICY = "none"
FEATURE_CALCULATOR_DEFAULT_NORMALIZATION_SCOPE = "whole_well"
FEATURE_CALCULATOR_NORMALIZATION_SCOPES = {"whole_well", "interval"}
FEATURE_CALCULATOR_DEPTH_PRECISION = 6
FEATURE_CALCULATOR_STEP_TOLERANCE = 1e-9
FEATURE_CALCULATOR_ALLOWED_FORMULA_FUNCTIONS = {"log", "abs", "sqrt"}



FEATURE_CALCULATOR_STATUS_LABELS = {
    "missing_curve": "Missing curve",
    "depth_grid_mismatch": "Depth grid mismatch",
    "invalid_math_domain": "Invalid math domain",
    "invalid_log_domain": "Invalid math domain",
    "invalid_sqrt_domain": "Invalid math domain",
    "not_enough_points": "Not enough points",
    "non_finite_result": "Non-finite result",
    "no_common_calculator_depths": "No common calculator depths",
    "no_common_formula_depths": "No common formula depths",
}

FEATURE_CALCULATOR_RECOMMENDATIONS = {
    "missing_curve": "Проверьте, что нужная canonical-кривая или её alias загружены во всех скважинах текущего dataset.",
    "depth_grid_mismatch": "Проверьте, что входные кривые загружены с одинаковым шагом и одинаковой глубинной сеткой. Интерполяция в текущей версии калькулятора не выполняется.",
    "invalid_math_domain": "Проверьте область определения операции: log требует X > 0, sqrt требует X >= 0.",
    "invalid_log_domain": "Проверьте область определения log: все значения должны быть больше нуля.",
    "invalid_sqrt_domain": "Проверьте область определения sqrt: все значения должны быть неотрицательными.",
    "division_by_zero": "Измените формулу или входные кривые так, чтобы знаменатель не обращался в ноль на выбранном интервале.",
    "not_enough_points": "Расширьте интервал или выберите кривую, где достаточно точек для операции.",
    "non_finite_result": "Проверьте входные значения и параметры операции: результат не должен содержать NaN или inf.",
    "no_common_calculator_depths": "Проверьте выбранные интервалы и глубинные сетки calculator-признаков: для COLLECT нужна хотя бы одна глубина, где рассчитаны все выбранные calculator-признаки.",
    "no_common_formula_depths": "Проверьте выбранные интервалы и глубинные сетки входных кривых формулы: нужна хотя бы одна общая глубина без интерполяции.",
}

FEATURE_CALCULATOR_UNARY_OPERATIONS = {
    "log",
    "abs",
    "sqrt",
    "derivative",
    "rolling_mean",
    "rolling_median",
    "zscore",
    "robust",
    "minmax",
}


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
class WellLogCurveSeries:
    """One canonical curve resolved to a concrete WellLog curve in one well."""

    well_id: int
    canonical_id: int
    canonical_name: str
    depths: list[float]
    values: list[float | None]
    step: float
    begin: float
    end: float
    source_curve_name: str
    stats_depths: list[float] = field(default_factory=list)
    stats_values: list[float | None] = field(default_factory=list)


@dataclass(slots=True)
class FeatureCalculatorCoverageRow:
    """Dataset coverage diagnostics for one well and one calculator feature."""

    well_id: int
    well_name: str | None
    top_md: float
    bottom_md: float
    status: str
    points: int = 0
    errors: list[FeatureCalculatorError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and self.status == "OK"


@dataclass(slots=True)
class FeatureCalculatorCoverageSummary:
    """Aggregated applicability summary for a calculator feature in one dataset."""

    dataset_id: int
    calculator_id: int | None
    feature_name: str | None
    wells_total: int
    wells_ok: int
    error_count: int
    min_points: int | None
    max_points: int | None
    input_curves: list[str]
    mode: str
    normalization_scope: str

    @property
    def can_be_added(self) -> bool:
        return self.wells_total > 0 and self.error_count == 0 and self.wells_ok == self.wells_total


@dataclass(slots=True)
class FeatureCalculatorCoverageReport:
    """Full dataset-level diagnostics report for one calculator feature."""

    summary: FeatureCalculatorCoverageSummary
    rows: list[FeatureCalculatorCoverageRow] = field(default_factory=list)


@dataclass(slots=True)
class FeatureCalculatorEvaluationResult:
    """Calculation result for one well and one calculated Well Log feature."""

    values: list[float | None] = field(default_factory=list)
    depths: list[float] = field(default_factory=list)
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
    well_id: int | None = None,
    well_name: str | None = None,
    canonical_name: str | None = None,
) -> FeatureCalculatorError:
    return FeatureCalculatorError(
        code=code,
        message=message,
        calculator_id=calculator_id,
        feature_name=feature_name,
        well_id=well_id,
        well_name=well_name,
        canonical_name=canonical_name,
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
    """Validate a calculated-feature config supported by the current MVP stages."""
    errors: list[FeatureCalculatorError] = []
    if config.version != FEATURE_CALCULATOR_CONFIG_VERSION:
        errors.append(_error("unsupported_config_version", f"Версия конфигурации {config.version} не поддерживается.", config.calculator_id, config.feature_name))
    if config.mode not in FEATURE_CALCULATOR_SUPPORTED_MODES:
        errors.append(_error("unsupported_mode", f"Режим '{config.mode}' не поддерживается.", config.calculator_id, config.feature_name))
    if not config.inputs:
        errors.append(_error("missing_inputs", "Не задан список входных canonical-кривых.", config.calculator_id, config.feature_name))
    if config.mode == "operation":
        if not config.operation:
            errors.append(_error("missing_operation", "Для mode=operation необходимо поле operation.", config.calculator_id, config.feature_name))
        elif str(config.operation).lower() not in FEATURE_CALCULATOR_UNARY_OPERATIONS:
            errors.append(_error("unsupported_operation", f"Операция '{config.operation}' не поддерживается для unary backend.", config.calculator_id, config.feature_name))
        if len(config.inputs) != 1:
            errors.append(_error("invalid_inputs", "Операции этапа 2 поддерживают ровно одну входную canonical-кривую.", config.calculator_id, config.feature_name))
    if config.mode == "formula" and not config.expression:
        errors.append(_error("missing_expression", "Для mode=formula необходимо поле expression.", config.calculator_id, config.feature_name))
    if config.normalization_scope not in FEATURE_CALCULATOR_NORMALIZATION_SCOPES:
        errors.append(_error("unsupported_normalization_scope", "Разрешены только normalization_scope=whole_well или interval.", config.calculator_id, config.feature_name))
    if config.depth_grid_policy != FEATURE_CALCULATOR_DEPTH_GRID_POLICY:
        errors.append(_error("unsupported_depth_grid_policy", "В MVP разрешён только depth_grid_policy=strict_equal.", config.calculator_id, config.feature_name))
    if config.invalid_math_policy != FEATURE_CALCULATOR_INVALID_MATH_POLICY:
        errors.append(_error("unsupported_invalid_math_policy", "В MVP разрешён только invalid_math_policy=block.", config.calculator_id, config.feature_name))
    if config.outlier_policy != FEATURE_CALCULATOR_DEFAULT_OUTLIER_POLICY:
        errors.append(_error("unsupported_outlier_policy", "В MVP разрешён только outlier_policy=none.", config.calculator_id, config.feature_name))
    return errors


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _blocking_if_any_missing(values: Sequence[float | None], code: str, message: str) -> FeatureCalculatorError | None:
    if any(value is None or not math.isfinite(value) for value in values):
        return _error(code, message)
    return None


def _non_finite_result_error(values: Sequence[float | None]) -> FeatureCalculatorError | None:
    if any(value is None or not math.isfinite(value) for value in values):
        return _error("non_finite_result", "Результат содержит NaN/inf или пустые значения; расчёт признака заблокирован.")
    return None


def _copy_error_context(
    error: FeatureCalculatorError,
    *,
    config: FeatureCalculatorConfig | None = None,
    series: WellLogCurveSeries | None = None,
    well_id: int | None = None,
    canonical_name: str | None = None,
) -> FeatureCalculatorError:
    return FeatureCalculatorError(
        code=error.code,
        message=error.message,
        calculator_id=config.calculator_id if config else error.calculator_id,
        feature_name=config.feature_name if config else error.feature_name,
        well_id=series.well_id if series else well_id or error.well_id,
        well_name=error.well_name,
        canonical_name=series.canonical_name if series else canonical_name or error.canonical_name,
        severity=error.severity,
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[int(position)]
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def _quartiles(values: Sequence[float]) -> tuple[float, float]:
    return _percentile(values, 0.25), _percentile(values, 0.75)


def apply_log(values: Sequence[float | None]) -> tuple[list[float | None], list[FeatureCalculatorError]]:
    if any(value is None or not math.isfinite(value) or value <= 0 for value in values):
        return [], [_error("invalid_math_domain", "log(X) определён только для положительных конечных значений.")]
    result = [math.log(float(value)) for value in values]
    return result, []


def apply_abs(values: Sequence[float | None]) -> tuple[list[float | None], list[FeatureCalculatorError]]:
    source_error = _blocking_if_any_missing(values, "non_finite_result", "abs(X) требует конечные значения входной кривой.")
    if source_error:
        return [], [source_error]
    return [abs(float(value)) for value in values], []


def apply_sqrt(values: Sequence[float | None]) -> tuple[list[float | None], list[FeatureCalculatorError]]:
    if any(value is None or not math.isfinite(value) or value < 0 for value in values):
        return [], [_error("invalid_math_domain", "sqrt(X) определён только для неотрицательных конечных значений.")]
    return [math.sqrt(float(value)) for value in values], []


def apply_derivative(values: Sequence[float | None], step: float) -> tuple[list[float | None], list[FeatureCalculatorError]]:
    if step <= 0 or not math.isfinite(step):
        return [], [_error("invalid_step", "Для производной требуется положительный конечный шаг глубины.")]
    source_error = _blocking_if_any_missing(values, "non_finite_result", "dX/dz требует конечные значения входной кривой.")
    if source_error:
        return [], [source_error]
    if len(values) < 2:
        return [], [_error("not_enough_points", "Для производной требуется минимум две точки.")]
    numeric = [float(value) for value in values]
    if len(numeric) == 2:
        slope = (numeric[1] - numeric[0]) / step
        return [slope, slope], []
    result: list[float] = [(numeric[1] - numeric[0]) / step]
    result.extend((numeric[index + 1] - numeric[index - 1]) / (2 * step) for index in range(1, len(numeric) - 1))
    result.append((numeric[-1] - numeric[-2]) / step)
    return result, []


def _validate_window(window: Any, values_count: int) -> tuple[int | None, FeatureCalculatorError | None]:
    try:
        window_size = int(window)
    except (TypeError, ValueError):
        return None, _error("invalid_window", "Параметр window должен быть целым числом.")
    if window_size < 1:
        return None, _error("invalid_window", "Параметр window должен быть больше нуля.")
    if values_count < window_size:
        return None, _error("not_enough_points", "Недостаточно точек для выбранного окна сглаживания.")
    return window_size, None


def apply_rolling_mean(values: Sequence[float | None], window: int = 3) -> tuple[list[float | None], list[FeatureCalculatorError]]:
    source_error = _blocking_if_any_missing(values, "non_finite_result", "rolling_mean требует конечные значения входной кривой.")
    if source_error:
        return [], [source_error]
    window_size, window_error = _validate_window(window, len(values))
    if window_error:
        return [], [window_error]
    numeric = [float(value) for value in values]
    half_window = window_size // 2
    result = []
    for index in range(len(numeric)):
        start = max(0, index - half_window)
        end = min(len(numeric), start + window_size)
        start = max(0, end - window_size)
        result.append(sum(numeric[start:end]) / (end - start))
    return result, []


def apply_rolling_median(values: Sequence[float | None], window: int = 3) -> tuple[list[float | None], list[FeatureCalculatorError]]:
    source_error = _blocking_if_any_missing(values, "non_finite_result", "rolling_median требует конечные значения входной кривой.")
    if source_error:
        return [], [source_error]
    window_size, window_error = _validate_window(window, len(values))
    if window_error:
        return [], [window_error]
    numeric = [float(value) for value in values]
    half_window = window_size // 2
    result = []
    for index in range(len(numeric)):
        start = max(0, index - half_window)
        end = min(len(numeric), start + window_size)
        start = max(0, end - window_size)
        result.append(float(statistics.median(numeric[start:end])))
    return result, []


def _finite_values(values: Sequence[float | None]) -> list[float]:
    """Return finite values, silently ignoring gaps outside the calculation interval."""
    return [float(value) for value in values if value is not None and math.isfinite(value)]


def apply_zscore(values: Sequence[float | None], stats_values: Sequence[float | None] | None = None) -> tuple[list[float | None], list[FeatureCalculatorError]]:
    source = _finite_values(values if stats_values is None else stats_values)
    source_error = _blocking_if_any_missing(values, "non_finite_result", "zscore требует конечные значения расчётного интервала.")
    if source_error:
        return [], [source_error]
    if len(source) < 2:
        return [], [_error("not_enough_points", "Для zscore требуется минимум две конечные точки в области нормировки.")]
    mean = sum(source) / len(source)
    variance = sum((value - mean) ** 2 for value in source) / len(source)
    std = math.sqrt(variance)
    if std == 0:
        return [], [_error("zero_std", "Стандартное отклонение равно нулю; zscore невозможен.")]
    return [(float(value) - mean) / std for value in values], []


def apply_robust(values: Sequence[float | None], stats_values: Sequence[float | None] | None = None) -> tuple[list[float | None], list[FeatureCalculatorError]]:
    source = _finite_values(values if stats_values is None else stats_values)
    source_error = _blocking_if_any_missing(values, "non_finite_result", "robust требует конечные значения расчётного интервала.")
    if source_error:
        return [], [source_error]
    if len(source) < 2:
        return [], [_error("not_enough_points", "Для robust-нормировки требуется минимум две конечные точки в области нормировки.")]
    median_value = statistics.median(source)
    q1, q3 = _quartiles(source)
    iqr = q3 - q1
    if iqr == 0:
        return [], [_error("zero_iqr", "IQR равен нулю; robust-нормировка невозможна.")]
    return [(float(value) - median_value) / iqr for value in values], []


def apply_minmax(values: Sequence[float | None], stats_values: Sequence[float | None] | None = None) -> tuple[list[float | None], list[FeatureCalculatorError]]:
    source = _finite_values(values if stats_values is None else stats_values)
    source_error = _blocking_if_any_missing(values, "non_finite_result", "minmax требует конечные значения расчётного интервала.")
    if source_error:
        return [], [source_error]
    if len(source) < 2:
        return [], [_error("not_enough_points", "Для minmax требуется минимум две конечные точки в области нормировки.")]
    min_value = min(source)
    max_value = max(source)
    value_range = max_value - min_value
    if value_range == 0:
        return [], [_error("zero_range", "Диапазон значений равен нулю; minmax невозможен.")]
    return [(float(value) - min_value) / value_range for value in values], []


def apply_unary_operation(
    operation: str,
    values: Sequence[float | None],
    *,
    step: float | None = None,
    window: int = 3,
    stats_values: Sequence[float | None] | None = None,
) -> tuple[list[float | None], list[FeatureCalculatorError]]:
    """Apply a stage-2 unary calculator operation as a pure function."""
    normalized_operation = str(operation or "").strip().lower()
    if normalized_operation == "log":
        result = apply_log(values)
    elif normalized_operation == "abs":
        result = apply_abs(values)
    elif normalized_operation == "sqrt":
        result = apply_sqrt(values)
    elif normalized_operation == "derivative":
        result = apply_derivative(values, float(step or 0))
    elif normalized_operation == "rolling_mean":
        result = apply_rolling_mean(values, window)
    elif normalized_operation == "rolling_median":
        result = apply_rolling_median(values, window)
    elif normalized_operation == "zscore":
        result = apply_zscore(values, stats_values)
    elif normalized_operation == "robust":
        result = apply_robust(values, stats_values)
    elif normalized_operation == "minmax":
        result = apply_minmax(values, stats_values)
    else:
        return [], [_error("unsupported_operation", f"Операция '{operation}' не поддерживается.")]

    result_values, errors = result
    if errors:
        return result_values, errors
    non_finite_error = _non_finite_result_error(result_values)
    if non_finite_error:
        return [], [non_finite_error]
    return result_values, []


_FORMULA_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
}
_FORMULA_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class _FormulaCalculationError(Exception):
    """Expected point-level formula calculation error with a stable error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class _SafeFormulaValidator(ast.NodeVisitor):
    """Validate a formula AST against the stage-3 formula whitelist."""

    def __init__(self, allowed_names: set[str]) -> None:
        self.allowed_names = allowed_names
        self.errors: list[FeatureCalculatorError] = []

    def _add_validation_error(self, message: str) -> None:
        self.errors.append(_error("formula_validation_error", message))

    def generic_visit(self, node: ast.AST) -> None:
        self._add_validation_error(f"Недопустимый элемент формулы: {type(node).__name__}.")

    def visit_Expression(self, node: ast.Expression) -> None:
        self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            self._add_validation_error("В формулах разрешены только числовые константы.")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in self.allowed_names:
            self._add_validation_error(f"Имя '{node.id}' не входит в список inputs расчётного признака.")

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if type(node.op) not in _FORMULA_BINARY_OPERATORS and not isinstance(node.op, ast.Div):
            self._add_validation_error(f"Оператор {type(node.op).__name__} не разрешён; доступны только +, -, *, /.")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if type(node.op) not in _FORMULA_UNARY_OPERATORS:
            self._add_validation_error("Разрешены только unary + и unary -.")
        self.visit(node.operand)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            self._add_validation_error("Вызовы атрибутов и выражений запрещены; доступны только log, abs, sqrt.")
        elif node.func.id not in FEATURE_CALCULATOR_ALLOWED_FORMULA_FUNCTIONS:
            self._add_validation_error(f"Функция '{node.func.id}' не разрешена; доступны только log, abs, sqrt.")
        if node.keywords:
            self._add_validation_error("Именованные аргументы функций в формулах не поддерживаются.")
        if len(node.args) != 1:
            self._add_validation_error("Функции log/abs/sqrt принимают ровно один аргумент.")
        for arg in node.args:
            self.visit(arg)


@dataclass(slots=True)
class SafeFormulaExpression:
    """Parsed and validated expression that can be evaluated without eval/builtins."""

    expression: str
    tree: ast.Expression
    allowed_names: set[str]


def extract_formula_input_names(expression: str) -> tuple[list[str], list[FeatureCalculatorError]]:
    """Return unique curve names referenced by a formula, excluding allowed function names."""
    if not str(expression or "").strip():
        return [], [_error("formula_parse_error", "Формула пуста.")]
    try:
        tree = ast.parse(str(expression), mode="eval")
    except SyntaxError as exc:
        return [], [_error("formula_parse_error", f"Синтаксическая ошибка формулы: {exc.msg}.")]
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in FEATURE_CALCULATOR_ALLOWED_FORMULA_FUNCTIONS and node.id not in names:
            names.append(node.id)
    return names, []

def parse_safe_formula(expression: str, allowed_names: set[str]) -> tuple[SafeFormulaExpression | None, list[FeatureCalculatorError]]:
    """Parse and validate formula expression using AST only, never eval."""
    if not str(expression or "").strip():
        return None, [_error("formula_parse_error", "Формула пуста.")]
    try:
        tree = ast.parse(str(expression), mode="eval")
    except SyntaxError as exc:
        return None, [_error("formula_parse_error", f"Синтаксическая ошибка формулы: {exc.msg}.")]
    validator = _SafeFormulaValidator(allowed_names)
    validator.visit(tree)
    if validator.errors:
        return None, validator.errors
    return SafeFormulaExpression(expression=str(expression), tree=tree, allowed_names=set(allowed_names)), []


def _evaluate_safe_formula_node(node: ast.AST, values_by_name: dict[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate_safe_formula_node(node.body, values_by_name)
    if isinstance(node, ast.Constant):
        value = float(node.value)
    elif isinstance(node, ast.Name):
        if node.id not in values_by_name:
            raise _FormulaCalculationError("missing_value_at_depth", f"Нет значения входной кривой '{node.id}' на текущей глубине.")
        value = values_by_name[node.id]
    elif isinstance(node, ast.BinOp):
        left = _evaluate_safe_formula_node(node.left, values_by_name)
        right = _evaluate_safe_formula_node(node.right, values_by_name)
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise _FormulaCalculationError("division_by_zero", "Деление на ноль в формуле расчётного признака.")
            value = left / right
        else:
            value = _FORMULA_BINARY_OPERATORS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = _evaluate_safe_formula_node(node.operand, values_by_name)
        value = _FORMULA_UNARY_OPERATORS[type(node.op)](operand)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        argument = _evaluate_safe_formula_node(node.args[0], values_by_name)
        if node.func.id == "log":
            if argument <= 0:
                raise _FormulaCalculationError("invalid_log_domain", "log(x) определён только для x > 0.")
            value = math.log(argument)
        elif node.func.id == "sqrt":
            if argument < 0:
                raise _FormulaCalculationError("invalid_sqrt_domain", "sqrt(x) определён только для x >= 0.")
            value = math.sqrt(argument)
        elif node.func.id == "abs":
            value = abs(argument)
        else:
            raise _FormulaCalculationError("formula_validation_error", f"Функция '{node.func.id}' не разрешена.")
    else:
        raise _FormulaCalculationError("formula_validation_error", f"Недопустимый элемент формулы: {type(node).__name__}.")

    if not math.isfinite(value):
        raise _FormulaCalculationError("non_finite_result", "Результат формулы содержит NaN/inf; расчёт признака заблокирован.")
    return value


def evaluate_safe_formula_point(
    formula: SafeFormulaExpression,
    values_by_name: dict[str, float | None],
) -> tuple[float | None, FeatureCalculatorError | None]:
    """Evaluate a parsed formula for one depth point and return a blocking error on failure."""
    finite_values: dict[str, float] = {}
    for name in formula.allowed_names:
        value = values_by_name.get(name)
        if value is None or not math.isfinite(value):
            return None, _error("missing_value_at_depth", f"На текущей глубине отсутствует конечное значение входной кривой '{name}'.")
        finite_values[name] = float(value)
    try:
        return _evaluate_safe_formula_node(formula.tree, finite_values), None
    except _FormulaCalculationError as exc:
        return None, _error(exc.code, exc.message)


def evaluate_formula_series(
    expression: str,
    series_list: Sequence[WellLogCurveSeries],
) -> tuple[list[float | None], list[FeatureCalculatorError]]:
    """Evaluate formula for aligned input series; any point error blocks the whole feature."""
    if not series_list:
        return [], [_error("missing_inputs", "Для расчёта формулы нужны входные canonical-кривые.")]
    names = [series.canonical_name for series in series_list]
    if len(names) != len(set(names)):
        return [], [_error("formula_validation_error", "Имена входных canonical-кривых формулы должны быть уникальными.")]
    formula, errors = parse_safe_formula(expression, set(names))
    if errors or formula is None:
        return [], errors

    result_values: list[float | None] = []
    for index in range(len(series_list[0].depths)):
        values_by_name = {series.canonical_name: series.values[index] for series in series_list}
        value, point_error = evaluate_safe_formula_point(formula, values_by_name)
        if point_error:
            depth = series_list[0].depths[index]
            point_error.message = f"{point_error.message} Глубина: {depth:g}."
            return [], [point_error]
        result_values.append(value)
    non_finite_error = _non_finite_result_error(result_values)
    if non_finite_error:
        return [], [non_finite_error]
    return result_values, []



def align_formula_series_on_common_depths(
    series_list: Sequence[WellLogCurveSeries],
    *,
    precision: int = FEATURE_CALCULATOR_DEPTH_PRECISION,
) -> tuple[list[WellLogCurveSeries], list[FeatureCalculatorError]]:
    """Align formula inputs by common rounded depths without interpolation.

    Clean canonical curves are collected by rounded depth, so formula features should
    be calculated on the same common-depth principle instead of failing when input
    curves have harmless extra edge points or different start/end depths.
    """
    if not series_list:
        return [], [_error("missing_inputs", "Для расчёта формулы нужны входные canonical-кривые.")]

    depth_maps: list[dict[float, tuple[float, float | None]]] = []
    for series in series_list:
        depth_map: dict[float, tuple[float, float | None]] = {}
        for depth, value in zip(series.depths, series.values):
            rounded_depth = round(float(depth), precision)
            depth_map.setdefault(rounded_depth, (float(depth), value))
        if not depth_map:
            return [], [
                _error(
                    "no_common_formula_depths",
                    f"У входной кривой '{series.canonical_name}' нет точек в расчётном интервале.",
                    well_id=series.well_id,
                    canonical_name=series.canonical_name,
                )
            ]
        depth_maps.append(depth_map)

    common_depths = sorted(set.intersection(*(set(depth_map) for depth_map in depth_maps)))
    if not common_depths:
        input_names = ", ".join(series.canonical_name for series in series_list)
        return [], [
            _error(
                "no_common_formula_depths",
                f"Нет общих глубин для входных кривых формулы: {input_names}. Формула рассчитывается только на совпадающих глубинах без интерполяции.",
                well_id=series_list[0].well_id,
            )
        ]

    aligned: list[WellLogCurveSeries] = []
    for series, depth_map in zip(series_list, depth_maps):
        aligned_values = [depth_map[depth][1] for depth in common_depths]
        aligned.append(
            WellLogCurveSeries(
                well_id=series.well_id,
                canonical_id=series.canonical_id,
                canonical_name=series.canonical_name,
                depths=[float(depth) for depth in common_depths],
                values=aligned_values,
                step=series.step,
                begin=float(common_depths[0]),
                end=float(common_depths[-1]),
                source_curve_name=series.source_curve_name,
                stats_depths=list(series.stats_depths),
                stats_values=list(series.stats_values),
            )
        )
    return aligned, []

def complete_calculator_depths(
    calculator_value_map: dict[float, dict[int, Any]],
    calculator_ids: Sequence[int],
) -> list[float]:
    """Return depths where every selected calculator has a computed value.

    COLLECT builds rows on a depth grid. Canonical curves may have extra edge
    depths or slightly different grids, so calculator-backed features should not
    be required on the union of all canonical depths.  A calculator row is
    complete only where all selected calculator ids are present.
    """
    normalized_ids = [int(calculator_id) for calculator_id in calculator_ids]
    if not normalized_ids:
        return []
    return sorted(
        float(depth)
        for depth, values_by_calculator in calculator_value_map.items()
        if all(calculator_id in values_by_calculator for calculator_id in normalized_ids)
    )


def _rounded_depths(depths: Sequence[float], precision: int = FEATURE_CALCULATOR_DEPTH_PRECISION) -> list[float]:
    return [round(float(depth), precision) for depth in depths]


def validate_depth_grid_compatibility(
    series_list: Sequence[WellLogCurveSeries],
    *,
    config: FeatureCalculatorConfig | None = None,
    precision: int = FEATURE_CALCULATOR_DEPTH_PRECISION,
    step_tolerance: float = FEATURE_CALCULATOR_STEP_TOLERANCE,
) -> list[FeatureCalculatorError]:
    """Strictly verify that formula inputs share one depth grid; no interpolation is used."""
    if len(series_list) <= 1:
        return []
    feature_name = config.feature_name if config else None
    well_id = series_list[0].well_id if series_list else None
    input_names = [series.canonical_name for series in series_list]
    details = []
    for series in series_list:
        first = f"{series.depths[0]:g}" if series.depths else "n/a"
        last = f"{series.depths[-1]:g}" if series.depths else "n/a"
        details.append(f"{series.canonical_name}: step={series.step:g}, range=[{first}, {last}], points={len(series.depths)}")

    def mismatch(reason: str) -> list[FeatureCalculatorError]:
        feature_label = feature_name or "без имени"
        message = (
            f"Расчётный признак '{feature_label}' для скважины id={well_id} заблокирован: {reason}. "
            f"Входные кривые: {', '.join(input_names)}. Детали: {'; '.join(details)}. "
            "Интерполяция в текущей версии не используется; выберите кривые с полностью совпадающей глубинной сеткой."
        )
        return [_error("depth_grid_mismatch", message, config.calculator_id if config else None, feature_name, well_id=well_id)]

    for series in series_list:
        if not series.depths or series.step <= 0 or not math.isfinite(series.step):
            return mismatch("не все входные кривые имеют валидный step и непустой интервал")

    reference = series_list[0]
    reference_step = float(reference.step)
    reference_depths = _rounded_depths(reference.depths, precision)
    reference_count = len(reference_depths)
    reference_first = reference_depths[0]
    reference_last = reference_depths[-1]

    for series in series_list[1:]:
        if not math.isclose(float(series.step), reference_step, rel_tol=step_tolerance, abs_tol=step_tolerance):
            return mismatch("различаются шаги глубинной сетки")
        candidate_depths = _rounded_depths(series.depths, precision)
        if len(candidate_depths) != reference_count:
            return mismatch("различается количество точек в расчётном интервале")
        if candidate_depths[0] != reference_first or candidate_depths[-1] != reference_last:
            return mismatch("различаются первая или последняя глубина расчётного интервала")
        if candidate_depths != reference_depths:
            return mismatch(f"списки глубин после округления до {precision} знаков не совпадают")
    return []


def _resolve_canonical(input_curve: FeatureCalculatorInput, db_session: Any) -> tuple[Any | None, FeatureCalculatorError | None]:
    from models_db.model_cluster import CanonicalWellLog

    query = db_session.query(CanonicalWellLog)
    canonical = None
    if input_curve.canonical_id is not None:
        canonical = query.filter(CanonicalWellLog.id == int(input_curve.canonical_id)).first()
    if canonical is None and input_curve.canonical_name:
        canonical = query.filter(CanonicalWellLog.canonical_name_norm == input_curve.canonical_name.strip().casefold()).first()
    if canonical is None:
        label = input_curve.canonical_name or input_curve.canonical_id or "unknown"
        return None, _error("missing_curve", f"Canonical-кривая '{label}' не найдена в справочнике.", canonical_name=str(label))
    return canonical, None


def _alias_names_for_canonical(canonical: Any, db_session: Any) -> set[str]:
    from models_db.model_cluster import AliasWellLog

    alias_rows = db_session.query(AliasWellLog.alias_name_norm).filter(AliasWellLog.canonical_id == canonical.id).all()
    names = {str(row[0]).strip().casefold() for row in alias_rows if row[0]}
    names.add(str(canonical.canonical_name).strip().casefold())
    return names


def _parse_curve_values(well_log: Any, canonical: Any) -> tuple[list[float | None], FeatureCalculatorError | None]:
    if well_log.curve_data is None or str(well_log.curve_data).strip() == "":
        return [], _error("empty_curve_data", f"Кривая '{well_log.curve_name}' не содержит curve_data.", canonical_name=canonical.canonical_name)
    try:
        raw_values = json.loads(well_log.curve_data)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return [], _error("invalid_curve_json", f"curve_data кривой '{well_log.curve_name}' не является корректным JSON: {exc}", canonical_name=canonical.canonical_name)
    if not isinstance(raw_values, list):
        return [], _error("invalid_curve_json", f"curve_data кривой '{well_log.curve_name}' должен быть JSON-массивом.", canonical_name=canonical.canonical_name)
    if not raw_values:
        return [], _error("empty_curve_data", f"Кривая '{well_log.curve_name}' содержит пустой массив curve_data.", canonical_name=canonical.canonical_name)
    return [_finite_float(value) for value in raw_values], None


def _build_depths(begin: float, step: float, count: int) -> list[float]:
    return [begin + index * step for index in range(count)]


def _slice_curve(
    depths: Sequence[float], values: Sequence[float | None], top_md: float, bottom_md: float
) -> tuple[list[float], list[float | None]]:
    sliced_depths: list[float] = []
    sliced_values: list[float | None] = []
    for depth, value in zip(depths, values):
        if top_md <= depth <= bottom_md:
            sliced_depths.append(float(depth))
            sliced_values.append(value)
    return sliced_depths, sliced_values


def load_well_log_curve_series(
    *,
    well_id: int,
    top_md: float,
    bottom_md: float,
    normalization_scope: str = FEATURE_CALCULATOR_DEFAULT_NORMALIZATION_SCOPE,
    canonical_id: int | None = None,
    canonical_name: str | None = None,
    db_session: Any = None,
) -> tuple[WellLogCurveSeries | None, list[FeatureCalculatorError]]:
    """Resolve a canonical curve through aliases and load interval/statistics series for one well."""
    from models_db.model import WellLog, session

    if db_session is None:
        db_session = session
    input_curve = FeatureCalculatorInput(canonical_id=canonical_id, canonical_name=canonical_name)
    canonical, canonical_error = _resolve_canonical(input_curve, db_session)
    if canonical_error:
        canonical_error.well_id = int(well_id)
        return None, [canonical_error]

    alias_names = _alias_names_for_canonical(canonical, db_session)
    well_logs = db_session.query(WellLog).filter(WellLog.well_id == int(well_id)).all()
    candidates = [row for row in well_logs if str(row.curve_name or "").strip().casefold() in alias_names]
    if not candidates:
        return None, [
            _error(
                "missing_curve",
                f"В скважине id={well_id} не найдена кривая для canonical '{canonical.canonical_name}' с учётом алиасов.",
                well_id=int(well_id),
                canonical_name=canonical.canonical_name,
            )
        ]

    candidate_errors: list[FeatureCalculatorError] = []
    for well_log in candidates:
        try:
            begin = float(well_log.begin)
            step = float(well_log.step)
        except (TypeError, ValueError):
            candidate_errors.append(_error("invalid_step", f"У кривой '{well_log.curve_name}' некорректные begin/step.", well_id=int(well_id), canonical_name=canonical.canonical_name))
            continue
        if step <= 0 or not math.isfinite(step):
            candidate_errors.append(_error("invalid_step", f"У кривой '{well_log.curve_name}' неположительный или бесконечный step.", well_id=int(well_id), canonical_name=canonical.canonical_name))
            continue
        values, parse_error = _parse_curve_values(well_log, canonical)
        if parse_error:
            parse_error.well_id = int(well_id)
            candidate_errors.append(parse_error)
            continue
        depths = _build_depths(begin, step, len(values))
        interval_depths, interval_values = _slice_curve(depths, values, float(top_md), float(bottom_md))
        if not interval_depths:
            candidate_errors.append(_error("not_enough_points", f"В интервале [{top_md:g}, {bottom_md:g}] нет точек кривой '{well_log.curve_name}'.", well_id=int(well_id), canonical_name=canonical.canonical_name))
            continue
        stats_depths, stats_values = (depths, values) if normalization_scope == "whole_well" else (interval_depths, interval_values)
        if normalization_scope not in FEATURE_CALCULATOR_NORMALIZATION_SCOPES:
            return None, [_error("unsupported_normalization_scope", "Разрешены только normalization_scope=whole_well или interval.", well_id=int(well_id), canonical_name=canonical.canonical_name)]
        return WellLogCurveSeries(
            well_id=int(well_id),
            canonical_id=int(canonical.id),
            canonical_name=str(canonical.canonical_name),
            depths=interval_depths,
            values=interval_values,
            step=step,
            begin=begin,
            end=float(well_log.end) if well_log.end is not None else depths[-1],
            source_curve_name=str(well_log.curve_name or ""),
            stats_depths=list(stats_depths),
            stats_values=list(stats_values),
        ), []

    return None, candidate_errors or [_error("missing_curve", f"Не удалось загрузить кривая canonical '{canonical.canonical_name}'.", well_id=int(well_id), canonical_name=canonical.canonical_name)]


def load_input_series_for_calculator(
    config: FeatureCalculatorConfig,
    well_id: int,
    top_md: float,
    bottom_md: float,
    *,
    db_session: Any = None,
) -> tuple[list[WellLogCurveSeries], list[FeatureCalculatorError]]:
    """Load all canonical input curves for one well/calculator interval."""
    series_list: list[WellLogCurveSeries] = []
    errors: list[FeatureCalculatorError] = []
    for input_curve in config.inputs:
        series, load_errors = load_well_log_curve_series(
            well_id=int(well_id),
            canonical_id=input_curve.canonical_id,
            canonical_name=input_curve.canonical_name,
            top_md=float(top_md),
            bottom_md=float(bottom_md),
            normalization_scope=config.normalization_scope,
            db_session=db_session,
        )
        if load_errors or series is None:
            errors.extend(
                _copy_error_context(error, config=config, well_id=int(well_id), canonical_name=input_curve.canonical_name)
                for error in load_errors
            )
            continue
        series_list.append(series)
    return series_list, errors


def _attach_well_context(error: FeatureCalculatorError, well_row: Any) -> FeatureCalculatorError:
    well_id = getattr(well_row, "well_id", None)
    well = getattr(well_row, "well", None)
    well_name = getattr(well, "name", None) if well is not None else None
    return FeatureCalculatorError(
        code=error.code,
        message=error.message,
        calculator_id=error.calculator_id,
        feature_name=error.feature_name,
        well_id=int(well_id) if well_id is not None else error.well_id,
        well_name=str(well_name) if well_name else error.well_name,
        canonical_name=error.canonical_name,
        severity=error.severity,
    )


def feature_calculator_status_for_errors(errors: Sequence[FeatureCalculatorError]) -> str:
    """Return a concise user-facing coverage status for calculator errors."""
    if not errors:
        return "OK"
    code = errors[0].code
    return FEATURE_CALCULATOR_STATUS_LABELS.get(code, str(code).replace("_", " ").title())


def feature_calculator_recommendation(error: FeatureCalculatorError | None) -> str:
    """Return a short operator recommendation for a detailed diagnostic error."""
    if error is None:
        return "Расчётный признак успешно рассчитывается для выбранной скважины."
    return FEATURE_CALCULATOR_RECOMMENDATIONS.get(
        error.code,
        "Проверьте параметры признака, входные canonical-кривые и выбранный интервал скважины.",
    )


def _config_input_names(config: FeatureCalculatorConfig) -> list[str]:
    names: list[str] = []
    for input_curve in config.inputs:
        label = input_curve.canonical_name or (f"canonical_id={input_curve.canonical_id}" if input_curve.canonical_id is not None else "unknown")
        names.append(str(label))
    return names


def build_feature_calculator_coverage_report(
    dataset_id: int,
    config: FeatureCalculatorConfig,
    *,
    db_session: Any = None,
) -> FeatureCalculatorCoverageReport:
    """Evaluate one calculator feature against every dataset well and summarize coverage.

    This helper is intentionally read-only: it performs the same strict backend
    evaluation as COLLECT preflight but returns per-well rows suitable for the
    constructor UI coverage report and preview diagnostics.
    """
    from models_db.model import session
    from models_db.model_cluster import WellForCluster

    if db_session is None:
        db_session = session

    well_rows = (
        db_session.query(WellForCluster)
        .filter(WellForCluster.dataset_id == int(dataset_id))
        .order_by(WellForCluster.well_id)
        .all()
    )
    rows: list[FeatureCalculatorCoverageRow] = []
    for well_row in well_rows:
        result = evaluate_feature_calculator_for_well(
            config,
            well_id=int(well_row.well_id),
            top_md=float(well_row.top_md),
            bottom_md=float(well_row.bottom_md),
            db_session=db_session,
        )
        attached_errors = [_attach_well_context(error, well_row) for error in result.errors]
        rows.append(
            FeatureCalculatorCoverageRow(
                well_id=int(well_row.well_id),
                well_name=getattr(getattr(well_row, "well", None), "name", None),
                top_md=float(well_row.top_md),
                bottom_md=float(well_row.bottom_md),
                status=feature_calculator_status_for_errors(attached_errors),
                points=len(result.depths) if result.ok else 0,
                errors=attached_errors,
            )
        )

    ok_points = [row.points for row in rows if row.ok]
    error_count = sum(len(row.errors) for row in rows)
    summary = FeatureCalculatorCoverageSummary(
        dataset_id=int(dataset_id),
        calculator_id=config.calculator_id,
        feature_name=config.feature_name,
        wells_total=len(rows),
        wells_ok=sum(1 for row in rows if row.ok),
        error_count=error_count,
        min_points=min(ok_points) if ok_points else None,
        max_points=max(ok_points) if ok_points else None,
        input_curves=_config_input_names(config),
        mode=config.mode,
        normalization_scope=config.normalization_scope,
    )
    return FeatureCalculatorCoverageReport(summary=summary, rows=rows)


def validate_calculators_for_dataset(
    dataset_id: int,
    *,
    calculator_ids: Sequence[int] | None = None,
    db_session: Any = None,
) -> list[FeatureCalculatorError]:
    """Fully validate selected calculated features for every well/interval in a dataset.

    The validation intentionally performs a real evaluation pass for each selected
    calculator and each dataset well interval.  This preflight is used by COLLECT
    so calculation errors are reported before any previous dataset ``data`` row is
    deleted or replaced.
    """
    from models_db.model import session
    from models_db.model_cluster import (
        ClusterWellLogParameterFromCalculator,
        FeatureCalculator,
        WellForCluster,
    )

    if db_session is None:
        db_session = session

    well_rows = (
        db_session.query(WellForCluster)
        .filter(WellForCluster.dataset_id == int(dataset_id))
        .order_by(WellForCluster.well_id)
        .all()
    )
    if not well_rows:
        return [
            _error(
                "empty_dataset_wells",
                f"В Well Log dataset id={int(dataset_id)} нет скважин для проверки calculator-признаков.",
            )
        ]

    if calculator_ids is not None:
        normalized_ids = [int(calculator_id) for calculator_id in calculator_ids]
        if not normalized_ids:
            return []
        query = db_session.query(FeatureCalculator).filter(FeatureCalculator.id.in_(normalized_ids))
    else:
        query = (
            db_session.query(FeatureCalculator)
            .join(
                ClusterWellLogParameterFromCalculator,
                ClusterWellLogParameterFromCalculator.calculator_id == FeatureCalculator.id,
            )
            .filter(ClusterWellLogParameterFromCalculator.dataset_id == int(dataset_id))
        )
    calculators = query.order_by(FeatureCalculator.feature_name, FeatureCalculator.id).all()
    if not calculators:
        return []

    errors: list[FeatureCalculatorError] = []
    for calculator in calculators:
        config, parse_errors = parse_feature_calculator_config(calculator)
        if parse_errors:
            errors.extend(parse_errors)
        if config is None:
            continue
        config_errors = validate_feature_calculator_config(config)
        if config_errors:
            errors.extend(config_errors)
            continue
        for well_row in well_rows:
            result = evaluate_feature_calculator_for_well(
                config,
                well_id=int(well_row.well_id),
                top_md=float(well_row.top_md),
                bottom_md=float(well_row.bottom_md),
                db_session=db_session,
            )
            if result.errors:
                errors.extend(_attach_well_context(error, well_row) for error in result.errors)
    return errors


def validate_feature_calculator_for_dataset(dataset_id: int, calculator_id: int) -> list[FeatureCalculatorError]:
    """Validate one calculated feature against all wells/intervals of a dataset."""
    return validate_calculators_for_dataset(int(dataset_id), calculator_ids=[int(calculator_id)])


def evaluate_feature_calculator_for_well(
    config: FeatureCalculatorConfig,
    well_id: int,
    top_md: float,
    bottom_md: float,
    *,
    db_session: Any = None,
) -> FeatureCalculatorEvaluationResult:
    """Evaluate a unary operation or strict formula for one well and one interval."""
    config_errors = validate_feature_calculator_config(config)
    if config_errors:
        return FeatureCalculatorEvaluationResult(errors=config_errors)

    if config.mode == "formula":
        series_list, load_errors = load_input_series_for_calculator(
            config,
            well_id=int(well_id),
            top_md=float(top_md),
            bottom_md=float(bottom_md),
            db_session=db_session,
        )
        if load_errors:
            return FeatureCalculatorEvaluationResult(errors=load_errors)
        aligned_series, alignment_errors = align_formula_series_on_common_depths(series_list)
        if alignment_errors:
            return FeatureCalculatorEvaluationResult(
                errors=[_copy_error_context(error, config=config, well_id=int(well_id)) for error in alignment_errors]
            )
        values, formula_errors = evaluate_formula_series(str(config.expression), aligned_series)
        return FeatureCalculatorEvaluationResult(
            values=values,
            depths=aligned_series[0].depths if not formula_errors and aligned_series else [],
            errors=[_copy_error_context(error, config=config, well_id=int(well_id)) for error in formula_errors],
        )

    input_curve = config.inputs[0]
    series, load_errors = load_well_log_curve_series(
        well_id=int(well_id),
        canonical_id=input_curve.canonical_id,
        canonical_name=input_curve.canonical_name,
        top_md=float(top_md),
        bottom_md=float(bottom_md),
        normalization_scope=config.normalization_scope,
        db_session=db_session,
    )
    if load_errors or series is None:
        return FeatureCalculatorEvaluationResult(
            errors=[_copy_error_context(error, config=config, well_id=int(well_id), canonical_name=input_curve.canonical_name) for error in load_errors]
        )

    window = config.raw_params.get("window", 3)
    stats_values = series.stats_values if str(config.operation).lower() in {"zscore", "robust", "minmax"} else None
    values, operation_errors = apply_unary_operation(
        str(config.operation),
        series.values,
        step=series.step,
        window=window,
        stats_values=stats_values,
    )
    return FeatureCalculatorEvaluationResult(
        values=values,
        depths=series.depths if not operation_errors else [],
        errors=[_copy_error_context(error, config=config, series=series) for error in operation_errors],
    )
