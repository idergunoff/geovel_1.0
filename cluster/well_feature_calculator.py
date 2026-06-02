from __future__ import annotations

import json
import math
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


def apply_zscore(values: Sequence[float | None], stats_values: Sequence[float | None] | None = None) -> tuple[list[float | None], list[FeatureCalculatorError]]:
    source = list(values if stats_values is None else stats_values)
    source_error = _blocking_if_any_missing(values, "non_finite_result", "zscore требует конечные значения расчётного интервала.")
    stats_error = _blocking_if_any_missing(source, "non_finite_result", "zscore требует конечные значения области нормировки.")
    if source_error or stats_error:
        return [], [error for error in (source_error, stats_error) if error]
    if len(source) < 2:
        return [], [_error("not_enough_points", "Для zscore требуется минимум две точки.")]
    mean = sum(float(value) for value in source) / len(source)
    variance = sum((float(value) - mean) ** 2 for value in source) / len(source)
    std = math.sqrt(variance)
    if std == 0:
        return [], [_error("zero_std", "Стандартное отклонение равно нулю; zscore невозможен.")]
    return [(float(value) - mean) / std for value in values], []


def apply_robust(values: Sequence[float | None], stats_values: Sequence[float | None] | None = None) -> tuple[list[float | None], list[FeatureCalculatorError]]:
    source = list(values if stats_values is None else stats_values)
    source_error = _blocking_if_any_missing(values, "non_finite_result", "robust требует конечные значения расчётного интервала.")
    stats_error = _blocking_if_any_missing(source, "non_finite_result", "robust требует конечные значения области нормировки.")
    if source_error or stats_error:
        return [], [error for error in (source_error, stats_error) if error]
    if len(source) < 2:
        return [], [_error("not_enough_points", "Для robust-нормировки требуется минимум две точки.")]
    numeric_source = [float(value) for value in source]
    median_value = statistics.median(numeric_source)
    q1, q3 = _quartiles(numeric_source)
    iqr = q3 - q1
    if iqr == 0:
        return [], [_error("zero_iqr", "IQR равен нулю; robust-нормировка невозможна.")]
    return [(float(value) - median_value) / iqr for value in values], []


def apply_minmax(values: Sequence[float | None], stats_values: Sequence[float | None] | None = None) -> tuple[list[float | None], list[FeatureCalculatorError]]:
    source = list(values if stats_values is None else stats_values)
    source_error = _blocking_if_any_missing(values, "non_finite_result", "minmax требует конечные значения расчётного интервала.")
    stats_error = _blocking_if_any_missing(source, "non_finite_result", "minmax требует конечные значения области нормировки.")
    if source_error or stats_error:
        return [], [error for error in (source_error, stats_error) if error]
    min_value = min(float(value) for value in source)
    max_value = max(float(value) for value in source)
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
    *,
    db_session: Any = None,
) -> FeatureCalculatorEvaluationResult:
    """Evaluate a stage-2 unary operation for one well and one interval."""
    config_errors = validate_feature_calculator_config(config)
    if config_errors:
        return FeatureCalculatorEvaluationResult(errors=config_errors)
    if config.mode != "operation":
        return FeatureCalculatorEvaluationResult(
            errors=[
                _error(
                    "unsupported_operation",
                    "Formula mode будет реализован на следующем этапе; этап 2 поддерживает только mode=operation.",
                    config.calculator_id,
                    config.feature_name,
                    well_id=int(well_id),
                )
            ]
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
