"""Human-readable labels for automatically calculated well parameters."""
from __future__ import annotations


SOURCE_TITLES = {
    "boundary": "глубина границы скважины",
    "well_data": "информация о скважине",
    "well_log": "каротажная кривая",
}
AGGREGATION_TITLES = {
    "median": "медиана",
    "mean": "среднее арифметическое",
    "max": "максимум",
    "min": "минимум",
}
OPERATION_TITLES = {
    "single": "значение на всём интервале",
    "upper_lower_ratio": "отношение верхней половины интервала к нижней",
    "lower_upper_ratio": "отношение нижней половины интервала к верхней",
    "difference": "разность значений нижней и верхней половин интервала",
}
POSITION_TITLES = {
    "below": "ниже опорной глубины",
    "above": "выше опорной глубины",
    "centered": "симметрично относительно опорной глубины",
}


def parameter_header_tooltip(canonical_name, settings, boundary_name="") -> str:
    """Describe provenance and calculation behind a parameter table column."""
    lines = [
        f"Параметр: {canonical_name} (каноническое название).",
        f"Источник: {SOURCE_TITLES.get(settings.source, settings.source)}.",
    ]
    if settings.source == "boundary":
        lines.append("Значение: глубина выбранной записи канонической границы.")
    elif settings.source == "well_data":
        parsing = "только строго числовые значения" if settings.strict_numeric else "число извлекается из текстового значения"
        addition = ("; явные слагаемые, разделённые знаком «+», складываются"
                    if settings.allow_explicit_sum else "; выражения со знаком «+» не складываются")
        lines.append(f"Разбор исходного значения: {parsing}{addition}.")
    elif settings.source == "well_log":
        lines.append(f"Агрегация отсчётов кривой: {AGGREGATION_TITLES.get(settings.aggregation, settings.aggregation)}.")
        lines.append(f"Операция: {OPERATION_TITLES.get(settings.operation, settings.operation)}.")
        if settings.depth_mode == "boundary":
            reference = boundary_name or f"каноническая граница ID {settings.boundary_canonical_id}"
            lines.append(f"Опорная глубина: граница «{reference}».")
        else:
            lines.append(f"Опорная глубина: фиксированная, {settings.fixed_depth:g}.")
        lines.append(
            f"Интервал: {settings.interval:g}; положение — "
            f"{POSITION_TITLES.get(settings.interval_position, settings.interval_position)}.")
    lines.append("Ячейки содержат итоговое рассчитанное значение; подробности конкретной исходной записи показаны в подсказке ячейки.")
    return "\n".join(lines)
