"""Convert radar traces to depth without inventing thickness for thin intervals."""

import math
from numbers import Real


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def build_velocity_depth_signal(signal, boundaries, interpolate, smooth):
    """Boundaries contain (title, sample coordinates, predicted depths).

    Preserve the supplied binding order and the legacy 0.4 depth units/sample
    below the last boundary. Quantize absolute depth coordinates, so rounding
    thin intervals cannot shift every subsequent boundary.
    """
    if not isinstance(signal, list) or not signal:
        raise ValueError('Нет сигнала для отображения.')
    for trace, measure in enumerate(signal, 1):
        if not isinstance(measure, list) or len(measure) < 2:
            raise ValueError(f'Трасса {trace}: сигнал содержит меньше двух отсчётов.')
    if not boundaries:
        raise ValueError('Нет привязанных слоёв для отображения (Velocity - Prediction).')

    prepared = []
    for title, coordinates, depths in boundaries:
        for values, label in [(coordinates, 'границы'), (depths, 'prediction')]:
            if not isinstance(values, list) or len(values) != len(signal):
                raise ValueError(f'Слой «{title}»: длина {label} не совпадает с числом трасс ({len(signal)}).')
            for trace, value in enumerate(values, 1):
                if not _finite(value):
                    raise ValueError(f'Слой «{title}», трасса {trace}: некорректное значение {label}.')
        for trace, coordinate in enumerate(coordinates, 1):
            if int(coordinate) != coordinate or not 0 <= coordinate < len(signal[trace - 1]):
                raise ValueError(f'Слой «{title}», трасса {trace}: граница вне сигнала или нецелый отсчёт.')
        smoothed = list(smooth(depths, 175)) if len(depths) >= 5 else depths
        if len(smoothed) != len(signal) or not all(_finite(value) for value in smoothed):
            raise ValueError(f'Слой «{title}»: некорректный результат сглаживания prediction.')
        prepared.append((title, list(map(int, coordinates)), smoothed))

    result = []
    for trace, measure in enumerate(signal):
        top_sample, top_depth, top_title = 0, 0, 'поверхность'
        converted = []
        intervals = [(title, coordinates[trace], depths[trace]) for title, coordinates, depths in prepared]
        last_sample = len(measure) - 1
        last_boundary = intervals[-1]
        intervals.append(('низ сигнала', last_sample,
                          last_boundary[2] + 0.4 * (last_sample - last_boundary[1])))
        for index, (title, bottom_sample, bottom_depth) in enumerate(intervals):
            location = f'Интервал «{top_title} → {title}», трасса {trace + 1}'
            if bottom_sample < top_sample or (0 < index < len(prepared) and bottom_sample == top_sample):
                raise ValueError(f'{location}: отрисованные границы пересекаются или совпадают; проверьте порядок привязок.')
            if bottom_depth < top_depth:
                raise ValueError(f'{location}: prediction нижней границы выше верхней ({bottom_depth:g} < {top_depth:g}).')
            segment = measure[top_sample:bottom_sample]
            if not segment and bottom_depth > top_depth:
                raise ValueError(f'{location}: пустой участок сигнала при положительной толщине по prediction.')
            if segment and bottom_depth == top_depth:
                raise ValueError(f'{location}: нулевая толщина по prediction при разных отрисованных границах.')

            length = int(bottom_depth) - int(top_depth)
            if length:
                if len(segment) == 1:
                    converted.extend([int(segment[0])] * length)
                elif length == 1:
                    converted.append(int(segment[0]))
                else:
                    converted.extend(interpolate(segment, length))
            top_sample, top_depth, top_title = bottom_sample, bottom_depth, title
        result.append(converted)
    if max(map(len, result)) < 2:
        raise ValueError('Глубина профиля после пересчёта меньше двух отсчётов; проверьте единицы и значения prediction.')
    return result
