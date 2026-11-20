"""Matplotlib layout for continuous maps exported from GeoVel."""

from math import ceil, floor

import numpy as np
from matplotlib import colormaps, patheffects
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.spatial import ConvexHull, QhullError


def parse_contour_interval(text):
    if text.strip() in ('', 'Автоматически'):
        return None
    try:
        interval = float(text.replace(',', '.'))
    except ValueError:
        interval = 0
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError('Шаг изолиний должен быть положительным числом.')
    return interval


def parse_scale_bounds(minimum, maximum):
    bounds = []
    for text, label in ((minimum, 'Минимум'), (maximum, 'Максимум')):
        if not text.strip():
            bounds.append(None)
            continue
        try:
            value = float(text.replace(',', '.'))
        except ValueError:
            value = np.nan
        if not np.isfinite(value):
            raise ValueError(f'{label} шкалы должен быть числом.')
        bounds.append(value)
    if all(value is not None for value in bounds) and bounds[0] >= bounds[1]:
        raise ValueError('Минимум шкалы должен быть меньше максимума.')
    return tuple(bounds)


def _color_levels(low, high, interval):
    if interval is None:
        return MaxNLocator(nbins=10).tick_values(low, high)

    start, end = low / interval, high / interval
    if not np.isfinite([start, end]).all() or end - start > 100:
        raise ValueError('Слишком мелкий шаг изолиний. Увеличьте шаг или оставьте поле пустым.')
    first, last = floor(start), ceil(end)
    last = max(last, first + 1)
    if last - first > 100:
        raise ValueError('Слишком мелкий шаг изолиний. Увеличьте шаг или оставьте поле пустым.')
    levels = (float(first) + np.arange(last - first + 1)) * interval
    if not np.isfinite(levels).all() or not (np.diff(levels) > 0).all():
        raise ValueError('Не удалось построить уровни с заданным шагом. Оставьте поле пустым.')
    return levels


def build_report_map(
    xx, yy, surface, samples, *, title, parameter, cmap, grid_step, method,
    contour_interval=None, show_contours=True, show_points=True, clip_to_samples=True,
    profiles=(), wells=(), sample_names=None, scale_min=None, scale_max=None,
):
    """Build the report figure from the existing grid."""
    x_min, x_max = xx.min(), xx.max()
    y_min, y_max = yy.min(), yy.max()
    height = np.clip(2.5 + 8.5 * (y_max - y_min) / (x_max - x_min), 5, 10)
    figure = Figure(figsize=(11, height), layout='constrained')
    ax = figure.subplots()
    ax.set_title(title or parameter, loc='left', fontweight='bold', pad=12)
    ax.set(xlabel='X, м', ylabel='Y, м', aspect='equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.ticklabel_format(style='plain', useOffset=False)

    valid = np.isfinite(samples).all(axis=1)
    valid_samples = samples[valid]
    surface = np.ma.masked_invalid(surface)
    if surface.count() == 0:
        raise ValueError('Нет расчётных значений для построения отчётной карты.')
    surface_low, surface_high = float(surface.min()), float(surface.max())
    data_low, data_high = surface_low, surface_high
    if show_points and len(valid_samples):
        data_low = min(data_low, float(valid_samples[:, 2].min()))
        data_high = max(data_high, float(valid_samples[:, 2].max()))
    low = data_low if scale_min is None else scale_min
    high = data_high if scale_max is None else scale_max
    if low == high and (scale_min is None) != (scale_max is None):
        auto_low, *_, auto_high = _color_levels(low, high, None)
        low = auto_low if scale_min is None else low
        high = auto_high if scale_max is None else high
    if low > high or (low == high and scale_min is not None and scale_max is not None):
        raise ValueError('Минимум шкалы должен быть меньше максимума. Проверьте обе границы.')
    levels = _color_levels(low, high, contour_interval if show_contours else None)
    if scale_min is not None:
        levels = np.r_[low, levels[levels > low]]
    if scale_max is not None:
        levels = np.r_[levels[levels < high], high]
    palette = colormaps.get_cmap(cmap)
    if palette.N < len(levels) - 1:
        palette = palette.resampled(len(levels) - 1)
    norm = BoundaryNorm(levels, palette.N)

    extend_low, extend_high = data_low < levels[0], data_high > levels[-1]
    extend = {(True, True): 'both', (True, False): 'min',
              (False, True): 'max', (False, False): 'neither'}[extend_low, extend_high]
    fill = ax.contourf(xx, yy, surface, levels=levels, cmap=palette, norm=norm, extend=extend)
    colorbar = figure.colorbar(fill, cax=ax.inset_axes([1.03, 0, 0.045, 1]),
                               spacing='proportional', label=parameter)
    tick_step = max(1, ceil((len(levels) - 1) / 10))
    ticks = np.r_[levels[:-1:tick_step], levels[-1]] if low < high else [low]
    colorbar.set_ticks(ticks)

    coverage = None
    if clip_to_samples and len(valid_samples) >= 3:
        coordinates = valid_samples[:, :2]
        try:
            boundary = coordinates[ConvexHull(coordinates).vertices]
        except QhullError:
            pass
        else:
            coverage = Polygon(boundary, transform=ax.transData)
            fill.set_clip_path(coverage)
            ax.plot(*np.vstack((boundary, boundary[0])).T, color='0.35', linewidth=0.8, zorder=3)

    contour_levels = levels[(levels > surface_low) & (levels < surface_high)]
    if show_contours and contour_levels.size:
        lines = ax.contour(xx, yy, surface, levels=contour_levels, colors='0.2', linewidths=0.65)
        if coverage is not None:
            lines.set_clip_path(coverage)
        manual = False
        if coverage is not None:
            path = coverage.get_path()
            manual = []
            for segments in lines.allsegs:
                visible_segments = [segment[path.contains_points(segment)] for segment in segments]
                segment = max(visible_segments, key=len, default=[])
                if len(segment) > 1:
                    manual.append(segment[len(segment) // 2])
        # Pass all levels together: Matplotlib's manual placement indexes the full contour set.
        labels = ax.clabel(lines, manual=manual, inline=False, fontsize=8, fmt=lambda level: f'{level:.8g}')
        for label in labels:
            label.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground='white')])

    for index, (x, y) in enumerate(profiles):
        ax.plot(x, y, color='0.3', linewidth=1, zorder=4, label='Профили' if index == 0 else None)

    if show_points and len(valid_samples):
        x, y, values = valid_samples.T
        ax.scatter(x, y, c=values, cmap=palette, norm=norm, s=20, edgecolors='black',
                   linewidths=0.45, zorder=5, label='Точки замеров')
        if sample_names is not None:
            for (sx, sy, _), name, is_valid in zip(samples, sample_names, valid):
                if is_valid:
                    ax.annotate(name, (sx, sy), (4, 4), textcoords='offset points', fontsize=7, zorder=6)

    for index, (x, y, name, color) in enumerate(wells):
        ax.scatter(x, y, c=[color], marker='^', s=55, linewidths=0.6,
                   edgecolors='black', zorder=7, label='Скважины' if index == 0 else None)
        ax.annotate(name, (x, y), (4, 5), textcoords='offset points', fontsize=8, zorder=8)

    target_length = (x_max - x_min) / 5
    magnitude = 10.0 ** floor(np.log10(target_length))
    scale_length = max(value for value in (magnitude, 2 * magnitude, 5 * magnitude) if value <= target_length)
    scale_label = f'{scale_length / 1000:g} км' if scale_length >= 1000 else f'{scale_length:g} м'
    scale = AnchoredSizeBar(ax.transData, scale_length, scale_label, 'lower left', frameon=True, borderpad=0.7)
    scale.set_zorder(9)
    ax.add_artist(scale)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='upper right', fontsize=8)

    footer = f'{method}   ·   Шаг сетки {grid_step:g} м   ·   Узлы {xx.shape[0]} × {xx.shape[1]}'
    if clip_to_samples:
        footer += '   ·   Обрезка по области замеров' if coverage is not None else '   ·   Область замеров не образует контур'
    figure.supxlabel(footer, fontsize=9, color='0.3')
    return figure
