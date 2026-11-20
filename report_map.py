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
    if not text.strip():
        return None
    try:
        interval = float(text.replace(',', '.'))
    except ValueError:
        interval = 0
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError('Шаг изолиний должен быть положительным числом.')
    return interval


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
    profiles=(), wells=(), sample_names=None,
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
    low, high = surface_low, surface_high
    if show_points and len(valid_samples):
        low = min(low, float(valid_samples[:, 2].min()))
        high = max(high, float(valid_samples[:, 2].max()))
    levels = _color_levels(low, high, contour_interval if show_contours else None)
    palette = colormaps.get_cmap(cmap)
    if palette.N < len(levels) - 1:
        palette = palette.resampled(len(levels) - 1)
    norm = BoundaryNorm(levels, palette.N)

    fill = ax.contourf(xx, yy, surface, levels=levels, cmap=palette, norm=norm)
    colorbar = figure.colorbar(fill, cax=ax.inset_axes([1.03, 0, 0.045, 1]),
                               spacing='proportional', label=parameter)
    ticks = levels[::max(1, ceil((len(levels) - 1) / 10))] if low < high else [low]
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
        label_style = dict(inline=False, fontsize=8, fmt=lambda level: f'{level:.8g}')
        labels = list(ax.clabel(lines, **label_style))
        if coverage is not None:
            path = coverage.get_path()
            visible_labels = []
            for label in labels:
                if path.contains_point(label.get_position()):
                    visible_labels.append(label)
                else:
                    label.remove()
            labeled = {label.get_text() for label in visible_labels}
            for level, segments in zip(lines.levels, lines.allsegs):
                if label_style['fmt'](level) in labeled:
                    continue
                visible_segments = [segment[path.contains_points(segment)] for segment in segments]
                segment = max(visible_segments, key=len, default=[])
                if len(segment):
                    placed = ax.clabel(lines, levels=[level], manual=[segment[len(segment) // 2]], **label_style)
                    visible_labels.append(placed[-1])
            labels = visible_labels
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
