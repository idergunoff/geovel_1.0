"""Well markers and value legends for maps."""

from math import ceil

import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator

DEFAULT_WELL_MARKER_COLOR = "red"
WELL_LOG_MARKER_COLOR = "#D500F9"


def well_marker_color(well):
    """Return a bright accent color when a well has logging curves."""
    return WELL_LOG_MARKER_COLOR if well.well_logs else DEFAULT_WELL_MARKER_COLOR


def _values_at_wells(xx, yy, grid, wells, *, class_labels, coverage):
    positions = [(x, y) for x, y, _, _ in wells]
    values = np.ma.asarray(grid, dtype=float).filled(np.nan)
    sampler = RegularGridInterpolator(
        (xx[:, 0], yy[0, :]), values, method='linear' if class_labels is None else 'nearest',
        bounds_error=False, fill_value=np.nan,
    )
    sampled = sampler(positions)
    if coverage is not None:
        sampled[~coverage.contains_points(positions, radius=1e-6)] = np.nan
    if class_labels is None:
        return [value if np.isfinite(value) else None for value in sampled]
    return [class_labels[int(index)] if np.isfinite(index) else None for index in sampled]


def _well_value_text(value, categorical):
    if categorical:
        if value is None:
            return 'нет метки'
        return 'Шум (-1)' if value == -1 else f'Кластер {value}'
    return 'нет значения' if value is None else f'{value:.8g}'


def draw_map_wells(ax, wells, xx, yy, grid, *, marker='o', class_labels=None,
                   well_clusters=None, coverage=None):
    """Draw the well overlay and an exportable list beside the map."""
    if not wells:
        return

    categorical = class_labels is not None or well_clusters is not None
    if well_clusters is None:
        values = _values_at_wells(xx, yy, grid, wells, class_labels=class_labels, coverage=coverage)
        well_rows = [((None, value, None),) for value in values]
    else:
        well_rows = well_clusters

    handles = []
    for index, ((x, y, name, color), matches) in enumerate(zip(wells, well_rows)):
        ax.scatter(x, y, c=[color], marker=marker, s=55, linewidths=0.6,
                   edgecolors='black', zorder=7, label='Скважины' if index == 0 else None)
        ax.annotate(name, (x, y), (4, 5), textcoords='offset points', fontsize=8, zorder=8)
        for profile_title, value, distance in matches or ((None, None, None),):
            location = f'{name} · {profile_title} ({distance:.2f} м)' if profile_title else name
            text = f'{location} — {_well_value_text(value, categorical)}'
            handles.append(Line2D([], [], linestyle='none', marker=marker, markersize=5,
                                  markerfacecolor=color, markeredgecolor='black', label=text))

    figure = ax.figure
    width, height = figure.get_size_inches()
    rows_per_column = max(1, int((height - 0.5) * 5))
    legend = figure.legend(
        handles=handles, loc='outside right upper', ncols=ceil(len(handles) / rows_per_column),
        title='Кластеры у скважин' if categorical else 'Значения карты у скважин',
        fontsize=8, title_fontsize=9, handletextpad=0.5,
    )
    legend_width = legend.get_window_extent().width / figure.dpi
    figure.set_size_inches(width + max(0, legend_width - 1), height)
