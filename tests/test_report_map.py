import numpy as np
import pytest

from report_map import build_report_map, parse_contour_interval


def test_report_map_uses_the_same_scale_and_clips_the_field():
    xx, yy = np.mgrid[0:600:25, 0:400:25]
    surface = xx / 100
    samples = np.array([[50, 50, 0.5], [500, 50, 5.0], [300, 325, 3.0]])

    figure = build_report_map(
        xx, yy, surface, samples, title='Скорость', parameter='м/с', cmap='viridis',
        grid_step=25, method='Кригинг', contour_interval=1,
        profiles=[([50, 300], [50, 325])], wells=[(260, 100, 'Скв. 1', 'orange')],
    )

    ax = figure.axes[0]
    fill, contours, points, _ = ax.collections
    assert ax.get_aspect() == 1
    assert fill.norm is points.norm
    assert fill.get_clip_path() is not None
    assert contours.get_clip_path() is not None
    assert {'1', '2', '3', '4', 'Скв. 1'} <= {label.get_text() for label in ax.texts}
    assert ax.get_legend_handles_labels()[1] == ['Профили', 'Точки замеров', 'Скважины']


def test_report_exports_constant_surface_without_a_coverage_polygon(tmp_path):
    xx, yy = np.mgrid[0:600:75, 0:400:75]
    surface = np.full(xx.shape, 2.0)
    samples = np.array([[50, 50, 2.0], [150, 50, 2.0], [250, 50, 2.0]])

    figure = build_report_map(
        xx, yy, surface, samples, title='Постоянная поверхность', parameter='Значение',
        cmap='jet', grid_step=75, method='Постоянная поверхность', show_points=False,
    )
    ax = figure.axes[0]
    assert len(ax.collections) == 1
    assert ax.collections[0].get_clip_path() is None
    assert any('Область замеров не образует контур' in text.get_text() for text in figure.texts)

    for suffix, signature in [('pdf', b'%PDF'), ('png', b'\x89PNG')]:
        target = tmp_path / f'карта.{suffix}'
        figure.savefig(target)
        assert target.read_bytes().startswith(signature)


def test_report_rejects_invalid_or_unreasonably_fine_contour_steps():
    assert parse_contour_interval(' 0,25 ') == 0.25
    assert parse_contour_interval('') is None
    with pytest.raises(ValueError, match='положительным'):
        parse_contour_interval('nan')

    xx, yy = np.mgrid[0:600:75, 0:400:75]
    with pytest.raises(ValueError, match='Слишком мелкий шаг'):
        build_report_map(
            xx, yy, xx.copy(), np.array([[0, 0, 0]]), title='Скорость',
            parameter='м/с', cmap='jet', grid_step=75, method='Кригинг',
            contour_interval=0.1,
        )
