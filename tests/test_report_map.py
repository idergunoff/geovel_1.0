import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from report_map import build_categorical_report_map, build_report_map, parse_contour_interval, parse_scale_bounds


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
    assert [label.get_text() for label in figure.legends[0].get_texts()] == ['Скв. 1 — 2.6']


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
    assert parse_contour_interval('Автоматически') is None
    with pytest.raises(ValueError, match='положительным'):
        parse_contour_interval('nan')

    xx, yy = np.mgrid[0:600:75, 0:400:75]
    with pytest.raises(ValueError, match='Слишком мелкий шаг'):
        build_report_map(
            xx, yy, xx.copy(), np.array([[0, 0, 0]]), title='Скорость',
            parameter='м/с', cmap='jet', grid_step=75, method='Кригинг',
            contour_interval=0.1,
        )


def test_report_labels_higher_levels_when_coverage_excludes_lower_contours():
    xx, yy = np.mgrid[0:1000:10, 0:400:10]
    figure = build_report_map(
        xx, yy, xx / 100, np.array([[620, 40, 6.2], [950, 40, 9.5], [790, 105, 7.9]]),
        title='Скорость', parameter='м/с', cmap='viridis', grid_step=10, method='Кригинг',
        contour_interval=1,
    )
    labels = {label.get_text() for label in figure.axes[0].texts}
    assert labels == {'7', '8', '9'}


def test_report_respects_exact_scale_bounds_and_can_hide_points():
    xx, yy = np.mgrid[0:600:25, 0:400:25]
    figure = build_report_map(
        xx, yy, xx / 100, np.array([[50, 50, -1], [500, 50, 7], [300, 325, 3]]),
        title='Скорость', parameter='м/с', cmap='viridis', grid_step=25, method='Кригинг',
        contour_interval=1, scale_min=1.3, scale_max=4.8, show_points=False,
        sample_names=['Замер 1', 'Замер 2', 'Замер 3'], wells=[(260, 100, 'Скв. 1', 'orange')],
    )
    ax = figure.axes[0]
    fill, contours, _ = ax.collections
    assert fill.extend == 'both'
    assert fill.get_clip_path() is not None
    np.testing.assert_allclose(fill.levels, [1.3, 2, 3, 4, 4.8])
    np.testing.assert_allclose(contours.levels, fill.levels)
    np.testing.assert_allclose(ax.child_axes[0].get_yticks(), fill.levels)
    assert ax.get_legend_handles_labels()[1] == ['Скважины']
    assert {text.get_text() for text in ax.texts}.isdisjoint(['Замер 1', 'Замер 2', 'Замер 3'])


def test_report_scale_fields_accept_independent_bounds():
    assert parse_scale_bounds('', '') == (None, None)
    assert parse_scale_bounds('-1,5', '') == (-1.5, None)
    assert parse_scale_bounds('', '0') == (None, 0)
    with pytest.raises(ValueError, match='Минимум шкалы должен быть меньше максимума'):
        parse_scale_bounds('5', '5')
    with pytest.raises(ValueError, match='Максимум шкалы должен быть числом'):
        parse_scale_bounds('', 'inf')


def test_categorical_report_draws_only_boundaries_between_visible_classes(tmp_path):
    xx, yy = np.mgrid[0:300:100, 0:200:100]
    cells = np.ma.array([[0, 0], [1, 1], [2, 2]], mask=[[False, False], [False, False], [True, False]])
    points = np.array([[0, 0], [100, 100], [200, 100]])
    labels = np.array([2, 8, -1])
    colors = {2: 'red', 8: 'blue', -1: 'gray'}

    figure = build_categorical_report_map(
        xx, yy, cells, points, labels, colors, title='Кластерная карта', grid_step=100,
        method='Probability', clip_to_samples=True, uncertainty=cells / 2,
        profiles=[([0, 200], [25, 75])], wells=[(100, 50, 'Скв. 1', 'orange')],
    )

    ax = figure.axes[0]
    fill, overlay, boundaries, *_ = ax.collections
    assert ax.get_aspect() == 1
    np.testing.assert_array_equal(np.ma.getmaskarray(fill.get_array()).reshape(cells.shape), cells.mask)
    assert fill.cmap.colors == ['red', 'blue', 'gray']
    assert overlay.get_clim() == (0, 1)
    assert ax.get_legend().get_texts()[0].get_text() == 'Кластер 2'
    assert [label.get_text() for label in ax.get_legend().get_texts()] == [
        'Кластер 2', 'Кластер 8', 'Шум', 'Профили', 'Точки замеров', 'Шумовые точки', 'Скважины',
    ]
    np.testing.assert_allclose(boundaries.get_segments(), [
        [[50, -50], [50, 50]], [[50, 50], [50, 150]], [[150, 50], [150, 150]],
    ])

    for suffix, signature in [('pdf', b'%PDF'), ('png', b'\x89PNG')]:
        target = tmp_path / f'кластеры.{suffix}'
        figure.savefig(target)
        assert target.read_bytes().startswith(signature)


def test_categorical_report_can_hide_points_and_boundaries_with_one_class():
    xx, yy = np.mgrid[0:300:100, 0:200:100]
    figure = build_categorical_report_map(
        xx, yy, np.zeros(xx.shape, dtype=int), np.array([[0, 0]]), np.array([7]), {7: 'blue'},
        title='Один класс', grid_step=100, method='Nearest', show_points=False, show_contours=False,
    )

    ax = figure.axes[0]
    assert len(ax.collections) == 1
    assert [label.get_text() for label in ax.get_legend().get_texts()] == ['Кластер 7']
    assert not figure.legends


def test_report_wells_have_no_value_outside_the_visible_surface():
    xx, yy = np.mgrid[0:600:25, 0:400:25]
    samples = np.array([[50, 50, 0.5], [500, 50, 5], [300, 325, 3]])
    figure = build_report_map(
        xx, yy, xx / 100, samples, title='Скорость', parameter='м/с', cmap='viridis',
        grid_step=25, method='Кригинг', show_points=False,
        wells=[(50, 50, 'Граница', 'red'), (50, 350, 'Вне контура', 'red'),
               (1000, 0, 'Вне сетки', 'red')],
    )

    assert [label.get_text() for label in figure.legends[0].get_texts()] == [
        'Граница — 0.5', 'Вне контура — нет значения', 'Вне сетки — нет значения',
    ]


def test_categorical_well_legend_uses_cluster_labels_from_each_profile():
    xx, yy = np.mgrid[0:300:100, 0:200:100]
    figure = build_categorical_report_map(
        xx, yy, np.zeros(xx.shape, dtype=int), np.array([[100, 0]]), np.array([7]), {7: 'blue'},
        title='Карта кластеров', grid_step=100, method='Nearest',
        wells=[(100, 0, 'Скв. 1', 'red'), (100, 100, 'Скв. 2', 'red')],
        well_clusters=[[('П1', 4, 5), ('П2', -1, 2.5), ('П3', None, 0)], ()],
    )

    assert len(figure.legends) == 1
    assert [label.get_text() for label in figure.legends[0].get_texts()] == [
        'Скв. 1 · П1 (5.00 м) — Кластер 4', 'Скв. 1 · П2 (2.50 м) — Шум (-1)',
        'Скв. 1 · П3 (0.00 м) — нет метки', 'Скв. 2 — нет метки',
    ]
    assert [label.get_text() for label in figure.axes[0].get_legend().get_texts()][0] == 'Кластер 7'


def test_other_categorical_maps_sample_the_visible_grid_and_respect_its_mask():
    xx, yy = np.mgrid[0:300:100, 0:200:100]
    cells = np.ma.array([[0, 0], [1, 1], [0, 0]], mask=[[False, False], [False, False], [False, True]])
    figure = build_categorical_report_map(
        xx, yy, cells, np.array([[0, 0], [100, 0]]), np.array([7, -1]), {7: 'blue', -1: 'gray'},
        title='Категории', grid_step=100, method='Nearest',
        wells=[(10, 50, 'Класс', 'red'), (100, 0, 'Шум', 'red'), (200, 100, 'Пропуск', 'red')],
    )

    assert [label.get_text() for label in figure.legends[0].get_texts()] == [
        'Класс — Кластер 7', 'Шум — Шум (-1)', 'Пропуск — нет метки',
    ]


def test_long_well_legend_leaves_room_for_the_map():
    xx, yy = np.mgrid[0:1000:50, 0:200:50]
    wells = [(i, 100, f'Скважина №{i + 1000}', 'red') for i in range(60)]
    figure = build_categorical_report_map(
        xx, yy, np.zeros(xx.shape), np.array([[50, 0]]), np.array([4]), {4: 'blue'},
        title='Карта', grid_step=50, method='Nearest', wells=wells,
        well_clusters=[[(f'Продольный профиль {i}', 4, 9)] for i in range(len(wells))],
    )

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    map_bounds = figure.axes[0].get_window_extent()
    legend = figure.legends[0]
    assert len(legend.get_texts()) == len(wells)
    assert not map_bounds.overlaps(legend.get_window_extent())
    assert map_bounds.width / figure.dpi > 5


def test_report_labels_follow_valid_observations_and_group_coincident_samples():
    xx, yy = np.mgrid[0:300:100, 0:200:100]
    samples = np.array([[100, 100, 1], [0, 0, np.nan], [100, 100, 2], [100, 100, 3], [200, 0, 2]])
    figure = build_report_map(
        xx, yy, xx / 100, samples, title='Геохимия', parameter='Значение', cmap='viridis',
        grid_step=100, method='Кригинг', show_contours=False,
        point_labels=['Проба 1', 'Пропуск', 'Проба 2', 'Проба 1', ' '],
    )
    labels = figure.axes[0].texts
    assert [(label.xy, label.get_text()) for label in labels] == [((100, 100), 'Проба 1\nПроба 2')]


@pytest.mark.parametrize('show_points', [True, False])
def test_categorical_report_uses_observation_names_only_when_points_are_visible(show_points):
    xx, yy = np.mgrid[0:300:100, 0:200:100]
    figure = build_categorical_report_map(
        xx, yy, np.zeros(xx.shape, dtype=int), np.array([[100, 0]]), np.array([7]), {7: 'blue'},
        title='Геохимия', grid_step=100, method='Nearest', show_points=show_points, point_labels=['Проба 7'],
    )
    assert [label.get_text() for label in figure.axes[0].texts] == (['Проба 7'] if show_points else [])
