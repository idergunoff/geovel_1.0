import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pyqtgraph as pg
import pytest

from profile_graph_scale import fit_profile_graph_y


@pytest.fixture
def graph():
    app = pg.mkQApp()
    widget = pg.PlotWidget()
    yield widget
    widget.close()


def test_range_includes_all_curves_and_ignores_annotations_and_nonfinite_values(graph):
    graph.addItem(pg.PlotCurveItem([10, 20, float('nan'), float('inf')]))
    graph.addItem(pg.PlotCurveItem([-2, 25]))
    graph.addItem(pg.InfiniteLine(pos=1000, angle=0))
    label = pg.TextItem('Parameter')
    label.setPos(0, -1000)
    graph.addItem(label)

    fit_profile_graph_y(graph)

    assert graph.viewRange()[1] == pytest.approx([-2, 25])


def test_switching_between_classification_and_regression_resets_range(graph):
    graph.addItem(pg.PlotCurveItem([-0.05, 1.05]))
    fit_profile_graph_y(graph, classification=True)
    assert graph.viewRange()[1] == pytest.approx([0, 1])

    graph.clear()
    graph.addItem(pg.PlotCurveItem([100, 250]))
    fit_profile_graph_y(graph)
    assert graph.viewRange()[1] == pytest.approx([100, 250])

    fit_profile_graph_y(graph, classification=True)
    assert graph.viewRange()[1] == pytest.approx([0, 1])


@pytest.mark.parametrize('value', [0, -100, 100])
def test_constant_series_has_finite_nonzero_range(graph, value):
    graph.addItem(pg.PlotCurveItem([value, value]))
    fit_profile_graph_y(graph)
    lower, upper = graph.viewRange()[1]
    assert lower < value < upper


def test_empty_and_nonfinite_data_reset_previous_scale(graph):
    graph.setYRange(100, 250)
    graph.addItem(pg.PlotCurveItem([float('nan'), float('inf')]))
    fit_profile_graph_y(graph)
    assert graph.viewRange()[1] == pytest.approx([0, 1])
