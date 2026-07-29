from types import SimpleNamespace

from well_map_style import DEFAULT_WELL_MARKER_COLOR, WELL_LOG_MARKER_COLOR, well_marker_color


def test_well_with_logging_curves_uses_bright_purple_marker():
    well = SimpleNamespace(well_logs=[object()])

    assert well_marker_color(well) == WELL_LOG_MARKER_COLOR == "#D500F9"


def test_well_without_logging_curves_keeps_default_marker():
    well = SimpleNamespace(well_logs=[])

    assert well_marker_color(well) == DEFAULT_WELL_MARKER_COLOR == "red"
