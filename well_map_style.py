"""Shared styling rules for wells displayed on maps."""

DEFAULT_WELL_MARKER_COLOR = "red"
WELL_LOG_MARKER_COLOR = "#D500F9"


def well_marker_color(well):
    """Return a bright accent color when a well has logging curves."""
    return WELL_LOG_MARKER_COLOR if well.well_logs else DEFAULT_WELL_MARKER_COLOR
