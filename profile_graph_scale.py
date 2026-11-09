"""Vertical scale shared by the lower data graph."""

import numpy as np
import pyqtgraph as pg


def fit_profile_graph_y(graph, *, classification=False):
    """Fit all visible data curves, excluding labels and reference lines."""
    lower, upper = 0.0, 1.0
    if not classification:
        bounds = []
        for item in graph.plotItem.items:
            if not isinstance(item, (pg.PlotCurveItem, pg.PlotDataItem)) or not item.isVisible():
                continue
            _, values = item.getData()
            if values is None:
                continue
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                bounds.append((float(values.min()), float(values.max())))
        if bounds:
            lower = min(bound[0] for bound in bounds)
            upper = max(bound[1] for bound in bounds)
            # A constant series still needs a nonzero viewport height.
            if lower == upper:
                margin = max(abs(lower) * 0.01, 0.5)
                lower, upper = lower - margin, upper + margin
    graph.setYRange(lower, upper, padding=0)
