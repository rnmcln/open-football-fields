"""Event-density field.

Spatial intensity of located events (pass origins, pressures, shots, ...) via a
Gaussian KDE evaluated on the grid. This is the most robust field on open data
and a sensible backbone. The result integrates to 1 over the pitch (a density),
so its spatial entropy is directly interpretable.

Caveat carried in ``meta``: the density of *recorded* events conflates true
activity with where the ball went and with annotation coverage. It is not a
density of play, only of logged events.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import gaussian_kde

from ..geometry import Grid
from . import FieldResult


@dataclass
class EventDensityField:
    grid: Grid
    bandwidth: str | float = "scott"

    def estimate(self, points_metric: np.ndarray, name: str = "event_density") -> FieldResult:
        """KDE density from (n, 2) metric points, evaluated on grid centres.

        Returns a field normalised to integrate to ~1 over the pitch. Requires
        at least 3 non-degenerate points.
        """
        pts = np.asarray(points_metric, dtype=float)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) < 3:
            raise ValueError("EventDensityField needs >= 3 finite points")

        kde = gaussian_kde(pts.T, bw_method=self.bandwidth)
        centres = self.grid.flat_centres().T  # (2, ncells)
        dens = kde(centres).reshape(self.grid.nx, self.grid.ny)

        # renormalise so the discrete integral over the pitch is 1
        integral = dens.sum() * self.grid.cell_area
        if integral > 0:
            dens = dens / integral

        return FieldResult(
            name=name,
            values=dens,
            grid=self.grid,
            mask=None,
            meta={
                "estimator": "gaussian_kde",
                "bandwidth": self.bandwidth,
                "n_points": int(len(pts)),
                "caveat": "density of recorded events, not of play",
            },
        )
