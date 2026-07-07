"""Pitch geometry and discretisation.

Coordinate systems
------------------
* **StatsBomb native**: x in [0, 120], y in [0, 80], origin top-left, y
  increasing downward. This is the data's own system.
* **Metric**: x in [0, L], y in [0, W] with L, W in metres (default 105 x 68),
  obtained by independent linear scaling of each axis. Distances in metric
  units are physically interpretable (needed by the control model's speeds).
* **Normalised**: x, y in [0, 1].

We keep the StatsBomb y-orientation throughout (no vertical flip); it is
internally consistent and the choice does not affect rotation-free field
quantities. Attacking-direction normalisation is handled separately in the
ingestion layer, because it depends on team and period, not on geometry.

The continuous-manifold language in the thesis is a convenience: every field
is in fact estimated on the discrete ``Grid`` defined here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pitch:
    """Pitch dimensions and linear transforms between coordinate systems."""

    sb_length: float = 120.0
    sb_width: float = 80.0
    metric_length: float = 105.0
    metric_width: float = 68.0

    def sb_to_metric(self, xy: np.ndarray) -> np.ndarray:
        """Map StatsBomb coordinates -> metric. ``xy`` has shape (..., 2)."""
        xy = np.asarray(xy, dtype=float)
        out = np.empty_like(xy)
        out[..., 0] = xy[..., 0] / self.sb_length * self.metric_length
        out[..., 1] = xy[..., 1] / self.sb_width * self.metric_width
        return out

    def metric_to_normalised(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        out = np.empty_like(xy)
        out[..., 0] = xy[..., 0] / self.metric_length
        out[..., 1] = xy[..., 1] / self.metric_width
        return out

    @property
    def attacked_goal_metric(self) -> np.ndarray:
        """Centre of the goal a left->right attacking team aims at (metric)."""
        return np.array([self.metric_length, self.metric_width / 2.0])


@dataclass(frozen=True)
class Grid:
    """A regular grid over the pitch in **metric** coordinates.

    Cell (i, j) has centre ``centres[i, j]``. ``i`` indexes x (length), ``j``
    indexes y (width). Field arrays are shaped (nx, ny) to match.
    """

    pitch: Pitch
    nx: int = 48
    ny: int = 32

    @property
    def dx(self) -> float:
        return self.pitch.metric_length / self.nx

    @property
    def dy(self) -> float:
        return self.pitch.metric_width / self.ny

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy

    @property
    def x_centres(self) -> np.ndarray:
        return (np.arange(self.nx) + 0.5) * self.dx

    @property
    def y_centres(self) -> np.ndarray:
        return (np.arange(self.ny) + 0.5) * self.dy

    @property
    def centres(self) -> np.ndarray:
        """(nx, ny, 2) array of cell-centre metric coordinates."""
        xx, yy = np.meshgrid(self.x_centres, self.y_centres, indexing="ij")
        return np.stack([xx, yy], axis=-1)

    def cell_of(self, xy_metric: np.ndarray) -> np.ndarray:
        """Index of the containing cell for metric point(s). Clipped to grid.

        Returns an int array of shape (..., 2) giving (i, j).
        """
        xy = np.asarray(xy_metric, dtype=float)
        i = np.clip((xy[..., 0] / self.dx).astype(int), 0, self.nx - 1)
        j = np.clip((xy[..., 1] / self.dy).astype(int), 0, self.ny - 1)
        return np.stack([i, j], axis=-1)

    def flat_centres(self) -> np.ndarray:
        """(nx*ny, 2) cell centres, row-major over (i, j)."""
        return self.centres.reshape(-1, 2)
