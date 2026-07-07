"""Threat field T: expected threat (xT) as a value surface.

Design and provenance
---------------------
This module deliberately does **not** reimplement a Markov move/shot solver.
The threat surface is the community-standard *expected threat* (xT) of Singh
(2019), fit once offline with ``socceraction`` (see ``scripts/fit_xt.py``) and
exported to a small JSON artefact (``data/models/xt_grid.json``). Reusing the
reference implementation is the anti-decoration choice and keeps T comparable
with the wider literature. The runtime here consumes only the artefact, so it
carries no socceraction dependency.

What this module adds on top of the artefact
-------------------------------------------
* resampling of the native (coarse, 16x12) xT surface onto the project ``Grid``
  by bilinear interpolation, returned as a :class:`FieldResult`;
* the spatial gradient ``grad T`` (per-metre), the field-operator the thesis
  asks for, with a magnitude field;
* point queries and an xT action value ``delta xT = T(end) - T(start)`` for
  passes/carries, the standard literature quantity.

Honesty notes
-------------
* xT is **associational**: it scores locations by the historical value of
  possession reaching them, not by any causal effect of an action.
* The surface is fit on recorded open events; it inherits their coverage.
* xT is defined over the whole pitch (it needs no camera), so unlike pitch
  control it is reported unmasked; partial observation enters T only through the
  events used to *evaluate* moves, not through the surface itself.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..geometry import Grid, Pitch
from . import FieldResult


@dataclass
class ExpectedThreat:
    """An xT value surface resampled onto the project grid.

    Parameters
    ----------
    xt : np.ndarray
        Native xT grid of shape ``(w, l)`` = ``(n_y, n_x)``; rows index y
        (width, 0..W), columns index x (length, 0..L), attack toward +x. This
        is socceraction's convention.
    pitch : Pitch
        Pitch geometry giving the metric extent the grid spans.
    meta : dict
        Provenance carried from the artefact (competitions, counts, commit).
    """

    xt: np.ndarray
    pitch: Pitch
    meta: dict[str, Any]

    def __post_init__(self) -> None:
        self.xt = np.asarray(self.xt, dtype=float)
        if self.xt.ndim != 2:
            raise ValueError("xT grid must be 2-D (w, l)")
        w, l = self.xt.shape
        L, W = self.pitch.metric_length, self.pitch.metric_width
        # cell-centre coordinates of the native xT grid (metric)
        self._x_centres = (np.arange(l) + 0.5) * (L / l)
        self._y_centres = (np.arange(w) + 0.5) * (W / w)
        # interpolator indexed (x, y): transpose xt from (w,l)->(l,w)
        self._interp = RegularGridInterpolator(
            (self._x_centres, self._y_centres),
            self.xt.T,
            method="linear",
            bounds_error=False,
            fill_value=None,  # extrapolate at the thin border outside cell centres
        )

    # -- constructors ---------------------------------------------------------
    @classmethod
    def from_artefact(cls, path: str | pathlib.Path, pitch: Pitch) -> "ExpectedThreat":
        """Load an xT surface from the JSON artefact written by ``fit_xt.py``."""
        path = pathlib.Path(path)
        obj = json.loads(path.read_text(encoding="utf-8"))
        xt = np.asarray(obj["xT"], dtype=float)
        meta = {k: v for k, v in obj.items() if k != "xT"}
        meta["artefact_path"] = str(path)
        return cls(xt=xt, pitch=pitch, meta=meta)

    # -- queries --------------------------------------------------------------
    def value_at(self, xy_metric: np.ndarray) -> np.ndarray:
        """xT value at metric point(s) of shape (..., 2). Coords are clipped to
        the pitch before interpolation so border points are well defined."""
        xy = np.asarray(xy_metric, dtype=float)
        L, W = self.pitch.metric_length, self.pitch.metric_width
        x = np.clip(xy[..., 0], 0.0, L)
        y = np.clip(xy[..., 1], 0.0, W)
        pts = np.stack([x, y], axis=-1)
        return self._interp(pts)

    def field(self, grid: Grid, name: str = "expected_threat") -> FieldResult:
        """Resample xT onto ``grid`` cell centres -> (nx, ny) FieldResult."""
        centres = grid.centres  # (nx, ny, 2)
        vals = self.value_at(centres.reshape(-1, 2)).reshape(grid.nx, grid.ny)
        return FieldResult(
            name=name,
            values=vals,
            grid=grid,
            mask=None,
            meta={"estimator": "xT (Singh 2019) via socceraction artefact", **self.meta},
        )

    def gradient(self, grid: Grid) -> dict[str, FieldResult]:
        """Spatial gradient of T on ``grid`` (per-metre), plus its magnitude.

        ``grad T`` points toward increasing threat (toward the attacked goal).
        Returned as fields ``grad_x``, ``grad_y`` and ``grad_mag``.
        """
        t = self.field(grid)
        gx, gy = np.gradient(t.values, grid.dx, grid.dy, edge_order=2)
        mag = np.hypot(gx, gy)
        common = {"derived_from": "expected_threat", "units": "xT per metre"}
        return {
            "grad_x": FieldResult("xT_grad_x", gx, grid, meta=common),
            "grad_y": FieldResult("xT_grad_y", gy, grid, meta=common),
            "grad_mag": FieldResult("xT_grad_mag", mag, grid, meta=common),
        }

    def rate_moves(
        self,
        x0: np.ndarray,
        y0: np.ndarray,
        x1: np.ndarray,
        y1: np.ndarray,
    ) -> np.ndarray:
        """Standard xT action value: ``delta xT = T(end) - T(start)`` for moves
        (passes/carries) given attack-oriented metric start/end coordinates."""
        start = np.stack([np.asarray(x0, float), np.asarray(y0, float)], axis=-1)
        end = np.stack([np.asarray(x1, float), np.asarray(y1, float)], axis=-1)
        return self.value_at(end) - self.value_at(start)
