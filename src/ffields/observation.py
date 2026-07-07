"""The partial-observation operator O.

Under continuous tracking, O is the identity on player positions. Under open
data it is lossy: at a 360 event we see only the players the broadcast camera
captured, inside a ``visible_area`` polygon, with no velocities. This module
makes O explicit:

* it orients freeze-frame positions and the visible polygon so the team in
  possession attacks +x (reusing the event's ``att_sign``);
* it splits players into possession vs opponent using the ``teammate`` flag;
* it produces a boolean grid mask of observed cells, so fields can be reported
  *masked* outside the visible area rather than silently extrapolated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point, Polygon

from .geometry import Grid, Pitch


def orient_xy(xy_sb: np.ndarray, att_sign: int, pitch: Pitch) -> np.ndarray:
    """StatsBomb coords -> metric, rotated 180 deg if ``att_sign == -1`` so the
    reference team attacks +x. ``xy_sb`` has shape (..., 2)."""
    xy_m = pitch.sb_to_metric(np.asarray(xy_sb, dtype=float))
    if att_sign == 1:
        return xy_m
    out = np.empty_like(xy_m)
    out[..., 0] = pitch.metric_length - xy_m[..., 0]
    out[..., 1] = pitch.metric_width - xy_m[..., 1]
    return out


@dataclass
class FreezeFrame:
    """An oriented 360 snapshot for a single event.

    Positions are metric and oriented so the team in possession attacks +x.
    ``attackers`` are the actor's team (teammate=True or the actor); ``keeper``
    membership is retained separately for optional exclusion.
    """

    attackers: np.ndarray  # (na, 2)
    defenders: np.ndarray  # (nd, 2)
    attacker_keeper: np.ndarray  # (na,) bool
    defender_keeper: np.ndarray  # (nd,) bool
    visible_polygon: Polygon
    att_sign: int

    @classmethod
    def from_raw(cls, raw_360: dict, att_sign: int, pitch: Pitch) -> "FreezeFrame":
        ff = raw_360["freeze_frame"]
        att_xy, att_k, def_xy, def_k = [], [], [], []
        for p in ff:
            loc = orient_xy(np.array(p["location"]), att_sign, pitch)
            if p.get("teammate", False) or p.get("actor", False):
                att_xy.append(loc)
                att_k.append(bool(p.get("keeper", False)))
            else:
                def_xy.append(loc)
                def_k.append(bool(p.get("keeper", False)))
        va = np.asarray(raw_360["visible_area"], dtype=float).reshape(-1, 2)
        va_m = orient_xy(va, att_sign, pitch)
        poly = Polygon(va_m)
        if not poly.is_valid:
            poly = poly.buffer(0)  # repair self-touching polygons
        return cls(
            attackers=np.asarray(att_xy).reshape(-1, 2),
            defenders=np.asarray(def_xy).reshape(-1, 2),
            attacker_keeper=np.asarray(att_k, dtype=bool),
            defender_keeper=np.asarray(def_k, dtype=bool),
            visible_polygon=poly,
            att_sign=att_sign,
        )

    @property
    def n_visible(self) -> int:
        return len(self.attackers) + len(self.defenders)


@dataclass
class ObservationOperator:
    """Maps a visible polygon onto a grid mask of observed cells."""

    grid: Grid

    def visible_mask(self, polygon: Polygon) -> np.ndarray:
        """Boolean (nx, ny): True where the cell centre lies in ``polygon``.

        Vectorised containment via a prepared geometry for speed.
        """
        from shapely.prepared import prep

        centres = self.grid.flat_centres()
        pp = prep(polygon)
        inside = np.fromiter(
            (pp.contains(Point(x, y)) for x, y in centres),
            count=len(centres),
            dtype=bool,
        )
        return inside.reshape(self.grid.nx, self.grid.ny)
