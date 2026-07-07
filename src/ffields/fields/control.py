"""Positional pitch control.

This is the **position-only degenerate case** of a time-to-intercept pitch
control model (Spearman, 2018; cf. Fernandez & Bornn, 2018; Taki & Hasegawa,
2000). 360 freeze frames give player *positions* but no velocities, so the
kinematic terms of the full model are unavailable. We therefore assume every
player starts at rest and reaches a cell after a reaction delay plus
constant-max-speed travel:

    tti_p(c) = reaction_time + ||c - x_p|| / max_speed

Control for the possession team at cell c is a softmin aggregation over arrival
times:

    w_p(c)  = exp(-tti_p(c) / tau)
    C_att(c) = sum_{p in att} w_p(c) / sum_{all p} w_p(c)

Honesty notes
-------------
* This is a *simplification*, not the Spearman model. It must not be presented
  as kinematic pitch control. Its role is a defensible, reproducible baseline
  estimable from open 360 data.
* Players outside the camera are unobserved. The result is masked to the
  ``visible_area``; control outside the mask is not reported.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

from ..geometry import Grid
from ..observation import FreezeFrame, ObservationOperator
from . import FieldResult


@dataclass
class PositionalPitchControl:
    grid: Grid
    max_speed_m_s: float = 7.0
    reaction_time_s: float = 0.7
    tti_temperature_s: float = 0.45
    keeper_included: bool = True

    def _tti_weights(self, players: np.ndarray) -> np.ndarray:
        """(n_players, n_cells) softmin weights from arrival times."""
        centres = self.grid.flat_centres()  # (ncells, 2)
        dist = cdist(players, centres)  # (nplayers, ncells)
        tti = self.reaction_time_s + dist / self.max_speed_m_s
        return np.exp(-tti / self.tti_temperature_s)

    def estimate(self, frame: FreezeFrame) -> FieldResult:
        """Compute the possession-team control field for a freeze frame."""
        att, deff = frame.attackers, frame.defenders
        ak, dk = frame.attacker_keeper, frame.defender_keeper
        if not self.keeper_included:
            att = att[~ak] if len(att) else att
            deff = deff[~dk] if len(deff) else deff

        ncells = self.grid.nx * self.grid.ny
        w_att = (
            self._tti_weights(att).sum(axis=0) if len(att) else np.zeros(ncells)
        )
        w_def = (
            self._tti_weights(deff).sum(axis=0) if len(deff) else np.zeros(ncells)
        )
        denom = w_att + w_def
        with np.errstate(invalid="ignore", divide="ignore"):
            control = np.where(denom > 0, w_att / denom, 0.5)
        control = control.reshape(self.grid.nx, self.grid.ny)

        mask = ObservationOperator(self.grid).visible_mask(frame.visible_polygon)
        return FieldResult(
            name="positional_pitch_control",
            values=control,
            grid=self.grid,
            mask=mask,
            meta={
                "model": "positional (zero-velocity) time-to-intercept",
                "max_speed_m_s": self.max_speed_m_s,
                "reaction_time_s": self.reaction_time_s,
                "tti_temperature_s": self.tti_temperature_s,
                "n_attackers": int(len(att)),
                "n_defenders": int(len(deff)),
                "n_visible": int(frame.n_visible),
            },
        )


@dataclass
class KinematicPitchControl:
    """Velocity-bearing time-to-intercept control.

    Generalises :class:`PositionalPitchControl` by letting each player continue
    at their current velocity during the reaction window before sprinting at
    ``max_speed`` toward a cell:

        anchor_p   = x_p + v_p * reaction_time
        tti_p(c)   = reaction_time + ||c - anchor_p|| / max_speed

    Setting all velocities to zero recovers the position-only model **exactly**,
    so the difference between the two is attributable solely to velocity. This is
    the instrument used to quantify the open-data fidelity gap: open 360 data
    carry no velocity, forcing the position-only special case, while continuous
    tracking (Metrica) supplies the velocity the full model needs.

    Unlike the 360 case there is no camera mask: continuous tracking observes the
    whole pitch, so the field is returned unmasked.
    """

    grid: Grid
    max_speed_m_s: float = 7.0
    reaction_time_s: float = 0.7
    tti_temperature_s: float = 0.45

    def _weights(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        centres = self.grid.flat_centres()
        anchor = pos + vel * self.reaction_time_s
        dist = cdist(anchor, centres)
        tti = self.reaction_time_s + dist / self.max_speed_m_s
        return np.exp(-tti / self.tti_temperature_s)

    def estimate(
        self,
        att_pos: np.ndarray,
        att_vel: np.ndarray,
        def_pos: np.ndarray,
        def_vel: np.ndarray,
        name: str = "kinematic_pitch_control",
    ) -> FieldResult:
        """Reference-team (``att``) control over the whole grid."""
        att_pos = np.asarray(att_pos, float).reshape(-1, 2)
        def_pos = np.asarray(def_pos, float).reshape(-1, 2)
        att_vel = np.asarray(att_vel, float).reshape(-1, 2)
        def_vel = np.asarray(def_vel, float).reshape(-1, 2)
        ncells = self.grid.nx * self.grid.ny
        w_att = self._weights(att_pos, att_vel).sum(axis=0) if len(att_pos) else np.zeros(ncells)
        w_def = self._weights(def_pos, def_vel).sum(axis=0) if len(def_pos) else np.zeros(ncells)
        denom = w_att + w_def
        with np.errstate(invalid="ignore", divide="ignore"):
            control = np.where(denom > 0, w_att / denom, 0.5)
        control = control.reshape(self.grid.nx, self.grid.ny)
        speed = float(np.nanmean(np.hypot(att_vel[:, 0], att_vel[:, 1]))) if len(att_vel) else 0.0
        return FieldResult(
            name=name, values=control, grid=self.grid, mask=None,
            meta={
                "model": "kinematic (velocity-bearing) time-to-intercept",
                "max_speed_m_s": self.max_speed_m_s,
                "reaction_time_s": self.reaction_time_s,
                "tti_temperature_s": self.tti_temperature_s,
                "n_attackers": int(len(att_pos)),
                "n_defenders": int(len(def_pos)),
                "mean_attacker_speed_m_s": speed,
            },
        )
