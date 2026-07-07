"""Observation-operator and pitch-control tests on synthetic freeze frames.

These run fully offline: we hand-build a freeze frame so the expected control
geometry is known.
"""
import numpy as np
from shapely.geometry import Polygon

from ffields.fields import PositionalPitchControl
from ffields.geometry import Grid, Pitch
from ffields.observation import FreezeFrame, ObservationOperator, orient_xy


def _full_pitch_polygon(pitch: Pitch) -> Polygon:
    L, W = pitch.metric_length, pitch.metric_width
    return Polygon([(0, 0), (L, 0), (L, W), (0, W)])


def test_orient_flips_when_negative():
    p = Pitch()
    xy = np.array([0.0, 0.0])  # sb origin -> metric origin
    assert np.allclose(orient_xy(xy, 1, p), [0, 0])
    # att_sign -1 rotates 180 deg about pitch centre
    assert np.allclose(orient_xy(xy, -1, p), [105, 68])


def test_visible_mask_full_and_partial():
    g = Grid(Pitch(), nx=20, ny=10)
    oo = ObservationOperator(g)
    full = oo.visible_mask(_full_pitch_polygon(g.pitch))
    assert full.all()
    half = oo.visible_mask(Polygon([(0, 0), (52.5, 0), (52.5, 68), (0, 68)]))
    # left half observed, right half not
    assert half[: g.nx // 2].all()
    assert not half[g.nx // 2 :].any()


def test_control_favours_closer_team():
    """A lone attacker near +x goal should dominate control there vs a distant
    defender near the halfway line."""
    p = Pitch()
    g = Grid(p, nx=48, ny=32)
    frame = FreezeFrame(
        attackers=np.array([[95.0, 34.0]]),
        defenders=np.array([[52.5, 34.0]]),
        attacker_keeper=np.array([False]),
        defender_keeper=np.array([False]),
        visible_polygon=_full_pitch_polygon(p),
        att_sign=1,
    )
    res = PositionalPitchControl(g).estimate(frame)
    # control is a probability surface in [0,1]
    v = res.values
    assert np.nanmin(v) >= 0.0 and np.nanmax(v) <= 1.0
    # near the attacker, control strongly favours attackers
    cij = g.cell_of(np.array([95.0, 34.0]))
    assert v[tuple(cij)] > 0.8
    # near the defender, it favours defenders
    dij = g.cell_of(np.array([52.5, 34.0]))
    assert v[tuple(dij)] < 0.2


def test_control_mask_blocks_unobserved_region():
    p = Pitch()
    g = Grid(p, nx=24, ny=16)
    frame = FreezeFrame(
        attackers=np.array([[60.0, 34.0]]),
        defenders=np.array([[40.0, 34.0]]),
        attacker_keeper=np.array([False]),
        defender_keeper=np.array([False]),
        visible_polygon=Polygon([(0, 0), (52.5, 0), (52.5, 68), (0, 68)]),
        att_sign=1,
    )
    res = PositionalPitchControl(g).estimate(frame)
    masked = res.masked_values()
    # right half (unobserved) must be NaN
    assert np.isnan(masked[g.nx // 2 :]).all()
