"""Kinematic pitch-control tests (offline, synthetic)."""
import numpy as np

from ffields.fields import KinematicPitchControl
from ffields.geometry import Grid, Pitch


def _grid():
    return Grid(Pitch(), nx=48, ny=32)


def test_zero_velocity_mirror_antisymmetry():
    g = _grid()
    kpc = KinematicPitchControl(g)
    # attacker and defender mirrored about the pitch centre (x = 52.5)
    att = np.array([[42.5, 34.0]]); deff = np.array([[62.5, 34.0]])
    z = np.zeros((1, 2))
    fr = kpc.estimate(att, z, deff, z)
    # mirroring x swaps the two players' roles, so C(i,j) + C(nx-1-i,j) == 1
    mirrored = fr.values + fr.values[::-1, :]
    assert np.allclose(mirrored, 1.0, atol=1e-6)


def test_velocity_toward_cell_increases_control_there():
    g = _grid()
    kpc = KinematicPitchControl(g)
    att = np.array([[40.0, 34.0]]); deff = np.array([[65.0, 34.0]])
    z = np.zeros((1, 2))
    target = np.array([62.0, 34.0])
    i, j = g.cell_of(target)
    static = kpc.estimate(att, z, deff, z).values[i, j]
    # attacker sprinting toward the contested cell gains control there
    moving = kpc.estimate(att, np.array([[6.0, 0.0]]), deff, z).values[i, j]
    assert moving > static


def test_field_shape_and_bounds():
    g = _grid()
    kpc = KinematicPitchControl(g)
    rng = np.random.default_rng(0)
    att = rng.uniform([0, 0], [105, 68], size=(11, 2))
    deff = rng.uniform([0, 0], [105, 68], size=(11, 2))
    fr = kpc.estimate(att, rng.normal(0, 2, (11, 2)), deff, rng.normal(0, 2, (11, 2)))
    assert fr.values.shape == (48, 32)
    assert fr.values.min() >= 0.0 and fr.values.max() <= 1.0
    assert fr.mask is None  # continuous tracking observes the whole pitch
