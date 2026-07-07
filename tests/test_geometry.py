"""Geometry tests: coordinate transforms and grid indexing."""
import numpy as np

from ffields.geometry import Grid, Pitch


def test_sb_to_metric_corners():
    p = Pitch()
    corners_sb = np.array([[0, 0], [120, 80], [60, 40]], dtype=float)
    m = p.sb_to_metric(corners_sb)
    assert np.allclose(m[0], [0, 0])
    assert np.allclose(m[1], [105, 68])
    assert np.allclose(m[2], [52.5, 34.0])


def test_grid_partition_and_area():
    g = Grid(Pitch(), nx=48, ny=32)
    # cell areas tile the pitch exactly
    assert np.isclose(g.cell_area * g.nx * g.ny, 105 * 68)
    assert g.centres.shape == (48, 32, 2)
    assert g.flat_centres().shape == (48 * 32, 2)


def test_cell_of_bounds_and_centre():
    g = Grid(Pitch(), nx=10, ny=10)
    # point at origin -> cell (0,0); far corner clipped to (9,9)
    assert tuple(g.cell_of(np.array([0.0, 0.0]))) == (0, 0)
    assert tuple(g.cell_of(np.array([105.0, 68.0]))) == (9, 9)
    # a centre maps back to its own cell
    c = g.centres[3, 7]
    assert tuple(g.cell_of(c)) == (3, 7)
