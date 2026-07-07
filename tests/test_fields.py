"""Density-field and descriptor tests (offline, synthetic)."""
import numpy as np

from ffields.fields import EventDensityField, FieldResult
from ffields.geometry import Grid, Pitch


def test_density_integrates_to_one():
    rng = np.random.default_rng(0)
    pts = rng.normal(loc=[52.5, 34.0], scale=[12.0, 8.0], size=(400, 2))
    g = Grid(Pitch(), nx=48, ny=32)
    res = EventDensityField(g).estimate(pts)
    integral = res.values.sum() * g.cell_area
    assert np.isclose(integral, 1.0, atol=1e-6)


def test_entropy_ordering_uniform_gt_concentrated():
    g = Grid(Pitch(), nx=16, ny=16)
    uniform = FieldResult("u", np.ones((16, 16)), g)
    concentrated = np.zeros((16, 16))
    concentrated[8, 8] = 1.0
    conc = FieldResult("c", concentrated, g)
    # a uniform field has maximal entropy; a spike has ~0
    assert uniform.spatial_entropy() > conc.spatial_entropy()
    assert np.isclose(uniform.spatial_entropy(), np.log2(16 * 16))
    assert np.isclose(conc.spatial_entropy(), 0.0)


def test_entropy_rejects_negative_field():
    g = Grid(Pitch(), nx=4, ny=4)
    neg = FieldResult("n", -np.ones((4, 4)), g)
    try:
        neg.spatial_entropy()
    except ValueError:
        return
    raise AssertionError("expected ValueError on negative field")
