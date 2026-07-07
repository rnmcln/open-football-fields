"""RQ2 robustness-harness tests (offline, synthetic)."""
import numpy as np

from ffields import robustness as rb
from ffields.fields import EventDensityField, FieldResult
from ffields.geometry import Grid, Pitch


def test_spatial_correlation_identity_and_negation():
    g = Grid(Pitch(), nx=16, ny=16)
    rng = np.random.default_rng(0)
    v = rng.random((16, 16))
    a = FieldResult("a", v, g)
    b = FieldResult("b", v.copy(), g)
    c = FieldResult("c", -v, g)
    assert np.isclose(rb.spatial_correlation(a, b), 1.0)
    assert np.isclose(rb.spatial_correlation(a, c), -1.0)


def test_jitter_is_seeded_and_zero_mean():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    pts = np.zeros((1000, 2))
    j1 = rb.jitter(pts, 1.0, rng1)
    j2 = rb.jitter(pts, 1.0, rng2)
    assert np.allclose(j1, j2)  # reproducible given seed
    assert abs(j1.mean()) < 0.1   # ~zero-mean noise


def test_battery_passes_for_stable_operator():
    pitch = Pitch()
    g = Grid(pitch, nx=48, ny=32)
    rng = np.random.default_rng(0)
    pts = rng.normal([52.5, 34.0], [15.0, 9.0], size=(600, 2))
    mk = lambda p: EventDensityField(g).estimate(p)  # noqa: E731
    rep = rb.run_battery("density_noise", make_field=mk, points=pts,
                         sigmas_m=(0.5, 1.0), n_rep=5, rng=rng)
    assert rep.passed["noise"] is True
    assert rep.ok


def test_battery_fails_when_threshold_unreachable():
    pitch = Pitch()
    g = Grid(pitch, nx=48, ny=32)
    rng = np.random.default_rng(0)
    pts = rng.normal([52.5, 34.0], [15.0, 9.0], size=(600, 2))
    mk = lambda p: EventDensityField(g).estimate(p)  # noqa: E731
    # an impossible correlation threshold must fail the gate
    rep = rb.run_battery("density_noise", make_field=mk, points=pts,
                         sigmas_m=(1.0,), min_mean_corr=1.01, n_rep=5, rng=rng)
    assert rep.passed["noise"] is False
    assert not rep.ok


def test_resolution_summary_efficiency_is_stable():
    # entropy efficiency (H / log2 Ncells) is resolution-robust; raw entropy is not
    pitch = Pitch()
    rng = np.random.default_rng(0)
    pts = rng.normal([52.5, 34.0], [15.0, 9.0], size=(800, 2))

    def eff(fr):
        n = fr.values.size
        return fr.spatial_entropy() / np.log2(n)

    grids = [Grid(pitch, nx, ny) for nx, ny in [(32, 22), (48, 32), (64, 44)]]
    rep = rb.run_battery("density_resolution",
                         make_field_for_grid=lambda gg: EventDensityField(gg).estimate(pts),
                         grids=grids, summary=eff, max_resolution_cv=0.1)
    assert rep.passed["resolution"] is True
