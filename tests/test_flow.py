"""Possession-flow field and vector-operator tests (offline, synthetic)."""
import numpy as np

from ffields.fields import PossessionFlowField
from ffields.fields.flow import FlowResult
from ffields.geometry import Grid, Pitch


def _grid():
    return Grid(Pitch(), nx=48, ny=32)


def test_aggregation_mean_displacement():
    g = _grid()
    flow = PossessionFlowField(g, min_count=3)
    # 10 identical moves from (10, 34) with displacement (+5, 0)
    x0 = np.full(10, 10.0); y0 = np.full(10, 34.0)
    x1 = x0 + 5.0; y1 = y0.copy()
    fr = flow.estimate(x0, y0, x1, y1)
    i, j = g.cell_of(np.array([10.0, 34.0]))
    assert np.isclose(fr.jx[i, j], 5.0)
    assert np.isclose(fr.jy[i, j], 0.0)
    assert fr.count[i, j] == 10
    assert fr.mask[i, j]


def test_divergence_of_linear_field_is_constant():
    g = _grid()
    flow = PossessionFlowField(g)
    centres = g.centres
    # analytic field J = (x, 0) -> div J = 1 everywhere
    fr = FlowResult(jx=centres[..., 0].copy(), jy=np.zeros((g.nx, g.ny)),
                    count=np.full((g.nx, g.ny), 99), grid=g,
                    mask=np.ones((g.nx, g.ny), bool))
    div = flow.divergence(fr)
    assert np.allclose(div.values, 1.0, atol=1e-6)


def test_curl_of_rotational_field():
    g = _grid()
    flow = PossessionFlowField(g)
    centres = g.centres
    # J = (0, x) -> curl = dJy/dx - dJx/dy = 1
    fr = FlowResult(jx=np.zeros((g.nx, g.ny)), jy=centres[..., 0].copy(),
                    count=np.full((g.nx, g.ny), 99), grid=g,
                    mask=np.ones((g.nx, g.ny), bool))
    cur = flow.curl(fr)
    assert np.allclose(cur.values, 1.0, atol=1e-6)
    # irrotational field J = (x, 0) has ~zero curl
    fr2 = FlowResult(jx=centres[..., 0].copy(), jy=np.zeros((g.nx, g.ny)),
                     count=np.full((g.nx, g.ny), 99), grid=g,
                     mask=np.ones((g.nx, g.ny), bool))
    assert np.allclose(flow.curl(fr2).values, 0.0, atol=1e-6)


def test_flux_matches_divergence_theorem():
    g = _grid()
    flow = PossessionFlowField(g)
    centres = g.centres
    # J = (x, 0): integral of div over the whole pitch = area; net flux out of
    # the full-grid region should match, up to a one-cell boundary discretisation.
    fr = FlowResult(jx=centres[..., 0].copy(), jy=np.zeros((g.nx, g.ny)),
                    count=np.full((g.nx, g.ny), 99), grid=g,
                    mask=np.ones((g.nx, g.ny), bool))
    region = np.ones((g.nx, g.ny), bool)
    flux = flow.flux(fr, region)
    area = g.pitch.metric_length * g.pitch.metric_width
    assert abs(flux - area) / area < 0.05


def test_bootstrap_ci_shapes_and_order():
    g = _grid()
    flow = PossessionFlowField(g, min_count=1)
    rng = np.random.default_rng(0)
    x0 = rng.uniform(0, 105, 300); y0 = rng.uniform(0, 68, 300)
    x1 = x0 + rng.normal(2, 5, 300); y1 = y0 + rng.normal(0, 5, 300)
    out = flow.bootstrap_ci(x0, y0, x1, y1, n_boot=30, rng=rng)
    assert out["point"].shape == (g.nx, g.ny)
    assert out["lo"].shape == out["hi"].shape == (g.nx, g.ny)
    # lo <= hi wherever both finite
    fin = np.isfinite(out["lo"]) & np.isfinite(out["hi"])
    assert np.all(out["lo"][fin] <= out["hi"][fin] + 1e-9)
