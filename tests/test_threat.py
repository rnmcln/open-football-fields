"""Expected-threat (xT) field tests (offline, synthetic + artefact-gated)."""
import json
import pathlib

import numpy as np
import pytest

from ffields import repo_root
from ffields.fields import ExpectedThreat
from ffields.geometry import Grid, Pitch


def _monotone_xt(w=12, l=16):
    """A synthetic xT grid (w, l) increasing toward +x (the attacked goal)."""
    col = np.linspace(0.0, 0.3, l)  # value rises with x index
    return np.tile(col, (w, 1))


def test_field_increases_toward_goal():
    pitch = Pitch()
    xt = ExpectedThreat(_monotone_xt(), pitch, meta={})
    g = Grid(pitch, nx=48, ny=32)
    T = xt.field(g)
    assert T.values.shape == (48, 32)
    # mean threat in the attacking third exceeds the defending third
    assert T.values[-16:].mean() > T.values[:16].mean()


def test_gradient_points_up_x():
    pitch = Pitch()
    xt = ExpectedThreat(_monotone_xt(), pitch, meta={})
    g = Grid(pitch, nx=48, ny=32)
    grad = xt.gradient(g)
    assert grad["grad_x"].values.shape == (48, 32)
    # grad along x is positive on average (threat grows toward goal)
    assert np.nanmean(grad["grad_x"].values) > 0
    # grad along y is ~0 for an x-only surface
    assert abs(np.nanmean(grad["grad_y"].values)) < 1e-6


def test_value_at_clips_to_pitch():
    pitch = Pitch()
    xt = ExpectedThreat(_monotone_xt(), pitch, meta={})
    # points outside the pitch are clipped, not errored
    v = xt.value_at(np.array([[-10.0, -10.0], [200.0, 200.0]]))
    assert np.all(np.isfinite(v))
    assert v[1] > v[0]  # far +x is more threatening than far -x


def test_rate_moves_forward_positive():
    pitch = Pitch()
    xt = ExpectedThreat(_monotone_xt(), pitch, meta={})
    # a forward move (toward +x) should add threat
    dxt = xt.rate_moves([10.0], [34.0], [90.0], [34.0])
    assert dxt[0] > 0
    # a backward move should subtract threat
    dxt_b = xt.rate_moves([90.0], [34.0], [10.0], [34.0])
    assert dxt_b[0] < 0


@pytest.mark.skipif(
    not (repo_root() / "data" / "models" / "xt_grid.json").exists(),
    reason="fitted xT artefact not present (run scripts/fit_xt.py)",
)
def test_real_artefact_shape_and_range():
    pitch = Pitch()
    path = repo_root() / "data" / "models" / "xt_grid.json"
    xt = ExpectedThreat.from_artefact(path, pitch)
    assert xt.xt.shape == (12, 16)
    assert xt.xt.min() >= 0.0
    # xT peaks near the goal and is well under 1
    assert 0.1 < xt.xt.max() < 1.0
    obj = json.loads(pathlib.Path(path).read_text())
    assert obj["n_actions"] > 10000
