"""Metrica tracking-ingestion tests (network-gated, real open sample data)."""
import numpy as np
import pytest

from ffields.geometry import Pitch
from ffields.ingest import MetricaClient, parse_tracking


@pytest.mark.network
def test_parse_sample_game_one(tmp_path):
    pitch = Pitch()
    cl = MetricaClient(cache_dir=tmp_path / "metrica")
    m = parse_tracking(cl.tracking_csv(1, "home"), cl.tracking_csv(1, "away"), pitch)
    # ~25 fps, two outfield+GK squads of ~14 tracked ids each
    assert m.n_frames > 100000
    assert 24 <= m.meta["fps"] <= 26
    assert m.meta["n_home"] >= 11 and m.meta["n_away"] >= 11
    # a mid-match frame yields finite positions and velocities within the pitch
    hp, hv, ap, av, ball = m.frame_arrays(50000)
    assert hp.shape[1] == 2 and ap.shape[1] == 2
    assert np.isfinite(hp).all() and np.isfinite(hv).all()
    assert (hp[:, 0] >= -5).all() and (hp[:, 0] <= 110).all()
    # velocities are capped to a plausible sprint ceiling
    assert np.hypot(hv[:, 0], hv[:, 1]).max() <= 12.0 + 1e-6
