"""Integration tests against real StatsBomb open data.

Marked ``network`` so they are skipped in offline CI. They validate that the
ingestion layer matches the live data structure and that attacking-direction
normalisation behaves sensibly (shots concentrate toward the attacked goal).
"""
import numpy as np
import pytest

from ffields import load_config, repo_root
from ffields.geometry import Pitch
from ffields.ingest import (
    StatsBombClient,
    attach_freeze_frames,
    events_to_frame,
    normalise_attacking_direction,
)
from ffields.observation import FreezeFrame

# A 360-enabled Euro 2020 match (Ukraine vs North Macedonia).
MATCH_ID = 3788758


@pytest.fixture(scope="module")
def client():
    cfg = load_config()
    cache = repo_root() / cfg["paths"]["data_cache"]
    return StatsBombClient(cache_dir=cache, base_url=cfg["statsbomb"]["base_url"])


@pytest.mark.network
def test_events_frame_builds(client):
    raw = client.events(MATCH_ID)
    df = events_to_frame(raw, Pitch())
    assert len(df) > 1000
    assert {"x_m", "y_m", "type", "team"}.issubset(df.columns)
    # locations within metric pitch where present
    xy = df[["x_m", "y_m"]].dropna().to_numpy()
    assert (xy[:, 0].min() >= -1e-6) and (xy[:, 0].max() <= 105 + 1e-6)
    assert (xy[:, 1].min() >= -1e-6) and (xy[:, 1].max() <= 68 + 1e-6)


@pytest.mark.network
def test_attacking_normalisation_orients_shots(client):
    raw = client.events(MATCH_ID)
    df = normalise_attacking_direction(events_to_frame(raw, Pitch()), Pitch())
    shots = df[(df["type"] == "Shot") & df["x_att"].notna()]
    # after orientation every team attacks +x, so shots sit in the attacking
    # half on average and well beyond the centre line
    assert shots["x_att"].mean() > 70.0


@pytest.mark.network
def test_three_sixty_attaches_and_orients(client):
    raw = client.events(MATCH_ID)
    df = normalise_attacking_direction(events_to_frame(raw, Pitch()), Pitch())
    ts = attach_freeze_frames(client.three_sixty(MATCH_ID))
    # pick a passing event that has a freeze frame
    cand = df[(df["type"] == "Pass") & df["id"].isin(ts.keys())]
    assert len(cand) > 0
    row = cand.iloc[0]
    ff = FreezeFrame.from_raw(ts[row["id"]], int(row["att_sign"]), Pitch())
    assert ff.n_visible >= 1
    # oriented positions lie within (a small tolerance of) the pitch
    allpos = np.vstack([p for p in (ff.attackers, ff.defenders) if len(p)])
    assert allpos[:, 0].min() > -5 and allpos[:, 0].max() < 110
