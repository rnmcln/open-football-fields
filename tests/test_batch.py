"""Batch-ingestion tests (network-gated, real StatsBomb open data)."""
import pathlib
import tempfile

import pytest

from ffields.geometry import Pitch
from ffields.ingest import BatchIngestor, StatsBombClient


@pytest.mark.network
def test_ingest_competition_limit_two(tmp_path):
    pitch = Pitch()
    client = StatsBombClient(cache_dir=tmp_path / "cache")
    ing = BatchIngestor(client=client, pitch=pitch, out_dir=tmp_path / "out")
    # Euro 2020 (competition 55, season 43); first two matches only
    manifest = ing.ingest_competition(55, 43, limit=2)
    assert len(manifest) == 2
    for col in ["match_id", "n_events", "n_passes", "has_360", "n_360_frames"]:
        assert col in manifest.columns
    # Euro 2020 matches carry 360 data
    assert manifest["has_360"].all()
    assert (manifest["n_events"] > 0).all()
    # manifest + provenance written
    assert (tmp_path / "out" / "manifest_55_43.parquet").exists()
    assert (tmp_path / "out" / "provenance_55_43.json").exists()
    # per-match event parquet written
    assert (tmp_path / "out" / "events" / f"{int(manifest.iloc[0]['match_id'])}.parquet").exists()
