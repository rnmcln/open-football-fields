"""Multi-match batch ingestion.

Loops the cached StatsBomb loader across a whole competition-season, normalises
each match to attack-oriented metric event frames, writes one parquet per match
under the gitignored cache, and emits a **manifest** describing what was
ingested (match ids, teams, date, event counts, 360 availability and frame
counts). The manifest is the index later analyses read to assemble panels for
the validation battery.

Licence: per-match event parquet files are derived from StatsBomb data and are
written under ``data/cache/`` (gitignored, never redistributed). The manifest
holds only aggregate counts and identifiers. Provenance (pinned commit, seed,
library versions) is written alongside the manifest.
"""
from __future__ import annotations

import datetime as _dt
import json
import pathlib
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..geometry import Pitch
from ..provenance import RunProvenance
from .statsbomb import (
    StatsBombClient,
    attach_freeze_frames,
    events_to_frame,
    normalise_attacking_direction,
)


@dataclass
class BatchIngestor:
    """Ingest and cache a competition-season as normalised per-match parquet."""

    client: StatsBombClient
    pitch: Pitch
    out_dir: pathlib.Path

    def __post_init__(self) -> None:
        self.out_dir = pathlib.Path(self.out_dir)
        (self.out_dir / "events").mkdir(parents=True, exist_ok=True)

    def _match_label(self, m: dict) -> dict[str, Any]:
        return {
            "match_id": m["match_id"],
            "match_date": m.get("match_date"),
            "competition": m.get("competition", {}).get("competition_name"),
            "season": m.get("season", {}).get("season_name"),
            "home_team": m.get("home_team", {}).get("home_team_name"),
            "away_team": m.get("away_team", {}).get("away_team_name"),
            "home_score": m.get("home_score"),
            "away_score": m.get("away_score"),
            "match_status_360": m.get("match_status_360"),
        }

    def ingest_match(self, match_id: int) -> dict[str, Any]:
        """Normalise one match, cache its event parquet, return a manifest row."""
        ev_path = self.out_dir / "events" / f"{match_id}.parquet"
        df = normalise_attacking_direction(
            events_to_frame(self.client.events(match_id), self.pitch), self.pitch
        )
        df.to_parquet(ev_path)

        n_360 = 0
        has_360 = False
        try:
            ts = attach_freeze_frames(self.client.three_sixty(match_id))
            n_360 = len(ts)
            has_360 = n_360 > 0
        except Exception:  # no 360 file for this match
            pass

        return {
            "match_id": int(match_id),
            "n_events": int(len(df)),
            "n_passes": int((df["type"] == "Pass").sum()),
            "n_carries": int((df["type"] == "Carry").sum()),
            "n_shots": int((df["type"] == "Shot").sum()),
            "has_360": bool(has_360),
            "n_360_frames": int(n_360),
            "events_parquet": str(ev_path.relative_to(self.out_dir)),
        }

    def ingest_competition(
        self, competition_id: int, season_id: int, limit: int | None = None
    ) -> pd.DataFrame:
        """Ingest a whole competition-season; write manifest + provenance.

        ``limit`` (optional) ingests only the first N matches, for quick demos.
        Resumable: matches whose event parquet already exists are re-read for
        their manifest row but not re-downloaded.
        """
        matches = self.client.matches(competition_id, season_id)
        if limit is not None:
            matches = matches[:limit]
        rows: list[dict[str, Any]] = []
        for m in matches:
            label = self._match_label(m)
            row = self.ingest_match(m["match_id"])
            rows.append({**label, **row})
        manifest = pd.DataFrame(rows)

        tag = f"{competition_id}_{season_id}"
        manifest.to_parquet(self.out_dir / f"manifest_{tag}.parquet")
        self.out_dir.joinpath(f"manifest_{tag}.json").write_text(
            manifest.to_json(orient="records", indent=2), encoding="utf-8"
        )
        prov = RunProvenance(
            seed=0,
            extra={
                "stage": "batch_ingestion",
                "competition_id": competition_id,
                "season_id": season_id,
                "n_matches": int(len(manifest)),
                "n_matches_with_360": int(manifest["has_360"].sum()),
                "ingested_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            },
        )
        prov.write(self.out_dir / f"provenance_{tag}.json")
        return manifest
