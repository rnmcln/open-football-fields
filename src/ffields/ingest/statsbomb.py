"""StatsBomb open-data ingestion.

Design constraints
------------------
* **Licence**: download on demand, cache locally, never redistribute. The
  cache directory is gitignored.
* **Provenance**: the pinned open-data commit SHA travels with cached files.
* **Honesty about orientation**: StatsBomb event coordinates are absolute
  pitch positions; a team does not always attack +x. We *infer* attacking
  direction per (team, period) empirically (from shot locations, falling back
  to net carry/pass progression) and expose the inference so it can be tested,
  rather than asserting a convention. After normalisation each team attacks
  toward +x in metric coordinates.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests

from ..geometry import Pitch

_DEFAULT_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


@dataclass
class StatsBombClient:
    """Caching reader for StatsBomb open data.

    Parameters
    ----------
    cache_dir : path to a local cache (created if absent; gitignored).
    base_url  : raw-content base for the open-data repo.
    timeout   : per-request timeout (seconds).
    """

    cache_dir: pathlib.Path
    base_url: str = _DEFAULT_BASE
    timeout: float = 60.0

    def __post_init__(self) -> None:
        self.cache_dir = pathlib.Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- low-level cached JSON fetch ------------------------------------------
    def _get_json(self, rel_path: str) -> Any:
        cache_path = self.cache_dir / rel_path
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        url = f"{self.base_url}/{rel_path}"
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # atomic write: temp then replace
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp.write_text(resp.text, encoding="utf-8")
        tmp.replace(cache_path)
        return json.loads(resp.text)

    # -- typed accessors ------------------------------------------------------
    def competitions(self) -> list[dict]:
        return self._get_json("competitions.json")

    def matches(self, competition_id: int, season_id: int) -> list[dict]:
        return self._get_json(f"matches/{competition_id}/{season_id}.json")

    def events(self, match_id: int) -> list[dict]:
        return self._get_json(f"events/{match_id}.json")

    def lineups(self, match_id: int) -> list[dict]:
        return self._get_json(f"lineups/{match_id}.json")

    def three_sixty(self, match_id: int) -> list[dict]:
        return self._get_json(f"three-sixty/{match_id}.json")


# -- event normalisation ------------------------------------------------------
_MOVE_TYPES = {"Pass": "pass", "Carry": "carry", "Shot": "shot"}


def events_to_frame(raw_events: list[dict], pitch: Pitch) -> pd.DataFrame:
    """Flatten raw StatsBomb events into a tidy DataFrame with metric coords.

    Keeps a deliberately small, well-defined column set. ``end_*`` columns are
    populated for passes, carries and shots; NaN otherwise.
    """
    rows: list[dict] = []
    for e in raw_events:
        loc = e.get("location")
        type_name = e["type"]["name"]
        end = None
        sub = _MOVE_TYPES.get(type_name)
        if sub is not None and sub in e and "end_location" in e[sub]:
            end = e[sub]["end_location"]
        rows.append(
            {
                "id": e["id"],
                "index": e["index"],
                "period": e["period"],
                "minute": e.get("minute"),
                "second": e.get("second"),
                "type": type_name,
                "team": e.get("team", {}).get("name"),
                "player": e.get("player", {}).get("name"),
                "possession": e.get("possession"),
                "possession_team": e.get("possession_team", {}).get("name"),
                "x_sb": loc[0] if loc else np.nan,
                "y_sb": loc[1] if loc else np.nan,
                "end_x_sb": end[0] if end else np.nan,
                "end_y_sb": end[1] if end else np.nan,
            }
        )
    df = pd.DataFrame(rows)

    # metric coordinates
    xy = df[["x_sb", "y_sb"]].to_numpy()
    xy_m = pitch.sb_to_metric(xy)
    df["x_m"], df["y_m"] = xy_m[:, 0], xy_m[:, 1]
    exy = df[["end_x_sb", "end_y_sb"]].to_numpy()
    exy_m = pitch.sb_to_metric(exy)
    df["end_x_m"], df["end_y_m"] = exy_m[:, 0], exy_m[:, 1]
    return df


def infer_attacking_direction(df: pd.DataFrame, team: str, period: int, pitch: Pitch) -> int:
    """Infer whether ``team`` attacks +x (returns +1) or -x (returns -1) in
    ``period``.

    Primary signal: median x of the team's shots (shots occur near the attacked
    goal). Fallback: sign of net forward progression of the team's passes and
    carries. Returns +1 if undetermined (with the caveat that this is a guess).
    """
    sub = df[(df["team"] == team) & (df["period"] == period)]
    shots = sub[sub["type"] == "Shot"]
    mid = pitch.metric_length / 2.0
    if len(shots) >= 3 and shots["x_m"].notna().any():
        return 1 if shots["x_m"].median() >= mid else -1
    moves = sub[sub["type"].isin(["Pass", "Carry"])]
    prog = (moves["end_x_m"] - moves["x_m"]).dropna()
    if len(prog) > 0:
        return 1 if prog.median() >= 0 else -1
    return 1


def normalise_attacking_direction(df: pd.DataFrame, pitch: Pitch) -> pd.DataFrame:
    """Add ``*_att`` metric columns oriented so every team attacks toward +x.

    A 180-degree rotation (x -> L-x, y -> W-y) is applied to rows whose
    (team, period) attacks -x. The applied sign is recorded in ``att_sign`` so
    that downstream consumers (e.g. freeze-frame orientation) can reuse it.
    """
    df = df.copy()
    L, W = pitch.metric_length, pitch.metric_width
    signs: dict[tuple[str, int], int] = {}
    for (team, period), _ in df.groupby(["team", "period"]):
        signs[(team, period)] = infer_attacking_direction(df, team, period, pitch)
    df["att_sign"] = [signs.get((t, p), 1) for t, p in zip(df["team"], df["period"])]

    def _orient(x, y, sign):
        x2 = np.where(sign == 1, x, L - x)
        y2 = np.where(sign == 1, y, W - y)
        return x2, y2

    s = df["att_sign"].to_numpy()
    df["x_att"], df["y_att"] = _orient(df["x_m"].to_numpy(), df["y_m"].to_numpy(), s)
    df["end_x_att"], df["end_y_att"] = _orient(
        df["end_x_m"].to_numpy(), df["end_y_m"].to_numpy(), s
    )
    return df


def attach_freeze_frames(three_sixty: list[dict]) -> dict[str, dict]:
    """Index 360 frames by ``event_uuid`` for joining to events.

    Each value is ``{"visible_area": [...], "freeze_frame": [...]}`` in native
    StatsBomb coordinates. Orientation to a team's attacking direction is the
    caller's responsibility (use ``att_sign`` from the event).
    """
    return {f["event_uuid"]: f for f in three_sixty}
