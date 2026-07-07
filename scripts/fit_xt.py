"""Fit an expected-threat (xT) value surface and export it as a model artefact.

Why this script exists
----------------------
The threat field T reuses the community-standard
*expected threat* (xT) of Singh (2019) rather than reimplementing a Markov
move/shot solver (the anti-decoration rule: reuse validated work). xT is fit
here with ``socceraction`` (the reference open-source implementation) on
StatsBomb open events, and the resulting value grid is written to a small,
provenance-stamped JSON artefact under ``data/models/``.

The artefact is the *only* thing the ffields runtime consumes
(``ffields.fields.threat``); the runtime therefore carries **no** socceraction
dependency and stays on the project's pinned pandas 3.x. ``socceraction`` (and
its pandas<3 constraint) lives behind the optional ``[xt]`` extra and is used
only by this offline fitting step.

Reproducibility and licence
---------------------------
* The grid is an aggregate model parameter array (l x w cell values), not raw
  StatsBomb data; the artefact records the competitions, game/action counts,
  the pinned open-data commit, and library versions.
* Raw events and per-game SPADL actions are cached under the gitignored
  ``data/cache/`` and never redistributed (StatsBomb Public Data User
  Agreement; see ATTRIBUTION.md).
* The script is **resumable**: per-game SPADL actions are cached to parquet, so
  re-running continues where it stopped and only fits once all games are cached.

Run:  python scripts/fit_xt.py
Output: data/models/xt_grid.json  (+ data/cache/spadl/<comp>_<season>/*.parquet)
"""
from __future__ import annotations

import datetime as _dt
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ffields import load_config, repo_root  # noqa: E402
from ffields.provenance import statsbomb_commit_sha  # noqa: E402

# Primary competitions (men's open data with strong coverage; the same source
# as the 360 competitions used elsewhere). Team-level analysis; xT needs events
# only (not 360). (competition_id, season_id, label)
COMPETITIONS = [
    (55, 43, "UEFA Euro 2020"),
    (43, 106, "FIFA World Cup 2022"),
]


def _spadl_cache_dir(root: pathlib.Path, comp: int, season: int) -> pathlib.Path:
    d = root / "data" / "cache" / "spadl" / f"{comp}_{season}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_one_game(SBL, spadl, game) -> pathlib.Path | None:
    """Fetch + convert one game to SPADL actions (left-to-right) and cache it."""
    root = repo_root()
    cdir = _spadl_cache_dir(root, int(game["competition_id"]), int(game["season_id"]))
    out = cdir / f"{int(game['game_id'])}.parquet"
    if out.exists():
        return out
    try:
        ev = SBL.events(int(game["game_id"]))
        acts = spadl.statsbomb.convert_to_actions(ev, int(game["home_team_id"]))
        acts = spadl.play_left_to_right(acts, int(game["home_team_id"]))
        acts.to_parquet(out)
        return out
    except Exception as exc:  # pragma: no cover - network/data robustness
        print(f"  ! game {int(game['game_id'])} failed: {exc}")
        return None


def main() -> int:
    import socceraction
    import socceraction.spadl as spadl
    import socceraction.xthreat as xt
    from socceraction.data.statsbomb import StatsBombLoader

    cfg = load_config()
    root = repo_root()
    SBL = StatsBombLoader(getter="remote")

    # 1. enumerate games and ensure every game's SPADL is cached (resumable)
    all_cached: list[pathlib.Path] = []
    n_games_total = 0
    pending = 0
    for comp, season, label in COMPETITIONS:
        games = SBL.games(competition_id=comp, season_id=season)
        n_games_total += len(games)
        for _, game in games.iterrows():
            cdir = _spadl_cache_dir(root, comp, season)
            out = cdir / f"{int(game['game_id'])}.parquet"
            if out.exists():
                all_cached.append(out)
                continue
            p = _cache_one_game(SBL, spadl, game)
            if p is not None:
                all_cached.append(p)
            else:
                pending += 1

    print(f"games enumerated: {n_games_total}; SPADL cached: {len(all_cached)}; failed: {pending}")
    if len(all_cached) < n_games_total:
        print("Not all games cached yet (likely interrupted). Re-run to continue.")
        # still proceed to fit on what we have only if we have a solid majority
        if len(all_cached) < 0.95 * n_games_total:
            return 1

    # 2. concatenate and fit xT (l x w = 16 x 12, the Singh default)
    frames = [pd.read_parquet(p) for p in all_cached]
    actions = pd.concat(frames, ignore_index=True)
    n_actions = len(actions)
    n_shots = int((actions["type_id"] == spadl.config.actiontypes.index("shot")).sum())

    model = xt.ExpectedThreat(l=16, w=12)
    model.fit(actions)
    grid = np.asarray(model.xT, dtype=float)  # shape (w, l) = (12, 16): rows=y, cols=x

    # 3. export artefact with provenance
    models_dir = root / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    artefact = {
        "model": "expected_threat",
        "reference": "Singh (2019), expected threat; fit via socceraction",
        "socceraction_version": socceraction.__version__,
        "grid_l_x": 16,
        "grid_w_y": 12,
        "axis_order": "rows=y(width,0..W), cols=x(length,0..L); attack toward +x",
        "xT": grid.tolist(),
        "xT_min": float(grid.min()),
        "xT_max": float(grid.max()),
        "competitions": [{"competition_id": c, "season_id": s, "label": l} for c, s, l in COMPETITIONS],
        "n_games": len(all_cached),
        "n_games_enumerated": n_games_total,
        "n_actions": int(n_actions),
        "n_shot_actions": n_shots,
        "statsbomb_commit_pinned": statsbomb_commit_sha(),
        "seed": cfg["seed"],
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "note": (
            "Aggregate xT value surface (model parameters), not raw StatsBomb "
            "data. Associational. attack-normalised (play_left_to_right)."
        ),
    }
    out = models_dir / "xt_grid.json"
    out.write_text(json.dumps(artefact, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in artefact.items() if k != "xT"}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
