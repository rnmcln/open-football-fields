"""Pass-only directness sensitivity for the cross-provider comparison (Chapter 6).

The Wyscout open data record no explicit carries, so on that provider a move is a
pass. To check that the cross-provider agreement of directness is not an artefact of
the differing event taxonomies, this script recomputes StatsBomb directness on the
same men's league season (English Premier League 2015/16) using passes only, dropping
carries, and compares its reliability, discriminant validity, and team-versus-opponent
share with the pass-and-carry definition used in the main analysis. If the pass-only
results are close to the pass-and-carry results, the StatsBomb and Wyscout numbers are
being compared on a like operational definition.

Output: figures/analysis_directness_passonly.json
"""
from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd

from ffields import load_config, repo_root
from ffields.geometry import Grid, Pitch
from ffields.provenance import RunProvenance, seed_everything
from ffields.validation import (
    discriminant_permutation, split_half_reliability, two_way_variance)


def _rows(events_dir, manifest, move_types):
    man = pd.read_parquet(manifest); opp = {}
    for _, r in man.iterrows():
        opp[(r["match_id"], r["home_team"])] = r["away_team"]
        opp[(r["match_id"], r["away_team"])] = r["home_team"]
    rows = []
    for f in sorted(glob.glob(str(events_dir / "*.parquet"))):
        df = pd.read_parquet(f, columns=["type", "team", "x_att", "y_att", "end_x_att", "end_y_att"])
        mid = int(f.split("/")[-1].split(".")[0])
        for team, sub in df.groupby("team"):
            mv = sub[sub["type"].isin(move_types)].dropna(
                subset=["x_att", "y_att", "end_x_att", "end_y_att"])
            if len(mv) < 30:
                continue
            ends = mv[["end_x_att", "end_y_att"]].to_numpy()
            starts = mv[["x_att", "y_att"]].to_numpy()
            rows.append({"match_id": mid, "team": team, "opponent": opp.get((mid, team), "?"),
                         "directness": float((ends[:, 0] - starts[:, 0]).mean())})
    return pd.DataFrame(rows)


def _evaluate(tm, seed, rng):
    teams = tm["team"].unique()
    r = split_half_reliability(tm, "team", "directness", "match_id", n_perm=2000, seed=seed)
    bs = []
    for _ in range(300):
        samp = rng.choice(teams, size=len(teams), replace=True)
        boot = pd.concat([tm[tm["team"] == t].assign(team=f"{t}__{k}") for k, t in enumerate(samp)],
                         ignore_index=True)
        bs.append(split_half_reliability(boot, "team", "directness", "match_id", n_perm=1, seed=0)["pearson_r"])
    disc = discriminant_permutation(tm, "team", "directness", n_perm=2000, seed=seed)
    tw = two_way_variance(tm, "team", "opponent", "directness", n_perm=1000, seed=seed)
    return {
        "n_rows": int(len(tm)),
        "reliability_r": float(r["pearson_r"]),
        "reliability_ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
        "between_team_share": float(disc["between_share"]),
        "team_share_given_opponent": float(tw["a_partial"]),
        "opponent_share_given_team": float(tw["b_partial"]),
    }


def main() -> None:
    cfg = load_config(); seed_everything(cfg["seed"]); root = repo_root()
    figdir = root / cfg["paths"]["figures"]
    rng = np.random.default_rng(cfg["seed"])
    ev = root / "data" / "cache" / "batch_pl" / "events"
    man = root / "data" / "cache" / "batch_pl" / "manifest_2_27.parquet"

    pc = _rows(ev, man, ["Pass", "Carry"])
    po = _rows(ev, man, ["Pass"])
    summary = {
        "competition": "English Premier League 2015/16 (StatsBomb)",
        "pass_and_carry": _evaluate(pc, cfg["seed"], rng),
        "pass_only": _evaluate(po, cfg["seed"], np.random.default_rng(cfg["seed"])),
        "note": ("pass-only drops carries to match the Wyscout event taxonomy; "
                 "close agreement means the cross-provider directness comparison "
                 "is on a like operational definition"),
    }
    (figdir / "analysis_directness_passonly.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    RunProvenance(seed=cfg["seed"], extra={"stage": "directness_passonly"}).write(
        figdir / "analysis_directness_passonly_provenance.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
