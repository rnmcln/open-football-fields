"""Richer-outcome incremental-value test: expected goals (thesis Chapter 6).

The summary battery and analysis_descriptor_stats.py test whether the spatial
field descriptors add predictive value for a raw *shot count* beyond possession
volume. A shot count weights a tap-in and a half-chance equally. This script
repeats the incremental-value test against a richer outcome, summed StatsBomb
expected goals (xG) per team-match, which weights each shot by its modelled
conversion probability. This is the "richer outcome" extension flagged in the
discussion, implemented here entirely within the open StatsBomb data: the native
per-shot xG already ships in the event stream (shot.statsbomb_xg), so no new model
is trained and no new heavy dependency is added. The lineage of the idea is the
DTAI Sports Analytics Lab Soccer xG package; here we consume StatsBomb's shipped
values rather than re-estimating them.

To keep the comparison fair, both the shot-count (Poisson) and the xG (Gamma,
log link) models are fitted on the *identical* set of team-match rows, namely the
tournament team-matches for which a raw event file is cached. Each model uses a
possession-volume covariate (log number of move actions) and is compared with and
without an opponent-strength control.

Output: figures/analysis_xg_outcome.json (+ .png)
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, spearmanr

from ffields import load_config, repo_root
from ffields.fields import EventDensityField, ExpectedThreat
from ffields.geometry import Grid, Pitch
from ffields.provenance import RunProvenance, seed_everything

MOVE_SP = {0, 21}
SHOT_SP = {11, 12, 13}
DESCS = ["mean_xt_end", "directness", "att_third_frac", "sum_dxt"]


def _team_match_xg(root, game_id):
    """Summed StatsBomb xG per team_id for one match, from the cached raw events."""
    path = root / "data" / "cache" / "events" / f"{game_id}.json"
    if not path.exists():
        return None
    ev = json.loads(path.read_text(encoding="utf-8"))
    agg = {}
    for e in ev:
        if e.get("type", {}).get("name") == "Shot":
            tid = e.get("team", {}).get("id")
            xg = e.get("shot", {}).get("statsbomb_xg", 0.0) or 0.0
            agg[tid] = agg.get(tid, 0.0) + float(xg)
    return agg


def _rows(cfg, pitch, grid, root):
    xt = ExpectedThreat.from_artefact(root / cfg["threat"]["artefact"], pitch)
    rows = []
    for tag in ["55_43", "43_106"]:
        for f in sorted(glob.glob(str(root / "data" / "cache" / "spadl" / tag / "*.parquet"))):
            df = pd.read_parquet(f, columns=["game_id", "team_id", "type_id",
                                             "start_x", "start_y", "end_x", "end_y"])
            gid = int(df["game_id"].iloc[0])
            xg_map = _team_match_xg(root, gid)
            if xg_map is None:
                continue  # restrict to rows with a cached raw event file
            teams = df["team_id"].unique()
            for team in teams:
                sub = df[df["team_id"] == team]
                mv = sub[sub["type_id"].isin(MOVE_SP)].dropna(
                    subset=["start_x", "start_y", "end_x", "end_y"])
                if len(mv) < 30:
                    continue
                if int(team) not in xg_map:
                    continue
                ends = mv[["end_x", "end_y"]].to_numpy()
                starts = mv[["start_x", "start_y"]].to_numpy()
                opp = [t for t in teams if t != team]
                rows.append({"comp": tag, "game_id": gid, "team_id": int(team),
                             "opp_id": int(opp[0]) if opp else -1,
                             "n_moves": int(len(mv)),
                             "shots": int(sub["type_id"].isin(SHOT_SP).sum()),
                             "sum_xg": float(xg_map[int(team)]),
                             "mean_xt_end": float(xt.value_at(ends).mean()),
                             "sum_dxt": float(xt.rate_moves(starts[:, 0], starts[:, 1],
                                                            ends[:, 0], ends[:, 1]).sum()),
                             "directness": float((ends[:, 0] - starts[:, 0]).mean()),
                             "att_third_frac": float((ends[:, 0] > 70.0).mean())})
    d = pd.DataFrame(rows)
    # opponent defensive strength: mean xG the opponent concedes across its games
    sc = []
    for gid, g in d.groupby("game_id"):
        for _, r in g.iterrows():
            opp_xg = g[g["team_id"] == r["opp_id"]]["sum_xg"]
            sc.append({"team_id": r["team_id"],
                       "conceded_xg": float(opp_xg.iloc[0]) if len(opp_xg) else np.nan})
    sc = pd.DataFrame(sc)
    d["opp_def_strength"] = d["opp_id"].map(sc.groupby("team_id")["conceded_xg"].mean())
    return d


def _std(s):
    s = s.astype(float)
    return (s - s.mean()) / s.std()


def main() -> None:
    cfg = load_config(); seed_everything(cfg["seed"]); root = repo_root()
    figdir = root / cfg["paths"]["figures"]; pitch = Pitch()
    grid = Grid(pitch, nx=cfg["grid"]["nx"], ny=cfg["grid"]["ny"])

    tm = _rows(cfg, pitch, grid, root)
    n = len(tm)
    base_X = sm.add_constant(pd.DataFrame({"log_moves": np.log(tm["n_moves"].to_numpy(float))}))

    # outcome relationship: how related is xG to a raw shot count?
    rho, rho_p = spearmanr(tm["shots"], tm["sum_xg"])

    y_shots = tm["shots"].to_numpy(float)
    y_xg = tm["sum_xg"].to_numpy(float)

    # baselines
    p0 = sm.GLM(y_shots, base_X, family=sm.families.Poisson()).fit()
    g0 = sm.GLM(y_xg, base_X, family=sm.families.Gamma(link=sm.families.links.Log())).fit()

    out = {}
    for d in DESCS:
        X1 = base_X.copy(); X1[d] = _std(tm[d])
        X2 = X1.copy()
        X2["opp_def_strength"] = _std(tm["opp_def_strength"].fillna(tm["opp_def_strength"].mean()))

        # shot-count model (Poisson), for side-by-side comparison
        p1 = sm.GLM(y_shots, X1, family=sm.families.Poisson()).fit()
        p2 = sm.GLM(y_shots, X2, family=sm.families.Poisson()).fit()
        lr_p = 2 * (p1.llf - p0.llf)

        # xG model (Gamma, log link)
        g1 = sm.GLM(y_xg, X1, family=sm.families.Gamma(link=sm.families.links.Log())).fit()
        g2 = sm.GLM(y_xg, X2, family=sm.families.Gamma(link=sm.families.links.Log())).fit()
        lr_g = 2 * (g1.llf - g0.llf)

        out[d] = {
            "shots_poisson": {
                "coef": float(p1.params[d]), "p_value": float(p1.pvalues[d]),
                "lr_stat_vs_baseline": float(lr_p), "lr_p": float(chi2.sf(lr_p, 1)),
                "coef_with_opp_control": float(p2.params[d]),
                "p_with_opp_control": float(p2.pvalues[d]),
            },
            "xg_gamma_log": {
                "coef": float(g1.params[d]), "p_value": float(g1.pvalues[d]),
                "lr_stat_vs_baseline": float(lr_g), "lr_p": float(chi2.sf(lr_g, 1)),
                "aic_baseline": float(g0.aic), "aic_with_descriptor": float(g1.aic),
                "coef_with_opp_control": float(g2.params[d]),
                "p_with_opp_control": float(g2.pvalues[d]),
                "deviance_explained_increment": float(
                    (g0.deviance - g1.deviance) / g0.null_deviance),
            },
        }

    summary = {
        "n_team_match_rows": int(n),
        "n_games": int(tm["game_id"].nunique()),
        "outcome_note": ("summed StatsBomb xG per team-match (native shot.statsbomb_xg); "
                         "Gamma GLM with log link; identical rows as the Poisson shot model"),
        "shots_vs_xg_spearman": {"rho": float(rho), "p_value": float(rho_p)},
        "sum_xg_summary": {"min": float(tm["sum_xg"].min()),
                           "median": float(tm["sum_xg"].median()),
                           "max": float(tm["sum_xg"].max())},
        "incremental_value": out,
    }
    (figdir / "analysis_xg_outcome.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RunProvenance(seed=cfg["seed"], extra={"stage": "xg_outcome"}).write(
        figdir / "analysis_xg_outcome_provenance.json")

    # figure: descriptor coefficient on each outcome (standardised effect)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(DESCS)); w = 0.38
    cp = [out[d]["shots_poisson"]["coef"] for d in DESCS]
    cg = [out[d]["xg_gamma_log"]["coef"] for d in DESCS]
    ax.bar(x - w / 2, cp, w, label="shots (Poisson)", color="#7fbf7b")
    ax.bar(x + w / 2, cg, w, label="xG (Gamma, log)", color="#1b7837")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(DESCS, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("coefficient on +1 SD descriptor (log scale)")
    ax.set_title(f"Incremental value for a richer outcome: expected goals\n"
                 f"summed xG vs raw shot count, identical {n} team-match rows", fontsize=11)
    ax.legend()
    fig.tight_layout(); fig.savefig(figdir / "analysis_xg_outcome.png", dpi=300); plt.close(fig)

    print(json.dumps(summary, indent=1)[:2200])


if __name__ == "__main__":
    main()
