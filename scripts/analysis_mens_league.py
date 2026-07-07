"""Men's-league external validation (thesis Chapter 6).

Repeats the descriptor validation on a complete men's league season (English
Premier League 2015/16; 380 matches, 20 teams, 38 per team), the best-powered
reliability test in the thesis, and compares it with the two women's seasons.
This tests whether the descriptor results generalise across sex and league, the
principal external-validity gap identified in review.

Output: figures/analysis_mens_league.json (+ .png)
"""
from __future__ import annotations

import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ffields import load_config, repo_root
from ffields.fields import EventDensityField, ExpectedThreat
from ffields.geometry import Grid, Pitch
from ffields.provenance import RunProvenance, seed_everything
from ffields.validation import (
    discriminant_permutation, split_half_reliability, two_way_variance)

DESCS = ["mean_xt_end", "entropy_end", "directness", "att_third_frac", "sum_dxt"]


def assemble(cfg, pitch, grid, root, events_dir, manifest):
    xt = ExpectedThreat.from_artefact(root / cfg["threat"]["artefact"], pitch)
    man = pd.read_parquet(manifest); opp = {}
    for _, r in man.iterrows():
        opp[(r["match_id"], r["home_team"])] = r["away_team"]
        opp[(r["match_id"], r["away_team"])] = r["home_team"]
    rows = []
    for f in sorted(glob.glob(str(events_dir / "*.parquet"))):
        df = pd.read_parquet(f, columns=["type", "team", "x_att", "y_att", "end_x_att", "end_y_att"])
        mid = int(f.split("/")[-1].split(".")[0])
        for team, sub in df.groupby("team"):
            mv = sub[sub["type"].isin(["Pass", "Carry"])].dropna(subset=["x_att","y_att","end_x_att","end_y_att"])
            if len(mv) < 30:
                continue
            ends = mv[["end_x_att","end_y_att"]].to_numpy(); starts = mv[["x_att","y_att"]].to_numpy()
            rows.append({"match_id": mid, "team": team, "opponent": opp.get((mid, team), "?"),
                         "mean_xt_end": float(xt.value_at(ends).mean()),
                         "sum_dxt": float(xt.rate_moves(starts[:,0],starts[:,1],ends[:,0],ends[:,1]).sum()),
                         "entropy_end": float(EventDensityField(grid).estimate(ends).spatial_entropy()),
                         "directness": float((ends[:,0]-starts[:,0]).mean()),
                         "att_third_frac": float((ends[:,0]>70.0).mean())})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config(); seed_everything(cfg["seed"]); root = repo_root()
    figdir = root / cfg["paths"]["figures"]; pitch = Pitch()
    grid = Grid(pitch, nx=cfg["grid"]["nx"], ny=cfg["grid"]["ny"]); rng = np.random.default_rng(cfg["seed"])

    tm = assemble(cfg, pitch, grid, root,
                  root / "data" / "cache" / "batch_pl" / "events",
                  root / "data" / "cache" / "batch_pl" / "manifest_2_27.parquet")
    teams = tm["team"].unique()

    rel, disc, tw = {}, {}, {}
    for d in DESCS:
        disc[d] = discriminant_permutation(tm, "team", d, n_perm=2000, seed=cfg["seed"])
        r = split_half_reliability(tm, "team", d, "match_id", n_perm=2000, seed=cfg["seed"])
        # bootstrap CI on reliability (resample teams)
        bs = []
        for _ in range(300):
            samp = rng.choice(teams, size=len(teams), replace=True)
            boot = pd.concat([tm[tm["team"] == t].assign(team=f"{t}__{k}") for k, t in enumerate(samp)],
                             ignore_index=True)
            bs.append(split_half_reliability(boot, "team", d, "match_id", n_perm=1, seed=0)["pearson_r"])
        rel[d] = {"r": float(r["pearson_r"]), "p": float(r["p_value"]),
                  "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5))}
        tw[d] = two_way_variance(tm, "team", "opponent", d, n_perm=1000, seed=cfg["seed"])

    # comparison with women's seasons
    prev = {}
    for f, lbl in [("demo5_summary.json", "FAWSL_2018_19"), ("demo6_summary.json", "FAWSL_2019_20")]:
        p = figdir / f
        if p.exists():
            j = json.loads(p.read_text())
            key = "reliability_league" if "reliability_league" in j else "reliability_2019_20"
            prev[lbl] = {d: j[key][d]["pearson_r"] for d in DESCS}

    summary = {
        "competition": "English Premier League 2015/16",
        "n_team_match_rows": int(len(tm)), "n_teams": int(len(teams)),
        "median_matches_per_team": int(tm.groupby("team").size().median()),
        "discriminant": {d: disc[d] for d in DESCS},
        "reliability_pl": rel,
        "team_vs_opponent_pl": {d: tw[d] for d in DESCS},
        "reliability_womens": prev,
    }
    (figdir / "analysis_mens_league.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RunProvenance(seed=cfg["seed"], extra={"stage": "mens_league", "competition": "PL2015_16"}).write(
        figdir / "analysis_mens_league_provenance.json")

    # figure: reliability across three competitions + PL variance decomposition
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    x = np.arange(len(DESCS))
    pl = [rel[d]["r"] for d in DESCS]
    lo = [rel[d]["r"] - rel[d]["lo"] for d in DESCS]; hi = [rel[d]["hi"] - rel[d]["r"] for d in DESCS]
    axes[0].errorbar(x - 0.15, pl, yerr=[lo, hi], fmt="o", color="#08519c", capsize=4, label="PL 2015/16 (men)")
    if "FAWSL_2018_19" in prev:
        axes[0].scatter(x, [prev["FAWSL_2018_19"][d] for d in DESCS], marker="s", color="#1b7837", label="FAWSL 18/19")
    if "FAWSL_2019_20" in prev:
        axes[0].scatter(x + 0.15, [prev["FAWSL_2019_20"][d] for d in DESCS], marker="^", color="#5aae61", label="FAWSL 19/20")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(DESCS, rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("split-half reliability r"); axes[0].set_ylim(-0.6, 1.05); axes[0].legend(fontsize=8)
    axes[0].set_title("A. Reliability generalises across sex and league\n"
                      "(PL with 95% bootstrap CIs; women's seasons for comparison)", fontsize=11)

    team = [tw[d]["a_partial"] for d in DESCS]; oppo = [tw[d]["b_partial"] for d in DESCS]
    resid = [tw[d]["residual"] for d in DESCS]
    axes[1].bar(x, team, 0.6, color="#2166ac", label="team | opponent")
    axes[1].bar(x, oppo, 0.6, bottom=team, color="#f4a582", label="opponent | team")
    axes[1].bar(x, resid, 0.6, bottom=np.array(team)+np.array(oppo), color="#dddddd", label="residual")
    axes[1].set_xticks(x); axes[1].set_xticklabels(DESCS, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("variance share"); axes[1].legend(fontsize=8)
    axes[1].set_title("B. Team vs opponent decomposition (PL 2015/16)\n"
                      "directness remains a team property", fontsize=11)
    fig.suptitle("Men's-league external validation: English Premier League 2015/16 "
                 "(attribution: StatsBomb) | associational", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(figdir / "analysis_mens_league.png", dpi=300); plt.close(fig)

    print(f"PL rows {len(tm)} teams {len(teams)} median matches/team {summary['median_matches_per_team']}")
    print("RELIABILITY r [95% CI] | discriminant eta2 (p) | team-share:")
    for d in DESCS:
        print(f"  {d:16s} r={rel[d]['r']:.3f} [{rel[d]['lo']:.3f},{rel[d]['hi']:.3f}] "
              f"| eta2={disc[d]['between_share']:.3f} (p={disc[d]['p_value']:.3f}) | team={tw[d]['a_partial']:.3f}")


if __name__ == "__main__":
    main()
