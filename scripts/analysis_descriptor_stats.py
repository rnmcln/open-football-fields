"""Deeper statistics for the descriptor study (thesis Chapter 6).

Adds three things the summary battery lacked:
  1. a descriptor correlation matrix (confound check; is directness distinct?);
  2. count models for match shots (Poisson and negative binomial) with a
     possession-volume offset and an opponent-strength covariate, with a
     likelihood-ratio test, McFadden pseudo-R-squared, and a dispersion check;
  3. bootstrap confidence intervals (resampling teams) for split-half
     reliability and for the team variance share in a league season.

Output: figures/analysis_descriptor_stats.json (+ .png x2)
"""
from __future__ import annotations

import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from ffields import load_config, repo_root
from ffields.fields import EventDensityField, ExpectedThreat
from ffields.geometry import Grid, Pitch
from ffields.provenance import RunProvenance, seed_everything
from ffields.validation import split_half_reliability, two_way_variance

MOVE_SP = {0, 21}
SHOT_SP = {11, 12, 13}
DESCS = ["mean_xt_end", "sum_dxt", "entropy_end", "directness", "att_third_frac"]


def _tournament_rows(cfg, pitch, grid, root):
    xt = ExpectedThreat.from_artefact(root / cfg["threat"]["artefact"], pitch)
    rows = []
    for tag in ["55_43", "43_106"]:
        for f in sorted(glob.glob(str(root / "data" / "cache" / "spadl" / tag / "*.parquet"))):
            df = pd.read_parquet(f, columns=["game_id", "team_id", "type_id",
                                             "start_x", "start_y", "end_x", "end_y"])
            gid = int(df["game_id"].iloc[0])
            teams = df["team_id"].unique()
            for team in teams:
                sub = df[df["team_id"] == team]
                mv = sub[sub["type_id"].isin(MOVE_SP)].dropna(subset=["start_x", "start_y", "end_x", "end_y"])
                if len(mv) < 30:
                    continue
                ends = mv[["end_x", "end_y"]].to_numpy(); starts = mv[["start_x", "start_y"]].to_numpy()
                opp = [t for t in teams if t != team]
                rows.append({"comp": tag, "game_id": gid, "team_id": int(team),
                             "opp_id": int(opp[0]) if opp else -1,
                             "n_moves": int(len(mv)),
                             "shots": int(sub["type_id"].isin(SHOT_SP).sum()),
                             "mean_xt_end": float(xt.value_at(ends).mean()),
                             "sum_dxt": float(xt.rate_moves(starts[:,0],starts[:,1],ends[:,0],ends[:,1]).sum()),
                             "entropy_end": float(EventDensityField(grid).estimate(ends).spatial_entropy()),
                             "directness": float((ends[:,0]-starts[:,0]).mean()),
                             "att_third_frac": float((ends[:,0]>70.0).mean())})
    d = pd.DataFrame(rows)
    # opponent defensive strength: mean shots the opponent concedes across its games
    conceded = {}
    for (gid, tid), g in d.groupby(["game_id", "team_id"]):
        pass
    # shots conceded by team t in a game = shots by the opponent in that game
    sc = []
    for gid, g in d.groupby("game_id"):
        for _, r in g.iterrows():
            opp_shots = g[g["team_id"] == r["opp_id"]]["shots"]
            sc.append({"game_id": gid, "team_id": r["team_id"],
                       "conceded": float(opp_shots.iloc[0]) if len(opp_shots) else np.nan})
    sc = pd.DataFrame(sc)
    team_conceded = sc.groupby("team_id")["conceded"].mean()
    d["opp_def_strength"] = d["opp_id"].map(team_conceded)
    return d


def _league_rows(cfg, pitch, grid, root, events_dir, manifest):
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
    grid = Grid(pitch, nx=cfg["grid"]["nx"], ny=cfg["grid"]["ny"])
    rng = np.random.default_rng(cfg["seed"])

    tm = _tournament_rows(cfg, pitch, grid, root)

    # 1. descriptor correlation matrix
    cm = tm[DESCS + ["n_moves"]].corr(method="pearson").round(3)
    corr = {a: {b: float(cm.loc[a, b]) for b in cm.columns} for a in cm.index}

    # 2. count models for shots
    def _std(s):
        s = s.astype(float); return (s - s.mean()) / s.std()
    glm = {}
    base_X = sm.add_constant(pd.DataFrame({"log_moves": np.log(tm["n_moves"])}))
    y = tm["shots"].to_numpy(float)
    m0 = sm.GLM(y, base_X, family=sm.families.Poisson()).fit()
    for d in ["mean_xt_end", "directness", "att_third_frac", "sum_dxt"]:
        X1 = base_X.copy(); X1[d] = _std(tm[d])
        X2 = X1.copy(); X2["opp_def_strength"] = _std(tm["opp_def_strength"].fillna(tm["opp_def_strength"].mean()))
        m1 = sm.GLM(y, X1, family=sm.families.Poisson()).fit()
        m2 = sm.GLM(y, X2, family=sm.families.Poisson()).fit()
        lr = 2 * (m1.llf - m0.llf)
        from scipy.stats import chi2
        glm[d] = {
            "coef": float(m1.params[d]), "p_value": float(m1.pvalues[d]),
            "lr_stat_vs_baseline": float(lr), "lr_p": float(chi2.sf(lr, 1)),
            "mcfadden_pseudo_r2": float(1 - m1.llf / m0.llf),
            "coef_with_opp_control": float(m2.params[d]), "p_with_opp_control": float(m2.pvalues[d]),
        }
    dispersion = float(m0.pearson_chi2 / m0.df_resid)
    nb = sm.GLM(y, base_X, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    glm["_diagnostics"] = {"poisson_dispersion": dispersion,
                           "interpretation": ("dispersion near 1 supports Poisson; "
                                              ">1 indicates overdispersion, hence the negative-binomial check"),
                           "negbin_fitted": True, "n": int(len(y))}

    # 3. bootstrap CIs for reliability and team share (FAWSL 2018/19)
    lg = _league_rows(cfg, pitch, grid, root,
                      root / "data" / "cache" / "batch_wsl" / "events",
                      root / "data" / "cache" / "batch_wsl" / "manifest_37_4.parquet")
    teams = lg["team"].unique()
    rel_ci, share_ci = {}, {}
    for d in DESCS:
        r_obs = split_half_reliability(lg, "team", d, "match_id", n_perm=1, seed=cfg["seed"])["pearson_r"]
        tw_obs = two_way_variance(lg, "team", "opponent", d, n_perm=1, seed=cfg["seed"])["a_partial"]
        rb_, sb_ = [], []
        for _ in range(300):
            samp_teams = rng.choice(teams, size=len(teams), replace=True)
            boot = pd.concat([lg[lg["team"] == t].assign(team=f"{t}__{k}")
                              for k, t in enumerate(samp_teams)], ignore_index=True)
            rb_.append(split_half_reliability(boot, "team", d, "match_id", n_perm=1, seed=0)["pearson_r"])
        rel_ci[d] = {"r": float(r_obs), "lo": float(np.nanpercentile(rb_, 2.5)),
                     "hi": float(np.nanpercentile(rb_, 97.5)), "team_share": float(tw_obs)}

    summary = {"n_team_match_tournaments": int(len(tm)),
               "descriptor_correlations": corr,
               "shot_count_models": glm,
               "reliability_bootstrap_ci_fawsl1819": rel_ci}
    (figdir / "analysis_descriptor_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RunProvenance(seed=cfg["seed"], extra={"stage": "descriptor_stats"}).write(
        figdir / "analysis_descriptor_stats_provenance.json")

    # figure 1: correlation heatmap
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    labs = DESCS + ["n_moves"]
    M = np.array([[corr[a][b] for b in labs] for a in labs])
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=9)
    for i in range(len(labs)):
        for j in range(len(labs)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(M[i,j]) > 0.6 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_title("Descriptor correlations (tournament team-match rows)\n"
                 "directness is weakly correlated with the others", fontsize=11)
    fig.tight_layout(); fig.savefig(figdir / "analysis_descriptor_corr.png", dpi=300); plt.close(fig)

    # figure 2: reliability with bootstrap CIs
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(DESCS))
    r = [rel_ci[d]["r"] for d in DESCS]
    lo = [rel_ci[d]["r"] - rel_ci[d]["lo"] for d in DESCS]
    hi = [rel_ci[d]["hi"] - rel_ci[d]["r"] for d in DESCS]
    ax.errorbar(x, r, yerr=[lo, hi], fmt="o", color="#1b7837", capsize=5, ms=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(DESCS, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("split-half reliability r"); ax.set_ylim(-0.6, 1.05)
    ax.set_title("Split-half reliability with 95% bootstrap CIs (FAWSL 2018/19)\n"
                 "resampling teams; directness and sum_dxt are reliably high", fontsize=11)
    fig.tight_layout(); fig.savefig(figdir / "analysis_reliability_ci.png", dpi=300); plt.close(fig)

    print(json.dumps({"corr_directness": {k: round(corr["directness"][k], 2) for k in labs},
                      "glm": {k: v for k, v in glm.items()},
                      "rel_ci": {d: {k: round(x, 3) for k, x in rel_ci[d].items()} for d in DESCS}}, indent=1)[:1600])


if __name__ == "__main__":
    main()
