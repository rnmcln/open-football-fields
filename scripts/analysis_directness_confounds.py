"""Directness confounds: volume residualisation and context (thesis Chapter 6).

Addresses the review concern that directness (mean forward displacement per move)
correlates about -0.76 with move volume, so its reliability and team-specificity
might be primarily the inverse geometry of possession granularity rather than a
genuine style signal. It regresses directness on log move volume (and on home
advantage), and re-runs the reliability, between-team variance share, and
team-versus-opponent decomposition on the residual. If the residual retains
reliability and team specificity, directness carries an independent style signal;
if not, it is largely a volume artefact.

Uses the cached EPL 2015/16 descriptor rows produced by
analysis_reliability_curve.py (assemble stage). Output:
figures/analysis_directness_confounds.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ffields import load_config, repo_root
from ffields.provenance import RunProvenance, seed_everything
from ffields.validation import split_half_reliability, two_way_variance, variance_components

CACHE = "data/cache/_pl_desc_rows.parquet"


def _residualise(df, target, covars):
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in covars])
    y = df[target].to_numpy(float)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def _battery(df, col, seed):
    rel = split_half_reliability(df, "team", col, "match_id", n_perm=1000, seed=seed)
    vc = variance_components(df["team"].to_numpy(), df[col].to_numpy(float))
    tw = two_way_variance(df, "team", "opponent", col, n_perm=1000, seed=seed)
    return {"reliability_r": rel["pearson_r"], "between_team_share": vc["between_share"],
            "team_given_opponent": tw["a_partial"], "opponent_given_team": tw["b_partial"]}


def main():
    cfg = load_config()
    seed_everything(cfg["seed"])
    root = repo_root()
    figdir = root / cfg["paths"]["figures"]
    df = pd.read_parquet(root / CACHE).copy()

    # home/away from the manifest
    man = pd.read_parquet(root / "data" / "cache" / "batch_pl" / "manifest_2_27.parquet")
    home = {(r["match_id"], r["home_team"]): 1 for _, r in man.iterrows()}
    df["home"] = [home.get((m, t), 0) for m, t in zip(df["match_id"], df["team"])]
    df["log_moves"] = np.log(df["n_moves"].to_numpy(float))

    corr_vol = float(np.corrcoef(df["directness"], df["log_moves"])[0, 1])

    df["directness_resid_vol"] = _residualise(df, "directness", ["log_moves"])
    df["directness_resid_vol_home"] = _residualise(df, "directness", ["log_moves", "home"])

    seed = cfg["seed"]
    out = {
        "season": "English Premier League 2015/16",
        "n_rows": int(len(df)),
        "corr_directness_log_moves": corr_vol,
        "directness_raw": _battery(df, "directness", seed),
        "directness_residual_on_volume": _battery(df, "directness_resid_vol", seed),
        "directness_residual_on_volume_and_home": _battery(df, "directness_resid_vol_home", seed),
        "note": ("Residual directness is the OLS residual of directness on log move volume "
                 "(and home). If the residual keeps high reliability and between-team share, "
                 "directness carries a volume-independent team signal."),
    }
    (figdir / "analysis_directness_confounds.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    RunProvenance(seed=cfg["seed"], extra={"stage": "directness_confounds"}).write(
        figdir / "analysis_directness_confounds_provenance.json")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
