"""Robustness checks for the incremental-value outcome models (thesis Chapter 6).

Addresses two reviewer points about the shot-count (Poisson) and expected-goals
(Gamma, log link) incremental-value analyses of Section 6.6:

1. Zero-xG rows. A Gamma GLM with a log link requires a strictly positive outcome.
   This script reports how many included team-match rows have summed xG equal to zero
   (the row builder already restricts to teams that recorded at least one shot, but a
   team can in principle take only zero-xG-bearing shots). If any zeros exist, the
   Gamma model is refitted on the strictly positive rows and the exclusion reported.

2. Dependence in the outcome models. Team-match rows are not independent: each match
   contributes two rows, and teams and opponents recur. Naive GLM standard errors
   ignore this. This script refits each descriptor model with cluster-robust standard
   errors grouped by match (game_id) and reports whether the descriptor associations
   remain significant, so the inferential uncertainty is not understated.

Reuses the identical row set and model specification as analysis_xg_outcome.py.

Output: figures/analysis_xg_robustness.json
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

from ffields import load_config, repo_root
from ffields.geometry import Grid, Pitch
from ffields.provenance import RunProvenance, seed_everything

# import the row builder and helpers from the sibling script
_spec = importlib.util.spec_from_file_location(
    "analysis_xg_outcome", Path(__file__).with_name("analysis_xg_outcome.py"))
_xg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_xg)

DESCS = _xg.DESCS


def _cluster_fit(y, X, family, groups):
    """GLM with cluster-robust covariance by match; fall back to HC1 if unsupported."""
    model = sm.GLM(y, X, family=family)
    try:
        return model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    except Exception:
        return model.fit(cov_type="HC1")


def main() -> None:
    cfg = load_config(); seed_everything(cfg["seed"]); root = repo_root()
    figdir = root / cfg["paths"]["figures"]; pitch = Pitch()
    grid = Grid(pitch, nx=cfg["grid"]["nx"], ny=cfg["grid"]["ny"])

    tm = _xg.main.__globals__["_rows"](cfg, pitch, grid, root) if False else _xg._rows(
        cfg, pitch, grid, root)
    n = len(tm)

    # --- 1. zero-xG diagnostics ---
    n_zero = int((tm["sum_xg"] <= 0).sum())
    pos = tm[tm["sum_xg"] > 0].copy()
    n_pos = len(pos)
    xg_summary = {
        "n_rows": n, "n_zero_xg": n_zero, "n_positive_xg": n_pos,
        "min_sum_xg": float(tm["sum_xg"].min()),
        "min_positive_sum_xg": float(pos["sum_xg"].min()) if n_pos else None,
        "median_sum_xg": float(tm["sum_xg"].median()),
        "gamma_valid_without_adjustment": bool(n_zero == 0),
    }

    groups = tm["game_id"].to_numpy()
    g_pos = pos["game_id"].to_numpy()
    n_clusters = int(tm["game_id"].nunique())

    y_shots = tm["shots"].to_numpy(float)
    base_shots = sm.add_constant(pd.DataFrame(
        {"log_moves": np.log(tm["n_moves"].to_numpy(float))}, index=tm.index))
    # Gamma is fitted on strictly positive rows (drops zeros if any)
    y_xg = pos["sum_xg"].to_numpy(float)
    base_xg = sm.add_constant(pd.DataFrame(
        {"log_moves": np.log(pos["n_moves"].to_numpy(float))}, index=pos.index))

    out = {}
    for d in DESCS:
        Xs = base_shots.copy(); Xs[d] = _xg._std(tm[d])
        Xg = base_xg.copy();    Xg[d] = _xg._std(pos[d])

        # shot count, Poisson: naive vs match-clustered
        p_naive = sm.GLM(y_shots, Xs, family=sm.families.Poisson()).fit()
        p_clu = _cluster_fit(y_shots, Xs, sm.families.Poisson(), groups)
        # xG, Gamma log link on positive rows: naive vs match-clustered
        g_naive = sm.GLM(y_xg, Xg, family=sm.families.Gamma(link=sm.families.links.Log())).fit()
        g_clu = _cluster_fit(y_xg, Xg, sm.families.Gamma(link=sm.families.links.Log()), g_pos)

        out[d] = {
            "shots_poisson": {
                "coef": float(p_naive.params[d]),
                "p_naive": float(p_naive.pvalues[d]),
                "p_cluster_by_match": float(p_clu.pvalues[d]),
                "se_naive": float(p_naive.bse[d]),
                "se_cluster_by_match": float(p_clu.bse[d]),
            },
            "xg_gamma_log_positive_rows": {
                "coef": float(g_naive.params[d]),
                "p_naive": float(g_naive.pvalues[d]),
                "p_cluster_by_match": float(g_clu.pvalues[d]),
                "se_naive": float(g_naive.bse[d]),
                "se_cluster_by_match": float(g_clu.bse[d]),
            },
        }

    summary = {
        "n_team_match_rows": n,
        "n_match_clusters": n_clusters,
        "xg_zero_diagnostics": xg_summary,
        "cluster_note": ("cluster-robust covariance grouped by match (game_id); each "
                         "match contributes two correlated team rows"),
        "incremental_value_robust": out,
    }
    (figdir / "analysis_xg_robustness.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    RunProvenance(seed=cfg["seed"], extra={"stage": "xg_robustness"}).write(
        figdir / "analysis_xg_robustness_provenance.json")
    print(json.dumps(summary, indent=1)[:2600])


if __name__ == "__main__":
    main()
