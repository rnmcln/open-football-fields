"""Leave-one-competition-out xT and threat-descriptor endogeneity (thesis Chapter 6).

The threat descriptors (mean endpoint threat, total threat generated) are computed
with an expected-threat surface fitted on the combined Euro 2020 and World Cup 2022
actions, then applied to teams within those same tournaments. The review flags this
as an endogeneity concern and asks for a leave-one-competition-out (cross-fitted)
surface.

This script fits a standard expected-threat surface (Singh, 2019) from the cached
SPADL actions, attack-normalised per (game, team, period), for (a) both competitions
combined, (b) the World Cup 2022 only, and (c) UEFA Euro 2020 only. It then:

* validates the combined surface against the shipped socceraction surface;
* compares the two competition-specific surfaces;
* recomputes each team-match's mean endpoint threat with a *disjoint-competition*
  surface (Euro teams valued by the World Cup surface and vice versa) and correlates
  it with the endogenous values;
* recomputes the between-team discriminant variance share with the cross-fitted
  values and compares it with the endogenous result.

If the two surfaces agree closely and the cross-fitted descriptors and discriminant
shares track the endogenous ones, the endogeneity concern is quantitatively small.

Output: figures/analysis_xt_crossfit.json.
"""
from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd

from ffields import load_config, repo_root
from ffields.provenance import seed_everything
from ffields.validation import variance_components

NX, NY = 16, 12
L, W = 105.0, 68.0
MOVE_TYPES = {0, 1, 2, 3, 4, 5, 6, 7, 21}   # passes, crosses, set-piece deliveries, take-ons, dribbles
SHOT_TYPES = {11, 12, 13}
COMPS = {"WC2022": "43_106", "Euro2020": "55_43"}


def _cell(x, y):
    cx = min(max(int(x / L * NX), 0), NX - 1)
    cy = min(max(int(y / W * NY), 0), NY - 1)
    return cy * NX + cx


def load_norm(spadl_dir):
    """Load SPADL for one competition, attack-normalised per (game, team, period)."""
    frames = []
    for f in sorted(glob.glob(f"{spadl_dir}/*.parquet")):
        d = pd.read_parquet(f, columns=["game_id", "team_id", "period_id", "start_x", "start_y",
                                        "end_x", "end_y", "type_id", "result_id"])
        # attack direction per (team, period): sign of mean forward progression of moves
        for (tid, per), sub in d.groupby(["team_id", "period_id"]):
            mv = sub[sub["type_id"].isin(MOVE_TYPES)]
            prog = float((mv["end_x"] - mv["start_x"]).mean()) if len(mv) else 0.0
            if prog < 0:  # attacking -x -> flip to +x
                idx = sub.index
                d.loc[idx, "start_x"] = L - d.loc[idx, "start_x"]
                d.loc[idx, "end_x"] = L - d.loc[idx, "end_x"]
                d.loc[idx, "start_y"] = W - d.loc[idx, "start_y"]
                d.loc[idx, "end_y"] = W - d.loc[idx, "end_y"]
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def fit_xt(d, n_iter=60):
    ncell = NX * NY
    shots = np.zeros(ncell); goals = np.zeros(ncell); moves = np.zeros(ncell)
    T = np.zeros((ncell, ncell))
    sh = d[d["type_id"].isin(SHOT_TYPES)]
    for x, y, r in zip(sh["start_x"], sh["start_y"], sh["result_id"]):
        c = _cell(x, y); shots[c] += 1
        if r == 1:
            goals[c] += 1
    mv = d[d["type_id"].isin(MOVE_TYPES) & (d["result_id"] == 1)]
    for sx, sy, ex, ey in zip(mv["start_x"], mv["start_y"], mv["end_x"], mv["end_y"]):
        c = _cell(sx, sy); c2 = _cell(ex, ey); moves[c] += 1; T[c, c2] += 1
    total = shots + moves
    with np.errstate(invalid="ignore", divide="ignore"):
        s_prob = np.where(total > 0, shots / total, 0.0)
        m_prob = np.where(total > 0, moves / total, 0.0)
        g_prob = np.where(shots > 0, goals / shots, 0.0)
        Tn = np.where(moves[:, None] > 0, T / np.where(moves[:, None] > 0, moves[:, None], 1), 0.0)
    xt = np.zeros(ncell)
    for _ in range(n_iter):
        xt = s_prob * g_prob + m_prob * (Tn @ xt)
    return xt  # flat (ncell,), index cy*NX+cx


def value_ends(d, xt):
    """Mean xT of successful-move endpoints per (game_id, team_id)."""
    mv = d[d["type_id"].isin(MOVE_TYPES) & (d["result_id"] == 1)].copy()
    mv["cell"] = [_cell(x, y) for x, y in zip(mv["end_x"], mv["end_y"])]
    mv["xt"] = xt[mv["cell"].to_numpy()]
    g = mv.groupby(["game_id", "team_id"])["xt"].mean().reset_index()
    return g


def main():
    cfg = load_config()
    seed_everything(cfg["seed"])
    root = repo_root()
    figdir = root / cfg["paths"]["figures"]

    data = {name: load_norm(str(root / "data" / "cache" / "spadl" / d)) for name, d in COMPS.items()}
    d_all = pd.concat(data.values(), ignore_index=True)
    xt_all = fit_xt(d_all)
    xt_wc = fit_xt(data["WC2022"])
    xt_euro = fit_xt(data["Euro2020"])

    # sanity vs shipped socceraction surface
    shipped = np.array(json.load(open(root / "data" / "models" / "xt_grid.json"))["xT"], dtype=float)
    shipped_flat = shipped.reshape(-1)  # (12,16) rows=y, cols=x -> cy*NX+cx matches
    corr_shipped = float(np.corrcoef(xt_all, shipped_flat)[0, 1])
    corr_surfaces = float(np.corrcoef(xt_wc, xt_euro)[0, 1])

    # endogenous vs cross-fitted mean endpoint threat, per team-match
    endo = {"WC2022": value_ends(data["WC2022"], xt_all),
            "Euro2020": value_ends(data["Euro2020"], xt_all)}
    cross = {"WC2022": value_ends(data["WC2022"], xt_euro),   # WC teams valued by Euro surface
             "Euro2020": value_ends(data["Euro2020"], xt_wc)}  # Euro teams valued by WC surface

    out = {"grid": [NX, NY], "corr_combined_vs_shipped": corr_shipped,
           "corr_wc_vs_euro_surface": corr_surfaces, "by_competition": {}}
    for comp in COMPS:
        e = endo[comp].rename(columns={"xt": "xt_endo"})
        c = cross[comp].rename(columns={"xt": "xt_cross"})
        m = e.merge(c, on=["game_id", "team_id"])
        corr_desc = float(np.corrcoef(m["xt_endo"], m["xt_cross"])[0, 1])
        eta_endo = variance_components(m["team_id"].to_numpy(), m["xt_endo"].to_numpy())["between_share"]
        eta_cross = variance_components(m["team_id"].to_numpy(), m["xt_cross"].to_numpy())["between_share"]
        out["by_competition"][comp] = {
            "n_team_matches": int(len(m)),
            "corr_endogenous_vs_crossfit_descriptor": corr_desc,
            "between_team_share_endogenous": eta_endo,
            "between_team_share_crossfit": eta_cross}

    out["note"] = ("Standard xT fitted from cached SPADL, attack-normalised per team-period. "
                   "Cross-fit values a competition's team-matches with the surface trained on the "
                   "other competition. High agreement means the threat-descriptor conclusions do "
                   "not depend on the surface being fitted on the same tournament.")
    (figdir / "analysis_xt_crossfit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
