"""Random-effects variance decomposition (thesis Chapter 6).

Replaces the fixed-effect team-versus-opponent decomposition with a crossed
random-effects mixed model, the textbook tool for partitioning variance in an
unbalanced panel. For each descriptor a linear mixed model with crossed random
intercepts for team and for opponent is fitted, and the variance is partitioned
into a team component, an opponent component, and a residual. The team proportion
is an intraclass-correlation-style measure of how much of the descriptor is a
property of the team. Fitted on the men's league (Premier League 2015/16) and a
women's season (FAWSL 2018/19).

Output: figures/analysis_mixed_models.json (+ .png)
"""
from __future__ import annotations

import glob
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

from ffields import load_config, repo_root
from ffields.fields import EventDensityField, ExpectedThreat
from ffields.geometry import Grid, Pitch
from ffields.provenance import RunProvenance, seed_everything

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
            rows.append({"team": str(team), "opponent": str(opp.get((mid, team), "?")),
                         "mean_xt_end": float(xt.value_at(ends).mean()),
                         "sum_dxt": float(xt.rate_moves(starts[:,0],starts[:,1],ends[:,0],ends[:,1]).sum()),
                         "entropy_end": float(EventDensityField(grid).estimate(ends).spatial_entropy()),
                         "directness": float((ends[:,0]-starts[:,0]).mean()),
                         "att_third_frac": float((ends[:,0]>70.0).mean())})
    return pd.DataFrame(rows)


def mixed_decompose(df, descriptor):
    """Crossed random-effects variance proportions for one descriptor."""
    d = df[[descriptor, "team", "opponent"]].dropna().copy()
    y = d[descriptor].to_numpy(float)
    d["yv"] = (y - y.mean()) / y.std()  # standardise for stability/interpretability
    d["grp"] = 1
    vcf = {"team": "0 + C(team)", "opp": "0 + C(opponent)"}
    md = smf.mixedlm("yv ~ 1", d, groups="grp", vc_formula=vcf)
    res = md.fit(reml=True, method="lbfgs")
    # map variance components by name (statsmodels orders vcomp alphabetically)
    names = list(md.exog_vc.names)
    vmap = {nm: float(v) for nm, v in zip(names, res.vcomp)}
    v_team = vmap["team"]; v_opp = vmap["opp"]; v_res = float(res.scale)
    tot = v_team + v_opp + v_res
    return {"team_prop": v_team / tot, "opp_prop": v_opp / tot, "resid_prop": v_res / tot,
            "n": int(len(d)), "converged": bool(res.converged)}


def main() -> None:
    cfg = load_config(); seed_everything(cfg["seed"]); root = repo_root()
    figdir = root / cfg["paths"]["figures"]; pitch = Pitch()
    grid = Grid(pitch, nx=cfg["grid"]["nx"], ny=cfg["grid"]["ny"])

    datasets = {
        "PL_2015_16": (root / "data/cache/batch_pl/events", root / "data/cache/batch_pl/manifest_2_27.parquet"),
        "FAWSL_2018_19": (root / "data/cache/batch_wsl/events", root / "data/cache/batch_wsl/manifest_37_4.parquet"),
    }
    out = {}
    for name, (ev, man) in datasets.items():
        tm = assemble(cfg, pitch, grid, root, ev, man)
        out[name] = {"n_rows": int(len(tm)), "n_teams": int(tm["team"].nunique()),
                     "descriptors": {d: mixed_decompose(tm, d) for d in DESCS}}

    summary = {"method": "crossed random-effects mixed model (REML); team and opponent "
                         "variance components; proportions of total variance",
               "datasets": out}
    (figdir / "analysis_mixed_models.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RunProvenance(seed=cfg["seed"], extra={"stage": "mixed_models"}).write(
        figdir / "analysis_mixed_models_provenance.json")

    # figure: team/opponent/residual proportions per descriptor, both datasets
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    for ax, name in zip(axes, ["PL_2015_16", "FAWSL_2018_19"]):
        x = np.arange(len(DESCS))
        team = [out[name]["descriptors"][d]["team_prop"] for d in DESCS]
        opp = [out[name]["descriptors"][d]["opp_prop"] for d in DESCS]
        res = [out[name]["descriptors"][d]["resid_prop"] for d in DESCS]
        ax.bar(x, team, 0.6, color="#2166ac", label="team")
        ax.bar(x, opp, 0.6, bottom=team, color="#f4a582", label="opponent")
        ax.bar(x, res, 0.6, bottom=np.array(team)+np.array(opp), color="#dddddd", label="residual")
        ax.set_xticks(x); ax.set_xticklabels(DESCS, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("variance proportion"); ax.legend(fontsize=8)
        ax.set_title(f"{name.replace('_','/')}  (n={out[name]['n_rows']})", fontsize=11)
    fig.suptitle("Crossed random-effects variance decomposition: team, opponent, residual "
                 "(associational)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(figdir / "analysis_mixed_models.png", dpi=300); plt.close(fig)

    for name in out:
        print(f"== {name} (n={out[name]['n_rows']}) ==")
        for d in DESCS:
            r = out[name]["descriptors"][d]
            print(f"  {d:16s} team={r['team_prop']:.3f} opp={r['opp_prop']:.3f} resid={r['resid_prop']:.3f} conv={r['converged']}")


if __name__ == "__main__":
    main()
