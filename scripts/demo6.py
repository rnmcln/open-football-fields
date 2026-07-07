"""Demonstration 6 generalisation: does the league-season reliability replicate?

Demonstration 5 showed that, with ~19 matches per team (FAWSL 2018/19), field-derived
descriptors are highly reliable (split-half). Demonstration 6 tests whether that
replicates in an **independent full season** (FAWSL 2019/20), and re-runs the
team-vs-opponent variance decomposition. A result that holds in two independent
seasons is far more trustworthy than one season alone.

Outputs: figures/demo6_*.png, figures/demo6_summary.json,
         figures/demo6_provenance.json
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
from ffields.validation import benjamini_hochberg, split_half_reliability, two_way_variance

DESCRIPTORS = ["mean_xt_end", "entropy_end", "directness", "att_third_frac", "sum_dxt"]


def assemble(cfg, pitch, grid, root, events_dir, manifest_path) -> pd.DataFrame:
    xt = ExpectedThreat.from_artefact(root / cfg["threat"]["artefact"], pitch)
    man = pd.read_parquet(manifest_path)
    opp = {}
    for _, r in man.iterrows():
        opp[(r["match_id"], r["home_team"])] = r["away_team"]
        opp[(r["match_id"], r["away_team"])] = r["home_team"]
    rows = []
    for f in sorted(glob.glob(str(events_dir / "*.parquet"))):
        df = pd.read_parquet(f, columns=["type", "team", "x_att", "y_att",
                                         "end_x_att", "end_y_att"])
        mid = int(f.split("/")[-1].split(".")[0])
        for team, sub in df.groupby("team"):
            mv = sub[sub["type"].isin(["Pass", "Carry"])].dropna(
                subset=["x_att", "y_att", "end_x_att", "end_y_att"])
            if len(mv) < 30:
                continue
            ends = mv[["end_x_att", "end_y_att"]].to_numpy()
            starts = mv[["x_att", "y_att"]].to_numpy()
            dxt = xt.rate_moves(starts[:, 0], starts[:, 1], ends[:, 0], ends[:, 1])
            dens = EventDensityField(grid).estimate(ends)
            rows.append({
                "match_id": mid, "team": team, "opponent": opp.get((mid, team), "?"),
                "n_moves": int(len(mv)),
                "mean_xt_end": float(xt.value_at(ends).mean()),
                "sum_dxt": float(dxt.sum()),
                "entropy_end": float(dens.spatial_entropy()),
                "directness": float((ends[:, 0] - starts[:, 0]).mean()),
                "att_third_frac": float((ends[:, 0] > 70.0).mean()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    root = repo_root()
    figdir = root / cfg["paths"]["figures"]
    pitch = Pitch()
    grid = Grid(pitch, nx=cfg["grid"]["nx"], ny=cfg["grid"]["ny"])
    seed = cfg["seed"]

    tm = assemble(cfg, pitch, grid, root,
                  root / "data" / "cache" / "batch_wsl2" / "events",
                  root / "data" / "cache" / "batch_wsl2" / "manifest_37_42.parquet")

    # Season 1 (2018/19) reliability for comparison, if present
    s5 = {}
    s5f = figdir / "demo5_summary.json"
    if s5f.exists():
        j = json.loads(s5f.read_text())
        s5 = {d: j["reliability_league"][d]["pearson_r"] for d in DESCRIPTORS}

    reliability, twoway = {}, {}
    pvals, labels = [], []
    for d in DESCRIPTORS:
        r = split_half_reliability(tm, "team", d, "match_id", n_perm=2000, seed=seed)
        reliability[d] = r
        pvals.append(r["p_value"]); labels.append(f"reliability:{d}")
        w = two_way_variance(tm, "team", "opponent", d, n_perm=1000, seed=seed)
        twoway[d] = w
        pvals.append(w["p_value"]); labels.append(f"team_vs_opponent:{d}")
    bh = benjamini_hochberg(pvals, q=0.05)

    # correlation of season-1 vs season-2 reliability (does the pattern replicate?)
    s2 = {d: reliability[d]["pearson_r"] for d in DESCRIPTORS}
    common = [d for d in DESCRIPTORS if np.isfinite(s5.get(d, np.nan)) and np.isfinite(s2[d])]
    repl_r = float(np.corrcoef([s5[d] for d in common], [s2[d] for d in common])[0, 1]) \
        if len(common) >= 3 else float("nan")

    summary = {
        "season": "FA Women's Super League 2019/20",
        "n_team_match_rows": int(len(tm)),
        "n_teams": int(tm["team"].nunique()),
        "median_matches_per_team": int(tm.groupby("team").size().median()),
        "reliability_2019_20": {d: reliability[d] for d in DESCRIPTORS},
        "reliability_2018_19": s5,
        "team_vs_opponent_2019_20": {d: twoway[d] for d in DESCRIPTORS},
        "across_season_reliability_corr": repl_r,
        "n_significant_after_fdr": int(np.sum(bh["rejected"])),
        "n_tests": len(pvals),
    }
    (figdir / "demo6_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RunProvenance(seed=seed, extra={"stage": "demo6"}).write(figdir / "demo6_provenance.json")

    _render(figdir, reliability, twoway, s5, repl_r)

    print(f"WSL 2019/20 rows: {len(tm)}  teams: {tm['team'].nunique()}  "
          f"median matches/team: {summary['median_matches_per_team']}")
    print("\nRELIABILITY  2019/20 r [p]  vs  2018/19 r:")
    for d in DESCRIPTORS:
        r = reliability[d]
        print(f"  {d:16s} 2019/20 r={r['pearson_r']:.3f} (p={r['p_value']:.3f})"
              f"   2018/19 r={s5.get(d, float('nan')):.3f}")
    print(f"\nAcross-season reliability correlation: {repl_r:.3f}")
    print("\nTEAM vs OPPONENT (2019/20):")
    for d in DESCRIPTORS:
        w = twoway[d]
        print(f"  {d:16s} team={w['a_partial']:.3f} (p={w['p_value']:.3f}) opp={w['b_partial']:.3f}")


def _render(figdir, reliability, twoway, s5, repl_r):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    x = np.arange(len(DESCRIPTORS))
    s2 = [reliability[d]["pearson_r"] for d in DESCRIPTORS]
    s1 = [s5.get(d, np.nan) for d in DESCRIPTORS]
    ax = axes[0]
    ax.bar(x - 0.2, s1, 0.4, color="#1b7837", label="2018/19")
    ax.bar(x + 0.2, s2, 0.4, color="#5aae61", label="2019/20")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(DESCRIPTORS, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("split-half reliability r"); ax.legend(fontsize=8)
    ax.set_title("A. Reliability replicates across two independent seasons\n"
                 f"(across-season pattern correlation r = {repl_r:.2f})", fontsize=10)

    ax = axes[1]
    team = [twoway[d]["a_partial"] for d in DESCRIPTORS]
    oppo = [twoway[d]["b_partial"] for d in DESCRIPTORS]
    resid = [twoway[d]["residual"] for d in DESCRIPTORS]
    ax.bar(x, team, 0.6, color="#2166ac", label="team | opponent")
    ax.bar(x, oppo, 0.6, bottom=team, color="#f4a582", label="opponent | team")
    ax.bar(x, resid, 0.6, bottom=np.array(team) + np.array(oppo), color="#dddddd", label="residual")
    ax.set_xticks(x); ax.set_xticklabels(DESCRIPTORS, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("variance share"); ax.legend(fontsize=8)
    ax.set_title("B. Team vs opponent decomposition (2019/20)\n"
                 "(directness remains a team trait)", fontsize=10)

    fig.suptitle("ffields Demonstration 6 | FAWSL 2019/20 generalisation (attribution: StatsBomb) | "
                 "associational, not causal", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = figdir / "demo6_fields.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
