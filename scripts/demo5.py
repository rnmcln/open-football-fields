"""End-to-end Demonstration 5 demonstration: league-season replication + confounders.

Demonstration 4 found that field/threat descriptors discriminate teams and add
incremental value for shots, but their split-half **temporal reliability** was
weak -- plausibly because knockout tournaments give only ~6 matches per team
(~3 per split-half). Demonstration 5 addresses that head-on:

1. **League-season replication.** Re-run split-half reliability on a *complete*
   league season (FA Women's Super League 2018/19; ~19 matches per team, ~10 per
   half) and compare with the tournament result from Demonstration 4.
2. **Confounder-controlled decomposition.** Partition each descriptor's variance
   into a **team** component (controlling for opponent), an **opponent**
   component, and residual, with a permutation null for the team effect -- the
   proper successor to the one-way discriminant eta^2.

Associational and open-data only. xT is reused as a fixed value prior fit on
men's tournaments; applied to the women's league it is a generic spatial-value
reference (a domain-transfer caveat, logged). The non-xT style descriptors carry
no such caveat and are the primary reliability targets.

Outputs: figures/demo5_*.png, figures/demo5_summary.json,
         figures/demo5_provenance.json
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
WSL_DIR = "data/cache/batch_wsl"


def _assemble(cfg, pitch, grid, root) -> pd.DataFrame:
    xt = ExpectedThreat.from_artefact(root / cfg["threat"]["artefact"], pitch)
    man = pd.read_parquet(root / WSL_DIR / "manifest_37_4.parquet")
    opp = {}
    for _, r in man.iterrows():
        opp[(r["match_id"], r["home_team"])] = r["away_team"]
        opp[(r["match_id"], r["away_team"])] = r["home_team"]
    rows = []
    for f in sorted(glob.glob(str(root / WSL_DIR / "events" / "*.parquet"))):
        df = pd.read_parquet(f, columns=["type", "team", "x_att", "y_att",
                                         "end_x_att", "end_y_att"])
        mid = int(f.split("/")[-1].split(".")[0])  # match_id from filename
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

    tm = _assemble(cfg, pitch, grid, root)

    # tournament reliability (Demonstration 4) for comparison, if present
    s4 = {}
    s4f = figdir / "demo4_summary.json"
    if s4f.exists():
        j = json.loads(s4f.read_text())
        s4 = {d: j["temporal_stability"]["Euro2020"][d]["pearson_r"] for d in DESCRIPTORS}

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
    fdr = [{"test": lb, "p": float(p), "q": float(q), "reject": bool(rej)}
           for lb, p, q, rej in zip(labels, pvals, bh["qvalues"], bh["rejected"])]

    summary = {
        "season": "FA Women's Super League 2018/19",
        "n_team_match_rows": int(len(tm)),
        "n_teams": int(tm["team"].nunique()),
        "median_matches_per_team": int(tm.groupby("team").size().median()),
        "reliability_league": {d: reliability[d] for d in DESCRIPTORS},
        "reliability_tournament_euro2020": s4,
        "team_vs_opponent": {d: twoway[d] for d in DESCRIPTORS},
        "fdr_family": fdr,
        "n_significant_after_fdr": int(np.sum(bh["rejected"])),
        "note": ("xT reused as a fixed men's-tournament value prior (domain "
                 "transfer caveat); style descriptors need no xT."),
    }
    (figdir / "demo5_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RunProvenance(seed=seed, extra={"stage": "demo5"}).write(figdir / "demo5_provenance.json")

    _render(figdir, tm, reliability, twoway, s4)

    print(f"WSL 2018/19 team-match rows: {len(tm)}  teams: {tm['team'].nunique()}  "
          f"median matches/team: {summary['median_matches_per_team']}")
    print("\nSPLIT-HALF RELIABILITY  league r [perm p]  vs  tournament r (Euro2020):")
    for d in DESCRIPTORS:
        r = reliability[d]
        print(f"  {d:16s} league r={r['pearson_r']:.3f} (p={r['p_value']:.3f}, n={r['n_groups']})"
              f"   tournament r={s4.get(d, float('nan')):.3f}")
    print("\nTEAM vs OPPONENT variance (controlling each other):")
    for d in DESCRIPTORS:
        w = twoway[d]
        print(f"  {d:16s} team={w['a_partial']:.3f} (p={w['p_value']:.3f}) "
              f"opponent={w['b_partial']:.3f} residual={w['residual']:.3f}")
    print(f"\nFDR: {summary['n_significant_after_fdr']}/{len(pvals)} significant at q<=0.05")


def _render(figdir, tm, reliability, twoway, s4):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    x = np.arange(len(DESCRIPTORS))

    # A: league vs tournament split-half reliability
    ax = axes[0, 0]
    league = [reliability[d]["pearson_r"] for d in DESCRIPTORS]
    tour = [s4.get(d, np.nan) for d in DESCRIPTORS]
    ax.bar(x - 0.2, league, 0.4, color="#1b7837", label="league (WSL 18/19, ~19 mt/team)")
    ax.bar(x + 0.2, tour, 0.4, color="#999999", label="tournament (Euro 2020, ~6 mt/team)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(DESCRIPTORS, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("split-half reliability r")
    ax.legend(fontsize=8)
    ax.set_title("A. Reliability with adequate matches/team\n"
                 "(league season vs tournament)", fontsize=10)

    # B: split-half scatter, most reliable descriptor in the league
    ax = axes[0, 1]
    best = max(DESCRIPTORS, key=lambda d: reliability[d]["pearson_r"]
               if np.isfinite(reliability[d]["pearson_r"]) else -1)
    A, B = {}, {}
    for gg, s in tm.groupby("team"):
        s = s.sort_values("match_id")
        A[gg] = s.iloc[0::2][best].mean(); B[gg] = s.iloc[1::2][best].mean()
    ax.scatter([A[g] for g in A], [B[g] for g in A], c="#1b7837", s=45)
    ax.set_xlabel(f"{best} (half A)"); ax.set_ylabel(f"{best} (half B)")
    ax.set_title(f"B. League split-half of '{best}'\n"
                 f"r = {reliability[best]['pearson_r']:.2f} "
                 f"(p = {reliability[best]['p_value']:.3f})", fontsize=10)

    # C: variance decomposition (team controlling opponent, opponent, residual)
    ax = axes[1, 0]
    team = [twoway[d]["a_partial"] for d in DESCRIPTORS]
    oppo = [twoway[d]["b_partial"] for d in DESCRIPTORS]
    resid = [twoway[d]["residual"] for d in DESCRIPTORS]
    ax.bar(x, team, 0.6, color="#2166ac", label="team | opponent")
    ax.bar(x, oppo, 0.6, bottom=team, color="#f4a582", label="opponent | team")
    ax.bar(x, resid, 0.6, bottom=np.array(team) + np.array(oppo), color="#dddddd", label="residual")
    ax.set_xticks(x); ax.set_xticklabels(DESCRIPTORS, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("variance share"); ax.legend(fontsize=8)
    ax.set_title("C. Team vs opponent variance decomposition\n"
                 "(blue = team signal net of opponent)", fontsize=10)

    # D: team-partial with permutation significance
    ax = axes[1, 1]
    teamp = [twoway[d]["a_partial"] for d in DESCRIPTORS]
    sig = [twoway[d]["p_value"] < 0.05 for d in DESCRIPTORS]
    colors = ["#2166ac" if s else "#bbbbbb" for s in sig]
    ax.bar(x, teamp, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(DESCRIPTORS, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("team variance share | opponent")
    ax.set_title("D. Team effect controlling for opponent\n"
                 "(blue = permutation p < 0.05)", fontsize=10)

    fig.suptitle("ffields Demonstration 5 | FAWSL 2018/19 (attribution: StatsBomb) | "
                 "associational, not causal", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = figdir / "demo5_fields.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
