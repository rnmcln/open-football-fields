"""Reliability as a function of matches per team (thesis Chapter 6).

Addresses two review points. First, it estimates the split-half reliability of
each descriptor as a function of the number of matches per team, R(n), by
repeatedly subsampling the best-powered season (English Premier League 2015/16,
38 matches per team) down to n matches per team. Second, it thereby tests
directly whether the entropy descriptor's high reliability in the men's league,
against its failure in the shorter women's seasons, is a statistical-power effect:
if EPL entropy reliability at n approximately 19 falls to the women's-season
level, power explains the discrepancy; if it stays high, a competition difference
is implicated.

This is a reliability distribution, not a single estimate: for each n and each
descriptor, matches are subsampled without replacement per team many times and the
deterministic parity split-half correlation is recomputed, giving a distribution
of R at that n.

Runs in two stages so each fits a short job:
    python scripts/analysis_reliability_curve.py assemble   # -> cached rows
    python scripts/analysis_reliability_curve.py curve       # -> json + png
    python scripts/analysis_reliability_curve.py all
"""
from __future__ import annotations

import glob
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ffields import load_config, repo_root
from ffields.fields import EventDensityField, ExpectedThreat
from ffields.geometry import Grid, Pitch
from ffields.provenance import RunProvenance, seed_everything

DESCS = ["directness", "sum_dxt", "att_third_frac", "mean_xt_end", "entropy_end"]
NS = [6, 10, 15, 19, 27, 38]
N_REP = 300
CACHE = "data/cache/_pl_desc_rows.parquet"


def assemble(cfg, pitch, grid, root):
    xt = ExpectedThreat.from_artefact(root / cfg["threat"]["artefact"], pitch)
    events_dir = root / "data" / "cache" / "batch_pl" / "events"
    man = pd.read_parquet(root / "data" / "cache" / "batch_pl" / "manifest_2_27.parquet")
    opp = {}
    for _, r in man.iterrows():
        opp[(r["match_id"], r["home_team"])] = r["away_team"]
        opp[(r["match_id"], r["away_team"])] = r["home_team"]
    rows = []
    for f in sorted(glob.glob(str(events_dir / "*.parquet"))):
        df = pd.read_parquet(f, columns=["type", "team", "x_att", "y_att", "end_x_att", "end_y_att"])
        mid = int(f.split("/")[-1].split(".")[0])
        for team, sub in df.groupby("team"):
            mv = sub[sub["type"].isin(["Pass", "Carry"])].dropna(
                subset=["x_att", "y_att", "end_x_att", "end_y_att"])
            if len(mv) < 30:
                continue
            ends = mv[["end_x_att", "end_y_att"]].to_numpy()
            starts = mv[["x_att", "y_att"]].to_numpy()
            rows.append({"match_id": mid, "team": team, "opponent": opp.get((mid, team), "?"),
                         "n_moves": int(len(mv)),
                         "mean_xt_end": float(xt.value_at(ends).mean()),
                         "sum_dxt": float(xt.rate_moves(starts[:, 0], starts[:, 1], ends[:, 0], ends[:, 1]).sum()),
                         "entropy_end": float(EventDensityField(grid).estimate(ends).spatial_entropy()),
                         "directness": float((ends[:, 0] - starts[:, 0]).mean()),
                         "att_third_frac": float((ends[:, 0] > 70.0).mean())})
    out = pd.DataFrame(rows)
    out.to_parquet(root / CACHE)
    print(f"assembled {len(out)} rows, {out['team'].nunique()} teams -> {CACHE}", file=sys.stderr)
    return out


def _split_half_r(df, desc):
    """Deterministic parity split-half Pearson r on whatever matches are present."""
    a, b = {}, {}
    for g, sub in df.groupby("team"):
        sub = sub.sort_values("match_id")
        va = sub.iloc[0::2][desc].to_numpy(float)
        vb = sub.iloc[1::2][desc].to_numpy(float)
        if len(va) >= 1 and len(vb) >= 1:
            a[g] = np.nanmean(va)
            b[g] = np.nanmean(vb)
    common = [g for g in a if g in b]
    if len(common) < 4:
        return np.nan
    A = np.array([a[g] for g in common])
    B = np.array([b[g] for g in common])
    if np.std(A) < 1e-12 or np.std(B) < 1e-12:
        return np.nan
    return float(np.corrcoef(A, B)[0, 1])


def curve(cfg, root):
    df = pd.read_parquet(root / CACHE)
    rng = np.random.default_rng(cfg["seed"])
    teams = df["team"].unique()
    by_team = {t: df[df["team"] == t] for t in teams}
    result = {}
    for desc in DESCS:
        result[desc] = {}
        for n in NS:
            rs = []
            reps = 1 if n >= 38 else N_REP
            for _ in range(reps):
                parts = []
                for t, sub in by_team.items():
                    k = min(n, len(sub))
                    idx = rng.choice(len(sub), size=k, replace=False)
                    parts.append(sub.iloc[idx])
                samp = pd.concat(parts, ignore_index=True)
                r = _split_half_r(samp, desc)
                if np.isfinite(r):
                    rs.append(r)
            a = np.array(rs)
            result[desc][str(n)] = {
                "mean": float(a.mean()) if a.size else None,
                "lo": float(np.percentile(a, 2.5)) if a.size else None,
                "hi": float(np.percentile(a, 97.5)) if a.size else None,
                "n_rep": int(a.size)}
    return result


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfg = load_config()
    seed_everything(cfg["seed"])
    root = repo_root()
    figdir = root / cfg["paths"]["figures"]
    pitch = Pitch(metric_length=cfg["pitch"]["metric_length"], metric_width=cfg["pitch"]["metric_width"])
    grid = Grid(pitch, nx=cfg["grid"]["nx"], ny=cfg["grid"]["ny"])

    if stage in ("assemble", "all"):
        assemble(cfg, pitch, grid, root)
    if stage in ("curve", "all"):
        res = curve(cfg, root)
        summary = {"season": "English Premier League 2015/16", "n_per_team_values": NS,
                   "n_repeats": N_REP, "reliability_curve": res,
                   "note": ("Split-half reliability R(n) by subsampling n matches per team from "
                            "the 38-match season; distribution over subsamples. Tests whether the "
                            "entropy descriptor's men's-league reliability is a power effect.")}
        (figdir / "analysis_reliability_curve.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        RunProvenance(seed=cfg["seed"], extra={"stage": "reliability_curve"}).write(
            figdir / "analysis_reliability_curve_provenance.json")
        fig, ax = plt.subplots(figsize=(9, 6))
        colours = {"directness": "#2166ac", "sum_dxt": "#4393c3", "att_third_frac": "#92c5de",
                   "mean_xt_end": "#f4a582", "entropy_end": "#b2182b"}
        labels = {"directness": "directness", "sum_dxt": "total threat generated",
                  "att_third_frac": "attacking-third share", "mean_xt_end": "mean endpoint threat",
                  "entropy_end": "endpoint dispersion entropy"}
        for desc in DESCS:
            m = [res[desc][str(n)]["mean"] for n in NS]
            lo = [res[desc][str(n)]["lo"] for n in NS]
            hi = [res[desc][str(n)]["hi"] for n in NS]
            ax.plot(NS, m, "-o", color=colours[desc], label=labels[desc])
            ax.fill_between(NS, lo, hi, color=colours[desc], alpha=0.15)
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(19, color="grey", ls=":", lw=0.8)
        ax.set_xlabel("matches per team (subsampled)")
        ax.set_ylabel("split-half reliability (Pearson r)")
        ax.set_title("Reliability versus matches per team (EPL 2015/16 subsampled)\n"
                     "grey line marks 19 matches, the women's-season depth", fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(figdir / "analysis_reliability_curve.png", dpi=300)
        plt.close(fig)
        # console summary
        print("descriptor          " + "".join(f"n={n:>3} " for n in NS))
        for desc in DESCS:
            print(f"{desc:18s} " + "".join(
                f"{(res[desc][str(n)]['mean'] if res[desc][str(n)]['mean'] is not None else float('nan')):>5.2f} "
                for n in NS))


if __name__ == "__main__":
    main()
