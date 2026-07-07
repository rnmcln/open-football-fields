"""End-to-end Demonstration 4 demonstration: the validation battery.

This is where the field/threat descriptors are *tested*, not just displayed. The
unit of analysis is a (team, match) row, assembled from the cached SPADL actions
for both primary competitions (UEFA Euro 2020 and FIFA World Cup 2022) -- no new
network access. Each test uses a permutation null; the whole family is then
FDR-controlled (Benjamini-Hochberg). Everything is associational.

Battery
-------
1. Discriminant validity: between-team variance share (eta^2) per descriptor,
   per competition, with a label-shuffle permutation null.
2. Temporal stability: split-half reliability of per-team descriptor means.
3. Cross-competition replication: does each descriptor's discriminability hold
   in both tournaments?
4. Incremental value: does a descriptor explain match shot counts beyond a
   possession-volume baseline (nested OLS Delta-R^2, permutation p)?
5. Multiple comparisons: BH-FDR across all p-values.

Outputs: figures/demo4_atlas.png, figures/demo4_summary.json,
         figures/demo4_provenance.json
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
    benjamini_hochberg,
    discriminant_permutation,
    incremental_value_ols,
    split_half_reliability,
)

# SPADL action type ids (socceraction standard ordering)
MOVE_TYPES = {0, 21}          # pass, dribble (carry)
SHOT_TYPES = {11, 12, 13}     # shot, shot_penalty, shot_freekick
COMPS = {"Euro2020": "55_43", "WC2022": "43_106"}
STYLE_DESCRIPTORS = ["mean_xt_end", "entropy_end", "directness", "att_third_frac", "sum_dxt"]


def _assemble(cfg, pitch, grid, root) -> pd.DataFrame:
    xt = ExpectedThreat.from_artefact(root / cfg["threat"]["artefact"], pitch)
    rows = []
    for comp, tag in COMPS.items():
        for f in sorted(glob.glob(str(root / "data" / "cache" / "spadl" / tag / "*.parquet"))):
            df = pd.read_parquet(f, columns=["game_id", "team_id", "type_id",
                                             "start_x", "start_y", "end_x", "end_y"])
            gid = int(df["game_id"].iloc[0])
            for team, sub in df.groupby("team_id"):
                mv = sub[sub["type_id"].isin(MOVE_TYPES)].dropna(
                    subset=["start_x", "start_y", "end_x", "end_y"])
                if len(mv) < 30:
                    continue
                ends = mv[["end_x", "end_y"]].to_numpy()
                starts = mv[["start_x", "start_y"]].to_numpy()
                dxt = xt.rate_moves(starts[:, 0], starts[:, 1], ends[:, 0], ends[:, 1])
                dens = EventDensityField(grid).estimate(ends)
                shots = int(sub["type_id"].isin(SHOT_TYPES).sum())
                rows.append({
                    "competition": comp, "game_id": gid, "team_id": int(team),
                    "n_moves": int(len(mv)),
                    "mean_xt_end": float(xt.value_at(ends).mean()),
                    "sum_dxt": float(dxt.sum()),
                    "entropy_end": float(dens.spatial_entropy()),
                    "directness": float((ends[:, 0] - starts[:, 0]).mean()),
                    "att_third_frac": float((ends[:, 0] > 70.0).mean()),
                    "shots": shots,
                })
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    root = repo_root()
    figdir = root / cfg["paths"]["figures"]
    figdir.mkdir(parents=True, exist_ok=True)
    pitch = Pitch()
    grid = Grid(pitch, nx=cfg["grid"]["nx"], ny=cfg["grid"]["ny"])
    seed = cfg["seed"]

    tm = _assemble(cfg, pitch, grid, root)

    pvals, labels = [], []
    # 1. discriminant validity (per competition) + 3. cross-competition
    discriminant = {}
    for comp in COMPS:
        sub = tm[tm["competition"] == comp]
        discriminant[comp] = {}
        for d in STYLE_DESCRIPTORS:
            r = discriminant_permutation(sub, "team_id", d, n_perm=2000, seed=seed)
            discriminant[comp][d] = r
            pvals.append(r["p_value"]); labels.append(f"discriminant[{comp}]:{d}")

    # 2. temporal stability (pooled within competition, then averaged report)
    stability = {}
    for comp in COMPS:
        sub = tm[tm["competition"] == comp]
        stability[comp] = {}
        for d in STYLE_DESCRIPTORS:
            r = split_half_reliability(sub, "team_id", d, "game_id", n_perm=2000, seed=seed)
            stability[comp][d] = r
            if np.isfinite(r["p_value"]):
                pvals.append(r["p_value"]); labels.append(f"stability[{comp}]:{d}")

    # 4. incremental value over possession volume, pooled across competitions
    incremental = {}
    for d in [c for c in STYLE_DESCRIPTORS if c != "entropy_end"]:
        r = incremental_value_ols(tm, "shots", ["n_moves"], d, n_perm=2000, seed=seed)
        incremental[d] = r
        pvals.append(r["p_value"]); labels.append(f"incremental:{d}")

    # 5. FDR across the whole family
    bh = benjamini_hochberg(pvals, q=0.05)
    fdr = [{"test": lb, "p": float(p), "q": float(q), "reject": bool(rej)}
           for lb, p, q, rej in zip(labels, pvals, bh["qvalues"], bh["rejected"])]

    summary = {
        "n_team_match_rows": int(len(tm)),
        "competitions": {c: int((tm["competition"] == c).sum()) for c in COMPS},
        "descriptors": STYLE_DESCRIPTORS,
        "discriminant": discriminant,
        "temporal_stability": stability,
        "incremental_value": incremental,
        "fdr_family": fdr,
        "n_tests": len(pvals),
        "n_significant_after_fdr": int(np.sum(bh["rejected"])),
    }
    (figdir / "demo4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RunProvenance(seed=seed, extra={"stage": "demo4"}).write(figdir / "demo4_provenance.json")

    _render(figdir, tm, discriminant, stability, incremental, fdr)

    # console digest
    print(f"team-match rows: {len(tm)}  ({summary['competitions']})")
    print("\nDISCRIMINANT between-team share (eta^2) [perm p]:")
    for d in STYLE_DESCRIPTORS:
        e = discriminant["Euro2020"][d]; w = discriminant["WC2022"][d]
        print(f"  {d:16s} Euro {e['between_share']:.3f} (p={e['p_value']:.3f}) | "
              f"WC {w['between_share']:.3f} (p={w['p_value']:.3f})")
    print("\nTEMPORAL split-half r [perm p] (Euro2020):")
    for d in STYLE_DESCRIPTORS:
        r = stability["Euro2020"][d]
        print(f"  {d:16s} r={r['pearson_r']:.3f} (p={r['p_value']:.3f}, n={r['n_groups']})")
    print("\nINCREMENTAL value over n_moves -> shots:")
    for d, r in incremental.items():
        print(f"  {d:16s} dR2={r['delta_r2']:.3f} (p={r['p_value']:.3f}, n={r['n']})")
    print(f"\nFDR: {summary['n_significant_after_fdr']}/{summary['n_tests']} tests significant at q<=0.05")


def _render(figdir, tm, discriminant, stability, incremental, fdr):
    descs = STYLE_DESCRIPTORS
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # A: discriminant between-team share, Euro vs WC, with permutation-null markers
    ax = axes[0, 0]
    x = np.arange(len(descs))
    eu = [discriminant["Euro2020"][d]["between_share"] for d in descs]
    wc = [discriminant["WC2022"][d]["between_share"] for d in descs]
    eu_null = [discriminant["Euro2020"][d]["null_mean"] for d in descs]
    wc_null = [discriminant["WC2022"][d]["null_mean"] for d in descs]
    ax.bar(x - 0.2, eu, 0.4, color="#2166ac", label="Euro 2020")
    ax.bar(x + 0.2, wc, 0.4, color="#b2182b", label="WC 2022")
    ax.plot(x - 0.2, eu_null, "k_", ms=12, label="perm-null mean")
    ax.plot(x + 0.2, wc_null, "k_", ms=12)
    ax.set_xticks(x); ax.set_xticklabels(descs, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("between-team variance share (eta^2)")
    ax.legend(fontsize=8)
    ax.set_title("A. Discriminant validity vs permutation null\n"
                 "(bar above the dash = team signal beyond chance)", fontsize=10)

    # B: temporal split-half scatter for the most reliable descriptor (Euro2020)
    ax = axes[0, 1]
    best_d = max(descs, key=lambda d: (stability["Euro2020"][d]["pearson_r"]
                                       if np.isfinite(stability["Euro2020"][d]["pearson_r"]) else -1))
    sub = tm[tm["competition"] == "Euro2020"]
    A, B = {}, {}
    for gg, s in sub.groupby("team_id"):
        s = s.sort_values("game_id")
        a = s.iloc[0::2][best_d].mean(); b = s.iloc[1::2][best_d].mean()
        A[gg] = a; B[gg] = b
    common = [g for g in A if np.isfinite(A[g]) and np.isfinite(B[g])]
    ax.scatter([A[g] for g in common], [B[g] for g in common], c="#1b7837", s=40)
    r = stability["Euro2020"][best_d]["pearson_r"]
    ax.set_xlabel(f"{best_d} (half A)"); ax.set_ylabel(f"{best_d} (half B)")
    ax.set_title(f"B. Temporal stability of '{best_d}' (Euro 2020)\n"
                 f"split-half r = {r:.2f}", fontsize=10)

    # C: cross-competition replication of discriminability
    ax = axes[1, 0]
    ax.scatter(eu, wc, c="#762a83", s=50)
    lim = [0, max(max(eu), max(wc)) * 1.1 + 1e-6]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
    for i, d in enumerate(descs):
        ax.annotate(d, (eu[i], wc[i]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("between-team share, Euro 2020")
    ax.set_ylabel("between-team share, WC 2022")
    ax.set_title("C. Cross-competition replication\n(points near the diagonal replicate)", fontsize=10)

    # D: incremental value Delta-R^2 with FDR significance
    ax = axes[1, 1]
    dd = list(incremental.keys())
    dr2 = [incremental[d]["delta_r2"] for d in dd]
    sig = {f["test"].split(":", 1)[1]: f["reject"] for f in fdr if f["test"].startswith("incremental:")}
    colors = ["#2166ac" if sig.get(d, False) else "#bbbbbb" for d in dd]
    ax.bar(np.arange(len(dd)), dr2, color=colors)
    ax.set_xticks(np.arange(len(dd))); ax.set_xticklabels(dd, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("incremental R^2 over possession volume")
    ax.set_title("D. Incremental value -> match shots\n"
                 "(blue = significant after BH-FDR; grey = not)", fontsize=10)

    fig.suptitle("ffields Demonstration 4 | validation battery | StatsBomb open data "
                 "(attribution: StatsBomb) | associational, not causal", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = figdir / "demo4_atlas.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
