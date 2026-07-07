"""Cross-provider convergent validity on an independent event provider (Chapter 6).

The descriptor studies elsewhere use StatsBomb event data. This script repeats the
reliability and team-versus-opponent analysis for the provider-independent
descriptors on a complete league season from a different provider, the Wyscout open
data of Pappalardo et al. (2019), Premier League 2017/18 (380 matches), loaded
through kloppy. It addresses the single-event-provider limitation: if a descriptor
is reliable and team-specific when measured from an independent provider's events,
its behaviour is not an artefact of one provider's annotation.

Only descriptors that need no fitted value surface are used (directness,
attacking-third share, endpoint entropy), so the test is free of the
expected-threat endogeneity that affects the threat-based descriptors, and Wyscout's
lack of explicit carries (moves are passes) is itself a useful provider difference.

Per-match descriptor rows are cached under data/cache/wyscout_rows/ so the heavy
parsing is resumable across runs.

Output: figures/analysis_xprovider_wyscout.json (+ .png)
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ffields import load_config, repo_root
from ffields.fields import EventDensityField
from ffields.geometry import Grid, Pitch
from ffields.provenance import RunProvenance, seed_everything
from ffields.validation import discriminant_permutation, split_half_reliability, two_way_variance

DESCS = ["directness", "att_third_frac", "entropy_end"]
PL_X, PL_Y = 105.0, 68.0


def _assemble(cfg, grid, root):
    """Build (and cache) per-match team rows from the Wyscout JSONs via kloppy."""
    from kloppy import wyscout
    wdir = root / cfg["paths"]["data_cache"] / "wyscout"
    rdir = root / cfg["paths"]["data_cache"] / "wyscout_rows"
    rdir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(wdir / "2*.json")))
    for f in files:
        mid = os.path.basename(f).split(".")[0]
        cache = rdir / f"{mid}.json"
        if cache.exists():
            continue
        try:
            ds = wyscout.load(event_data=f, coordinates="wyscout")
        except Exception:
            cache.write_text("[]"); continue
        names = {t.team_id: t.name for t in ds.metadata.teams}
        df = ds.to_df()
        df = df[(df["event_type"] == "PASS")].dropna(
            subset=["coordinates_x", "coordinates_y", "end_coordinates_x", "end_coordinates_y"])
        out = []
        tids = list(names)
        for tid, sub in df.groupby("team_id"):
            if len(sub) < 30:
                continue
            sx = sub["coordinates_x"].to_numpy() * (PL_X / 100.0)
            sy = sub["coordinates_y"].to_numpy() * (PL_Y / 100.0)
            ex = sub["end_coordinates_x"].to_numpy() * (PL_X / 100.0)
            ey = sub["end_coordinates_y"].to_numpy() * (PL_Y / 100.0)
            ends = np.column_stack([ex, ey])
            opp = [names[t] for t in tids if t != tid]
            out.append({"match_id": mid, "team": names.get(tid, str(tid)),
                        "opponent": opp[0] if opp else "?",
                        "n_moves": int(len(sub)),
                        "directness": float((ex - sx).mean()),
                        "att_third_frac": float((ex > 70.0).mean()),
                        "entropy_end": float(EventDensityField(grid).estimate(ends).spatial_entropy())})
        cache.write_text(json.dumps(out))
    # aggregate
    rows = []
    for c in sorted(glob.glob(str(rdir / "*.json"))):
        rows.extend(json.loads(open(c).read()))
    return pd.DataFrame(rows), len(files), len(glob.glob(str(rdir / "*.json")))


def main() -> None:
    cfg = load_config(); seed_everything(cfg["seed"]); root = repo_root()
    figdir = root / cfg["paths"]["figures"]; pitch = Pitch()
    grid = Grid(pitch, nx=cfg["grid"]["nx"], ny=cfg["grid"]["ny"]); rng = np.random.default_rng(cfg["seed"])

    tm, n_files, n_cached = _assemble(cfg, grid, root)
    if len(tm) == 0:
        print(json.dumps({"status": "no rows yet", "files": n_files, "cached": n_cached})); return
    teams = tm["team"].unique()

    rel, disc, tw = {}, {}, {}
    for d in DESCS:
        disc[d] = discriminant_permutation(tm, "team", d, n_perm=2000, seed=cfg["seed"])
        r = split_half_reliability(tm, "team", d, "match_id", n_perm=2000, seed=cfg["seed"])
        bs = []
        for _ in range(300):
            samp = rng.choice(teams, size=len(teams), replace=True)
            boot = pd.concat([tm[tm["team"] == t].assign(team=f"{t}__{k}") for k, t in enumerate(samp)],
                             ignore_index=True)
            bs.append(split_half_reliability(boot, "team", d, "match_id", n_perm=1, seed=0)["pearson_r"])
        rel[d] = {"r": float(r["pearson_r"]), "p": float(r["p_value"]),
                  "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5))}
        tw[d] = two_way_variance(tm, "team", "opponent", d, n_perm=1000, seed=cfg["seed"])

    # StatsBomb PL 2015/16 comparison (same descriptors), if present
    sb = {}
    mp = figdir / "analysis_mens_league.json"
    if mp.exists():
        j = json.loads(mp.read_text())
        for d in DESCS:
            sb[d] = {"r": j["reliability_pl"][d]["r"], "team": j["team_vs_opponent_pl"][d]["a_partial"],
                     "eta2": j["discriminant"][d]["between_share"]}

    summary = {
        "provider": "Wyscout open (Pappalardo et al. 2019)",
        "competition": "English Premier League 2017/18",
        "n_team_match_rows": int(len(tm)), "n_teams": int(len(teams)),
        "median_matches_per_team": int(tm.groupby("team").size().median()),
        "n_matches_cached": n_cached, "n_matches_total": n_files,
        "note": "moves are passes (Wyscout has no explicit carries); provider-independent descriptors only",
        "reliability": rel,
        "discriminant": {d: disc[d] for d in DESCS},
        "team_vs_opponent": {d: tw[d] for d in DESCS},
        "statsbomb_pl_2015_16": sb,
    }
    (figdir / "analysis_xprovider_wyscout.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RunProvenance(seed=cfg["seed"], extra={"stage": "xprovider_wyscout"}).write(
        figdir / "analysis_xprovider_wyscout_provenance.json")

    # figure: reliability + team share, Wyscout vs StatsBomb
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(DESCS))
    wy = [rel[d]["r"] for d in DESCS]
    lo = [rel[d]["r"] - rel[d]["lo"] for d in DESCS]; hi = [rel[d]["hi"] - rel[d]["r"] for d in DESCS]
    axes[0].errorbar(x - 0.1, wy, yerr=[lo, hi], fmt="o", color="#08519c", capsize=4,
                     label="Wyscout PL 17/18")
    if sb:
        axes[0].scatter(x + 0.1, [sb[d]["r"] for d in DESCS], marker="s", color="#b2182b",
                        label="StatsBomb PL 15/16")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(DESCS, rotation=20, ha="right", fontsize=9)
    axes[0].set_ylabel("split-half reliability r"); axes[0].set_ylim(-0.6, 1.05); axes[0].legend(fontsize=9)
    axes[0].set_title("A. Reliability replicates on an independent provider\n"
                      "(Wyscout PL with 95% bootstrap CIs)", fontsize=11)
    team = [tw[d]["a_partial"] for d in DESCS]; oppo = [tw[d]["b_partial"] for d in DESCS]
    resid = [tw[d]["residual"] for d in DESCS]
    axes[1].bar(x, team, 0.6, color="#2166ac", label="team | opponent")
    axes[1].bar(x, oppo, 0.6, bottom=team, color="#f4a582", label="opponent | team")
    axes[1].bar(x, resid, 0.6, bottom=np.array(team) + np.array(oppo), color="#dddddd", label="residual")
    axes[1].set_xticks(x); axes[1].set_xticklabels(DESCS, rotation=20, ha="right", fontsize=9)
    axes[1].set_ylabel("variance share"); axes[1].legend(fontsize=8)
    axes[1].set_title("B. Team vs opponent (Wyscout PL 17/18)\n"
                      "directness remains a team property", fontsize=11)
    fig.suptitle("Cross-provider convergent validity: Wyscout open data, Premier League 2017/18 "
                 "(attribution: Wyscout/Pappalardo et al.) | associational", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(figdir / "analysis_xprovider_wyscout.png", dpi=300); plt.close(fig)

    print(json.dumps({"rows": len(tm), "teams": len(teams),
                      "cached": f"{n_cached}/{n_files}",
                      "directness_r": rel["directness"]["r"],
                      "directness_team_share": tw["directness"]["a_partial"],
                      "sb": sb.get("directness")}, indent=1))


if __name__ == "__main__":
    main()
