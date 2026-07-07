"""Combined observation-ceiling comparison across competitions (thesis Chapter 4).

Aggregates the per-competition observation-ceiling summaries produced by
``analysis_obs_ceiling.py`` (UEFA Euro 2020), ``analysis_obs_ceiling_wc2022.py``
(FIFA World Cup 2022) and any number of ``analysis_obs_ceiling_any.py`` runs
(e.g. UEFA Euro 2024, FIFA Women's World Cup 2023, UEFA Women's Euro 2022) into a
single comparison table and figure. This extends the two-competition replication
of Section 4.5.2 across further competitions, sexes, and formats.

It reads whatever ``figures/analysis_obs_ceiling*.json`` files are present, so run
the individual scripts first, then this one. Output:
figures/analysis_obs_ceiling_multi.json (+ .png) and a provenance file, plus a
ready-to-paste markdown table on stdout.
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ffields import load_config, repo_root
from ffields.provenance import RunProvenance, seed_everything

EXCLUDE = ("comparison", "provenance", "multi")


def _summarise(d: dict) -> dict:
    name = d.get("competition", "?")
    bt = d.get("by_third", {})

    def third(k, f):
        return bt.get(k, {}).get(f)

    return {
        "competition": name,
        "gender": "women" if "women" in name.lower() else "men",
        "n_matches": d.get("n_matches"),
        "n_frames": d.get("n_frames"),
        "vf_mean": d["visible_fraction"]["mean"],
        "vf_median": d["visible_fraction"]["median"],
        "vf_p10": d["visible_fraction"]["p10"],
        "vf_p90": d["visible_fraction"]["p90"],
        "np_mean": d["n_players"]["mean"],
        "np_median": d["n_players"]["median"],
        "frac_all22": d.get("fraction_frames_all_22"),
        "frac_ge20": d.get("fraction_frames_ge_20"),
        "mid_vf": third("mid", "vis_frac_mean"),
        "att_vf": third("att", "vis_frac_mean"),
        "def_players": third("def", "n_players_mean"),
    }


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    figdir = repo_root() / cfg["paths"]["figures"]

    files = []
    for f in sorted(glob.glob(str(figdir / "analysis_obs_ceiling*.json"))):
        base = os.path.basename(f)
        if any(x in base for x in EXCLUDE):
            continue
        files.append(f)

    comps = []
    seen = set()
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if "visible_fraction" not in d or "competition" not in d:
            continue
        s = _summarise(d)
        key = (s["competition"], s["n_frames"])
        if key in seen:
            continue
        seen.add(key)
        comps.append(s)

    if not comps:
        print("No per-competition observation-ceiling JSONs found. Run "
              "analysis_obs_ceiling.py / analysis_obs_ceiling_any.py first.")
        return

    # order: men then women, by frames descending within group
    comps.sort(key=lambda s: (s["gender"] == "women", -(s["n_frames"] or 0)))

    total_frames = sum(c["n_frames"] or 0 for c in comps)
    out = {"n_competitions": len(comps), "total_frames": total_frames, "competitions": comps}
    (figdir / "analysis_obs_ceiling_multi.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    RunProvenance(seed=cfg["seed"], extra={"stage": "obs_ceiling_multi",
                  "n_competitions": len(comps)}).write(figdir / "analysis_obs_ceiling_multi_provenance.json")

    # figure: median visible fraction (with p10-p90) and mean visible players, coloured by sex
    labels = [c["competition"].replace("FIFA ", "").replace("UEFA ", "") for c in comps]
    x = np.arange(len(comps))
    colours = ["#2166ac" if c["gender"] == "men" else "#b2182b" for c in comps]
    fig, ax = plt.subplots(1, 2, figsize=(max(11, 1.5 * len(comps)), 5))
    vf_med = [c["vf_median"] for c in comps]
    lo = [c["vf_median"] - c["vf_p10"] for c in comps]
    hi = [c["vf_p90"] - c["vf_median"] for c in comps]
    ax[0].bar(x, vf_med, color=colours, yerr=[lo, hi], capsize=3, edgecolor="white")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax[0].set_ylabel("visible-area fraction (median, p10 to p90)")
    ax[0].set_title("A. Pitch coverage per 360 frame", fontsize=11)
    ax[1].bar(x, [c["np_mean"] for c in comps], color=colours, edgecolor="white")
    ax[1].axhline(22, color="k", ls=":", lw=0.8, label="all 22 players")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax[1].set_ylabel("visible players per frame (mean of 22)")
    ax[1].set_title("B. Player coverage per 360 frame", fontsize=11)
    ax[1].legend(fontsize=8)
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(color="#2166ac", label="men"), Patch(color="#b2182b", label="women")],
                 fontsize=8)
    fig.suptitle("Observation ceiling across competitions | StatsBomb 360 open data "
                 "(attribution: StatsBomb) | associational", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(figdir / "analysis_obs_ceiling_multi.png", dpi=300)
    plt.close(fig)

    # markdown table for the thesis
    print(f"\n{len(comps)} competitions, {total_frames:,} frames total\n")
    print("| Competition | Sex | Matches | Frames | Visible fraction (median, p10-p90) | "
          "Visible players (mean, median) | All 22 | >=20 | mid third | att third | def players |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in comps:
        print(f"| {c['competition']} | {c['gender']} | {c['n_matches']} | {c['n_frames']:,} | "
              f"{c['vf_median']:.3f} ({c['vf_p10']:.3f} to {c['vf_p90']:.3f}) | "
              f"{c['np_mean']:.1f}, {c['np_median']:.0f} | {c['frac_all22']*100:.2f}% | "
              f"{c['frac_ge20']*100:.1f}% | {c['mid_vf']:.3f} | {c['att_vf']:.3f} | {c['def_players']:.1f} |")


if __name__ == "__main__":
    main()
