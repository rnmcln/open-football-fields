"""Observation ceiling for any StatsBomb 360 competition (thesis Chapters 4, 8).

Generalises ``analysis_obs_ceiling.py`` to an arbitrary competition-season so the
observation-ceiling replication of Section 4.5.2 can be extended beyond UEFA Euro
2020 and the FIFA World Cup 2022 as further open 360 competitions are released
(Section 8.2). The measurement is identical; only the competition changes.

Usage
-----
    python scripts/analysis_obs_ceiling_any.py <competition_id> <season_id> "<name>"

Examples (StatsBomb open-data identifiers; confirm against competitions.json):
    python scripts/analysis_obs_ceiling_any.py 55  282 "UEFA Euro 2024"
    python scripts/analysis_obs_ceiling_any.py 72  107 "FIFA Women's World Cup 2023"
    python scripts/analysis_obs_ceiling_any.py 53  106 "UEFA Women's Euro 2022"

The script downloads and caches the events and 360 data on first run (internet
required, and subject to the StatsBomb Public Data User Agreement); it writes an
aggregate JSON and a figure named after the competition. No raw data are stored in
the repository. Output: figures/analysis_obs_ceiling_<slug>.json (+ .png).
"""
from __future__ import annotations

import json
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from ffields import load_config, repo_root
from ffields.geometry import Pitch
from ffields.ingest import StatsBombClient, events_to_frame, normalise_attacking_direction
from ffields.provenance import RunProvenance, seed_everything


def _dist(s: pd.Series) -> dict:
    return {"n": int(s.size), "mean": float(s.mean()), "median": float(s.median()),
            "p10": float(s.quantile(0.10)), "p90": float(s.quantile(0.90)),
            "min": float(s.min()), "max": float(s.max())}


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    comp, season, name = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

    cfg = load_config()
    seed_everything(cfg["seed"])
    root = repo_root()
    figdir = root / cfg["paths"]["figures"]
    pitch = Pitch()
    pitch_area = pitch.metric_length * pitch.metric_width
    client = StatsBombClient(cache_dir=root / cfg["paths"]["data_cache"])

    matches = client.matches(comp, season)
    rows = []
    for m in matches:
        mid = m["match_id"]
        try:
            ts = client.three_sixty(mid)
            ev = normalise_attacking_direction(events_to_frame(client.events(mid), pitch), pitch)
        except Exception:
            continue
        ex = dict(zip(ev["id"], ev["x_att"]))
        for f in ts:
            uuid = f["event_uuid"]
            va = np.asarray(f["visible_area"], dtype=float).reshape(-1, 2)
            poly = Polygon(pitch.sb_to_metric(va))
            if not poly.is_valid:
                poly = poly.buffer(0)
            x = ex.get(uuid, np.nan)
            third = ("def" if x < 35 else "mid" if x < 70 else "att") if np.isfinite(x) else "na"
            rows.append({"match_id": mid, "visible_fraction": float(poly.area / pitch_area),
                         "n_players": len(f["freeze_frame"]), "third": third})
    if not rows:
        print(f"No 360 frames found for {name} (comp {comp}, season {season}). "
              "Confirm the competition carries 360 data in competitions.json.")
        sys.exit(2)
    df = pd.DataFrame(rows)

    summary = {
        "competition": name, "competition_id": comp, "season_id": season,
        "n_matches": int(df["match_id"].nunique()), "n_frames": int(len(df)),
        "visible_fraction": _dist(df["visible_fraction"]),
        "n_players": _dist(df["n_players"]),
        "fraction_frames_all_22": float((df["n_players"] >= 22).mean()),
        "fraction_frames_ge_20": float((df["n_players"] >= 20).mean()),
        "by_third": {t: {"n": int(g.shape[0]),
                         "vis_frac_mean": float(g["visible_fraction"].mean()),
                         "n_players_mean": float(g["n_players"].mean())}
                     for t, g in df.groupby("third")},
    }
    (figdir / f"analysis_obs_ceiling_{slug}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RunProvenance(seed=cfg["seed"], extra={"stage": "obs_ceiling_any", "competition": name}).write(
        figdir / f"analysis_obs_ceiling_{slug}_provenance.json")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df["visible_fraction"], bins=40, color="#4393c3", edgecolor="white")
    axes[0].axvline(df["visible_fraction"].median(), color="k", ls="--",
                    label=f"median {df['visible_fraction'].median():.2f}")
    axes[0].set_xlabel("visible-area fraction of pitch"); axes[0].set_ylabel("frames"); axes[0].legend()
    axes[0].set_title(f"A. Visible-area fraction ({summary['n_frames']:,} frames, "
                      f"{summary['n_matches']} matches)", fontsize=11)
    axes[1].hist(df["n_players"], bins=range(0, 24), color="#d6604d", edgecolor="white", align="left")
    axes[1].axvline(df["n_players"].median(), color="k", ls="--",
                    label=f"median {int(df['n_players'].median())}")
    axes[1].set_xlabel("visible players (of 22)"); axes[1].set_ylabel("frames"); axes[1].legend()
    axes[1].set_title(f"B. Visible players; all-22 frames {summary['fraction_frames_all_22']*100:.2f}%",
                      fontsize=11)
    fig.suptitle(f"Observation ceiling: {name} | StatsBomb 360 open data "
                 "(attribution: StatsBomb) | associational", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(figdir / f"analysis_obs_ceiling_{slug}.png", dpi=300)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
