"""Multi-match observation-ceiling study (thesis Chapter 4).

Generalises the single-match observation-ceiling result to every 360-enabled
match of a competition. For each broadcast-tracking freeze frame it records the
visible-area fraction of the pitch and the number of visible players, then
aggregates the distributions overall, by event type, and by pitch third, and
reports the fraction of frames that observe all 22 players.

Output: figures/analysis_obs_ceiling.json (+ .png)
"""
from __future__ import annotations

import json

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

COMP, SEASON = 55, 43  # UEFA Euro 2020 (all matches carry 360)


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    root = repo_root()
    figdir = root / cfg["paths"]["figures"]
    pitch = Pitch()
    pitch_area = pitch.metric_length * pitch.metric_width
    client = StatsBombClient(cache_dir=root / cfg["paths"]["data_cache"])

    matches = client.matches(COMP, SEASON)
    rows = []
    for m in matches:
        mid = m["match_id"]
        try:
            ts = client.three_sixty(mid)
            ev = normalise_attacking_direction(events_to_frame(client.events(mid), pitch), pitch)
        except Exception:
            continue
        etype = dict(zip(ev["id"], ev["type"]))
        ex = dict(zip(ev["id"], ev["x_att"]))
        for f in ts:
            uuid = f["event_uuid"]
            va = np.asarray(f["visible_area"], dtype=float).reshape(-1, 2)
            va_m = pitch.sb_to_metric(va)
            poly = Polygon(va_m)
            if not poly.is_valid:
                poly = poly.buffer(0)
            frac = float(poly.area / pitch_area)
            npl = len(f["freeze_frame"])
            x = ex.get(uuid, np.nan)
            third = ("def" if x < 35 else "mid" if x < 70 else "att") if np.isfinite(x) else "na"
            rows.append({"match_id": mid, "visible_fraction": frac, "n_players": npl,
                         "event_type": etype.get(uuid, "unknown"), "third": third})
    df = pd.DataFrame(rows)

    def dist(s):
        return {"n": int(s.size), "mean": float(s.mean()), "median": float(s.median()),
                "p10": float(s.quantile(0.10)), "p90": float(s.quantile(0.90)),
                "min": float(s.min()), "max": float(s.max())}

    summary = {
        "competition": "UEFA Euro 2020",
        "n_matches": int(df["match_id"].nunique()),
        "n_frames": int(len(df)),
        "visible_fraction": dist(df["visible_fraction"]),
        "n_players": dist(df["n_players"]),
        "fraction_frames_all_22": float((df["n_players"] >= 22).mean()),
        "fraction_frames_ge_20": float((df["n_players"] >= 20).mean()),
        "by_event_type": {t: {"n": int(g.shape[0]),
                              "vis_frac_mean": float(g["visible_fraction"].mean()),
                              "n_players_mean": float(g["n_players"].mean())}
                          for t, g in df.groupby("event_type") if g.shape[0] >= 200},
        "by_third": {t: {"n": int(g.shape[0]),
                         "vis_frac_mean": float(g["visible_fraction"].mean()),
                         "n_players_mean": float(g["n_players"].mean())}
                     for t, g in df.groupby("third")},
        "per_match_fraction_all22_max": float(df.groupby("match_id").apply(
            lambda g: (g["n_players"] >= 22).mean()).max()),
    }
    (figdir / "analysis_obs_ceiling.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RunProvenance(seed=cfg["seed"], extra={"stage": "obs_ceiling", "competition": "Euro2020"}).write(
        figdir / "analysis_obs_ceiling_provenance.json")

    # figure: distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df["visible_fraction"], bins=40, color="#4393c3", edgecolor="white")
    axes[0].axvline(df["visible_fraction"].median(), color="k", ls="--",
                    label=f"median {df['visible_fraction'].median():.2f}")
    axes[0].set_xlabel("visible-area fraction of pitch"); axes[0].set_ylabel("frames")
    axes[0].legend(fontsize=9)
    axes[0].set_title(f"A. Visible-area fraction per 360 frame\n"
                      f"{summary['n_frames']:,} frames, {summary['n_matches']} matches (Euro 2020)",
                      fontsize=11)
    axes[1].hist(df["n_players"], bins=range(0, 24), color="#d6604d", edgecolor="white", align="left")
    axes[1].axvline(df["n_players"].median(), color="k", ls="--",
                    label=f"median {int(df['n_players'].median())}")
    axes[1].set_xlabel("visible players (of 22)"); axes[1].set_ylabel("frames")
    axes[1].legend(fontsize=9)
    axes[1].set_title(f"B. Visible players per 360 frame\n"
                      f"frames observing all 22: {summary['fraction_frames_all_22']*100:.2f}%",
                      fontsize=11)
    fig.suptitle("Observation ceiling across a full competition | StatsBomb 360 open data "
                 "(attribution: StatsBomb) | associational", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(figdir / "analysis_obs_ceiling.png", dpi=300)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
