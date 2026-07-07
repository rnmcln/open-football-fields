"""Observation-ceiling replication on a second 360 competition (thesis Chapter 4).

The primary observation-ceiling study (analysis_obs_ceiling.py) measures, across
every 360-enabled match of UEFA Euro 2020, how much of the pitch and how many
players a broadcast-tracking freeze frame records. A single competition leaves open
whether the ceiling is a property of that one tournament's broadcast production or a
general feature of broadcast-derived tracking. This script repeats the identical
measurement on a second, independent 360 competition, the FIFA World Cup 2022, and
reports the two side by side. The methodology is deliberately unchanged from the
primary study so that any difference is attributable to the competition and its
broadcast process, not to a change of method.

Output: figures/analysis_obs_ceiling_wc2022.json,
        figures/analysis_obs_ceiling_comparison.json (+ comparison .png)
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

COMP, SEASON, NAME = 43, 106, "FIFA World Cup 2022"


def dist(s):
    return {"n": int(s.size), "mean": float(s.mean()), "median": float(s.median()),
            "p10": float(s.quantile(0.10)), "p90": float(s.quantile(0.90)),
            "min": float(s.min()), "max": float(s.max())}


def build_frame(client, pitch, pitch_area):
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
            poly = Polygon(pitch.sb_to_metric(va))
            if not poly.is_valid:
                poly = poly.buffer(0)
            frac = float(poly.area / pitch_area)
            npl = len(f["freeze_frame"])
            x = ex.get(uuid, np.nan)
            third = ("def" if x < 35 else "mid" if x < 70 else "att") if np.isfinite(x) else "na"
            rows.append({"match_id": mid, "visible_fraction": frac, "n_players": npl,
                         "event_type": etype.get(uuid, "unknown"), "third": third})
    return pd.DataFrame(rows)


def summarise(df, name):
    return {
        "competition": name,
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
    }


def main() -> None:
    cfg = load_config(); seed_everything(cfg["seed"]); root = repo_root()
    figdir = root / cfg["paths"]["figures"]; pitch = Pitch()
    pitch_area = pitch.metric_length * pitch.metric_width
    client = StatsBombClient(cache_dir=root / cfg["paths"]["data_cache"])

    df = build_frame(client, pitch, pitch_area)
    wc = summarise(df, NAME)
    (figdir / "analysis_obs_ceiling_wc2022.json").write_text(json.dumps(wc, indent=2), encoding="utf-8")
    RunProvenance(seed=cfg["seed"], extra={"stage": "obs_ceiling", "competition": "WC2022"}).write(
        figdir / "analysis_obs_ceiling_wc2022_provenance.json")

    # load the primary Euro 2020 result for comparison
    euro = json.loads((figdir / "analysis_obs_ceiling.json").read_text(encoding="utf-8"))
    comp = {
        "euro2020": {k: euro[k] for k in ["competition", "n_matches", "n_frames",
                                          "visible_fraction", "n_players",
                                          "fraction_frames_all_22", "fraction_frames_ge_20", "by_third"]},
        "wc2022": {k: wc[k] for k in ["competition", "n_matches", "n_frames",
                                      "visible_fraction", "n_players",
                                      "fraction_frames_all_22", "fraction_frames_ge_20", "by_third"]},
    }
    (figdir / "analysis_obs_ceiling_comparison.json").write_text(json.dumps(comp, indent=2), encoding="utf-8")

    # comparison figure: overlaid distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df["visible_fraction"], bins=40, density=True, alpha=0.6,
                 color="#2166ac", label=f"WC 2022 (median {wc['visible_fraction']['median']:.2f})")
    axes[0].axvline(euro["visible_fraction"]["median"], color="#b2182b", ls="--",
                    label=f"Euro 2020 median {euro['visible_fraction']['median']:.2f}")
    axes[0].set_xlabel("visible-area fraction of pitch"); axes[0].set_ylabel("density")
    axes[0].legend(fontsize=9)
    axes[0].set_title("A. Visible-area fraction, World Cup 2022 vs Euro 2020", fontsize=11)
    axes[1].hist(df["n_players"], bins=range(0, 24), density=True, alpha=0.6, align="left",
                 color="#2166ac", label=f"WC 2022 (median {int(wc['n_players']['median'])})")
    axes[1].axvline(euro["n_players"]["median"], color="#b2182b", ls="--",
                    label=f"Euro 2020 median {int(euro['n_players']['median'])}")
    axes[1].set_xlabel("visible players (of 22)"); axes[1].set_ylabel("density")
    axes[1].legend(fontsize=9)
    axes[1].set_title(f"B. Visible players; frames with all 22: "
                      f"WC {wc['fraction_frames_all_22']*100:.2f}%, "
                      f"Euro {euro['fraction_frames_all_22']*100:.2f}%", fontsize=11)
    fig.suptitle("Observation ceiling replicated across two 360 competitions | StatsBomb 360 "
                 "open data (attribution: StatsBomb) | associational", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(figdir / "analysis_obs_ceiling_comparison.png", dpi=300)
    plt.close(fig)

    print(json.dumps(comp, indent=2))


if __name__ == "__main__":
    main()
