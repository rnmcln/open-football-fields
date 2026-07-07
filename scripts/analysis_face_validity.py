"""Descriptor face-validity / interpretability check (thesis Chapter 6).

Reports per-team descriptor means for a single league season (English Premier
League 2015/16), so the descriptors can be read concretely rather than only as
abstract reliability coefficients. Purely descriptive and associational: the
rankings characterise recorded behaviour, not quality or tactics.

Output: figures/analysis_face_validity.json (+ .png)
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
from ffields.fields import ExpectedThreat
from ffields.geometry import Grid, Pitch
from ffields.provenance import RunProvenance, seed_everything


def main() -> None:
    cfg = load_config(); seed_everything(cfg["seed"]); root = repo_root()
    figdir = root / cfg["paths"]["figures"]; pitch = Pitch()
    grid = Grid(pitch, nx=cfg["grid"]["nx"], ny=cfg["grid"]["ny"])
    xt = ExpectedThreat.from_artefact(root / cfg["threat"]["artefact"], pitch)

    rows = []
    for f in sorted(glob.glob(str(root / "data/cache/batch_pl/events/*.parquet"))):
        df = pd.read_parquet(f, columns=["type", "team", "x_att", "y_att", "end_x_att", "end_y_att"])
        mid = int(f.split("/")[-1].split(".")[0])
        for team, sub in df.groupby("team"):
            mv = sub[sub["type"].isin(["Pass", "Carry"])].dropna(subset=["x_att", "y_att", "end_x_att", "end_y_att"])
            if len(mv) < 30:
                continue
            ends = mv[["end_x_att", "end_y_att"]].to_numpy(); starts = mv[["x_att", "y_att"]].to_numpy()
            rows.append({"match_id": mid, "team": team,
                         "directness": float((ends[:, 0] - starts[:, 0]).mean()),
                         "mean_xt_end": float(xt.value_at(ends).mean()),
                         "att_third_frac": float((ends[:, 0] > 70.0).mean())})
    tm = pd.DataFrame(rows)
    team_means = tm.groupby("team").agg(
        n_matches=("match_id", "nunique"),
        directness=("directness", "mean"),
        mean_xt_end=("mean_xt_end", "mean"),
        att_third_frac=("att_third_frac", "mean")).reset_index().sort_values("directness", ascending=False)

    summary = {
        "competition": "English Premier League 2015/16",
        "directness_m": {"min": float(team_means["directness"].min()),
                         "max": float(team_means["directness"].max()),
                         "mean": float(team_means["directness"].mean()),
                         "sd": float(team_means["directness"].std())},
        "most_direct": team_means.head(5)[["team", "directness"]].to_dict(orient="records"),
        "least_direct": team_means.tail(5)[["team", "directness"]].to_dict(orient="records"),
        "team_table": team_means.round(3).to_dict(orient="records"),
    }
    (figdir / "analysis_face_validity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RunProvenance(seed=cfg["seed"], extra={"stage": "face_validity"}).write(
        figdir / "analysis_face_validity_provenance.json")

    fig, ax = plt.subplots(figsize=(9, 7))
    tms = team_means.sort_values("directness")
    y = np.arange(len(tms))
    ax.barh(y, tms["directness"], color="#2166ac")
    ax.set_yticks(y); ax.set_yticklabels(tms["team"], fontsize=8)
    ax.set_xlabel("mean forward displacement per move (m), season mean")
    ax.set_title("Team directness, English Premier League 2015/16\n"
                 "(descriptive; mean advancement per pass or carry)", fontsize=11)
    fig.tight_layout(); fig.savefig(figdir / "analysis_face_validity.png", dpi=300); plt.close(fig)

    print(json.dumps({"directness_m": summary["directness_m"],
                      "most_direct": summary["most_direct"],
                      "least_direct": summary["least_direct"]}, indent=1))


if __name__ == "__main__":
    main()
