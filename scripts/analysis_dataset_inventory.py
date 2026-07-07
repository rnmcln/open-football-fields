"""Dataset inventory for the thesis appendix.

Emits a real inventory of the data actually used: per dataset, the number of
matches, teams, team-match rows (with at least 30 moves), completed moves, and
shots, computed from the local cache. Numbers are not hand-entered.

Output: figures/analysis_dataset_inventory.json
"""
from __future__ import annotations

import glob
import json

import pandas as pd

from ffields import load_config, repo_root
from ffields.provenance import RunProvenance


def _event_dataset(events_dir):
    files = sorted(glob.glob(str(events_dir / "*.parquet")))
    n_matches = len(files)
    teams, rows, moves, shots = set(), 0, 0, 0
    for f in files:
        df = pd.read_parquet(f, columns=["type", "team", "x_att", "y_att", "end_x_att", "end_y_att"])
        for team, sub in df.groupby("team"):
            teams.add(str(team))
            mv = sub[sub["type"].isin(["Pass", "Carry"])].dropna(subset=["x_att", "y_att", "end_x_att", "end_y_att"])
            moves += len(mv)
            shots += int((sub["type"] == "Shot").sum())
            if len(mv) >= 30:
                rows += 1
    return {"matches": n_matches, "teams": len(teams), "team_match_rows": rows,
            "moves": int(moves), "shots": int(shots)}


def main() -> None:
    cfg = load_config(); root = repo_root()
    figdir = root / cfg["paths"]["figures"]
    cache = root / "data" / "cache"
    inv = {}
    ev = {
        "UEFA Euro 2020 (events+360)": cache / "batch" / "events",
        "FAWSL 2018/19": cache / "batch_wsl" / "events",
        "FAWSL 2019/20": cache / "batch_wsl2" / "events",
        "English Premier League 2015/16": cache / "batch_pl" / "events",
    }
    for name, d in ev.items():
        if d.exists():
            inv[name] = _event_dataset(d)

    # SPADL (xT training + tournament descriptor study)
    spadl = {}
    for tag, label in [("55_43", "UEFA Euro 2020"), ("43_106", "FIFA World Cup 2022")]:
        files = sorted(glob.glob(str(cache / "spadl" / tag / "*.parquet")))
        if files:
            n_act = sum(len(pd.read_parquet(f, columns=["type_id"])) for f in files)
            spadl[label] = {"games": len(files), "actions": int(n_act)}

    # Metrica
    metrica = {}
    for g in (1, 2):
        p = cache / "metrica" / f"Sample_Game_{g}" / f"Sample_Game_{g}_RawTrackingData_Home_Team.csv"
        metrica[f"game {g}"] = {"present": p.exists()}

    # 360 frames (Euro 2020)
    n360 = len(glob.glob(str(cache / "three-sixty" / "*.json")))

    summary = {"event_datasets": inv, "spadl_datasets": spadl,
               "metrica_tracking": metrica, "statsbomb_360_files_cached": n360,
               "pinned_statsbomb_commit": (root / "data" / ".statsbomb_commit_sha.txt").read_text().strip()}
    (figdir / "analysis_dataset_inventory.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RunProvenance(seed=cfg["seed"], extra={"stage": "dataset_inventory"}).write(
        figdir / "analysis_dataset_inventory_provenance.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
