"""Pitch-control fidelity study on the Sportec open dataset (thesis Chapter 5).

Replicates the velocity-fidelity measurement of analysis_fidelity_extended.py on a
second, independent open continuous-tracking source: the Sportec/IDSSE open dataset
(Bassek et al., 2025, Scientific Data; CC-BY 4.0), distributed via kloppy and the
PySport HuggingFace mirror. Seven Bundesliga matches carry 25 Hz tracking for all
players and the ball. The coordinate convention and per-player speeds were validated
against kloppy (identical to load_open_tracking_data, coordinates="secondspectrum");
here a lightweight streaming sampler is used so that representative frames can be
drawn from across each full match without paying kloppy's full-match parse cost.

Method, identical to the Metrica study: for each sampled frame, the velocity-bearing
control field and the position-only field (velocities zeroed) are computed from the
same players; agreement is the spatial correlation and the controlling-team cell-flip
fraction, stratified by frame mean player speed, with bootstrap CIs. Velocities are
centred finite differences over a 7-frame (0.24 s) window, capped.

Output: figures/analysis_fidelity_sportec.json (+ .png)
"""
from __future__ import annotations

import glob
import json
import os
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ffields import load_config, repo_root
from ffields import robustness as rb
from ffields.fields import KinematicPitchControl
from ffields.geometry import Grid, Pitch
from ffields.provenance import RunProvenance, seed_everything

BINS = [0.0, 1.0, 2.0, 3.0, 4.0, 12.0]
BIN_LABELS = ["0-1", "1-2", "2-3", "3-4", ">4"]
FPS = 25
DT = 1.0 / FPS
STRIDE = 200          # one anchor frame every 200 frames (~8 s) across the match
HALF = 3             # +/- frames around an anchor for the velocity window
N_EVAL = 250          # anchors evaluated per match (random subsample)

COMP = {"J03WMX": "DFL-COM-000001", "J03WN1": "DFL-COM-000001",
        "J03WOH": "DFL-COM-000002", "J03WOY": "DFL-COM-000002",
        "J03WPY": "DFL-COM-000002", "J03WQQ": "DFL-COM-000002",
        "J03WR9": "DFL-COM-000002"}


def _team_ids(meta_path):
    """Return (home_club_id, away_club_id) from the Sportec match-information XML."""
    for _, el in ET.iterparse(meta_path, events=("end",)):
        if el.tag.endswith("General"):
            return el.attrib["HomeTeamId"], el.attrib["GuestTeamId"]
    raise ValueError("no General element in meta")


def _collect_windows(track_path, home_id, away_id):
    """Stream the positions XML, collecting player x,y for frames in anchor windows.

    Returns dict: (section, anchor) -> {"home": {pid: {off: (x,y)}}, "away": {...}}.
    Coordinates are shifted from the centred Sportec system to a [0,105]x[0,68] pitch.
    """
    windows = {}
    cur_section = None
    cur_ground = None
    cur_pid = None
    for ev, el in ET.iterparse(track_path, events=("start", "end")):
        tag = el.tag.split("}")[-1]
        if ev == "start" and tag == "FrameSet":
            cur_team = el.attrib.get("TeamId")
            cur_section = el.attrib.get("GameSection")
            cur_pid = el.attrib.get("PersonId")
            cur_ground = "home" if cur_team == home_id else "away" if cur_team == away_id else None
        elif ev == "end" and tag == "Frame":
            if cur_ground is not None and cur_pid is not None:
                n = int(el.attrib["N"])
                off = (n % STRIDE) - (STRIDE // 2)
                if -HALF <= off <= HALF:
                    anchor = (n // STRIDE) * STRIDE + (STRIDE // 2)
                    key = (cur_section, anchor)
                    x = float(el.attrib["X"]) + 52.5
                    y = float(el.attrib["Y"]) + 34.0
                    windows.setdefault(key, {"home": {}, "away": {}})
                    windows[key][cur_ground].setdefault(cur_pid, {})[off] = (x, y)
            el.clear()
        elif ev == "end" and tag == "FrameSet":
            cur_section = cur_ground = cur_pid = None
            el.clear()
    return windows


def _arrays_from_window(side, cap):
    """Positions (at anchor, off=0) and capped centred-difference velocities."""
    pos, vel = [], []
    span = 2 * HALF * DT
    for pid, frames in side.items():
        if 0 not in frames or -HALF not in frames or HALF not in frames:
            continue
        x0, y0 = frames[0]
        (xa, ya), (xb, yb) = frames[-HALF], frames[HALF]
        vx, vy = (xb - xa) / span, (yb - ya) / span
        sp = float(np.hypot(vx, vy))
        if sp > cap and sp > 0:
            vx, vy = vx * cap / sp, vy * cap / sp
        pos.append((x0, y0)); vel.append((vx, vy))
    return np.asarray(pos, float), np.asarray(vel, float)


def _eval_match(track_path, meta_path, kpc, cap, rng):
    home_id, away_id = _team_ids(meta_path)
    windows = _collect_windows(track_path, home_id, away_id)
    keys = list(windows.keys())
    if len(keys) > N_EVAL:
        keys = [keys[i] for i in np.sort(rng.choice(len(keys), N_EVAL, replace=False))]
    rows = []
    for key in keys:
        hp, hv = _arrays_from_window(windows[key]["home"], cap)
        ap, av = _arrays_from_window(windows[key]["away"], cap)
        if len(hp) < 7 or len(ap) < 7:
            continue
        Ck = kpc.estimate(hp, hv, ap, av)
        Cp = kpc.estimate(hp, np.zeros_like(hv), ap, np.zeros_like(av))
        corr = rb.spatial_correlation(Ck, Cp)
        flip = float(np.mean((Ck.values > 0.5) != (Cp.values > 0.5)))
        sp = np.r_[np.hypot(hv[:, 0], hv[:, 1]), np.hypot(av[:, 0], av[:, 1])].mean()
        rows.append({"corr": corr, "flip": flip, "mean_speed": float(sp)})
    return pd.DataFrame(rows)


def _boot_ci(x, n_boot=2000, rng=None):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return (float("nan"), float("nan"), float("nan"))
    rng = rng or np.random.default_rng(0)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return (float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main() -> None:
    cfg = load_config(); seed_everything(cfg["seed"]); root = repo_root()
    figdir = root / cfg["paths"]["figures"]; pitch = Pitch()
    grid = Grid(pitch, nx=cfg["grid"]["nx"], ny=cfg["grid"]["ny"])
    cap = cfg["metrica"]["max_speed_cap_m_s"]
    kpc = KinematicPitchControl(grid, max_speed_m_s=cfg["control"]["max_speed_m_s"],
                                reaction_time_s=cfg["control"]["reaction_time_s"],
                                tti_temperature_s=cfg["control"]["tti_temperature_s"])
    rng = np.random.default_rng(cfg["seed"])
    sdir = root / "data" / "cache" / "sportec"

    def _complete(path):
        """A fully downloaded positions file ends with the closing root tag."""
        if not path.exists() or path.stat().st_size < 50_000_000:
            return False
        with open(path, "rb") as fh:
            fh.seek(-200, os.SEEK_END)
            return b"</PutDataRequest>" in fh.read()

    per_match, allrows = {}, []
    for mid in COMP:
        tp = sdir / f"track_{mid}.xml"; mp = sdir / f"meta_{mid}.xml"
        rows_cache = figdir / f"_sportec_rows_{mid}.json"
        if rows_cache.exists():
            df = pd.read_json(rows_cache)
        else:
            if not (mp.exists() and _complete(tp)):
                continue
            try:
                df = _eval_match(str(tp), str(mp), kpc, cap, rng)
            except Exception as e:
                per_match[mid] = {"error": f"{type(e).__name__}: {e}"}; continue
            if len(df) == 0:
                continue
            df.to_json(rows_cache)  # cache so the run is resumable across matches
        df["match"] = mid; allrows.append(df)
        per_match[mid] = {"n_evaluated": int(len(df)),
                          "overall_corr": float(df["corr"].mean()),
                          "overall_flip": float(df["flip"].mean())}
    if not allrows:
        print(json.dumps({"error": "no Sportec matches available/parsed", "per_match": per_match}, indent=2))
        return
    df = pd.concat(allrows, ignore_index=True)
    df["bin"] = pd.cut(df["mean_speed"], bins=BINS, labels=BIN_LABELS)

    by_speed = {}
    for lbl in BIN_LABELS:
        g = df[df["bin"] == lbl]
        cm, clo, chi = _boot_ci(g["corr"], rng=rng)
        fm, flo, fhi = _boot_ci(g["flip"], rng=rng)
        by_speed[lbl] = {"n": int(len(g)), "corr_mean": cm, "corr_lo": clo, "corr_hi": chi,
                         "flip_mean": fm, "flip_lo": flo, "flip_hi": fhi}

    summary = {
        "dataset": "Sportec/IDSSE open (Bassek et al. 2025; CC-BY 4.0)",
        "design": {"anchors_per_match": N_EVAL, "stride_frames": STRIDE,
                   "velocity_window_frames": 2 * HALF + 1, "fps": FPS,
                   "min_players_per_side": 7, "velocity_cap_m_s": cap,
                   "smoothing": "centred finite difference over 7 frames (0.24 s)",
                   "speed_stratifier": "frame mean player speed (both teams)",
                   "coordinate_validation": "raw X,Y,S identical to kloppy load_open_tracking_data (secondspectrum)"},
        "n_matches": int(len([k for k, v in per_match.items() if "n_evaluated" in v])),
        "matches": per_match,
        "n_evaluated_total": int(len(df)),
        "overall_corr_mean": float(df["corr"].mean()),
        "overall_flip_mean": float(df["flip"].mean()),
        "by_speed_band": by_speed,
    }
    (figdir / "analysis_fidelity_sportec.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RunProvenance(seed=cfg["seed"], extra={"stage": "fidelity_sportec"}).write(
        figdir / "analysis_fidelity_sportec_provenance.json")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    x = np.arange(len(BIN_LABELS))
    cm = [by_speed[l]["corr_mean"] for l in BIN_LABELS]
    clo = [by_speed[l]["corr_mean"] - by_speed[l]["corr_lo"] for l in BIN_LABELS]
    chi = [by_speed[l]["corr_hi"] - by_speed[l]["corr_mean"] for l in BIN_LABELS]
    axes[0].errorbar(x, cm, yerr=[clo, chi], fmt="o-", color="#2166ac", capsize=4)
    axes[0].set_xticks(x); axes[0].set_xticklabels(BIN_LABELS)
    axes[0].set_xlabel("frame mean player speed (m/s)")
    axes[0].set_ylabel("spatial correlation (kinematic vs position-only)")
    axes[0].set_title(f"A. Control agreement declines with speed\n"
                      f"Sportec open data, {summary['n_matches']} matches (95% bootstrap CIs)", fontsize=11)
    fm = [by_speed[l]["flip_mean"] for l in BIN_LABELS]
    flo = [by_speed[l]["flip_mean"] - by_speed[l]["flip_lo"] for l in BIN_LABELS]
    fhi = [by_speed[l]["flip_hi"] - by_speed[l]["flip_mean"] for l in BIN_LABELS]
    axes[1].errorbar(x, fm, yerr=[flo, fhi], fmt="s-", color="#762a83", capsize=4)
    axes[1].set_xticks(x); axes[1].set_xticklabels(BIN_LABELS)
    axes[1].set_xlabel("frame mean player speed (m/s)")
    axes[1].set_ylabel("controlling-team cell-flip fraction")
    axes[1].set_title("B. Controlling-team disagreement grows with speed\n(95% bootstrap CIs)", fontsize=11)
    fig.suptitle("Open-data pitch-control fidelity gap, Sportec/IDSSE open dataset "
                 "(attribution: DFL/Sportec; Bassek et al. 2025) | associational", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(figdir / "analysis_fidelity_sportec.png", dpi=300)
    plt.close(fig)
    print(json.dumps(summary, indent=2)[:1600])


if __name__ == "__main__":
    main()
