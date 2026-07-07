"""Extended pitch-control fidelity study (thesis Chapter 5).

Extends the single-match demonstration to both Metrica sample matches, reports
sampling and preprocessing details, and adds bootstrap confidence intervals on
the per-speed-band agreement statistics. The kinematic model is a reference
approximation, not ground truth.

Output: figures/analysis_fidelity_extended.json (+ .png)
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ffields import load_config, repo_root
from ffields import robustness as rb
from ffields.fields import KinematicPitchControl
from ffields.geometry import Grid, Pitch
from ffields.ingest import MetricaClient, parse_tracking
from ffields.provenance import RunProvenance, seed_everything

BINS = [0.0, 1.0, 2.0, 3.0, 4.0, 12.0]
BIN_LABELS = ["0-1", "1-2", "2-3", "3-4", ">4"]


def _eval_game(m, kpc, n_eval, cap, rng):
    idx = np.unique(rng.integers(1000, m.n_frames - 1000, size=n_eval))
    rows = []
    for i in idx:
        hp, hv, ap, av, ball = m.frame_arrays(i, max_speed_cap=cap)
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
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return (float("nan"), float("nan"), float("nan"))
    rng = rng or np.random.default_rng(0)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return (float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    root = repo_root()
    figdir = root / cfg["paths"]["figures"]
    pitch = Pitch()
    grid = Grid(pitch, nx=cfg["grid"]["nx"], ny=cfg["grid"]["ny"])
    cap = cfg["metrica"]["max_speed_cap_m_s"]
    kpc = KinematicPitchControl(grid, max_speed_m_s=cfg["control"]["max_speed_m_s"],
                                reaction_time_s=cfg["control"]["reaction_time_s"],
                                tti_temperature_s=cfg["control"]["tti_temperature_s"])
    cl = MetricaClient(cache_dir=root / "data" / "cache" / "metrica")
    rng = np.random.default_rng(cfg["seed"])

    per_game = {}
    allrows = []
    for game in (1, 2):
        m = parse_tracking(cl.tracking_csv(game, "home"), cl.tracking_csv(game, "away"), pitch)
        df = _eval_game(m, kpc, cfg["metrica"]["n_eval_frames"], cap, rng)
        df["game"] = game
        per_game[game] = {"n_frames_total": int(m.n_frames), "fps": m.meta["fps"],
                          "n_evaluated": int(len(df)),
                          "overall_corr": float(df["corr"].mean()),
                          "overall_flip": float(df["flip"].mean())}
        allrows.append(df)
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
        "design": {"games": [1, 2], "frames_sampled_per_game": cfg["metrica"]["n_eval_frames"],
                   "min_players_per_side": 7, "velocity_cap_m_s": cap,
                   "smoothing": "centred moving average, 7 frames; finite-difference velocity",
                   "speed_stratifier": "frame mean player speed (both teams)"},
        "per_game": per_game,
        "n_evaluated_total": int(len(df)),
        "overall_corr_mean": float(df["corr"].mean()),
        "overall_flip_mean": float(df["flip"].mean()),
        "by_speed_band": by_speed,
    }
    (figdir / "analysis_fidelity_extended.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RunProvenance(seed=cfg["seed"], extra={"stage": "fidelity_extended"}).write(
        figdir / "analysis_fidelity_extended_provenance.json")

    # figure: corr and flip vs speed with 95% bootstrap CIs
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    x = np.arange(len(BIN_LABELS))
    cm = [by_speed[l]["corr_mean"] for l in BIN_LABELS]
    clo = [by_speed[l]["corr_mean"] - by_speed[l]["corr_lo"] for l in BIN_LABELS]
    chi = [by_speed[l]["corr_hi"] - by_speed[l]["corr_mean"] for l in BIN_LABELS]
    axes[0].errorbar(x, cm, yerr=[clo, chi], fmt="o-", color="#2166ac", capsize=4)
    axes[0].set_xticks(x); axes[0].set_xticklabels(BIN_LABELS)
    axes[0].set_xlabel("frame mean player speed (m/s)")
    axes[0].set_ylabel("spatial correlation (kinematic vs position-only)")
    axes[0].set_title("A. Control agreement declines with speed\n(95% bootstrap CIs; two matches)",
                      fontsize=11)
    fm = [by_speed[l]["flip_mean"] for l in BIN_LABELS]
    flo = [by_speed[l]["flip_mean"] - by_speed[l]["flip_lo"] for l in BIN_LABELS]
    fhi = [by_speed[l]["flip_hi"] - by_speed[l]["flip_mean"] for l in BIN_LABELS]
    axes[1].errorbar(x, fm, yerr=[flo, fhi], fmt="s-", color="#762a83", capsize=4)
    axes[1].set_xticks(x); axes[1].set_xticklabels(BIN_LABELS)
    axes[1].set_xlabel("frame mean player speed (m/s)")
    axes[1].set_ylabel("controlling-team cell-flip fraction")
    axes[1].set_title("B. Controlling-team disagreement grows with speed\n(95% bootstrap CIs)",
                      fontsize=11)
    fig.suptitle("Open-data pitch-control fidelity gap, two Metrica matches "
                 "(attribution: Metrica Sports) | associational", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(figdir / "analysis_fidelity_extended.png", dpi=300)
    plt.close(fig)
    print(json.dumps(summary, indent=2)[:1200])


if __name__ == "__main__":
    main()
