"""Mechanical null for the velocity-fidelity gap (thesis Chapter 5).

Addresses the review concern that the headline of Chapter 5, that the
position-only vs velocity-bearing discrepancy grows with player speed, is partly
forced by construction: the discrepancy is computed by zeroing velocity, so a
larger omitted velocity mechanically moves the field more. This script quantifies
how much of the speed-discrepancy curve is algebraically expected under the model
alone, with no football structure.

Synthetic frames place 11 vs 11 players uniformly at random on the pitch and give
each a velocity of random direction and a per-frame Rayleigh speed scale, so mean
frame speed varies. The same kinematic-vs-position-only comparison used on real
data is applied. The resulting curve is the purely mechanical expectation; it is
printed beside the real Metrica curve so the reader can see what, if anything, the
real data add beyond the model algebra.

Output: figures/analysis_ch5_mechanical_null.json (+ .png).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ffields import load_config, repo_root
from ffields.fields import KinematicPitchControl
from ffields.geometry import Grid, Pitch
from ffields.provenance import RunProvenance, seed_everything

BINS = [0, 1, 2, 3, 4, 99]
LABELS = ["0-1", "1-2", "2-3", "3-4", ">4"]
N_FRAMES = 4000
CAP = 12.0


def _corr_flip(Ck, Cp):
    a, b = Ck.values.reshape(-1), Cp.values.reshape(-1)
    corr = float(np.corrcoef(a, b)[0, 1]) if min(np.std(a), np.std(b)) > 1e-12 else np.nan
    flip = float(np.mean((a > 0.5) != (b > 0.5)))
    return corr, flip


def main():
    cfg = load_config()
    seed_everything(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    root = repo_root()
    figdir = root / cfg["paths"]["figures"]
    pitch = Pitch(metric_length=cfg["pitch"]["metric_length"], metric_width=cfg["pitch"]["metric_width"])
    grid = Grid(pitch, nx=cfg["grid"]["nx"], ny=cfg["grid"]["ny"])
    kpc = KinematicPitchControl(grid, max_speed_m_s=cfg["control"]["max_speed_m_s"],
                                reaction_time_s=cfg["control"]["reaction_time_s"],
                                tti_temperature_s=cfg["control"]["tti_temperature_s"])
    L, W = pitch.metric_length, pitch.metric_width

    rows = []
    for _ in range(N_FRAMES):
        hp = rng.uniform([0, 0], [L, W], size=(11, 2))
        ap = rng.uniform([0, 0], [L, W], size=(11, 2))
        scale = rng.uniform(0.2, 3.5)  # per-frame speed scale -> spread of mean speeds
        def vel(n):
            sp = rng.rayleigh(scale, size=n)
            sp = np.clip(sp, 0, CAP)
            th = rng.uniform(0, 2 * np.pi, size=n)
            return np.column_stack([sp * np.cos(th), sp * np.sin(th)])
        hv, av = vel(11), vel(11)
        Ck = kpc.estimate(hp, hv, ap, av)
        Cp = kpc.estimate(hp, np.zeros_like(hv), ap, np.zeros_like(av))
        corr, flip = _corr_flip(Ck, Cp)
        mean_speed = float(np.mean(np.hypot(np.r_[hv[:, 0], av[:, 0]], np.r_[hv[:, 1], av[:, 1]])))
        rows.append((mean_speed, corr, flip))
    arr = np.array(rows)

    by_band = {}
    for i, lbl in enumerate(LABELS):
        m = (arr[:, 0] >= BINS[i]) & (arr[:, 0] < BINS[i + 1])
        if m.sum() >= 10:
            by_band[lbl] = {"n": int(m.sum()),
                            "corr_mean": float(np.nanmean(arr[m, 1])),
                            "flip_mean": float(np.nanmean(arr[m, 2]))}

    # real Metrica curve for side-by-side
    real = {}
    try:
        rj = json.load(open(figdir / "analysis_fidelity_extended.json"))["by_speed_band"]
        for lbl in LABELS:
            if lbl in rj:
                real[lbl] = {"corr_mean": rj[lbl]["corr_mean"], "flip_mean": rj[lbl]["flip_mean"]}
    except Exception:
        pass

    summary = {"n_frames": N_FRAMES, "design": "11v11 uniform-random positions, random-direction "
               "Rayleigh-speed velocities; kinematic vs zero-velocity control, same model as Chapter 5",
               "mechanical_null_by_band": by_band, "real_metrica_by_band": real,
               "note": ("The mechanical null is the speed-discrepancy relation with no football "
                        "structure. Comparison with the real curve shows how much of the direction "
                        "is algebraically forced (shared) versus data-specific (any gap).")}
    (figdir / "analysis_ch5_mechanical_null.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RunProvenance(seed=cfg["seed"], extra={"stage": "ch5_mechanical_null"}).write(
        figdir / "analysis_ch5_mechanical_null_provenance.json")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(LABELS))
    nm_c = [by_band.get(l, {}).get("corr_mean", np.nan) for l in LABELS]
    rl_c = [real.get(l, {}).get("corr_mean", np.nan) for l in LABELS]
    nm_f = [by_band.get(l, {}).get("flip_mean", np.nan) for l in LABELS]
    rl_f = [real.get(l, {}).get("flip_mean", np.nan) for l in LABELS]
    ax[0].plot(x, nm_c, "-o", color="grey", label="mechanical null (random)")
    ax[0].plot(x, rl_c, "-o", color="#2166ac", label="real (Metrica)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(LABELS); ax[0].set_xlabel("mean player speed (m/s)")
    ax[0].set_ylabel("spatial correlation (kin vs pos-only)"); ax[0].legend(fontsize=8)
    ax[0].set_title("A. Correlation vs speed", fontsize=11)
    ax[1].plot(x, nm_f, "-o", color="grey", label="mechanical null (random)")
    ax[1].plot(x, rl_f, "-o", color="#b2182b", label="real (Metrica)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(LABELS); ax[1].set_xlabel("mean player speed (m/s)")
    ax[1].set_ylabel("controlling-team flip fraction"); ax[1].legend(fontsize=8)
    ax[1].set_title("B. Flip fraction vs speed", fontsize=11)
    fig.suptitle("Mechanical null vs real velocity-fidelity curve | associational", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(figdir / "analysis_ch5_mechanical_null.png", dpi=300)
    plt.close(fig)

    print(f"{'band':>5} {'null_corr':>9} {'real_corr':>9} {'null_flip':>9} {'real_flip':>9}")
    for l in LABELS:
        print(f"{l:>5} {by_band.get(l,{}).get('corr_mean',float('nan')):>9.3f} "
              f"{real.get(l,{}).get('corr_mean',float('nan')):>9.3f} "
              f"{by_band.get(l,{}).get('flip_mean',float('nan')):>9.3f} "
              f"{real.get(l,{}).get('flip_mean',float('nan')):>9.3f}")


if __name__ == "__main__":
    main()
