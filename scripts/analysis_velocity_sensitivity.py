"""Sensitivity of the velocity-fidelity gap to velocity estimation (thesis Chapter 5).

Because velocity is the single manipulated ingredient in Chapter 5, the review asks
that the fidelity gap be shown robust to how velocity is estimated, not only to the
control-model constants. This script recomputes the kinematic-versus-position-only
agreement on the cached Metrica matches under several velocity estimators:

* central finite difference (no smoothing);
* centred moving average over 3, 7 (the default), 11, and 15 frames;
* a Savitzky-Golay derivative (window 7, order 2);

and, for the default estimator, three velocity caps (8, 12, 15 m/s). If the overall
agreement and the high-speed-band agreement are stable across these choices, the
gap is a property of omitting velocity rather than of a particular velocity
estimator.

Output: figures/analysis_velocity_sensitivity.json (+ .png).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ffields import load_config, repo_root
from ffields import robustness as rb
from ffields.fields import KinematicPitchControl
from ffields.geometry import Grid, Pitch
from ffields.ingest import MetricaClient, parse_tracking
from ffields.provenance import RunProvenance, seed_everything

N_EVAL = 500


def make_velocity(pos, dt, method):
    """Return (N,2) velocity for one player's (N,2) positions under a chosen estimator."""
    out = np.full_like(pos, np.nan)
    for c in (0, 1):
        s = pd.Series(pos[:, c])
        if method == "central":
            xs = s.to_numpy()
            v = np.gradient(xs) / dt
        elif method.startswith("ma"):
            w = int(method[2:])
            xs = s.rolling(w, center=True, min_periods=1).mean().to_numpy()
            v = np.gradient(xs) / dt
        elif method == "savgol":
            xf = s.interpolate(limit=5, limit_direction="both").to_numpy()
            if np.isfinite(xf).all():
                v = savgol_filter(xf, 7, 2, deriv=1, delta=dt)
            else:
                v = np.gradient(pd.Series(pos[:, c]).rolling(7, center=True, min_periods=1).mean().to_numpy()) / dt
        out[:, c] = v
    out[~np.isfinite(pos)] = np.nan
    return out


def eval_match(m, vels, kpc, cap, n_eval, rng):
    rows = []
    idx = np.unique(rng.integers(1000, m.n_frames - 1000, size=n_eval))
    pids = list(m.positions.keys())
    for i in idx:
        hp, hv, ap, av = [], [], [], []
        for pid in pids:
            p = m.positions[pid][i]
            if not np.all(np.isfinite(p)):
                continue
            v = vels[pid][i].copy()
            if not np.all(np.isfinite(v)):
                v = np.zeros(2)
            sp = float(np.hypot(v[0], v[1]))
            if sp > cap:
                v = v * (cap / sp)
            if m.team_of[pid] == "home":
                hp.append(p); hv.append(v)
            else:
                ap.append(p); av.append(v)
        if len(hp) < 7 or len(ap) < 7:
            continue
        hp, hv, ap, av = map(np.array, (hp, hv, ap, av))
        Ck = kpc.estimate(hp, hv, ap, av)
        Cp = kpc.estimate(hp, np.zeros_like(hv), ap, np.zeros_like(av))
        corr = rb.spatial_correlation(Ck, Cp)
        flip = float(np.mean((Ck.values > 0.5) != (Cp.values > 0.5)))
        sp = np.r_[np.hypot(hv[:, 0], hv[:, 1]), np.hypot(av[:, 0], av[:, 1])].mean()
        rows.append({"corr": corr, "flip": flip, "mean_speed": float(sp)})
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    seed_everything(cfg["seed"])
    root = repo_root()
    figdir = root / cfg["paths"]["figures"]
    pitch = Pitch()
    grid = Grid(pitch, nx=cfg["grid"]["nx"], ny=cfg["grid"]["ny"])
    kpc = KinematicPitchControl(grid, max_speed_m_s=cfg["control"]["max_speed_m_s"],
                                reaction_time_s=cfg["control"]["reaction_time_s"],
                                tti_temperature_s=cfg["control"]["tti_temperature_s"])
    cl = MetricaClient(cache_dir=root / "data" / "cache" / "metrica")

    # parse both games once; keep raw positions
    games = []
    for g in (1, 2):
        m = parse_tracking(cl.tracking_csv(g, "home"), cl.tracking_csv(g, "away"), pitch)
        dt = float(np.nanmedian(np.diff(m.time)))
        games.append((m, dt))

    configs = [("central", 12.0), ("ma3", 12.0), ("ma7", 12.0), ("ma11", 12.0),
               ("ma15", 12.0), ("savgol", 12.0), ("ma7", 8.0), ("ma7", 15.0)]
    rng = np.random.default_rng(cfg["seed"])
    results = {}
    for method, cap in configs:
        rows = []
        for m, dt in games:
            vels = {pid: make_velocity(m.positions[pid], dt, method) for pid in m.positions}
            rows.append(eval_match(m, vels, kpc, cap, N_EVAL, rng))
        df = pd.concat(rows, ignore_index=True)
        fast = df[df["mean_speed"] > 4.0]
        key = f"{method}_cap{cap:g}"
        results[key] = {
            "method": method, "cap": cap, "n": int(len(df)),
            "overall_corr": float(df["corr"].mean()), "overall_flip": float(df["flip"].mean()),
            "fast_corr": float(fast["corr"].mean()) if len(fast) else None,
            "fast_flip": float(fast["flip"].mean()) if len(fast) else None,
            "n_fast": int(len(fast))}

    summary = {"dataset": "Metrica sample (both games)", "n_eval_per_game": N_EVAL,
               "baseline_key": "ma7_cap12", "by_config": results,
               "note": ("Velocity-estimation sensitivity of the Chapter 5 fidelity gap. If overall "
                        "and high-speed agreement are stable across estimators, the gap is a property "
                        "of omitting velocity, not of the velocity estimator. ma7_cap12 is the default.")}
    (figdir / "analysis_velocity_sensitivity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RunProvenance(seed=cfg["seed"], extra={"stage": "velocity_sensitivity"}).write(
        figdir / "analysis_velocity_sensitivity_provenance.json")

    keys = list(results.keys())
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].bar(range(len(keys)), [results[k]["overall_corr"] for k in keys], color="#4393c3")
    ax[0].set_xticks(range(len(keys))); ax[0].set_xticklabels(keys, rotation=40, ha="right", fontsize=7)
    ax[0].set_ylim(0.95, 1.0); ax[0].set_ylabel("overall spatial correlation")
    ax[0].set_title("A. Overall agreement by velocity estimator", fontsize=11)
    ax[1].bar(range(len(keys)), [results[k]["fast_flip"] for k in keys], color="#b2182b")
    ax[1].set_xticks(range(len(keys))); ax[1].set_xticklabels(keys, rotation=40, ha="right", fontsize=7)
    ax[1].set_ylabel("flip fraction, >4 m/s band")
    ax[1].set_title("B. High-speed flip fraction by velocity estimator", fontsize=11)
    fig.suptitle("Velocity-estimation sensitivity of the fidelity gap | Metrica | associational", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(figdir / "analysis_velocity_sensitivity.png", dpi=300)
    plt.close(fig)

    print(f"{'config':>14} {'overall_corr':>12} {'overall_flip':>12} {'fast_corr':>10} {'fast_flip':>10}")
    for k in keys:
        r = results[k]
        print(f"{k:>14} {r['overall_corr']:>12.4f} {r['overall_flip']:>12.4f} "
              f"{(r['fast_corr'] or float('nan')):>10.4f} {(r['fast_flip'] or float('nan')):>10.4f}")


if __name__ == "__main__":
    main()
