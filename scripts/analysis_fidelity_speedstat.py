"""Speed-stratifier sensitivity for the fidelity ceiling (thesis Chapter 5).

The main fidelity analysis stratifies the position-only-versus-velocity-bearing
agreement by the frame mean player speed. Mean speed can understate transition
intensity when only a few players are sprinting. This script repeats the measurement
on both Metrica sample matches and, for each evaluated frame, records three
alternative intensity summaries: the mean, the 90th-percentile, and the maximum
outfield player speed. It then reports, for each stratifier, the Spearman correlation
between the stratifier and the spatial agreement, and the mean agreement in the lowest
and highest stratifier quintiles. If the degradation is monotonic under the
90th-percentile and maximum speed as well as the mean, the fidelity ceiling is
transition-sensitive, not merely mean-speed-sensitive.

Output: figures/analysis_fidelity_speedstat.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ffields import load_config, repo_root
from ffields import robustness as rb
from ffields.fields import KinematicPitchControl
from ffields.geometry import Grid, Pitch
from ffields.ingest import MetricaClient, parse_tracking
from ffields.provenance import RunProvenance, seed_everything

N_EVAL = 700
STRATIFIERS = ["mean_speed", "p90_speed", "max_speed"]


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
        spd = np.r_[np.hypot(hv[:, 0], hv[:, 1]), np.hypot(av[:, 0], av[:, 1])]
        rows.append({"corr": corr, "flip": flip,
                     "mean_speed": float(spd.mean()),
                     "p90_speed": float(np.percentile(spd, 90)),
                     "max_speed": float(spd.max())})
    return pd.DataFrame(rows)


def _quintile_contrast(df, strat):
    q = pd.qcut(df[strat], 5, labels=False, duplicates="drop")
    lo = df.loc[q == q.min()]
    hi = df.loc[q == q.max()]
    rho_c, p_c = spearmanr(df[strat], df["corr"])
    rho_f, p_f = spearmanr(df[strat], df["flip"])
    return {
        "spearman_rho_corr": float(rho_c), "spearman_p_corr": float(p_c),
        "spearman_rho_flip": float(rho_f), "spearman_p_flip": float(p_f),
        "corr_lowest_quintile": float(lo["corr"].mean()),
        "corr_highest_quintile": float(hi["corr"].mean()),
        "flip_lowest_quintile": float(lo["flip"].mean()),
        "flip_highest_quintile": float(hi["flip"].mean()),
        "stratifier_lowest_quintile_mean": float(lo[strat].mean()),
        "stratifier_highest_quintile_mean": float(hi[strat].mean()),
    }


def main() -> None:
    cfg = load_config(); seed_everything(cfg["seed"]); root = repo_root()
    figdir = root / cfg["paths"]["figures"]; pitch = Pitch()
    grid = Grid(pitch, nx=cfg["grid"]["nx"], ny=cfg["grid"]["ny"])
    cap = cfg["metrica"]["max_speed_cap_m_s"]
    kpc = KinematicPitchControl(grid, max_speed_m_s=cfg["control"]["max_speed_m_s"],
                                reaction_time_s=cfg["control"]["reaction_time_s"],
                                tti_temperature_s=cfg["control"]["tti_temperature_s"])
    rng = np.random.default_rng(cfg["seed"])
    cl = MetricaClient(cache_dir=root / "data" / "cache" / "metrica")

    frames = []
    for game in (1, 2):
        m = parse_tracking(cl.tracking_csv(game, "home"), cl.tracking_csv(game, "away"), pitch)
        frames.append(_eval_game(m, kpc, N_EVAL, cap, rng))
    df = pd.concat(frames, ignore_index=True)

    summary = {
        "dataset": "Metrica sample (both matches)",
        "n_frames_evaluated": int(len(df)),
        "overall_corr": float(df["corr"].mean()),
        "stratifier_contrasts": {s: _quintile_contrast(df, s) for s in STRATIFIERS},
        "note": ("negative Spearman rho between a speed stratifier and the spatial "
                 "correlation indicates agreement falls as that intensity measure "
                 "rises; reported for mean, 90th-percentile, and maximum player speed"),
    }
    (figdir / "analysis_fidelity_speedstat.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    RunProvenance(seed=cfg["seed"], extra={"stage": "fidelity_speedstat"}).write(
        figdir / "analysis_fidelity_speedstat_provenance.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
