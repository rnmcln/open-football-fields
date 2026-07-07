"""Estimation impact of the broadcast coverage bias (thesis Chapter 4).

The observation-ceiling study quantifies *how much* of the player configuration a
360 frame fails to observe. It does not quantify *how much that missing portion
changes a player-configuration field estimate*. This script measures the second
quantity for position-only pitch control, using an internal proxy that needs no
external ground truth.

Design
------
Broadcast cameras follow the ball, so the players a frame omits are
disproportionately those far from the ball. Among the well-observed frames (many
visible players) we treat the full visible configuration as a local reference,
then emulate increasing ball-following partiality by removing the ``k`` players
farthest from the ball and recomputing control. The divergence between the
reduced and reference control fields, as a function of ``k``, is the estimation
impact of the coverage bias.

Comparisons are made inside the reference frame's own visible mask (the region
that would actually be reported), by:

* spatial correlation between reference and reduced control fields;
* controlling-team cell-flip fraction (fraction of reported cells whose
  controlling team changes);
* mean absolute control difference.

Caveats (carried into the thesis)
---------------------------------
* The reference is itself a well-observed *broadcast* frame, not the latent
  state, so this measures sensitivity to the systematically-missing far players,
  not absolute error against full tracking.
* Well-observed frames are a biased subset (settled play, wider camera), so the
  magnitudes are indicative; the monotone direction is the robust finding.
* External validation against fuller tracking (e.g. PFF FC 2022 broadcast
  tracking on the same World Cup matches) is left as future work.

Runs in two disk-checkpointed stages so each fits comfortably in a short job:

    python scripts/analysis_coverage_impact.py scan       # -> candidate list
    python scripts/analysis_coverage_impact.py compute     # -> json + png
    python scripts/analysis_coverage_impact.py all         # both (default)

Output: figures/analysis_coverage_impact.json (+ .png) and a provenance file.
"""
from __future__ import annotations

import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from ffields import load_config, repo_root
from ffields.geometry import Grid, Pitch
from ffields.ingest import (
    StatsBombClient,
    attach_freeze_frames,
    events_to_frame,
    normalise_attacking_direction,
)
from ffields.observation import FreezeFrame, ObservationOperator
from ffields.provenance import RunProvenance, seed_everything

# Competitions with 360 in the pinned cache: UEFA Euro 2020 and FIFA World Cup 2022.
COMPETITIONS = [(55, 43, "UEFA Euro 2020"), (43, 106, "FIFA World Cup 2022")]

MIN_VISIBLE = 18          # a frame is a "well-observed" reference only if >= this many players
K_REMOVE = [0, 1, 2, 3, 4, 5, 6]   # farthest-from-ball players removed
MIN_PER_SIDE = 2          # never reduce a side below this many players
N_SAMPLE = 2500           # seeded sample of reference frames (runtime control)
N_BOOT = 300              # bootstrap resamples over frames for CIs
CAND_PATH_NAME = "_cov_candidates.json"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _cand_path(root, cfg):
    return root / cfg["paths"]["data_cache"] / CAND_PATH_NAME


def scan(root, cfg, pitch) -> None:
    client = StatsBombClient(cache_dir=root / cfg["paths"]["data_cache"])
    candidates = []
    per_comp = {}
    for comp, season, name in COMPETITIONS:
        try:
            matches = client.matches(comp, season)
        except Exception:
            continue
        n_here = 0
        _log(f"scanning {name}: {len(matches)} matches")
        for mi, m in enumerate(matches):
            if mi % 15 == 0:
                _log(f"  {name} {mi}/{len(matches)}; candidates {len(candidates)}")
            mid = m["match_id"]
            try:
                ts = attach_freeze_frames(client.three_sixty(mid))
                ev = normalise_attacking_direction(events_to_frame(client.events(mid), pitch), pitch)
            except Exception:
                continue
            ev = ev[ev["id"].isin(ts.keys())].copy()
            if ev.empty:
                continue
            ev["nvis"] = ev["id"].map(lambda u: len(ts[u]["freeze_frame"]))
            keep = (ev["nvis"] >= MIN_VISIBLE) & ev["x_att"].notna() & ev["y_att"].notna()
            sel = ev.loc[keep, ["id", "att_sign", "x_att", "y_att"]]
            for uuid, att_sign, xa, ya in sel.itertuples(index=False, name=None):
                candidates.append([int(mid), str(uuid), int(att_sign), float(xa), float(ya)])
            n_here += int(keep.sum())
        per_comp[name] = n_here
    out = {"candidates": candidates, "per_competition": per_comp}
    _cand_path(root, cfg).write_text(json.dumps(out), encoding="utf-8")
    _log(f"scan done: {len(candidates)} candidates -> {_cand_path(root, cfg)}")


def _control_values(att, deff, ak, dk, centres, pc_cfg) -> np.ndarray:
    """Fast position-only control values (no mask), matching PositionalPitchControl."""
    max_speed, reaction, tau, keep_keeper = pc_cfg
    if not keep_keeper:
        att = att[~ak] if len(att) else att
        deff = deff[~dk] if len(dk) else deff
    ncells = centres.shape[0]

    def wsum(players):
        if len(players) == 0:
            return np.zeros(ncells)
        d = cdist(players, centres)
        return np.exp(-(reaction + d / max_speed) / tau).sum(axis=0)

    w_att, w_def = wsum(att), wsum(deff)
    denom = w_att + w_def
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, w_att / denom, 0.5)


def compute(root, cfg, pitch, grid) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    data = json.loads(_cand_path(root, cfg).read_text())
    candidates = data["candidates"]
    per_comp = data["per_competition"]
    n_candidates = len(candidates)
    if n_candidates > N_SAMPLE:
        idx = rng.choice(n_candidates, size=N_SAMPLE, replace=False)
        sample = [candidates[i] for i in idx]
    else:
        sample = candidates

    centres = grid.flat_centres()
    op = ObservationOperator(grid)
    pc_cfg = (cfg["control"]["max_speed_m_s"], cfg["control"]["reaction_time_s"],
              cfg["control"]["tti_temperature_s"], cfg["control"]["keeper_included"])
    client = StatsBombClient(cache_dir=root / cfg["paths"]["data_cache"])

    ts_cache: dict[int, dict] = {}
    per_k = {k: {"corr": [], "flip": [], "mad": []} for k in K_REMOVE}
    n_used = 0
    _log(f"candidates={n_candidates}, sampled={len(sample)}; computing")
    for si, (mid, uuid, att_sign, bx, by) in enumerate(sample):
        if si % 500 == 0:
            _log(f"  compute {si}/{len(sample)}; used {n_used}")
        if mid not in ts_cache:
            try:
                ts_cache[mid] = attach_freeze_frames(client.three_sixty(mid))
            except Exception:
                ts_cache[mid] = {}
        ts = ts_cache[mid]
        if uuid not in ts:
            continue
        frame = FreezeFrame.from_raw(ts[uuid], att_sign, pitch)
        na, nd = len(frame.attackers), len(frame.defenders)
        if na < MIN_PER_SIDE + 1 or nd < MIN_PER_SIDE + 1:
            continue
        mask = op.visible_mask(frame.visible_polygon).reshape(-1)
        if mask.sum() < 8:
            continue

        ref = _control_values(frame.attackers, frame.defenders,
                              frame.attacker_keeper, frame.defender_keeper, centres, pc_cfg)
        a = ref[mask]
        if np.std(a) < 1e-9:
            continue
        n_used += 1

        ball = np.array([bx, by])
        d_att = np.linalg.norm(frame.attackers - ball, axis=1)
        d_def = np.linalg.norm(frame.defenders - ball, axis=1)
        order = sorted(
            [("a", i, d_att[i]) for i in range(na)] + [("d", i, d_def[i]) for i in range(nd)],
            key=lambda t: -t[2],
        )
        for k in K_REMOVE:
            if k == 0:
                per_k[k]["corr"].append(1.0); per_k[k]["flip"].append(0.0); per_k[k]["mad"].append(0.0)
                continue
            drop_a, drop_d = set(), set()
            removed = 0
            for side, i, _d in order:
                if removed >= k:
                    break
                if side == "a" and (na - len(drop_a)) > MIN_PER_SIDE:
                    drop_a.add(i); removed += 1
                elif side == "d" and (nd - len(drop_d)) > MIN_PER_SIDE:
                    drop_d.add(i); removed += 1
            keep_att = np.array([i for i in range(na) if i not in drop_a], dtype=int)
            keep_def = np.array([i for i in range(nd) if i not in drop_d], dtype=int)
            red = _control_values(frame.attackers[keep_att], frame.defenders[keep_def],
                                 frame.attacker_keeper[keep_att], frame.defender_keeper[keep_def],
                                 centres, pc_cfg)
            b = red[mask]
            corr = float(np.corrcoef(a, b)[0, 1]) if np.std(b) > 1e-9 else np.nan
            per_k[k]["corr"].append(corr)
            per_k[k]["flip"].append(float(np.mean((a > 0.5) != (b > 0.5))))
            per_k[k]["mad"].append(float(np.mean(np.abs(a - b))))

    def agg(vals):
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"mean": None, "lo": None, "hi": None, "n": 0}
        boots = [float(np.mean(rng.choice(arr, size=arr.size, replace=True))) for _ in range(N_BOOT)]
        return {"mean": float(arr.mean()), "lo": float(np.percentile(boots, 2.5)),
                "hi": float(np.percentile(boots, 97.5)), "n": int(arr.size)}

    by_k = {str(k): {"spatial_corr": agg(per_k[k]["corr"]),
                     "flip_fraction": agg(per_k[k]["flip"]),
                     "mean_abs_diff": agg(per_k[k]["mad"])} for k in K_REMOVE}

    summary = {
        "analysis": "coverage-bias estimation impact (internal proxy)",
        "competitions": [n for _, _, n in COMPETITIONS],
        "min_visible_reference": MIN_VISIBLE,
        "min_per_side": MIN_PER_SIDE,
        "grid": {"nx": grid.nx, "ny": grid.ny},
        "n_candidate_frames": n_candidates,
        "n_sampled": len(sample),
        "n_used": n_used,
        "per_competition_candidates": per_comp,
        "k_removed": K_REMOVE,
        "by_k": by_k,
        "note": ("Reduced control fields are compared with the well-observed reference "
                 "inside the reference visible mask. Removing the farthest-from-ball "
                 "players emulates the broadcast coverage bias. Internal proxy; the "
                 "reference is a broadcast frame, not the latent state."),
    }
    return summary


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfg = load_config()
    seed_everything(cfg["seed"])
    root = repo_root()
    figdir = root / cfg["paths"]["figures"]
    figdir.mkdir(parents=True, exist_ok=True)
    pitch = Pitch(metric_length=cfg["pitch"]["metric_length"], metric_width=cfg["pitch"]["metric_width"])
    grid = Grid(pitch, nx=cfg["grid"]["nx"], ny=cfg["grid"]["ny"])

    if stage in ("scan", "all"):
        scan(root, cfg, pitch)
    if stage in ("compute", "all"):
        summary = compute(root, cfg, pitch, grid)
        (figdir / "analysis_coverage_impact.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        RunProvenance(seed=cfg["seed"], extra={"stage": "coverage_impact"}).write(
            figdir / "analysis_coverage_impact_provenance.json")

        ks = K_REMOVE
        gm = lambda key, s: [summary["by_k"][str(k)][key][s] for k in ks]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(ks, gm("spatial_corr", "mean"), "-o", color="#2166ac")
        axes[0].fill_between(ks, gm("spatial_corr", "lo"), gm("spatial_corr", "hi"), alpha=0.2, color="#2166ac")
        axes[0].set_xlabel("farthest-from-ball players removed")
        axes[0].set_ylabel("spatial correlation with reference control")
        axes[0].set_title("A. Control-field agreement vs coverage loss", fontsize=11)
        axes[1].plot(ks, gm("flip_fraction", "mean"), "-o", color="#b2182b")
        axes[1].fill_between(ks, gm("flip_fraction", "lo"), gm("flip_fraction", "hi"), alpha=0.2, color="#b2182b")
        axes[1].set_xlabel("farthest-from-ball players removed")
        axes[1].set_ylabel("controlling-team cell-flip fraction")
        axes[1].set_title("B. Controlling-team disagreement vs coverage loss", fontsize=11)
        fig.suptitle("Estimation impact of the broadcast coverage bias (internal proxy) | "
                     "StatsBomb 360 open data (attribution: StatsBomb) | associational", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(figdir / "analysis_coverage_impact.png", dpi=300)
        plt.close(fig)
        print(json.dumps(summary["by_k"], indent=2))


if __name__ == "__main__":
    main()
