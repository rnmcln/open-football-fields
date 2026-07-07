"""External validation of the coverage-bias result against PFF FC tracking (Section 4.5.7 / 8.2).

Section 4.5.7 measures, with an *internal* proxy, how much removing the
farthest-from-ball players changes position-only pitch control. This script runs
the *external* version on the PFF FC 2022 World Cup broadcast tracking, which
covers the same matches as the StatsBomb 360 data used elsewhere. For each
StatsBomb 360 event it compares:

* ``C_full`` : control from the full PFF configuration (all tracked players);
* ``C_obs``  : control from only the PFF players that fall inside the StatsBomb
               360 ``visible_area`` polygon (i.e. what a 360 frame would observe).

Agreement is summarised inside the visible mask by spatial correlation and the
controlling-team cell-flip fraction, exactly as in Table 4.5. If the coverage bias
matters, C_obs departs from C_full, and by how much is the external measurement.

The two providers are aligned automatically:
* an ID crosswalk is built from kickoff date and team names;
* everything is done in the absolute pitch frame (no possession-relative
  orientation), so PFF maps to StatsBomb by a single (sx, sy) axis transform per
  match, chosen by minimising the ball-position residual;
* the median ball residual per match is printed as a SELF-CHECK: if the alignment
  and transform are right it is small (a few metres). Treat a large residual as a
  red flag and do not trust that match's numbers.

Caveat carried into the thesis: PFF tracking is itself computer-vision broadcast
tracking, more complete than a 360 freeze frame but not stadium optical truth, so
it is a richer observation rather than the latent state.

Requirements: the PFF data under ``data/cache/pff/wc2022`` (Metadata/, Rosters/,
"Tracking Data/"), ``pip install kloppy``, and internet on first run so the
StatsBomb WC 2022 events and 360 download into the cache.

Modes
-----
    python scripts/analysis_pff_paired.py crosswalk   # build+print the ID map only
    python scripts/analysis_pff_paired.py check        # crosswalk + ONE match end-to-end (fast gate)
    python scripts/analysis_pff_paired.py all          # all matched matches -> json + png (slow)

Environment overrides: PFF_DIR, PFF_SAMPLE_RATE (default 0.2 = 5 fps),
PFF_MAX_EVENTS_PER_MATCH (default 400).
"""
from __future__ import annotations

import json
import os
import re
import resource
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Point, Polygon
from shapely.prepared import prep

from ffields import load_config, repo_root
from ffields.geometry import Grid, Pitch
from ffields.ingest import StatsBombClient, attach_freeze_frames
from ffields.observation import ObservationOperator
from ffields.provenance import RunProvenance, seed_everything

PFF_DIR = os.environ.get("PFF_DIR", "data/cache/pff/wc2022")
SAMPLE_RATE = float(os.environ.get("PFF_SAMPLE_RATE", "0.2"))   # 5 fps
MAX_EVENTS = int(os.environ.get("PFF_MAX_EVENTS_PER_MATCH", "400"))
SB_COMP, SB_SEASON = 43, 106   # FIFA World Cup 2022
MIN_INSIDE = 6                 # need at least this many observed players to compare


# ----------------------------------------------------------------------------- utils
def _peak_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3   # macOS bytes, Linux KB


def _parse_ts(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


_ALIAS = {
    "iran": "iran", "ir iran": "iran",
    "south korea": "south korea", "korea republic": "south korea",
    "usa": "united states", "united states": "united states",
}


def _norm(name: str) -> str:
    n = re.sub(r"[^a-z ]", "", (name or "").lower()).strip()
    return _ALIAS.get(n, n)


def control_values(att, deff, centres, cc):
    """Position-only softmin control values over grid centres."""
    max_speed, reaction, tau = cc
    ncells = centres.shape[0]

    def wsum(p):
        if len(p) == 0:
            return np.zeros(ncells)
        return np.exp(-(reaction + cdist(p, centres) / max_speed) / tau).sum(axis=0)

    w_att, w_def = wsum(att), wsum(deff)
    denom = w_att + w_def
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, w_att / denom, 0.5)


def compare(c_full, c_obs, mask_flat):
    a, b = c_full[mask_flat], c_obs[mask_flat]
    if a.size < 8 or min(np.std(a), np.std(b)) < 1e-9:
        return np.nan, float(np.mean((a > 0.5) != (b > 0.5))) if a.size else np.nan
    return float(np.corrcoef(a, b)[0, 1]), float(np.mean((a > 0.5) != (b > 0.5)))


# ----------------------------------------------------------------------------- crosswalk
def build_crosswalk(root):
    import glob
    meta_files = sorted(glob.glob(os.path.join(root, PFF_DIR, "Metadata", "*.json")))
    pff = {}
    for f in meta_files:
        m = json.load(open(f))
        m = m[0] if isinstance(m, list) else m
        gid = str(m.get("id") or os.path.splitext(os.path.basename(f))[0])
        pff[gid] = {
            "date": (m.get("date") or "")[:10],
            "home": _norm(m.get("homeTeam", {}).get("name", "")),
            "away": _norm(m.get("awayTeam", {}).get("name", "")),
        }
    client = StatsBombClient(cache_dir=os.path.join(root, load_config()["paths"]["data_cache"]))
    sb = client.matches(SB_COMP, SB_SEASON)
    sb_index = []
    for s in sb:
        sb_index.append({
            "match_id": s["match_id"], "date": str(s.get("match_date"))[:10],
            "home": _norm(s.get("home_team", {}).get("home_team_name", "")),
            "away": _norm(s.get("away_team", {}).get("away_team_name", "")),
        })
    mapping, unmatched = {}, []
    for gid, p in pff.items():
        cand = [s for s in sb_index if s["date"] == p["date"]
                and {s["home"], s["away"]} == {p["home"], p["away"]}]
        if len(cand) == 1:
            mapping[gid] = cand[0]["match_id"]
        else:  # fall back to team-set match ignoring date
            cand2 = [s for s in sb_index if {s["home"], s["away"]} == {p["home"], p["away"]}]
            if len(cand2) == 1:
                mapping[gid] = cand2[0]["match_id"]
            else:
                unmatched.append((gid, p))
    return mapping, pff, unmatched, client


# ----------------------------------------------------------------------------- PFF frames
def load_pff_compact(root, gid):
    """Return per-period arrays for nearest-frame lookup, plus player grounds.

    Uses kloppy to_df() (verified column names) and downsamples via SAMPLE_RATE to
    bound memory. Coordinates stay in the PFF centre-origin frame here.
    """
    from kloppy import pff
    base = os.path.join(root, PFF_DIR)
    ds = pff.load_tracking(
        meta_data=os.path.join(base, "Metadata", f"{gid}.json"),
        roster_meta_data=os.path.join(base, "Rosters", f"{gid}.json"),
        raw_data=os.path.join(base, "Tracking Data", f"{gid}.jsonl.bz2"),
        coordinates="pff", sample_rate=SAMPLE_RATE, limit=None,
    )
    pid_ground, home_name, away_name = {}, None, None
    for team in ds.metadata.teams:
        g = team.ground.value  # 'home' / 'away'
        if g == "home":
            home_name = _norm(team.name)
        elif g == "away":
            away_name = _norm(team.name)
        for pl in team.players:
            pid_ground[str(pl.player_id)] = g
    df = ds.to_df()
    df["t"] = df["timestamp"].dt.total_seconds()
    player_cols = [c for c in df.columns if re.fullmatch(r"\d+_x", c)]
    pids = [c[:-2] for c in player_cols]
    per_period = {}
    for pid_key, g in df.groupby("period_id"):
        g = g.sort_values("t")
        xs = g[[f"{p}_x" for p in pids]].to_numpy()
        ys = g[[f"{p}_y" for p in pids]].to_numpy()
        per_period[int(pid_key)] = {
            "t": g["t"].to_numpy(),
            "bx": g["ball_x"].to_numpy(), "by": g["ball_y"].to_numpy(),
            "px": xs, "py": ys,
        }
    grounds = np.array([pid_ground.get(p, "?") for p in pids])
    del ds, df
    return per_period, pids, grounds, home_name, away_name


def _nearest(per, period, t):
    d = per.get(period)
    if d is None:
        return None
    i = int(np.argmin(np.abs(d["t"] - t)))
    if abs(d["t"][i] - t) > 2.0:   # no frame within 2 s -> skip
        return None
    return i


# ----------------------------------------------------------------------------- one match
def run_match(root, gid, sb_id, client, pitch, grid, centres, cc, op, verbose=False):
    t0 = time.perf_counter()
    L, W = pitch.metric_length, pitch.metric_width

    # StatsBomb events with 360
    raw = client.events(sb_id)
    ts = attach_freeze_frames(client.three_sixty(sb_id))
    evs = []
    for e in raw:
        uuid = e.get("id")
        if uuid not in ts:
            continue
        loc = e.get("location")
        if not loc:
            continue
        poly_sb = ts[uuid].get("visible_area")
        if not poly_sb:
            continue
        bxy = pitch.sb_to_metric(np.array(loc, dtype=float))
        poly = Polygon(pitch.sb_to_metric(np.asarray(poly_sb, dtype=float).reshape(-1, 2)))
        if not poly.is_valid:
            poly = poly.buffer(0)
        evs.append({
            "period": int(e["period"]), "t": _parse_ts(e["timestamp"]),
            "poss": _norm(e.get("possession_team", {}).get("name", "")),
            "ball": bxy, "poly": poly, "type": e.get("type", {}).get("name", ""),
        })
    if len(evs) > MAX_EVENTS:
        rng = np.random.default_rng(0)
        evs = [evs[i] for i in sorted(rng.choice(len(evs), MAX_EVENTS, replace=False))]

    per, pids, grounds, pff_home, pff_away = load_pff_compact(root, gid)

    # Fit an axis transform (sx,sy) AND a constant time offset PER PERIOD, by
    # minimising the ball residual over event types whose location is the ball.
    # Per-period (not per-match) because providers may handle the half-time
    # end-swap differently, so a single global reflection cannot fit both halves.
    ball_faithful = {"Pass", "Carry", "Shot", "Ball Receipt*"}

    def _period_residual(period, sx, sy, off, evp):
        res = []
        for ev in evp:
            i = _nearest(per, period, ev["t"] + off)
            if i is None:
                continue
            bxv, byv = per[period]["bx"][i], per[period]["by"][i]
            if not (np.isfinite(bxv) and np.isfinite(byv)):
                continue
            res.append(np.hypot(L / 2 + sx * bxv - ev["ball"][0],
                                W / 2 + sy * byv - ev["ball"][1]))
        return res

    def fit_transform(period):
        evp = [e for e in evs if e["period"] == period and e["type"] in ball_faithful]
        evp = evp[:: max(1, len(evp) // 250)]
        best, bres, boff, table = (1, 1), 1e9, 0.0, {}
        for sx in (1, -1):
            for sy in (1, -1):
                bo, bm = 0.0, 1e9
                for off in np.arange(-1.5, 1.51, 0.1):
                    res = _period_residual(period, sx, sy, off, evp)
                    if len(res) >= 8:
                        med = float(np.median(res))
                        if med < bm:
                            bm, bo = med, float(off)
                table[f"{sx:+d},{sy:+d}"] = round(bm, 2)
                if bm < bres:
                    bres, best, boff = bm, (sx, sy), bo
        return {"sx": best[0], "sy": best[1], "off": boff, "res": round(bres, 2), "table": table}

    transforms = {int(p): fit_transform(int(p)) for p in per}

    # possession-ground map (which PFF ground is the possession team)
    def poss_ground(name):
        if name == pff_home:
            return "home"
        if name == pff_away:
            return "away"
        return None

    corrs, flips, residuals, maes, rmses = [], [], [], [], []
    # edge-distance stratification: [<2, 2-5, 5-10, 10-20, >20 m inside polygon]
    edge = {b: [0.0, 0.0, 0] for b in range(5)}  # sum|dC|, sum flips, n cells
    for ev in evs:
        tr = transforms.get(ev["period"])
        if tr is None:
            continue
        sx, sy, toff = tr["sx"], tr["sy"], tr["off"]
        i = _nearest(per, ev["period"], ev["t"] + toff)
        if i is None:
            continue
        d = per[ev["period"]]
        X = L / 2 + sx * d["px"][i]
        Y = W / 2 + sy * d["py"][i]
        ok = np.isfinite(X) & np.isfinite(Y)
        if ok.sum() < 12:
            continue
        pg = poss_ground(ev["poss"])
        if pg is None:
            continue
        att_mask = ok & (grounds == pg)
        def_mask = ok & (grounds != pg) & (grounds != "?")
        att = np.column_stack([X[att_mask], Y[att_mask]])
        deff = np.column_stack([X[def_mask], Y[def_mask]])
        if len(att) < 1 or len(deff) < 1:
            continue
        # observed subset: players inside the 360 visible polygon
        pp = prep(ev["poly"])
        inside = np.array([pp.contains(Point(x, y)) for x, y in zip(X[ok], Y[ok])])
        Xi, Yi = X[ok][inside], Y[ok][inside]
        gi = grounds[ok][inside]
        if inside.sum() < MIN_INSIDE:
            continue
        att_i = np.column_stack([Xi[gi == pg], Yi[gi == pg]])
        def_i = np.column_stack([Xi[(gi != pg) & (gi != "?")], Yi[(gi != pg) & (gi != "?")]])
        if len(att_i) < 1 or len(def_i) < 1:
            continue

        mask_flat = op.visible_mask(ev["poly"]).reshape(-1)
        if mask_flat.sum() < 8:
            continue
        c_full = control_values(att, deff, centres, cc)
        c_obs = control_values(att_i, def_i, centres, cc)
        corr, flip = compare(c_full, c_obs, mask_flat)
        if np.isfinite(corr):
            corrs.append(corr)
            flips.append(flip)
            # multi-metric agreement (MAE, RMSE) over masked cells
            mi = np.where(mask_flat)[0]
            dcell = np.abs(c_full[mi] - c_obs[mi])
            maes.append(float(dcell.mean()))
            rmses.append(float(np.sqrt(np.mean(dcell ** 2))))
            # edge-distance stratification on a subsample of events (cost control)
            if len(corrs) % 5 == 0:
                bnd = ev["poly"].boundary
                fl = (c_full[mi] > 0.5) != (c_obs[mi] > 0.5)
                for j, ci in enumerate(mi):
                    de = bnd.distance(Point(centres[ci, 0], centres[ci, 1]))
                    b = 0 if de < 2 else 1 if de < 5 else 2 if de < 10 else 3 if de < 20 else 4
                    edge[b][0] += float(dcell[j])
                    edge[b][1] += 1.0 if fl[j] else 0.0
                    edge[b][2] += 1
        # ball residual for this event (self-check): ball-faithful events, tracked ball
        bxv, byv = d["bx"][i], d["by"][i]
        if ev["type"] in ball_faithful and np.isfinite(bxv) and np.isfinite(byv):
            bx = L / 2 + sx * bxv
            by = W / 2 + sy * byv
            residuals.append(float(np.hypot(bx - ev["ball"][0], by - ev["ball"][1])))

    dt = time.perf_counter() - t0
    out = {
        "gid": gid, "sb_id": sb_id,
        "transforms": {p: {k: t[k] for k in ("sx", "sy", "off", "res")} for p, t in transforms.items()},
        "n_events_used": len(corrs),
        "median_ball_residual_m": float(np.median(residuals)) if residuals else None,
        "mean_corr": float(np.mean(corrs)) if corrs else None,
        "mean_flip": float(np.mean(flips)) if flips else None,
        "mean_mae": float(np.mean(maes)) if maes else None,
        "mean_rmse": float(np.mean(rmses)) if rmses else None,
        "corrs": corrs, "flips": flips,
        "edge": {str(b): edge[b] for b in edge},
        "seconds": round(dt, 1), "peak_mem_mb": round(_peak_mb(), 0),
    }
    if verbose:
        mbr = out["median_ball_residual_m"]
        trs = " ".join(f"P{p}({t['sx']:+d},{t['sy']:+d},{t['off']:+.1f}s r={t['res']})"
                       for p, t in sorted(transforms.items()))
        mbr_s = f"{mbr:.2f}" if mbr is not None else "n/a"
        print(f"  {gid}->{sb_id} | {trs} | events {len(corrs)} | "
              f"ball residual {mbr_s} m | "
              f"corr {out['mean_corr']:.3f} flip {out['mean_flip']:.3f} | "
              f"{dt:.1f}s peak {out['peak_mem_mb']:.0f} MB", flush=True)
        for p, t in sorted(transforms.items()):
            print(f"      period {p} transform table (best-offset median residual m): {t['table']}", flush=True)
    return out


# ----------------------------------------------------------------------------- main
def _agg(vals, rng, nboot=300):
    a = np.asarray(vals, float); a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    boots = [float(np.mean(rng.choice(a, a.size, replace=True))) for _ in range(nboot)]
    return {"mean": float(a.mean()), "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)), "n": int(a.size)}


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    cfg = load_config()
    seed_everything(cfg["seed"])
    root = str(repo_root())
    figdir = repo_root() / cfg["paths"]["figures"]
    pitch = Pitch(metric_length=cfg["pitch"]["metric_length"], metric_width=cfg["pitch"]["metric_width"])
    grid = Grid(pitch, nx=cfg["grid"]["nx"], ny=cfg["grid"]["ny"])
    centres = grid.flat_centres()
    op = ObservationOperator(grid)
    cc = (cfg["control"]["max_speed_m_s"], cfg["control"]["reaction_time_s"], cfg["control"]["tti_temperature_s"])

    print(f"[{time.strftime('%H:%M:%S')}] building crosswalk...", flush=True)
    mapping, pff, unmatched, client = build_crosswalk(root)
    print(f"matched {len(mapping)}/{len(pff)} PFF games to StatsBomb matches; "
          f"{len(unmatched)} unmatched", flush=True)
    if unmatched:
        for gid, p in unmatched:
            print(f"  UNMATCHED {gid}: {p}", flush=True)

    if mode == "crosswalk":
        print(json.dumps(mapping, indent=2))
        return

    items = sorted(mapping.items())
    if mode == "check":
        items = items[:1]

    rng = np.random.default_rng(cfg["seed"])
    results, t_start = [], time.perf_counter()
    for gid, sb_id in items:
        try:
            results.append(run_match(root, gid, sb_id, client, pitch, grid, centres, cc, op, verbose=True))
        except Exception as exc:
            print(f"  ERROR {gid}->{sb_id}: {exc}", flush=True)

    all_corr = [c for r in results for c in r["corrs"]]
    all_flip = [f for r in results for f in r["flips"]]
    all_mae = [r["mean_mae"] for r in results if r.get("mean_mae") is not None]
    all_rmse = [r["mean_rmse"] for r in results if r.get("mean_rmse") is not None]
    residuals = [r["median_ball_residual_m"] for r in results if r["median_ball_residual_m"] is not None]
    # aggregate edge-distance stratification across matches
    EDGE_LABELS = {0: "<2 m", 1: "2-5 m", 2: "5-10 m", 3: "10-20 m", 4: ">20 m"}
    edge_agg = {}
    for b in range(5):
        sd = sum(r["edge"][str(b)][0] for r in results if "edge" in r)
        sf = sum(r["edge"][str(b)][1] for r in results if "edge" in r)
        n = sum(r["edge"][str(b)][2] for r in results if "edge" in r)
        edge_agg[EDGE_LABELS[b]] = {"mean_abs_diff": (sd / n) if n else None,
                                    "flip_fraction": (sf / n) if n else None, "n_cells": int(n)}
    summary = {
        "analysis": "coverage-bias external validation vs PFF FC tracking",
        "n_matches": len(results), "n_events": len(all_corr),
        "sample_rate": SAMPLE_RATE,
        "median_ball_residual_m_over_matches": float(np.median(residuals)) if residuals else None,
        "spatial_corr_full_vs_observed": _agg(all_corr, rng),
        "flip_fraction_full_vs_observed": _agg(all_flip, rng),
        "mean_abs_diff_full_vs_observed": float(np.mean(all_mae)) if all_mae else None,
        "rmse_full_vs_observed": float(np.mean(all_rmse)) if all_rmse else None,
        "agreement_by_distance_inside_polygon": edge_agg,
        "per_match": [{k: r[k] for k in ("gid", "sb_id", "transforms",
                                          "n_events_used", "median_ball_residual_m",
                                          "mean_corr", "mean_flip", "seconds", "peak_mem_mb")}
                      for r in results],
        "note": ("C_full = control from all PFF players; C_obs = control from PFF players "
                 "inside the StatsBomb 360 visible polygon; compared inside the visible mask. "
                 "External counterpart to the internal proxy of Section 4.5.7. PFF is CV "
                 "broadcast tracking, a richer observation than a 360 frame, not optical truth."),
        "total_seconds": round(time.perf_counter() - t_start, 1),
    }
    print("\n=== SELF-CHECK: median ball residual per match should be small (a few m) ===")
    print(f"overall median ball residual: {summary['median_ball_residual_m_over_matches']}")
    print(f"C_full vs C_obs: corr {summary['spatial_corr_full_vs_observed']['mean']}, "
          f"flip {summary['flip_fraction_full_vs_observed']['mean']} "
          f"(n={summary['spatial_corr_full_vs_observed']['n']} events)")
    print(f"MAE {summary['mean_abs_diff_full_vs_observed']}, RMSE {summary['rmse_full_vs_observed']}")
    print("agreement by distance inside the visible polygon (nearer the edge = more affected):")
    for lbl, v in summary["agreement_by_distance_inside_polygon"].items():
        if v["n_cells"]:
            print(f"  {lbl:>8}: mean|dC| {v['mean_abs_diff']:.3f}, flip {v['flip_fraction']:.3f}, "
                  f"cells {v['n_cells']:,}")

    if mode == "all":
        (figdir / "analysis_pff_paired.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        RunProvenance(seed=cfg["seed"], extra={"stage": "pff_paired"}).write(
            figdir / "analysis_pff_paired_provenance.json")
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].hist([r["mean_corr"] for r in results if r["mean_corr"] is not None], bins=20, color="#2166ac")
        ax[0].set_xlabel("per-match mean spatial correlation (C_full vs C_obs)"); ax[0].set_ylabel("matches")
        ax[0].set_title("A. Control agreement, full vs 360-observed subset", fontsize=11)
        ax[1].hist([r["mean_flip"] for r in results if r["mean_flip"] is not None], bins=20, color="#b2182b")
        ax[1].set_xlabel("per-match mean controlling-team flip fraction"); ax[1].set_ylabel("matches")
        ax[1].set_title("B. Controlling-team disagreement", fontsize=11)
        fig.suptitle("Coverage-bias external validation vs PFF FC tracking | associational", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(figdir / "analysis_pff_paired.png", dpi=300)
        plt.close(fig)
        print(f"\nwrote figures/analysis_pff_paired.json and .png")
    print(json.dumps({k: summary[k] for k in ("n_matches", "n_events",
          "median_ball_residual_m_over_matches", "spatial_corr_full_vs_observed",
          "flip_fraction_full_vs_observed", "total_seconds")}, indent=2))


if __name__ == "__main__":
    main()
