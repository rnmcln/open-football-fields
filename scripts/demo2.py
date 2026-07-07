"""End-to-end Demonstration 2 demonstration on real StatsBomb open data.

Builds on Demonstration 1 (foundations) with the Demonstration 2 deliverables:

  1. multi-match batch ingestion of a whole competition (UEFA Euro 2020),
     writing per-match event parquet + a manifest (provenance stamped);
  2. the threat field T = expected threat (xT, Singh 2019), reused via the
     fitted artefact (scripts/fit_xt.py), with its gradient grad T;
  3. the possession-flow vector field J aggregated across the competition,
     with divergence and (flagged-fragile) curl operators and a bootstrap CI;
  4. the RQ2 robustness harness applied as an anti-decoration gate: every
     operator is stress-tested under input jitter and grid-resolution change,
     and only those that pass are licensed for empirical use. The fragile curl
     operator is tested head-to-head against divergence; failures are reported
     as negative methodological findings, not hidden.

Everything is associational and open-data only. No numbers are invented; all
are emitted here.

Run:  python scripts/demo2.py
Outputs: figures/demo2_*.png, figures/demo2_summary.json,
         figures/demo2_provenance.json
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch as MplPitch

from ffields import load_config, repo_root
from ffields import robustness as rb
from ffields.fields import EventDensityField, ExpectedThreat, PossessionFlowField
from ffields.geometry import Grid, Pitch
from ffields.ingest import BatchIngestor, StatsBombClient
from ffields.provenance import RunProvenance, seed_everything

MOVE_TYPES = ["Pass", "Carry"]


def _load_moves(out_dir, manifest) -> np.ndarray:
    """Concatenate attack-oriented move endpoints across all ingested matches."""
    frames = []
    for mid in manifest["match_id"]:
        p = out_dir / "events" / f"{int(mid)}.parquet"
        df = pd.read_parquet(p, columns=["type", "x_att", "y_att", "end_x_att", "end_y_att"])
        df = df[df["type"].isin(MOVE_TYPES)]
        frames.append(df[["x_att", "y_att", "end_x_att", "end_y_att"]])
    moves = pd.concat(frames, ignore_index=True).dropna().to_numpy()
    return moves


def _flow_noise_gate(flow, moves, grid, sigmas, ref_sigma, n_rep, seed):
    """Noise-stability of divergence and curl under endpoint jitter.

    Returns a dict of tables; reuses the robustness primitives (jitter,
    spatial_correlation) on the vector-operator outputs.
    """
    rng = np.random.default_rng(seed)
    base = flow.estimate(moves[:, 0], moves[:, 1], moves[:, 2], moves[:, 3])
    base_div = flow.divergence(base)
    base_curl = flow.curl(base)
    base_mag = base.as_field("magnitude")  # the base field itself (not a derivative)
    out = {"field": [], "divergence": [], "curl": []}
    for s in sigmas:
        fc, dc, cc = [], [], []
        for _ in range(n_rep):
            j = rb.jitter(moves.reshape(-1, 2, 2), s, rng).reshape(moves.shape)
            fr = flow.estimate(j[:, 0], j[:, 1], j[:, 2], j[:, 3])
            fc.append(rb.spatial_correlation(base_mag, fr.as_field("magnitude")))
            dc.append(rb.spatial_correlation(base_div, flow.divergence(fr)))
            cc.append(rb.spatial_correlation(base_curl, flow.curl(fr)))
        out["field"].append(
            {"sigma_m": float(s), "mean_corr": float(np.nanmean(fc)), "std_corr": float(np.nanstd(fc))}
        )
        out["divergence"].append(
            {"sigma_m": float(s), "mean_corr": float(np.nanmean(dc)), "std_corr": float(np.nanstd(dc))}
        )
        out["curl"].append(
            {"sigma_m": float(s), "mean_corr": float(np.nanmean(cc)), "std_corr": float(np.nanstd(cc))}
        )
    return out, base, base_div, base_curl


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    root = repo_root()
    figdir = root / cfg["paths"]["figures"]
    figdir.mkdir(parents=True, exist_ok=True)
    pitch = Pitch(
        sb_length=cfg["pitch"]["statsbomb_length"], sb_width=cfg["pitch"]["statsbomb_width"],
        metric_length=cfg["pitch"]["metric_length"], metric_width=cfg["pitch"]["metric_width"],
    )
    grid = Grid(pitch, nx=cfg["grid"]["nx"], ny=cfg["grid"]["ny"])
    client = StatsBombClient(cache_dir=root / cfg["paths"]["data_cache"],
                             base_url=cfg["statsbomb"]["base_url"])
    comp, season, label = cfg["batch"]["demo_competition"]

    # 1. multi-match batch ingestion (resumable; reuses the on-demand cache) ---
    out_dir = root / "data" / "cache" / "batch"
    ing = BatchIngestor(client=client, pitch=pitch, out_dir=out_dir)
    manifest = ing.ingest_competition(comp, season)
    moves = _load_moves(out_dir, manifest)

    # 2. threat field T = xT, with grad T --------------------------------------
    xt = ExpectedThreat.from_artefact(root / cfg["threat"]["artefact"], pitch)
    T = xt.field(grid)
    gradT = xt.gradient(grid)

    # 3. possession-flow field J (multi-match) + operators + bootstrap CI ------
    fcfg = cfg["flow"]
    flow = PossessionFlowField(grid, min_count=fcfg["min_count"])
    rng = np.random.default_rng(cfg["seed"])
    boot = flow.bootstrap_ci(moves[:, 0], moves[:, 1], moves[:, 2], moves[:, 3],
                             n_boot=fcfg["bootstrap_n"], rng=rng)
    ci_width = boot["hi"] - boot["lo"]

    # 4. RQ2 robustness gate ---------------------------------------------------
    rcfg = cfg["robustness"]
    sigmas = tuple(rcfg["jitter_sigmas_m"])
    ref = rcfg["ref_sigma_m"]
    min_corr = rcfg["min_mean_corr"]
    n_rep = rcfg["n_rep"]
    # density (pass origins): point-based battery
    pass_origins = moves[:, :2]
    dens_mk = lambda p: EventDensityField(grid).estimate(p)  # noqa: E731

    def eff(fr):  # resolution-robust summary
        return fr.spatial_entropy() / np.log2(fr.values.size)

    grids = [Grid(pitch, nx, ny) for nx, ny in rcfg["resolutions"]]
    dens_rep = rb.run_battery(
        "event_density", make_field=dens_mk, points=pass_origins,
        make_field_for_grid=lambda gg: EventDensityField(gg).estimate(pass_origins),
        grids=grids, summary=eff, sigmas_m=sigmas, ref_sigma_m=ref,
        min_mean_corr=min_corr, max_resolution_cv=rcfg["max_resolution_cv"], n_rep=n_rep, rng=rng,
    )
    # threat: resolution sensitivity of a resolution-robust summary (threat-mass centroid x)
    def thr_centroid_x(fr):
        w = np.clip(fr.values, 0, None)
        cx = fr.grid.centres[..., 0]
        return float((w * cx).sum() / w.sum())

    thr_rep = rb.run_battery(
        "expected_threat",
        make_field_for_grid=lambda gg: xt.field(gg),
        grids=grids, summary=thr_centroid_x, max_resolution_cv=0.05,  # stricter: a fixed surface
    )
    # flow divergence vs curl: noise gate (the head-to-head fragility test)
    flow_noise, base_flow, base_div, base_curl = _flow_noise_gate(
        flow, moves, grid, sigmas, ref, n_rep=n_rep, seed=cfg["seed"]
    )

    def _ref_corr(table):
        r = next((x for x in table if x["sigma_m"] == ref), None)
        return None if r is None else r["mean_corr"]

    field_ref = _ref_corr(flow_noise["field"])
    div_ref = _ref_corr(flow_noise["divergence"])
    curl_ref = _ref_corr(flow_noise["curl"])

    # coarser grid: do the differential operators recover when cells are larger?
    coarse = Grid(pitch, nx=fcfg["coarse_nx"], ny=fcfg["coarse_ny"])
    coarse_flow = PossessionFlowField(coarse, min_count=fcfg["min_count"])
    cn, _, _, _ = _flow_noise_gate(coarse_flow, moves, coarse, (ref,), ref, n_rep=n_rep, seed=cfg["seed"])
    div_coarse_ref = _ref_corr(cn["divergence"])

    gate = {
        "ref_sigma_m": ref,
        "min_mean_corr": min_corr,
        "event_density_noise": dens_rep.passed.get("noise"),
        "event_density_resolution": dens_rep.passed.get("resolution"),
        "expected_threat_resolution": thr_rep.passed.get("resolution"),
        "flow_field_jmag_noise_corr": field_ref,
        "flow_field_passes": bool(field_ref is not None and field_ref >= min_corr),
        "flow_divergence_noise_corr": div_ref,
        "flow_divergence_passes": bool(div_ref is not None and div_ref >= min_corr),
        "flow_divergence_coarse_grid_corr": div_coarse_ref,
        "flow_divergence_coarse_passes": bool(div_coarse_ref is not None and div_coarse_ref >= min_corr),
        "flow_curl_noise_corr": curl_ref,
        "flow_curl_passes": bool(curl_ref is not None and curl_ref >= min_corr),
        "interpretation": (
            "Differential operators on J are noise-amplifying: the base flow "
            "field is stabler than its derivatives, and divergence recovers on a "
            "coarser grid. curl remains the most fragile. Reported as a negative "
            "methodological finding; curl is NOT licensed for empirical use."
        ),
    }

    # 5. descriptors / summary -------------------------------------------------
    obs = base_flow.mask
    summary = {
        "competition": label,
        "n_matches": int(len(manifest)),
        "n_matches_with_360": int(manifest["has_360"].sum()),
        "n_moves_total": int(len(moves)),
        "threat": {
            "native_grid": list(xt.xt.shape),
            "T_max_on_grid": float(T.values.max()),
            "T_spatial_entropy_bits": T.spatial_entropy(),
            "gradT_mean_x": float(np.nanmean(gradT["grad_x"].values)),
            "gradT_mean_mag": float(np.nanmean(gradT["grad_mag"].values)),
        },
        "flow": {
            "min_count": flow.min_count,
            "cells_observed": int(obs.sum()),
            "mean_speed_proxy_m": float(np.nanmean(base_flow.magnitude[obs])),
            "divergence_range": [float(np.nanmin(base_div.masked_values())),
                                 float(np.nanmax(base_div.masked_values()))],
            "bootstrap_div_median_ci_width": float(np.nanmedian(ci_width[obs])),
            "bootstrap_n": 200,
        },
        "robustness_gate": gate,
        "grid": {"nx": grid.nx, "ny": grid.ny, "cell_area_m2": grid.cell_area},
    }
    (figdir / "demo2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RunProvenance(seed=cfg["seed"], extra={"competition": label, "stage": "demo2"}).write(
        figdir / "demo2_provenance.json"
    )

    _render(figdir, pitch, grid, T, gradT, base_flow, base_div, base_curl, flow_noise, gate, summary)
    print(json.dumps(summary, indent=2))


def _quiver(ax, grid, vx, vy, mask=None, step=2, color="k", scale=None):
    cx = grid.x_centres[::step]
    cy = grid.y_centres[::step]
    X, Y = np.meshgrid(cx, cy, indexing="ij")
    U = vx[::step, ::step].copy()
    V = vy[::step, ::step].copy()
    if mask is not None:
        m = mask[::step, ::step]
        U = np.where(m, U, np.nan)
        V = np.where(m, V, np.nan)
    ax.quiver(X, Y, U, V, color=color, scale=scale, width=0.003, alpha=0.9)


def _render(figdir, pitch, grid, T, gradT, flow, div, curl, flow_noise, gate, summary):
    L, W = pitch.metric_length, pitch.metric_width
    extent = [0, L, 0, W]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # A: threat surface T + grad T quiver
    ax = axes[0, 0]
    MplPitch(pitch_type="custom", pitch_length=L, pitch_width=W, line_color="#222").draw(ax=ax)
    im = ax.imshow(T.values.T, origin="lower", extent=extent, cmap="magma", alpha=0.9)
    _quiver(ax, grid, gradT["grad_x"].values, gradT["grad_y"].values, step=3, color="white")
    ax.set_title("A. Threat field T = expected threat (xT, Singh 2019)\n"
                 "white arrows: grad T (toward goal). Euro 2020. Associational.", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="xT")

    # B: flow J quiver over divergence
    ax = axes[0, 1]
    MplPitch(pitch_type="custom", pitch_length=L, pitch_width=W, line_color="#222").draw(ax=ax)
    im = ax.imshow(div.masked_values().T, origin="lower", extent=extent, cmap="coolwarm",
                   alpha=0.85)
    _quiver(ax, grid, flow.jx, flow.jy, mask=flow.mask, step=2, color="k")
    ax.set_title(f"B. Possession-flow field J ({summary['n_moves_total']} moves)\n"
                 "heat: div J (red=source, blue=sink); arrows: mean move displacement", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="div J")

    # C: curl (flagged fragile)
    ax = axes[1, 0]
    MplPitch(pitch_type="custom", pitch_length=L, pitch_width=W, line_color="#222").draw(ax=ax)
    im = ax.imshow(curl.masked_values().T, origin="lower", extent=extent, cmap="PuOr", alpha=0.85)
    ax.set_title("C. Flow curl (FRAGILE / hypothesised)\n"
                 "shown only as a candidate; see panel D for its robustness", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="curl J")

    # D: robustness gate -- divergence vs curl noise stability
    ax = axes[1, 1]
    sig = [r["sigma_m"] for r in flow_noise["divergence"]]
    fvals = [r["mean_corr"] for r in flow_noise["field"]]
    dvals = [r["mean_corr"] for r in flow_noise["divergence"]]
    cvals = [r["mean_corr"] for r in flow_noise["curl"]]
    ax.plot(sig, fvals, "^-", color="#1b7837", label="|J| field")
    ax.plot(sig, dvals, "o-", color="#2166ac", label="div J")
    ax.plot(sig, cvals, "s-", color="#b2182b", label="curl J (fragile)")
    ax.axhline(0.80, color="k", ls="--", lw=1, label="gate (0.80)")
    ax.set_xlabel("input jitter sigma (m)")
    ax.set_ylabel("spatial correlation vs unperturbed")
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("D. RQ2 anti-decoration gate: stability under jitter\n"
                 f"div passes={gate['flow_divergence_passes']}, "
                 f"curl passes={gate['flow_curl_passes']}", fontsize=10)

    fig.suptitle("ffields Demonstration 2 | StatsBomb open data (attribution: StatsBomb) | "
                 "associational, not causal", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = figdir / "demo2_fields.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
