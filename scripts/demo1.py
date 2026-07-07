"""End-to-end Demonstration 1 demonstration on real StatsBomb open data.

Pipeline:
  1. seed + provenance
  2. ingest one 360-enabled match (events + freeze frames), cache locally
  3. normalise attacking direction (each team attacks +x)
  4. pick the shot whose freeze frame has the most visible players
  5. compute the positional pitch-control field (masked to visible_area)
  6. compute an event-density field (all pass origins, both teams)
  7. compute spatial-entropy descriptors
  8. render a two-panel figure
  9. write provenance + a machine-readable summary

Run:  python scripts/demo1.py
Outputs: figures/demo1_*.png, figures/demo1_provenance.json,
         figures/demo1_summary.json
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch as MplPitch

from ffields import load_config, repo_root
from ffields.fields import EventDensityField, PositionalPitchControl
from ffields.geometry import Grid, Pitch
from ffields.ingest import (
    StatsBombClient,
    attach_freeze_frames,
    events_to_frame,
    normalise_attacking_direction,
)
from ffields.observation import FreezeFrame
from ffields.provenance import RunProvenance, seed_everything

MATCH_ID = 3788758  # Euro 2020, Ukraine vs North Macedonia (360 available)


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    root = repo_root()
    figdir = root / cfg["paths"]["figures"]
    figdir.mkdir(parents=True, exist_ok=True)

    pitch = Pitch(
        sb_length=cfg["pitch"]["statsbomb_length"],
        sb_width=cfg["pitch"]["statsbomb_width"],
        metric_length=cfg["pitch"]["metric_length"],
        metric_width=cfg["pitch"]["metric_width"],
    )
    grid = Grid(pitch, nx=cfg["grid"]["nx"], ny=cfg["grid"]["ny"])

    client = StatsBombClient(
        cache_dir=root / cfg["paths"]["data_cache"], base_url=cfg["statsbomb"]["base_url"]
    )

    # ingest + normalise
    df = normalise_attacking_direction(events_to_frame(client.events(MATCH_ID), pitch), pitch)
    ts = attach_freeze_frames(client.three_sixty(MATCH_ID))

    # choose the shot freeze frame with the most visible players
    shots = df[(df["type"] == "Shot") & df["id"].isin(ts.keys())]
    best_id, best_n, best_row = None, -1, None
    for _, row in shots.iterrows():
        n = len(ts[row["id"]]["freeze_frame"])
        if n > best_n:
            best_id, best_n, best_row = row["id"], n, row
    if best_id is None:  # fallback to passes if no shot has 360
        passes = df[(df["type"] == "Pass") & df["id"].isin(ts.keys())]
        best_row = passes.iloc[0]
        best_id = best_row["id"]

    frame = FreezeFrame.from_raw(ts[best_id], int(best_row["att_sign"]), pitch)

    # control field
    pc = PositionalPitchControl(
        grid,
        max_speed_m_s=cfg["control"]["max_speed_m_s"],
        reaction_time_s=cfg["control"]["reaction_time_s"],
        tti_temperature_s=cfg["control"]["tti_temperature_s"],
        keeper_included=cfg["control"]["keeper_included"],
    )
    control = pc.estimate(frame)

    # density field: all pass origins (oriented)
    pass_origins = df[(df["type"] == "Pass")][["x_att", "y_att"]].dropna().to_numpy()
    density = EventDensityField(grid, bandwidth=cfg["density"]["bandwidth"]).estimate(
        pass_origins, name="pass_origin_density"
    )

    # descriptors
    summary = {
        "match_id": MATCH_ID,
        "event_id": best_id,
        "event_team": best_row["team"],
        "event_type": best_row["type"],
        "freeze_frame_n_players": int(best_n),
        "control_meta": control.meta,
        "control_visible_fraction": float(control.mask.mean()),
        "control_mean_in_visible": float(np.nanmean(control.masked_values())),
        "control_spatial_entropy_bits": control.spatial_entropy(),
        "density_n_pass_origins": density.meta["n_points"],
        "density_integral": density.integral(),
        "density_spatial_entropy_bits": density.spatial_entropy(),
        "grid": {"nx": grid.nx, "ny": grid.ny, "cell_area_m2": grid.cell_area},
    }
    (figdir / "demo1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    prov = RunProvenance(seed=cfg["seed"], extra={"match_id": MATCH_ID, "event_id": best_id})
    prov.write(figdir / "demo1_provenance.json")

    _render(figdir, pitch, grid, frame, control, density, best_row)

    print(json.dumps(summary, indent=2))


def _render(figdir, pitch, grid, frame, control, density, row) -> None:
    L, W = pitch.metric_length, pitch.metric_width
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))

    # panel A: pitch control (masked) + players + visible area
    mp = MplPitch(pitch_type="custom", pitch_length=L, pitch_width=W, line_color="#222")
    mp.draw(ax=axes[0])
    masked = control.masked_values()
    extent = [0, L, 0, W]
    im = axes[0].imshow(
        masked.T, origin="lower", extent=extent, cmap="coolwarm", vmin=0, vmax=1, alpha=0.75
    )
    # visible-area outline
    vx, vy = frame.visible_polygon.exterior.xy
    axes[0].plot(vx, vy, color="k", lw=1.2, ls="--", alpha=0.7)
    if len(frame.attackers):
        axes[0].scatter(frame.attackers[:, 0], frame.attackers[:, 1], c="#b2182b",
                        edgecolor="white", s=80, zorder=5, label="possession team")
    if len(frame.defenders):
        axes[0].scatter(frame.defenders[:, 0], frame.defenders[:, 1], c="#2166ac",
                        edgecolor="white", s=80, zorder=5, label="opponents")
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    axes[0].set_title(
        f"Positional pitch control (masked to visible area)\n"
        f"{row['team']} {row['type'].lower()}, {frame.n_visible} visible players  "
        f"[red = possession-team control, blue = opponents]",
        fontsize=10,
    )
    fig.colorbar(im, ax=axes[0], fraction=0.03, pad=0.02, label="possession-team control")

    # panel B: pass-origin density
    mp2 = MplPitch(pitch_type="custom", pitch_length=L, pitch_width=W, line_color="#222")
    mp2.draw(ax=axes[1])
    im2 = axes[1].imshow(
        density.values.T, origin="lower", extent=extent, cmap="viridis", alpha=0.9
    )
    axes[1].set_title(
        f"Pass-origin density (both teams, oriented +x)\n"
        f"n = {density.meta['n_points']} passes; entropy = "
        f"{density.spatial_entropy():.2f} bits",
        fontsize=10,
    )
    fig.colorbar(im2, ax=axes[1], fraction=0.03, pad=0.02, label="density (1/m^2)")

    fig.suptitle(
        "ffields Demonstration 1 demonstration | StatsBomb open data (attribution: StatsBomb) | "
        "associational, not causal",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = figdir / "demo1_fields.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
