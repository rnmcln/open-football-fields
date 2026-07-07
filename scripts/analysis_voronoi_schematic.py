"""Schematic: from a Voronoi dominant region to a softened control field (Chapter 2).

Illustrative figure for the pitch-control lineage discussed in Section 2.2. The left
panel shows the hard dominant-region (Voronoi) partition with velocities set to zero,
in which each location is assigned to the nearest player's team. The right panel
shows the softmin position-only pitch-control field computed from the same players:
the hard boundary becomes a smooth iso-probability contour. The configuration is
synthetic and is used only to illustrate the relationship between the two
representations; no data are involved.

Output: figures/voronoi_schematic.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ffields import load_config, repo_root
from ffields.fields import KinematicPitchControl
from ffields.geometry import Grid, Pitch
from ffields.provenance import seed_everything


def _pitch_lines(ax, L, W):
    ax.plot([0, 0, L, L, 0], [0, W, W, 0, 0], color="black", lw=1.0)
    ax.plot([L / 2, L / 2], [0, W], color="black", lw=0.8)
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(L / 2 + 9.15 * np.cos(th), W / 2 + 9.15 * np.sin(th), color="black", lw=0.8)
    for x0 in (0, L - 16.5):
        ax.plot([x0, x0 + 16.5, x0 + 16.5, x0],
                [W / 2 - 20.16, W / 2 - 20.16, W / 2 + 20.16, W / 2 + 20.16],
                color="black", lw=0.8)
    ax.set_xlim(-2, L + 2); ax.set_ylim(-2, W + 2); ax.set_aspect("equal"); ax.axis("off")


def main() -> None:
    cfg = load_config(); seed_everything(cfg["seed"]); root = repo_root()
    figdir = root / cfg["paths"]["figures"]
    pitch = Pitch(); L, W = pitch.metric_length, pitch.metric_width
    grid = Grid(pitch, nx=cfg["grid"]["nx"], ny=cfg["grid"]["ny"])

    # synthetic, plausible-looking configuration (attacking right)
    home = np.array([[28, 34], [44, 18], [44, 50], [58, 28], [60, 40], [72, 34]], float)
    away = np.array([[80, 34], [64, 20], [64, 48], [52, 30], [52, 40], [40, 34]], float)

    kpc = KinematicPitchControl(grid, max_speed_m_s=cfg["control"]["max_speed_m_s"],
                                reaction_time_s=cfg["control"]["reaction_time_s"],
                                tti_temperature_s=cfg["control"]["tti_temperature_s"])
    C = kpc.estimate(home, np.zeros_like(home), away, np.zeros_like(away))

    # hard dominant region: nearest player's team per cell
    xs, ys = grid.x_centres, grid.y_centres
    XX, YY = np.meshgrid(xs, ys)  # (ny, nx)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    dh = np.min(((pts[:, None, :] - home[None, :, :]) ** 2).sum(-1), axis=1)
    da = np.min(((pts[:, None, :] - away[None, :, :]) ** 2).sum(-1), axis=1)
    region = (dh < da).reshape(XX.shape).astype(float)  # 1 home, 0 away

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].imshow(region, origin="lower", extent=[0, L, 0, W], cmap="RdBu_r", alpha=0.5,
                   vmin=0, vmax=1, aspect="equal")
    _pitch_lines(axes[0], L, W)
    axes[0].scatter(home[:, 0], home[:, 1], c="#b2182b", s=70, edgecolor="white", zorder=3, label="team A")
    axes[0].scatter(away[:, 0], away[:, 1], c="#2166ac", s=70, edgecolor="white", zorder=3, label="team B")
    axes[0].set_title("A. Dominant region (hard Voronoi, zero velocity)", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.8)

    im = axes[1].imshow(C.values.T, origin="lower", extent=[0, L, 0, W], cmap="RdBu_r",
                        vmin=0, vmax=1, aspect="equal")
    cs = axes[1].contour(XX, YY, C.values.T, levels=[0.5], colors="black", linewidths=1.2)
    _pitch_lines(axes[1], L, W)
    axes[1].scatter(home[:, 0], home[:, 1], c="#b2182b", s=70, edgecolor="white", zorder=3)
    axes[1].scatter(away[:, 0], away[:, 1], c="#2166ac", s=70, edgecolor="white", zorder=3)
    axes[1].set_title("B. Softmin pitch control (same players)", fontsize=11)
    cb = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cb.set_label("P(team A controls)", fontsize=9)

    fig.suptitle("From dominant region to softened control field (synthetic, illustrative)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(figdir / "voronoi_schematic.png", dpi=300)
    plt.close(fig)
    print("wrote", figdir / "voronoi_schematic.png")


if __name__ == "__main__":
    main()
