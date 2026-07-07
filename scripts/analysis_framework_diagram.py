"""Conceptual schematic of the four-level framework (thesis Chapter 1).

Draws the chain latent state -> observation (under the operator O) -> estimate ->
claim, with the three gaps each study measures annotated beneath: the observation
ceiling (state to observation), the fidelity ceiling (a rich estimate versus a poor
one), and the validation battery (estimate to claim). No data are involved.

Output: figures/framework_schematic.pdf (vector) and .png (high-resolution raster).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from ffields import load_config, repo_root


LEVELS = [
    ("Latent state", r"$s$", "all players and\nball, with velocities", "#d6e3f0"),
    ("Observation", r"$o = O(s)$", "events and a\ncamera polygon", "#cfe8d8"),
    ("Estimate", r"$\hat f$", "a field, masked\nto what is seen", "#fbe6c8"),
    ("Claim", "", "control, directness,\na stable trait", "#f3d4d4"),
]
GAPS = [
    ("Observation ceiling", "Chapter 4"),
    ("Fidelity ceiling", "Chapter 5"),
    ("Validation battery", "Chapter 6"),
]


def main() -> None:
    cfg = load_config(); root = repo_root()
    figdir = root / cfg["paths"]["figures"]

    # Drawn close to the final text-column size so that, scaled to the page width,
    # the vector text renders at its true point size with no blur and no clipping.
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.0); ax.axis("off")

    w, h, y = 1.62, 1.35, 2.05
    gap = (7.6 - 0.2 - 4 * w) / 3.0          # even spacing, 0.1 margin each side
    xs = [0.1 + i * (w + gap) for i in range(4)]
    centres = [x + w / 2 for x in xs]

    for (title, sym, desc, colour), x in zip(LEVELS, xs):
        ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=colour, edgecolor="#333333",
                                    boxstyle="round,pad=0.06", linewidth=1.2))
        cx = x + w / 2
        ax.text(cx, y + h - 0.30, title, ha="center", va="center", fontsize=11, fontweight="bold")
        if sym:
            ax.text(cx, y + h - 0.62, sym, ha="center", va="center", fontsize=10.5)
        ax.text(cx, y + 0.34, desc, ha="center", va="center", fontsize=8.0, linespacing=1.3)

    # plain arrows between boxes (no labels: the gap annotations below name each step)
    for i in range(3):
        ax.add_patch(FancyArrowPatch((xs[i] + w, y + h / 2), (xs[i + 1], y + h / 2),
                                     arrowstyle="-|>", mutation_scale=14, lw=1.6, color="#333333"))

    # the three gaps, annotated beneath each transition
    for i, (name, chap) in enumerate(GAPS):
        gx = (centres[i] + centres[i + 1]) / 2
        ax.annotate("", xy=(centres[i + 1] - 0.1, y - 0.22), xytext=(centres[i] + 0.1, y - 0.22),
                    arrowprops=dict(arrowstyle="<->", color="#7f7f7f", lw=1.0))
        ax.text(gx, y - 0.62, name, ha="center", va="center", fontsize=9, color="#1a1a1a",
                fontweight="bold")
        ax.text(gx, y - 0.92, chap, ha="center", va="center", fontsize=8, color="#555555")

    ax.text(3.8, 3.78, "The four levels, and the gap each study measures",
            ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(3.8, 0.30,
            "A claim is supported only by what the estimate, and behind it the observation, can bear.",
            ha="center", va="center", fontsize=8, style="italic", color="#333333")

    fig.savefig(figdir / "framework_schematic.pdf", bbox_inches="tight")        # vector (used in the PDF)
    fig.savefig(figdir / "framework_schematic.png", dpi=400, bbox_inches="tight")  # raster (used in the Word build)
    plt.close(fig)
    print("wrote", figdir / "framework_schematic.pdf", "and .png")


if __name__ == "__main__":
    main()
