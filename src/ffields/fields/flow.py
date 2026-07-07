"""Possession-flow vector field J and its operators.

The possession-flow field aggregates ball movement into a vector field on the
pitch. For every completed move (pass or carry) originating in a grid cell we
record its displacement ``(dx, dy) = (x_end - x_start, y_end - y_start)`` in
attack-oriented metric coordinates; ``J(cell)`` is the mean displacement of
moves originating there. J is therefore an *associational* description of where
possession tends to travel from each region, not a physical velocity field and
not a causal statement.

Field operators, all evaluated on the discrete grid:

* **divergence** ``div J = dJx/dx + dJy/dy`` -- a scalar field; positive cells
  are net *sources* of forward movement, negative cells net *sinks*.
* **flux** ``integral J . n`` across the boundary of a region -- net flow out of
  the region (discretised over boundary cell faces).
* **curl** (2-D scalar) ``dJy/dx - dJx/dy`` -- rotational structure. This is
  flagged in the thesis as a **fragile, hypothesised** quantity: it is
  implemented here but must clear the RQ2 robustness gate before it is used in
  any empirical claim. Until then it is reported only as a candidate.

Cells with fewer than ``min_count`` originating moves are treated as
unobserved (mask = False); operators are not reported there, because finite-
sample displacement means are noisy in sparse regions.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Any, Callable

import numpy as np

from ..geometry import Grid
from . import FieldResult


@dataclass
class FlowResult:
    """A vector field on the grid plus the originating-move counts."""

    jx: np.ndarray  # (nx, ny) mean x-displacement
    jy: np.ndarray  # (nx, ny) mean y-displacement
    count: np.ndarray  # (nx, ny) number of moves per origin cell
    grid: Grid
    mask: np.ndarray  # (nx, ny) bool: count >= min_count
    meta: dict[str, Any] = _field(default_factory=dict)

    @property
    def magnitude(self) -> np.ndarray:
        return np.hypot(self.jx, self.jy)

    def as_field(self, component: str = "magnitude") -> FieldResult:
        """Expose one component as a scalar :class:`FieldResult`."""
        vals = {"jx": self.jx, "jy": self.jy, "magnitude": self.magnitude}[component]
        return FieldResult(f"flow_{component}", vals, self.grid, mask=self.mask, meta=self.meta)


@dataclass
class PossessionFlowField:
    """Estimator for the possession-flow vector field J.

    Parameters
    ----------
    grid : the discretisation.
    min_count : minimum originating moves for a cell to be reported (mask).
    """

    grid: Grid
    min_count: int = 5

    def _bin_sums(self, x0, y0, x1, y1):
        """Accumulate per-cell displacement sums and counts (vectorised)."""
        g = self.grid
        x0 = np.asarray(x0, float)
        y0 = np.asarray(y0, float)
        x1 = np.asarray(x1, float)
        y1 = np.asarray(y1, float)
        ok = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(x1) & np.isfinite(y1)
        x0, y0, x1, y1 = x0[ok], y0[ok], x1[ok], y1[ok]
        ij = g.cell_of(np.stack([x0, y0], axis=-1))
        i, j = ij[..., 0], ij[..., 1]
        flat = i * g.ny + j
        n = g.nx * g.ny
        sx = np.bincount(flat, weights=(x1 - x0), minlength=n).reshape(g.nx, g.ny)
        sy = np.bincount(flat, weights=(y1 - y0), minlength=n).reshape(g.nx, g.ny)
        cnt = np.bincount(flat, minlength=n).reshape(g.nx, g.ny)
        return sx, sy, cnt

    def estimate(self, x0, y0, x1, y1) -> FlowResult:
        """Mean-displacement vector field from attack-oriented move endpoints."""
        sx, sy, cnt = self._bin_sums(x0, y0, x1, y1)
        with np.errstate(invalid="ignore", divide="ignore"):
            jx = np.where(cnt > 0, sx / cnt, 0.0)
            jy = np.where(cnt > 0, sy / cnt, 0.0)
        mask = cnt >= self.min_count
        return FlowResult(
            jx=jx,
            jy=jy,
            count=cnt,
            grid=self.grid,
            mask=mask,
            meta={
                "estimator": "mean per-cell move displacement",
                "min_count": self.min_count,
                "n_moves": int(cnt.sum()),
                "caveat": "associational; mean displacement of recorded moves, not velocity",
            },
        )

    # -- vector operators -----------------------------------------------------
    def divergence(self, flow: FlowResult) -> FieldResult:
        g = self.grid
        djx_dx = np.gradient(flow.jx, g.dx, axis=0, edge_order=2)
        djy_dy = np.gradient(flow.jy, g.dy, axis=1, edge_order=2)
        div = djx_dx + djy_dy
        return FieldResult("flow_divergence", div, g, mask=flow.mask,
                           meta={"operator": "div J = dJx/dx + dJy/dy", "units": "1/s-free (per move)"})

    def curl(self, flow: FlowResult) -> FieldResult:
        """2-D scalar curl. FRAGILE: do not use empirically until it clears the
        RQ2 robustness gate (see ``ffields.robustness``)."""
        g = self.grid
        djy_dx = np.gradient(flow.jy, g.dx, axis=0, edge_order=2)
        djx_dy = np.gradient(flow.jx, g.dy, axis=1, edge_order=2)
        cur = djy_dx - djx_dy
        return FieldResult("flow_curl", cur, g, mask=flow.mask,
                           meta={"operator": "curl J = dJy/dx - dJx/dy",
                                 "status": "FRAGILE/hypothesised; gate on RQ2 before use"})

    def flux(self, flow: FlowResult, region_mask: np.ndarray) -> float:
        """Net flux of J out of a region defined by a boolean (nx, ny) mask.

        Discretised as the sum over boundary cell faces of ``J . n`` times the
        face length. A face between an in-region cell and an out-of-region (or
        off-grid) neighbour contributes the in-region cell's outward component.
        """
        g = self.grid
        rm = np.asarray(region_mask, dtype=bool)
        if rm.shape != (g.nx, g.ny):
            raise ValueError("region_mask shape must match the grid")
        total = 0.0
        # x-faces (normal +/- x), face length = dy
        for i in range(g.nx):
            for j in range(g.ny):
                if not rm[i, j]:
                    continue
                # +x neighbour
                if i + 1 >= g.nx or not rm[i + 1, j]:
                    total += flow.jx[i, j] * g.dy
                # -x neighbour
                if i - 1 < 0 or not rm[i - 1, j]:
                    total += -flow.jx[i, j] * g.dy
                # +y neighbour
                if j + 1 >= g.ny or not rm[i, j + 1]:
                    total += flow.jy[i, j] * g.dx
                # -y neighbour
                if j - 1 < 0 or not rm[i, j - 1]:
                    total += -flow.jy[i, j] * g.dx
        return float(total)

    # -- uncertainty ----------------------------------------------------------
    def bootstrap_ci(
        self,
        x0,
        y0,
        x1,
        y1,
        statistic: Callable[[FlowResult], np.ndarray] | None = None,
        n_boot: int = 200,
        alpha: float = 0.05,
        rng: np.random.Generator | None = None,
    ) -> dict[str, np.ndarray]:
        """Nonparametric bootstrap over moves.

        Resamples moves with replacement ``n_boot`` times and recomputes a
        ``statistic`` of the resulting :class:`FlowResult` (default: the
        divergence field). Returns the point estimate and percentile CI bounds.
        """
        x0 = np.asarray(x0, float); y0 = np.asarray(y0, float)
        x1 = np.asarray(x1, float); y1 = np.asarray(y1, float)
        ok = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(x1) & np.isfinite(y1)
        x0, y0, x1, y1 = x0[ok], y0[ok], x1[ok], y1[ok]
        n = len(x0)
        if rng is None:
            rng = np.random.default_rng()
        if statistic is None:
            statistic = lambda fr: self.divergence(fr).values  # noqa: E731

        point = statistic(self.estimate(x0, y0, x1, y1))
        boots = np.empty((n_boot, *point.shape), dtype=float)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            fr = self.estimate(x0[idx], y0[idx], x1[idx], y1[idx])
            boots[b] = statistic(fr)
        lo = np.nanpercentile(boots, 100 * alpha / 2, axis=0)
        hi = np.nanpercentile(boots, 100 * (1 - alpha / 2), axis=0)
        return {"point": point, "lo": lo, "hi": hi, "n_moves": n, "n_boot": n_boot}
