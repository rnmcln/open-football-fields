"""RQ2 robustness harness: the anti-decoration gate.

A field operator earns its place in an empirical claim only if it is *stable*
under perturbations that should not change the conclusion. This module provides
two standard stress tests and a small report object:

1. **Input noise / jitter** -- perturb the input coordinates by Gaussian noise
   of a plausible magnitude (annotation / tracking error, in metres) and
   measure how much the resulting field changes. Stability is summarised by the
   spatial Pearson correlation between the perturbed and the unperturbed field
   over comparable cells, averaged over repetitions.

2. **Grid-resolution sensitivity** -- recompute a scalar summary of the field at
   several grid resolutions and report its coefficient of variation. A
   conclusion that flips with the (arbitrary) grid choice is not defensible.

Thresholds are deliberately explicit and configurable; they are **[Prop]**
defaults, not settled truths. Crucially, the numbers are always returned, so a
*failure is reported as a negative methodological finding* rather than hidden.
Nothing here decides significance; it only gates which operators may proceed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .fields import FieldResult
from .geometry import Grid


# -- comparison metrics -------------------------------------------------------
def spatial_correlation(a: FieldResult, b: FieldResult) -> float:
    """Pearson correlation between two fields over their common finite cells.

    Both fields must share a grid. Masked cells (in either field) are excluded.
    Returns NaN if fewer than 3 comparable cells or zero variance.
    """
    if a.values.shape != b.values.shape:
        raise ValueError("fields must share a grid shape")
    va = a.values.astype(float).ravel()
    vb = b.values.astype(float).ravel()
    good = np.isfinite(va) & np.isfinite(vb)
    for fr in (a, b):
        if fr.mask is not None:
            good &= fr.mask.ravel()
    if good.sum() < 3:
        return float("nan")
    va, vb = va[good], vb[good]
    if np.std(va) == 0 or np.std(vb) == 0:
        return float("nan")
    return float(np.corrcoef(va, vb)[0, 1])


# -- jitter -------------------------------------------------------------------
def jitter(points: np.ndarray, sigma_m: float, rng: np.random.Generator) -> np.ndarray:
    """Add isotropic Gaussian noise of s.d. ``sigma_m`` metres to (..., 2) pts."""
    pts = np.asarray(points, dtype=float)
    return pts + rng.normal(0.0, sigma_m, size=pts.shape)


@dataclass
class RobustnessReport:
    """Outcome of a robustness battery for one operator."""

    operator: str
    noise_table: list[dict[str, float]] = field(default_factory=list)
    resolution_table: list[dict[str, float]] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    passed: dict[str, bool] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True only if every executed sub-test passed."""
        return len(self.passed) > 0 and all(self.passed.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "noise_table": self.noise_table,
            "resolution_table": self.resolution_table,
            "thresholds": self.thresholds,
            "passed": self.passed,
            "ok": self.ok,
            "meta": self.meta,
        }


def noise_stability(
    make_field: Callable[[np.ndarray], FieldResult],
    points: np.ndarray,
    sigmas_m: list[float],
    n_rep: int = 20,
    rng: np.random.Generator | None = None,
) -> list[dict[str, float]]:
    """Field stability under input jitter.

    ``make_field`` maps a perturbed (n, 2) point array to a FieldResult. For
    each sigma we perturb ``points`` ``n_rep`` times and record the mean and s.d.
    of the spatial correlation against the unperturbed field.
    """
    if rng is None:
        rng = np.random.default_rng()
    base = make_field(np.asarray(points, dtype=float))
    table: list[dict[str, float]] = []
    for s in sigmas_m:
        corrs = []
        for _ in range(n_rep):
            pert = make_field(jitter(points, s, rng))
            corrs.append(spatial_correlation(base, pert))
        corrs = np.array(corrs, dtype=float)
        table.append(
            {
                "sigma_m": float(s),
                "mean_corr": float(np.nanmean(corrs)),
                "std_corr": float(np.nanstd(corrs)),
                "n_rep": int(n_rep),
            }
        )
    return table


def resolution_sensitivity(
    make_field_for_grid: Callable[[Grid], FieldResult],
    grids: list[Grid],
    summary: Callable[[FieldResult], float],
) -> list[dict[str, float]]:
    """Scalar-summary stability across grid resolutions."""
    table: list[dict[str, float]] = []
    for g in grids:
        fr = make_field_for_grid(g)
        table.append({"nx": g.nx, "ny": g.ny, "summary": float(summary(fr))})
    return table


def coefficient_of_variation(values: list[float]) -> float:
    v = np.array([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) < 2 or np.mean(v) == 0:
        return float("nan")
    return float(np.std(v) / abs(np.mean(v)))


def run_battery(
    operator: str,
    make_field: Callable[[np.ndarray], FieldResult] | None = None,
    points: np.ndarray | None = None,
    make_field_for_grid: Callable[[Grid], FieldResult] | None = None,
    grids: list[Grid] | None = None,
    summary: Callable[[FieldResult], float] | None = None,
    sigmas_m: tuple[float, ...] = (0.5, 1.0, 2.0),
    ref_sigma_m: float = 1.0,
    min_mean_corr: float = 0.80,
    max_resolution_cv: float = 0.25,
    n_rep: int = 20,
    rng: np.random.Generator | None = None,
) -> RobustnessReport:
    """Run the available sub-tests and decide pass/fail against thresholds.

    The noise sub-test runs if ``make_field`` and ``points`` are given; the
    resolution sub-test runs if ``make_field_for_grid``, ``grids`` and
    ``summary`` are given. Either or both may be supplied.
    """
    rep = RobustnessReport(
        operator=operator,
        thresholds={
            "ref_sigma_m": ref_sigma_m,
            "min_mean_corr": min_mean_corr,
            "max_resolution_cv": max_resolution_cv,
        },
    )

    if make_field is not None and points is not None:
        rep.noise_table = noise_stability(make_field, points, list(sigmas_m), n_rep=n_rep, rng=rng)
        ref = next((r for r in rep.noise_table if r["sigma_m"] == ref_sigma_m), None)
        if ref is not None:
            rep.passed["noise"] = bool(ref["mean_corr"] >= min_mean_corr)
            rep.meta["ref_mean_corr"] = ref["mean_corr"]

    if make_field_for_grid is not None and grids is not None and summary is not None:
        rep.resolution_table = resolution_sensitivity(make_field_for_grid, grids, summary)
        cv = coefficient_of_variation([r["summary"] for r in rep.resolution_table])
        rep.meta["resolution_cv"] = cv
        rep.passed["resolution"] = bool(np.isfinite(cv) and cv <= max_resolution_cv)

    return rep
