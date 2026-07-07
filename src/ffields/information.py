"""Information layer: information gain and compression team signatures (9.11).

This module carries a distinctive descriptor: a
**description-length / rate-distortion team signature**. It also provides the
supporting information-theoretic quantities (per-event self-information, field
KL divergence) that the signature builds on. Everything is associational.

Design
------
* **Per-event self-information.** Given a reference spatial distribution over the
  pitch (a density field normalised to a probability per cell), the information
  content of a move ending in cell c is ``-log2 p_ref(c)`` bits: rare locations
  are surprising. A team's mean self-information under the reference is its
  cross-entropy with the reference.
* **Field KL divergence.** ``D_KL(p || q)`` in bits between two normalised
  fields measures how much a team's spatial distribution departs from a
  reference; i.e. the information gained by modelling the team specifically.
* **Rate-distortion team signature (novel).** A team's set of spatial points
  (e.g. move endpoints or displacement vectors) is vector-quantised at a range
  of codebook sizes k. For each k we record the **rate** (the entropy of the
  codeword distribution, bits per move: the ideal variable-length code length)
  and the **distortion** (RMS quantisation error, metres). The resulting
  rate-distortion curve R(D) is the signature; scalar summaries such as "bits
  per move to localise play to within 5 m" make teams directly comparable. A
  more compressible team (lower rate at a given distortion) is more spatially
  structured/predictable. This is a compression view of team style, not a
  causal or prescriptive claim.

All descriptors are intended to clear the RQ2 robustness gate (jitter the input
points, check the signature summary is stable) before any empirical use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .fields import FieldResult
from .geometry import Grid


# -- information-theoretic primitives -----------------------------------------
def _as_prob_per_cell(densities: FieldResult) -> np.ndarray:
    """Convert a density field (integrates ~1) to probability mass per cell."""
    p = np.clip(densities.values.astype(float), 0.0, None) * densities.grid.cell_area
    s = p.sum()
    return p / s if s > 0 else p


def move_self_information(endpoints: np.ndarray, reference: FieldResult,
                          eps: float = 1e-12) -> np.ndarray:
    """``-log2 p_ref(cell)`` bits for each move endpoint (metric (n, 2))."""
    p_cell = _as_prob_per_cell(reference)
    grid = reference.grid
    ij = grid.cell_of(np.asarray(endpoints, float))
    p = p_cell[ij[..., 0], ij[..., 1]]
    return -np.log2(np.clip(p, eps, None))


def field_kl_divergence(p: FieldResult, q: FieldResult, eps: float = 1e-12) -> float:
    """KL divergence ``D_KL(p || q)`` in bits between two fields on one grid."""
    if p.values.shape != q.values.shape:
        raise ValueError("fields must share a grid")
    pp = _as_prob_per_cell(p).ravel()
    qq = _as_prob_per_cell(q).ravel()
    good = pp > 0
    return float(np.sum(pp[good] * np.log2(pp[good] / np.clip(qq[good], eps, None))))


def shannon_entropy_bits(counts: np.ndarray) -> float:
    """Entropy (bits) of a non-negative count/weight vector."""
    c = np.asarray(counts, float)
    c = c[c > 0]
    if c.sum() <= 0:
        return float("nan")
    p = c / c.sum()
    return float(-np.sum(p * np.log2(p)))


# -- rate-distortion team signature -------------------------------------------
@dataclass
class RateDistortionSignature:
    """Compression-based team signature via vector quantisation.

    Parameters
    ----------
    ks : codebook sizes to sweep.
    seed : RNG seed for k-means (reproducibility).
    """

    ks: tuple[int, ...] = (2, 4, 8, 16, 32, 64)
    seed: int = 20260618
    n_init: int = 10
    curve: list[dict[str, float]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def fit(self, points: np.ndarray) -> "RateDistortionSignature":
        """Compute the rate-distortion curve for (n, 2) metric points."""
        from sklearn.cluster import KMeans

        pts = np.asarray(points, float)
        pts = pts[np.isfinite(pts).all(axis=1)]
        n = len(pts)
        self.curve = []
        for k in self.ks:
            if k >= n:
                continue
            km = KMeans(n_clusters=k, random_state=self.seed, n_init=self.n_init)
            labels = km.fit_predict(pts)
            d2 = ((pts - km.cluster_centers_[labels]) ** 2).sum(axis=1)
            distortion_rmse = float(np.sqrt(d2.mean()))
            counts = np.bincount(labels, minlength=k)
            rate_bits = shannon_entropy_bits(counts)  # ideal variable-length code
            self.curve.append(
                {"k": int(k), "rate_bits": rate_bits, "distortion_rmse_m": distortion_rmse}
            )
        self.meta = {"n_points": int(n), "seed": self.seed,
                     "note": "associational compression signature; not causal"}
        return self

    def rate_at_distortion(self, target_m: float) -> float:
        """Interpolate the rate (bits/move) needed to reach distortion target_m.

        Uses the (monotone) distortion-decreasing curve; returns NaN if the
        target lies outside the swept range.
        """
        if not self.curve:
            return float("nan")
        d = np.array([c["distortion_rmse_m"] for c in self.curve])
        r = np.array([c["rate_bits"] for c in self.curve])
        order = np.argsort(d)  # ascending distortion
        d, r = d[order], r[order]
        if target_m <= d.min():
            return float(r[d.argmin()])
        if target_m >= d.max():
            return float(r[d.argmax()])
        return float(np.interp(target_m, d, r))

    def signature(self, targets_m: tuple[float, ...] = (5.0, 10.0)) -> dict[str, float]:
        """Scalar team signature: rate (bits/move) at each target distortion."""
        return {f"rate_bits_at_{t:g}m": self.rate_at_distortion(t) for t in targets_m}
