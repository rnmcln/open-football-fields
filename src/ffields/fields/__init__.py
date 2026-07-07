"""Field result container and shared descriptors.

A ``FieldResult`` is a scalar field sampled on a ``Grid``, optionally carrying a
boolean observation mask. Masked cells are *unobserved*, not zero: descriptors
that aggregate over space must either restrict to the mask or be reported as
mask-dependent. This enforces the rule that fields are never silently
extrapolated into unobserved regions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..geometry import Grid


@dataclass
class FieldResult:
    name: str
    values: np.ndarray  # (nx, ny)
    grid: Grid
    mask: np.ndarray | None = None  # (nx, ny) bool; True = observed
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.values.shape != (self.grid.nx, self.grid.ny):
            raise ValueError(
                f"values shape {self.values.shape} != grid ({self.grid.nx},{self.grid.ny})"
            )
        if self.mask is not None and self.mask.shape != self.values.shape:
            raise ValueError("mask shape must match values shape")

    def masked_values(self, fill: float = np.nan) -> np.ndarray:
        if self.mask is None:
            return self.values
        out = self.values.astype(float).copy()
        out[~self.mask] = fill
        return out

    def integral(self, restrict_to_mask: bool = True) -> float:
        """Integrate the field over the pitch (sum * cell_area)."""
        v = self.values
        if restrict_to_mask and self.mask is not None:
            v = np.where(self.mask, v, 0.0)
        return float(np.nansum(v) * self.grid.cell_area)

    def spatial_entropy(self, restrict_to_mask: bool = True) -> float:
        """Shannon entropy (bits) of the normalised non-negative field.

        Dispersion/predictability descriptor. Requires
        non-negative values; raises otherwise. Entropy is computed over the
        (optionally masked) cells, renormalised to a proper distribution.
        """
        v = self.values.astype(float)
        if restrict_to_mask and self.mask is not None:
            v = v[self.mask]
        v = v[np.isfinite(v)]
        if np.any(v < -1e-12):
            raise ValueError("spatial_entropy requires a non-negative field")
        v = np.clip(v, 0.0, None)
        total = v.sum()
        if total <= 0:
            return float("nan")
        p = v / total
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))


# Re-export concrete fields (placed after FieldResult to avoid circular import).
from .control import PositionalPitchControl, KinematicPitchControl  # noqa: E402
from .density import EventDensityField  # noqa: E402
from .threat import ExpectedThreat  # noqa: E402
from .flow import PossessionFlowField  # noqa: E402

__all__ = [
    "FieldResult",
    "PositionalPitchControl",
    "KinematicPitchControl",
    "EventDensityField",
    "ExpectedThreat",
    "PossessionFlowField",
]
