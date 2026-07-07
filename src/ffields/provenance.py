"""Provenance and reproducibility utilities.

Every artefact this pipeline writes should be reconstructible. To that end we
stamp each run with: the global seed, the wall-clock retrieval time (UTC), the
pinned StatsBomb open-data commit SHA, key library versions, and the Python
version. ``seed_everything`` centralises RNG control so that any stochastic
step (KDE resampling, bootstrap, model fitting) is deterministic given the
config seed.
"""
from __future__ import annotations

import datetime as _dt
import json
import pathlib
import platform
import random
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from . import __version__, repo_root


def seed_everything(seed: int) -> None:
    """Seed Python and NumPy RNGs. Extend here if torch/others are added."""
    random.seed(seed)
    np.random.seed(seed)


def _library_versions() -> dict[str, str]:
    mods = ["numpy", "scipy", "pandas", "pyarrow", "shapely", "matplotlib", "sklearn"]
    out: dict[str, str] = {}
    for m in mods:
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "unknown")
        except Exception:  # pragma: no cover - defensive
            out[m] = "not-installed"
    return out


def statsbomb_commit_sha() -> str | None:
    """Return the pinned open-data commit SHA, if recorded."""
    p = repo_root() / "data" / ".statsbomb_commit_sha.txt"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return None


@dataclass
class RunProvenance:
    """A serialisable record of the conditions under which an artefact was made."""

    seed: int
    created_utc: str = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat()
    )
    ffields_version: str = __version__
    python_version: str = field(default_factory=platform.python_version)
    statsbomb_commit: str | None = field(default_factory=statsbomb_commit_sha)
    libraries: dict[str, str] = field(default_factory=_library_versions)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
