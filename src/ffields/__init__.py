"""ffields: field-theoretic and information-theoretic football analytics.

Open-data, partial-observation framing. See the project README and the
accompanying doctoral thesis for the conceptual basis. Every numerical
default lives in ``config/default.yaml``; nothing that affects results is
hard-coded in module bodies.
"""
from __future__ import annotations

import pathlib
from typing import Any

__version__ = "0.1.0"

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def repo_root() -> pathlib.Path:
    """Return the repository root (two levels above this file)."""
    return _REPO_ROOT


def load_config(path: str | pathlib.Path | None = None) -> dict[str, Any]:
    """Load a YAML config. Defaults to ``config/default.yaml`` at repo root."""
    import yaml

    if path is None:
        path = _REPO_ROOT / "config" / "default.yaml"
    path = pathlib.Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
