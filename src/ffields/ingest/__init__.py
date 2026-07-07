"""Data ingestion. Currently: StatsBomb open data (on demand, cached, never
redistributed). See ATTRIBUTION.md for licence obligations."""
from __future__ import annotations

from .statsbomb import (
    StatsBombClient,
    attach_freeze_frames,
    events_to_frame,
    infer_attacking_direction,
    normalise_attacking_direction,
)
from .batch import BatchIngestor
from .metrica import MetricaClient, MetricaMatch, parse_tracking

__all__ = [
    "StatsBombClient",
    "events_to_frame",
    "infer_attacking_direction",
    "normalise_attacking_direction",
    "attach_freeze_frames",
    "BatchIngestor",
    "MetricaClient",
    "MetricaMatch",
    "parse_tracking",
]
