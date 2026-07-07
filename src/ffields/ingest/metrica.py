"""Metrica Sports sample tracking-data ingestion.

Why this exists
---------------
Open event/360 data carry **no player velocities** (a 360 freeze frame is a
single instant). The thesis treats that as the object of study, but to *quantify*
the resulting fidelity ceiling we need a source of continuous tracking with
velocities. Metrica Sports publish a small open sample (two matches in the
classic "Friends of Tracking" CSV format) which is, at the time of writing, the
only openly redistributable continuous-tracking source. This module ingests it.

Format (Metrica sample, 25 fps, normalised coordinates)
------------------------------------------------------
Each match has a Home and an Away tracking CSV with three header rows: a team
label row, a jersey-number row, and a column-name row (``Period, Frame,
Time [s], Player.., <y>, .., Ball, <y>``). Positions are in [0, 1] x [0, 1] and
are scaled to the metric pitch (105 x 68 by default). Velocities are derived by
smoothed finite differences (the data ship positions only).

Licence: Metrica sample data are openly published for research; cite Metrica
Sports. As with StatsBomb, files are downloaded on demand and cached under the
gitignored ``data/cache/`` and never redistributed. See ATTRIBUTION.md.
"""
from __future__ import annotations

import io
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..geometry import Pitch

_DEFAULT_BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"


@dataclass
class MetricaClient:
    """Caching reader for the Metrica open sample tracking data."""

    cache_dir: pathlib.Path
    base_url: str = _DEFAULT_BASE
    timeout: float = 60.0

    def __post_init__(self) -> None:
        self.cache_dir = pathlib.Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_text(self, rel_path: str) -> str:
        import requests

        cache_path = self.cache_dir / rel_path
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")
        url = f"{self.base_url}/{rel_path}"
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp.write_text(resp.text, encoding="utf-8")
        tmp.replace(cache_path)
        return resp.text

    def tracking_csv(self, game: int, side: str) -> str:
        side_cap = side.capitalize()  # "Home" / "Away"
        rel = f"Sample_Game_{game}/Sample_Game_{game}_RawTrackingData_{side_cap}_Team.csv"
        return self._get_text(rel)


def _smooth(a: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average along axis 0, NaN-aware, preserving length."""
    s = pd.Series(a)
    return s.rolling(window, center=True, min_periods=1).mean().to_numpy()


@dataclass
class MetricaMatch:
    """Parsed tracking for one match.

    Attributes
    ----------
    time : (N,) seconds
    period : (N,) int
    positions : dict[str, np.ndarray]  player_id -> (N, 2) metric position (NaN if off-frame)
    velocities : dict[str, np.ndarray] player_id -> (N, 2) metric m/s
    team_of : dict[str, str]           player_id -> "home"/"away"
    ball : (N, 2) metric
    """

    time: np.ndarray
    period: np.ndarray
    positions: dict[str, np.ndarray]
    velocities: dict[str, np.ndarray]
    team_of: dict[str, str]
    ball: np.ndarray
    pitch: Pitch
    meta: dict

    @property
    def n_frames(self) -> int:
        return len(self.time)

    def frame_arrays(self, i: int, max_speed_cap: float = 12.0):
        """Return (home_pos, home_vel, away_pos, away_vel, ball) at frame ``i``,
        dropping players that are off-frame (NaN). Velocities are clipped to a
        plausible cap to suppress finite-difference spikes."""
        hp, hv, ap, av = [], [], [], []
        for pid, pos in self.positions.items():
            p = pos[i]
            if not np.all(np.isfinite(p)):
                continue
            v = self.velocities[pid][i]
            sp = np.hypot(*v)
            if sp > max_speed_cap:
                v = v * (max_speed_cap / sp)
            (hp if self.team_of[pid] == "home" else ap).append(p)
            (hv if self.team_of[pid] == "home" else av).append(v)
        return (
            np.asarray(hp).reshape(-1, 2), np.asarray(hv).reshape(-1, 2),
            np.asarray(ap).reshape(-1, 2), np.asarray(av).reshape(-1, 2),
            self.ball[i],
        )


def parse_tracking(home_csv: str, away_csv: str, pitch: Pitch,
                   smooth_window: int = 7) -> MetricaMatch:
    """Parse Home + Away Metrica tracking CSVs into a :class:`MetricaMatch`."""
    positions: dict[str, np.ndarray] = {}
    team_of: dict[str, str] = {}
    time = period = ball = None

    for side, text in (("home", home_csv), ("away", away_csv)):
        df = pd.read_csv(io.StringIO(text), skiprows=2)
        cols = list(df.columns)
        if time is None:
            period = df[cols[0]].to_numpy()
            time = df[cols[2]].to_numpy()
        # columns from index 3: (x, y) pairs; named col = x, following = y
        k = 3
        while k < len(cols) - 1:
            name = str(cols[k])
            x = df[cols[k]].to_numpy(dtype=float) * pitch.metric_length
            y = df[cols[k + 1]].to_numpy(dtype=float) * pitch.metric_width
            xy = np.stack([x, y], axis=-1)
            if name.lower().startswith("ball") or name.lower() == "ball":
                ball = xy
            else:
                positions[name] = xy
                team_of[name] = side
            k += 2

    dt = float(np.nanmedian(np.diff(time)))  # ~0.04 s
    velocities: dict[str, np.ndarray] = {}
    for pid, pos in positions.items():
        xs = _smooth(pos[:, 0], smooth_window)
        ys = _smooth(pos[:, 1], smooth_window)
        vx = np.gradient(xs, time)
        vy = np.gradient(ys, time)
        velocities[pid] = np.stack([vx, vy], axis=-1)

    return MetricaMatch(
        time=time, period=period, positions=positions, velocities=velocities,
        team_of=team_of, ball=ball, pitch=pitch,
        meta={"source": "Metrica Sports sample data", "fps": round(1.0 / dt, 1),
              "dt_s": dt, "smooth_window": smooth_window,
              "n_home": sum(v == "home" for v in team_of.values()),
              "n_away": sum(v == "away" for v in team_of.values())},
    )
