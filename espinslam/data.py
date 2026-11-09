"""Leakage-safe event sequence and manifest loading."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class EventSequence:
    """One validated raw-event sequence and evaluation-only label."""

    sequence_id: str
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    p: np.ndarray
    omega_rad_s: float
    width: int
    height: int
    metadata: dict[str, Any]
    event_path: Path

    @property
    def event_count(self) -> int:
        return int(self.t.size)

    @property
    def duration_seconds(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.t.size > 1 else 0.0

    def before(self, duration_seconds: float) -> "EventSequence":
        """Return a causal prefix without exposing the evaluation label to inference."""

        if not np.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive and finite")
        if not self.t.size:
            return self
        stop = int(np.searchsorted(self.t, self.t[0] + duration_seconds, side="right"))
        return EventSequence(
            self.sequence_id,
            self.x[:stop], self.y[:stop], self.t[:stop], self.p[:stop],
            self.omega_rad_s, self.width, self.height, dict(self.metadata),
            self.event_path,
        )


def read_manifest(path: str | Path, data_root: str | Path) -> list[dict[str, str]]:
    """Read explicit split membership and constrain referenced paths to the root."""

    path, root = Path(path), Path(data_root).resolve()
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
    except (OSError, csv.Error) as error:
        raise ValueError(f"invalid manifest: {path}") from error
    required = {"sequence_id", "event_path", "split"}
    if not reader.fieldnames or required - set(reader.fieldnames):
        raise ValueError(f"manifest is missing columns: {sorted(required - set(reader.fieldnames or []))}")
    seen: set[str] = set()
    for row in rows:
        for name in required:
            row[name] = (row.get(name) or "").strip()
            if not row[name]:
                raise ValueError(f"manifest {name} must be non-empty")
        if row["sequence_id"] in seen:
            raise ValueError(f"duplicate sequence_id: {row['sequence_id']}")
        seen.add(row["sequence_id"])
        event_path = (root / row["event_path"]).resolve()
        if not event_path.is_relative_to(root):
            raise ValueError("manifest event_path resolves outside data root")
    return rows


def _numeric(archive, name: str, *, ndim: int) -> np.ndarray:
    if name not in archive.files:
        raise ValueError(f"event NPZ is missing {name!r}")
    value = archive[name]
    if value.ndim != ndim or value.dtype.kind not in "iuf" or not np.isfinite(value).all():
        raise ValueError(f"event array {name!r} must be finite numeric with ndim={ndim}")
    return value


def load_event_sequence(
    row: dict[str, str], data_root: str | Path, sensor_size: Iterable[int],
    *, max_duration_seconds: float | None = None,
) -> EventSequence:
    """Load source events; labels remain a field used only by the evaluator."""

    root = Path(data_root).resolve()
    event_path = (root / row["event_path"]).resolve()
    if not event_path.is_relative_to(root) or not event_path.is_file():
        raise ValueError(f"missing event NPZ: {event_path}")
    width, height = (int(value) for value in sensor_size)
    try:
        with np.load(event_path, allow_pickle=False) as archive:
            x = _numeric(archive, "x", ndim=1)
            y = _numeric(archive, "y", ndim=1)
            t = _numeric(archive, "t", ndim=1).astype(np.float64, copy=False)
            p = _numeric(archive, "p", ndim=1)
            omega = float(_numeric(archive, "omega_rad_s", ndim=0))
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"failed to load {event_path}: {error}") from error
    if len({x.size, y.size, t.size, p.size}) != 1:
        raise ValueError("event arrays must have equal lengths")
    x = x.astype(np.int32, copy=False)
    y = y.astype(np.int32, copy=False)
    if (np.any(x < 0) or np.any(x >= width)
            or np.any(y < 0) or np.any(y >= height)):
        raise ValueError("event coordinates are outside the sensor")
    if t.size and (np.any(t < 0) or np.any(t[1:] < t[:-1])):
        raise ValueError("event timestamps must be sorted and non-negative")
    if not np.isin(p, (-1, 0, 1)).all():
        raise ValueError("event polarity must use {-1,1} or {0,1}")
    if np.any(p == -1) and np.any(p == 0):
        raise ValueError("event polarity mixes two OFF encodings")
    if omega < 0:
        raise ValueError("omega_rad_s must be non-negative")
    p = (p > 0).astype(np.uint8)
    if max_duration_seconds is not None and t.size:
        stop = int(np.searchsorted(t, t[0] + max_duration_seconds, side="right"))
        x, y, t, p = x[:stop], y[:stop], t[:stop], p[:stop]
    metadata = {
        name: value.strip() for name, value in row.items()
        if isinstance(value, str) and value.strip()
    }
    scene_dir = event_path.parent
    for candidate in (scene_dir / "scene_config.json", scene_dir / "meta" / "scene_metadata.json"):
        if candidate.is_file():
            try:
                metadata[candidate.stem] = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata[candidate.stem] = {"invalid": str(candidate)}
    return EventSequence(
        row["sequence_id"], x, y, t, p, omega, width, height,
        metadata, event_path,
    )

