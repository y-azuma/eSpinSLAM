"""Event sequences, manifests, and event-file readers.

Two entry points exist:

* :func:`read_manifest` / :func:`load_event_sequence` for the debris NPZ
  dataset (explicit ``split`` column, ``omega_rad_s`` evaluation label);
* :func:`read_events` / :func:`load_events_as_sequence` for generic event
  files such as TOPSPIN: ``.npz``, ``.npy``, ``.h5/.hdf5``, ``.csv/.txt``,
  Prophesee ``.dat`` and Prophesee ``.raw`` (EVT 2.0 / EVT 3.0). The Prophesee
  decoders follow the public OpenEB format documentation.

Timestamps are converted to seconds (float64) and polarities to {0, 1}.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable

from numba import njit
import numpy as np


# Audit-only scene conditions that may be copied into evaluation artifacts.
# These fields are never consumed by the front-end, frequency estimator, or
# reconstruction backend. Labels, event paths, bounding boxes, and CAD paths
# are deliberately absent from this allowlist.
EVALUATION_CONDITION_FIELDS = (
    "object_id",
    "physical_object_id",
    "scene_id",
    "source_group_id",
    "periodic_order",
    "perspective",
    "speed_class",
    "speed",
    "frequency_source",
    "provenance.label_status",
    "provenance.generator.sample_index",
    "provenance.target.debris_type",
    "provenance.target.periodic_order",
    "provenance.target.initial_spin_phase_deg",
    "provenance.target.initial_euler_deg.x",
    "provenance.target.initial_euler_deg.y",
    "provenance.target.initial_euler_deg.z",
    "provenance.target.body_dims_m.x",
    "provenance.target.body_dims_m.y",
    "provenance.target.body_dims_m.z",
    "provenance.camera.initial_range_m",
    "provenance.camera.closing_rate_mps",
    "provenance.camera.orbit_rate_deg_s",
    "provenance.camera.jitter_std_deg",
    "provenance.lighting.sun_azimuth_deg",
    "provenance.lighting.sun_elevation_deg",
    "provenance.lighting.sun_energy",
    "provenance.lighting.earthshine_energy",
    "provenance.lighting.world_strength",
    "provenance.environment.earth_visible",
    "provenance.environment.star_count",
    "provenance.rotation.axis_body.x",
    "provenance.rotation.axis_body.y",
    "provenance.rotation.axis_body.z",
    "provenance.time.duration_s",
)


def evaluation_condition_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return only post-inference diagnostic conditions from ``metadata``."""

    result: dict[str, Any] = {}
    for name in EVALUATION_CONDITION_FIELDS:
        if name not in metadata:
            continue
        value = metadata[name]
        if value is None or (isinstance(value, str) and not value):
            continue
        result[name] = value
    return result


@dataclass(frozen=True)
class EventSequence:
    """One validated raw-event sequence and an evaluation-only label."""

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


# ---------------------------------------------------------------------------
# Debris dataset (explicit split manifest)
# ---------------------------------------------------------------------------

def read_manifest(path: str | Path, data_root: str | Path,
                  required: Iterable[str] = ("sequence_id", "event_path", "split"),
                  ) -> list[dict[str, str]]:
    """Read a CSV manifest and constrain referenced event paths to ``data_root``."""

    path, root = Path(path), Path(data_root).resolve()
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
    except (OSError, csv.Error) as error:
        raise ValueError(f"invalid manifest: {path}") from error
    required = set(required)
    if not reader.fieldnames or required - set(reader.fieldnames):
        raise ValueError(f"manifest is missing columns: {sorted(required - set(reader.fieldnames or []))}")
    seen: set[str] = set()
    for row in rows:
        for name in reader.fieldnames:
            row[name] = (row.get(name) or "").strip()
        for name in required:
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


def _validate(x, y, t, p, width: int, height: int):
    if len({x.size, y.size, t.size, p.size}) != 1:
        raise ValueError("event arrays must have equal lengths")
    x = x.astype(np.int32, copy=False)
    y = y.astype(np.int32, copy=False)
    if (np.any(x < 0) or np.any(x >= width) or np.any(y < 0) or np.any(y >= height)):
        raise ValueError("event coordinates are outside the sensor")
    if t.size and np.any(t[1:] < t[:-1]):
        order = np.argsort(t, kind="stable")
        x, y, t, p = x[order], y[order], t[order], p[order]
    if not np.isin(p, (-1, 0, 1)).all():
        raise ValueError("event polarity must use {-1,1} or {0,1}")
    if np.any(p == -1) and np.any(p == 0):
        raise ValueError("event polarity mixes two OFF encodings")
    return x, y, t.astype(np.float64, copy=False), (p > 0).astype(np.uint8)


def load_event_sequence(
    row: dict[str, str], data_root: str | Path, sensor_size: Iterable[int],
    *, max_duration_seconds: float | None = None,
) -> EventSequence:
    """Load a debris NPZ; ``omega_rad_s`` stays a field used only by the evaluator."""

    root = Path(data_root).resolve()
    event_path = (root / row["event_path"]).resolve()
    if not event_path.is_relative_to(root) or not event_path.is_file():
        raise ValueError(f"missing event NPZ: {event_path}")
    width, height = (int(value) for value in sensor_size)
    try:
        with np.load(event_path, allow_pickle=False) as archive:
            x = _numeric(archive, "x", ndim=1)
            y = _numeric(archive, "y", ndim=1)
            if "t" in archive.files and "p" in archive.files:
                t = _numeric(archive, "t", ndim=1).astype(np.float64, copy=False)
                p = _numeric(archive, "p", ndim=1)
                event_schema = "espinslam_xy_tp_seconds"
            elif "t_us" in archive.files and "p01" in archive.files:
                # event_snn_torch's fused event-stream cache preserves the
                # source event order and integer microsecond timestamps.  Read
                # it directly so preparing an eSpinSLAM view does not duplicate
                # tens of gigabytes of event arrays.
                t = _numeric(archive, "t_us", ndim=1).astype(np.float64) * 1e-6
                p = _numeric(archive, "p01", ndim=1)
                event_schema = "event_snn_torch_xy_t_us_p01"
            else:
                raise ValueError(
                    "event NPZ must contain either x/y/t/p or x/y/t_us/p01"
                )
            omega = float(_numeric(archive, "omega_rad_s", ndim=0))
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"failed to load {event_path}: {error}") from error
    if t.size and np.any(t < 0):
        raise ValueError("event timestamps must be non-negative")
    if t.size and np.any(t[1:] < t[:-1]):
        raise ValueError("event timestamps must be sorted and non-negative")
    x, y, t, p = _validate(x, y, t, p, width, height)
    if omega < 0:
        raise ValueError("omega_rad_s must be non-negative")
    if max_duration_seconds is not None and t.size:
        stop = int(np.searchsorted(t, t[0] + max_duration_seconds, side="right"))
        x, y, t, p = x[:stop], y[:stop], t[:stop], p[:stop]
    metadata = {
        name: value.strip() for name, value in row.items()
        if isinstance(value, str) and value.strip()
    }
    metadata["event_schema"] = event_schema
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


# ---------------------------------------------------------------------------
# Generic event files (TOPSPIN and other datasets)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EventArrays:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    width: int | None = None
    height: int | None = None


_UNIT_SCALE = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


def _to_seconds(t: np.ndarray, unit: str) -> np.ndarray:
    if unit == "auto":
        if t.size == 0:
            unit = "s"
        elif t.dtype.kind in "iu":
            unit = "us"  # Prophesee / most event SDKs use integer microseconds.
        else:
            span = float(np.max(t) - np.min(t))
            unit = "s" if span < 1e4 else ("us" if span < 1e10 else "ns")
    return t.astype(np.float64) * _UNIT_SCALE[unit]


def _pick(names: Iterable[str], *candidates: str) -> str | None:
    lookup = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def _from_columns(source, names: Iterable[str]) -> tuple:
    names = list(names)
    keys = (
        _pick(names, "t", "ts", "timestamp", "timestamps", "time"),
        _pick(names, "x"),
        _pick(names, "y"),
        _pick(names, "p", "pol", "polarity", "polarities"),
    )
    if None in keys:
        raise ValueError(f"cannot find t/x/y/p fields among {names}")
    return tuple(np.asarray(source[key]).ravel() for key in keys)


def _read_header(stream) -> tuple[list[str], int]:
    lines = []
    while True:
        position = stream.tell()
        line = stream.readline()
        if not line.startswith(b"%"):
            stream.seek(position)
            break
        text = line.decode("latin-1").strip()
        lines.append(text)
        if text.lower() == "% end":
            break
    return lines, stream.tell()


def _geometry_from_header(lines: list[str]) -> tuple[int | None, int | None]:
    text = "\n".join(lines)
    width = height = None
    match = re.search(r"width\s*[=\s]\s*(\d+)", text, re.IGNORECASE)
    if match:
        width = int(match.group(1))
    match = re.search(r"height\s*[=\s]\s*(\d+)", text, re.IGNORECASE)
    if match:
        height = int(match.group(1))
    match = re.search(r"geometry\s+(\d+)x(\d+)", text, re.IGNORECASE)
    if match:
        width, height = int(match.group(1)), int(match.group(2))
    return width, height


def _read_dat(path: Path) -> EventArrays:
    with path.open("rb") as stream:
        lines, offset = _read_header(stream)
        stream.seek(offset)
        stream.read(2)  # event type and event size bytes
        raw = np.fromfile(stream, dtype="<u4")
    raw = raw[: raw.size - raw.size % 2].reshape(-1, 2)
    t, data = raw[:, 0].astype(np.int64), raw[:, 1]
    x = (data & 0x3FFF).astype(np.int32)
    y = ((data >> 14) & 0x3FFF).astype(np.int32)
    p = ((data >> 28) & 0xF).astype(np.int8)
    width, height = _geometry_from_header(lines)
    return EventArrays(t, x, y, p, width, height)


@njit(cache=True)
def _decode_evt3(words, t, x, y, p):
    count = 0
    time_high = 0
    time_low = 0
    overflow = 0
    last_high = 0
    current_y = 0
    base_x = 0
    polarity = 0
    for word in words:
        kind = word >> 12
        if kind == 0x0:
            current_y = word & 0x7FF
        elif kind == 0x2:
            if count < t.shape[0]:
                t[count] = (overflow << 24) | (time_high << 12) | time_low
                x[count] = word & 0x7FF
                y[count] = current_y
                p[count] = (word >> 11) & 1
                count += 1
        elif kind == 0x3:
            base_x = word & 0x7FF
            polarity = (word >> 11) & 1
        elif kind == 0x4 or kind == 0x5:
            bits = 12 if kind == 0x4 else 8
            mask = word & ((1 << bits) - 1)
            for bit in range(bits):
                if (mask >> bit) & 1 and count < t.shape[0]:
                    t[count] = (overflow << 24) | (time_high << 12) | time_low
                    x[count] = base_x + bit
                    y[count] = current_y
                    p[count] = polarity
                    count += 1
            base_x += bits
        elif kind == 0x6:
            time_low = word & 0xFFF
        elif kind == 0x8:
            value = word & 0xFFF
            if value < last_high and last_high - value > 2048:
                overflow += 1
            last_high = value
            time_high = value
    return count


@njit(cache=True)
def _decode_evt2(words, t, x, y, p):
    count = 0
    time_high = 0
    for word in words:
        kind = word >> 28
        if kind == 0x0 or kind == 0x1:
            t[count] = (time_high << 6) | ((word >> 22) & 0x3F)
            x[count] = (word >> 11) & 0x7FF
            y[count] = word & 0x7FF
            p[count] = kind
            count += 1
        elif kind == 0x8:
            time_high = word & 0x0FFFFFFF
    return count


def _read_raw(path: Path) -> EventArrays:
    with path.open("rb") as stream:
        lines, offset = _read_header(stream)
    header = "\n".join(lines).lower()
    width, height = _geometry_from_header(lines)
    if "evt3" in header or "evt 3" in header:
        words = np.fromfile(path, dtype="<u2", offset=offset).astype(np.int64)
        capacity = int(np.count_nonzero((words >> 12) == 0x2)
                       + 12 * np.count_nonzero((words >> 12) == 0x4)
                       + 8 * np.count_nonzero((words >> 12) == 0x5))
        t = np.empty(capacity, dtype=np.int64)
        x = np.empty(capacity, dtype=np.int32)
        y = np.empty(capacity, dtype=np.int32)
        p = np.empty(capacity, dtype=np.int8)
        count = _decode_evt3(words, t, x, y, p)
    elif "evt2" in header or "evt 2" in header:
        words = np.fromfile(path, dtype="<u4", offset=offset).astype(np.int64)
        capacity = int(np.count_nonzero((words >> 28) <= 1))
        t = np.empty(capacity, dtype=np.int64)
        x = np.empty(capacity, dtype=np.int32)
        y = np.empty(capacity, dtype=np.int32)
        p = np.empty(capacity, dtype=np.int8)
        count = _decode_evt2(words, t, x, y, p)
    else:
        raise ValueError(
            f"{path}: unsupported RAW encoding (header has no EVT2/EVT3 tag); "
            "convert it to .npz (t, x, y, p) with the Metavision SDK",
        )
    return EventArrays(t[:count], x[:count], y[:count], p[:count], width, height)


def _read_hdf5(path: Path) -> EventArrays:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover
        raise ImportError("reading HDF5 events requires h5py") from error
    with h5py.File(path, "r") as handle:
        width = handle.attrs.get("width")
        height = handle.attrs.get("height")
        for group in ("events", "CD/events", "CD", ""):
            node = handle[group] if group else handle
            try:
                if isinstance(node, h5py.Dataset):
                    if node.dtype.names:
                        data = node[()]
                        columns = _from_columns(data, data.dtype.names)
                    elif node.ndim == 2 and node.shape[1] == 4:
                        data = node[()]
                        columns = (data[:, 0], data[:, 1], data[:, 2], data[:, 3])
                    else:
                        continue
                else:
                    columns = _from_columns(node, list(node.keys()))
            except (KeyError, ValueError):
                continue
            except OSError as error:
                raise ValueError(
                    f"{path}: cannot decode {group or '/'} (Prophesee ECF-compressed HDF5 "
                    "needs the Metavision HDF5 plugin; convert to .npz instead)",
                ) from error
            t, x, y, p = columns
            return EventArrays(t, x, y, p,
                               int(width) if width is not None else None,
                               int(height) if height is not None else None)
    raise ValueError(f"{path}: no t/x/y/p event datasets found")


def _read_text(path: Path, columns: str) -> EventArrays:
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        first = stream.readline()
    delimiter = "," if "," in first else None
    skip = 0 if re.match(r"^\s*[-+0-9.]", first) else 1
    data = np.loadtxt(path, delimiter=delimiter, skiprows=skip, comments=("#", "%"),
                      ndmin=2)
    if data.shape[1] < 4:
        raise ValueError(f"{path}: expected at least four columns")
    index = {name: columns.index(name) for name in "txyp"}
    t = data[:, index["t"]]
    if np.all(np.mod(t, 1) == 0) and t.max(initial=0) > 1e4:
        t = t.astype(np.int64)
    return EventArrays(t, data[:, index["x"]], data[:, index["y"]], data[:, index["p"]])


def read_events(path: str | Path, *, time_unit: str = "auto",
                text_columns: str = "txyp") -> EventArrays:
    """Read an event file and return t[s] (float64), x, y (int32), p (uint8)."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "events" in archive.files and archive["events"].ndim == 2:
                data = archive["events"]
                columns = (data[:, 0], data[:, 1], data[:, 2], data[:, 3])
            else:
                columns = _from_columns(archive, archive.files)
            width = int(archive["width"]) if "width" in archive.files else None
            height = int(archive["height"]) if "height" in archive.files else None
        arrays = EventArrays(*columns, width, height)
    elif suffix == ".npy":
        data = np.load(path, allow_pickle=False)
        if data.dtype.names:
            arrays = EventArrays(*_from_columns(data, data.dtype.names))
        elif data.ndim == 2 and data.shape[1] == 4:
            arrays = EventArrays(data[:, 0], data[:, 1], data[:, 2], data[:, 3])
        else:
            raise ValueError(f"{path}: expected a structured array or an [N,4] t/x/y/p array")
    elif suffix in (".h5", ".hdf5"):
        arrays = _read_hdf5(path)
    elif suffix in (".csv", ".txt"):
        arrays = _read_text(path, text_columns)
    elif suffix == ".dat":
        arrays = _read_dat(path)
    elif suffix == ".raw":
        arrays = _read_raw(path)
    else:
        raise ValueError(f"unsupported event file type: {path}")
    t = _to_seconds(np.asarray(arrays.t), time_unit)
    p = np.asarray(arrays.p)
    p = np.where(p > 0, 1, 0).astype(np.uint8)
    return EventArrays(t, np.asarray(arrays.x).astype(np.int32),
                       np.asarray(arrays.y).astype(np.int32), p,
                       arrays.width, arrays.height)


def load_events_as_sequence(path: str | Path, sequence_id: str, *,
                            width: int | None = None, height: int | None = None,
                            time_unit: str = "auto", text_columns: str = "txyp",
                            max_duration_seconds: float | None = None,
                            metadata: dict[str, Any] | None = None) -> EventSequence:
    """Read a generic event file into a time-sorted :class:`EventSequence`."""

    arrays = read_events(path, time_unit=time_unit, text_columns=text_columns)
    width = width or arrays.width
    height = height or arrays.height
    if not width or not height:
        if arrays.x.size == 0:
            raise ValueError(f"{path}: empty event file and unknown sensor size")
        width = width or int(arrays.x.max()) + 1
        height = height or int(arrays.y.max()) + 1
    x, y, t, p = _validate(arrays.x, arrays.y, arrays.t, arrays.p, int(width), int(height))
    if max_duration_seconds is not None and t.size:
        stop = int(np.searchsorted(t, t[0] + max_duration_seconds, side="right"))
        x, y, t, p = x[:stop], y[:stop], t[:stop], p[:stop]
    return EventSequence(sequence_id, x, y, t, p, float("nan"), int(width), int(height),
                         dict(metadata or {}), Path(path))
