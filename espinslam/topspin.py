"""Exploratory TOPSPIN evaluation for the eSpinSLAM reproduction.

The public release does not contain authoritative rotation-frequency labels or
the authors' CAD models. Consequently, public-data runs reproduce only the
observable protocol/diagnostics; they do not reproduce true frequency error or
CAD ICP millimetres. Exploratory non-public inputs can populate those fields
only with explicit provenance. A verified public-TOPSPIN run rejects a bare
manifest authority claim because record 10884694 authenticates neither asset.
Same-speed agreement is reported as a *consistency* statistic, never as
ground-truth error.

Outputs, per run directory:

* ``per_sequence.csv``  - frequency, loop closure, reprojection error, ICP
  alignment RMSE [mm], landmarks, runtime and throughput of every scene;
* ``table2.csv`` / ``table2.md`` - per-object mean +- std of reprojection
  error [px] and available ICP alignment RMSE [mm]. If the manifest has
  a ``split`` column (dev versus non-dev exploratory scenes), the tables,
  figures and summary use the test scenes and ``table2_dev.*`` /
  ``table2_all.*`` report the dev scenes and all scenes;
* ``fig6_frequency_cdf.png`` - exploratory CDF of estimated f per nominal
  speed; authoritative reference lines appear only when supplied;
* ``fig5_throughput_summary.png`` - front-end (the paper's ETC tracker),
  full-system, and scene event-rate throughput (paper Fig. 5);
* ``table_vs_paper.md`` / ``table3_vs_paper.md`` - created only by an eligible
  clean, verified, exact-all-72 paper-comparison run;
* ``fig7_throughput.png`` - per-object throughput box plot with the mean scene
  event rate (paper Fig. 7);
* ``history/<scene>_optimizations.csv`` and ``fig8_<scene>.png`` - mean
  reprojection error and number of world points over time (paper Fig. 8);
* ``reconstruction_<scene>.png`` - world points inside the estimated orbit
  (as in paper Figs. 1, 3, 4); ``per_sequence.csv`` also flags whether the
  system kept up with the scene's event rate (``real_time``, Fig. 7);
* ``metrics.json`` - everything above plus the configuration.

:func:`evaluate_topspin_ablations` runs the four settings of Table 3 and
writes ``table3.csv`` / ``table3.md``.

The manifest is a CSV with one row per scene. Required columns:
``sequence_id, event_path, object``. Optional columns: ``speed``,
``frequency_hz`` and ``frequency_source`` (evaluation only), ``cad_path`` and
``cad_source``,
``cad_unit_mm`` (CAD units -> mm, default 1), ``orbit_radius_mm`` (true
camera-to-spin-axis distance; fixes the monocular scale for ICP),
``fx, fy, cx, cy, k1, k2, p1, p2, k3`` (calibration overriding the config),
``width, height`` and ``time_unit``.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import replace
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

from .config import CameraConfig, ESpinSLAMConfig
from .data import load_events_as_sequence, read_manifest
from .geometry import align_similarity_icp
from .pipeline import ESpinSLAM, ESpinSLAMResult

TABLE2_OBJECTS = ("Hubble", "SOHO", "TDRS", "Camera", "Dualshock", "Switch")

AUTHORITATIVE_SOURCES = {
    "author", "author_provided", "official", "topspin_author", "dataset_author",
}

PROTOCOL_SCHEMA = 3
TOPSPIN_RECORD_ID = 10884694
TOPSPIN_RECORD_FINGERPRINT_SHA256 = (
    "c97ac90550208bd2225f9d116016cd188b9f9037638ee2c608a376de8f6208e7"
)
TOPSPIN_MARKER_NAME = ".topspin_provenance.json"
TOPSPIN_ACQUISITION_AUDIT = "acquisition_audit.json"
TOPSPIN_CONVERTER_SCHEMA = 2
SAFE_SEQUENCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
TOPSPIN_SCENES = frozenset(
    f"{obj}-{perspective}-{speed}"
    for obj in ("camera", "controller", "hubble", "soho", "switch", "tdrs")
    for perspective in ("diagonal", "perpendicular", "sideon", "topdown")
    for speed in ("slow", "med", "fast")
)
TOPSPIN_DEV_SCENES = frozenset({
    "hubble-perpendicular-slow", "tdrs-sideon-med", "camera-topdown-fast",
})
TOPSPIN_OBJECTS = {
    "camera": "Camera", "controller": "Dualshock", "hubble": "Hubble",
    "soho": "SOHO", "switch": "Switch", "tdrs": "TDRS",
}
# The FFT construction remains an experimental interpretation because the
# cited method does not uniquely specify all implementation details.
PAPER_COMPARABLE_SETTINGS = frozenset({
    "eSpinSLAM", "No landmark association",
})

REVIEW18 = (
    "camera-diagonal-slow",
    "camera-perpendicular-med",
    "camera-sideon-fast",
    "controller-perpendicular-slow",
    "controller-sideon-med",
    "controller-topdown-fast",
    "hubble-sideon-slow",
    "hubble-topdown-med",
    "hubble-diagonal-fast",
    "soho-topdown-slow",
    "soho-diagonal-med",
    "soho-sideon-fast",
    "switch-diagonal-slow",
    "switch-perpendicular-med",
    "switch-sideon-fast",
    "tdrs-perpendicular-slow",
    "tdrs-topdown-med",
    "tdrs-topdown-fast",
)

# Values reported in the paper (mean, std per object), for side-by-side reports.
# Note: Table 3's eSpinSLAM ICP row differs from Table 2 for Hubble (1.92 vs
# 0.92) and Switch (1.83 vs 0.83); both are kept as printed.
PAPER_TABLE2 = {
    "reprojection_px": {
        "Hubble": (3.25, 3.01), "SOHO": (2.99, 3.17), "TDRS": (1.54, 0.20),
        "Camera": (1.98, 0.15), "Dualshock": (2.19, 0.91), "Switch": (1.94, 0.35),
    },
    "icp_mm": {
        "Hubble": (0.92, 0.71), "SOHO": (0.56, 0.41), "TDRS": (0.66, 0.76),
        "Camera": (0.75, 0.57), "Dualshock": (0.93, 1.02), "Switch": (0.83, 0.66),
    },
}
PAPER_TABLE3 = {
    "FFT f estimation": {
        "reprojection_px": {"Hubble": (5.61, 7.93), "SOHO": (5.81, 5.58), "TDRS": (3.41, 3.56),
                            "Camera": (2.67, 1.34), "Dualshock": (2.99, 2.35), "Switch": (4.39, 5.11)},
        "icp_mm": {"Hubble": (3.17, 1.82), "SOHO": (3.51, 2.44), "TDRS": (5.95, 1.62),
                   "Camera": (2.13, 1.05), "Dualshock": (4.21, 1.33), "Switch": (3.16, 2.03)},
    },
    "No factor graph f constraint": {
        "reprojection_px": {"Hubble": (6.01, 3.79), "SOHO": (17.16, 9.03), "TDRS": (4.29, 2.33),
                            "Camera": (2.89, 3.11), "Dualshock": (5.27, 7.75), "Switch": (12.38, 6.91)},
        "icp_mm": {"Hubble": (8.14, 7.29), "SOHO": (15.49, 3.60), "TDRS": (5.11, 2.49),
                   "Camera": (1.12, 1.27), "Dualshock": (4.35, 0.74), "Switch": (9.38, 3.29)},
    },
    "No landmark association": {
        "reprojection_px": {"Hubble": (2.11, 1.93), "SOHO": (3.68, 0.71), "TDRS": (1.18, 0.71),
                            "Camera": (0.42, 0.80), "Dualshock": (1.67, 1.86), "Switch": (1.09, 1.50)},
        "icp_mm": {"Hubble": (13.74, 3.92), "SOHO": (12.47, 11.16), "TDRS": (19.61, 8.31),
                   "Camera": (9.19, 5.10), "Dualshock": (7.72, 3.27), "Switch": (13.11, 3.30)},
    },
    "eSpinSLAM": {
        "reprojection_px": PAPER_TABLE2["reprojection_px"],
        "icp_mm": {"Hubble": (1.92, 0.71), "SOHO": (0.56, 0.41), "TDRS": (0.66, 0.76),
                   "Camera": (0.75, 0.57), "Dualshock": (0.93, 1.02), "Switch": (1.83, 0.66)},
    },
}
ABLATIONS = {
    "fft_frequency": ("FFT f estimation", {"frequency": {"method": "fft"}}),
    "free_frequency": ("No factor graph f constraint", {"backend": {"optimize_frequency": True}}),
    "no_landmark_association": ("No landmark association",
                                {"backend": {"landmark_association": False}}),
    "espinslam": ("eSpinSLAM", {}),
}

# Frozen method/evaluation values from examples/configs/topspin_reference.jsonc.
# Logging is deliberately excluded: changing a W&B destination does not change
# the evaluated method.  Everything that can affect input selection, the
# estimator, reconstruction, or a reported metric is compared exactly.
REFERENCE_PROTOCOL = {
    "data.sensor_size": [1280, 720],
    "data.max_duration_seconds": None,
    "data.time_unit": "auto",
    "data.text_columns": "txyp",
    "data.evaluation_split": "test",
    "camera.fx": None, "camera.fy": None, "camera.cx": None, "camera.cy": None,
    "camera.distortion": [],
    "frontend.interpretation_mode": "paper",
    "frontend.chunk_events": 1000,
    "frontend.density_radius_px": 7.0,
    "frontend.density_mean_per_polarity": True,
    "frontend.cluster_min_size": 10,
    "frontend.cluster_min_samples": None,
    "frontend.cluster_selection_epsilon": 5.0,
    "frontend.cluster_time_scale_px_per_second": 1.0,
    "frontend.cluster_by_polarity": False,
    "frontend.association_radius_px": 30.0,
    "frontend.association_max_gap_seconds": 0.05,
    "frontend.association_head_events": 5,
    "frontend.retain_events_per_track": 1000,
    "frequency.method": "loop_closure",
    "frequency.alignment_mode": "paper",
    "frequency.update_interval_seconds": 0.1,
    "frequency.min_period_seconds": 0.1,
    "frequency.max_period_seconds": None,
    "frequency.time_scale_px_per_second": 1.0,
    "frequency.max_alignment_samples": 10000,
    "frequency.coarse_alignment_samples": 2000,
    "frequency.coarse_relative_step": 0.01,
    "frequency.icp_iterations": 30,
    "frequency.icp_tolerance_seconds": 0.00001,
    "frequency.harmonic_tolerance": 0.0,
    "frequency.min_samples_per_half": 50,
    "frequency.loop_std_threshold_hz": 0.001,
    "frequency.loop_history": 20,
    "frequency.fft_bin_seconds": 0.01,
    "backend.interpretation_mode": "paper",
    "backend.solver_backend": "equivalent_custom",
    "backend.orbit_radius": 10.0,
    "backend.min_track_events": 3,
    "backend.huber_threshold_px": 2.0,
    "backend.max_mean_reprojection_error_px": 10.0,
    "backend.landmark_merge_radius": 0.01,
    "backend.optimization_cadence_revolutions": 0.1,
    "backend.forget_revolutions": 3.0,
    "backend.max_iterations": 100,
    "backend.min_triangulation_angle_deg": 0.0,
    "backend.max_observations_per_track": 0,
    "backend.optimize_frequency": False,
    "backend.landmark_association": True,
    "backend.final_offline_refinement": False,
    "evaluation.observation_times_seconds": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
    "evaluation.pointcloud_max_reference_points": 30000,
    "evaluation.failure_policy": "report_and_mark_invalid",
}

REFERENCE_ABLATION_OVERRIDES = {
    "FFT f estimation": {"frequency.method": "fft"},
    "No factor graph f constraint": {"backend.optimize_frequency": True},
    "No landmark association": {"backend.landmark_association": False},
    "eSpinSLAM": {},
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _float_or_none(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _nullable(value: float | None) -> float | None:
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _json_default(value: Any) -> Any:
    """Convert NumPy scalar diagnostics without permitting NaN/Infinity."""

    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")).hexdigest()


def _source_tree_hash(repository: Path) -> str:
    """Hash executable/configuration sources, including uncommitted files."""

    digest = hashlib.sha256()
    roots = [
        repository / "espinslam", repository / "scripts",
        repository / "examples" / "configs", repository / "requirements",
    ]
    files = [repository / "pyproject.toml"]
    for root in roots:
        if root.is_dir():
            files.extend(
                path for path in root.rglob("*")
                if (path.is_file() and "__pycache__" not in path.parts
                    and path.suffix.lower() in {
                        ".py", ".json", ".jsonc", ".toml", ".txt", ".sh",
                        ".yml", ".yaml",
                    })
            )
    for path in sorted(set(files), key=lambda item: str(item.relative_to(repository))):
        relative = str(path.relative_to(repository)).encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _code_identity() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=False,
        capture_output=True, text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repository, check=False,
        capture_output=True, text=True,
    )
    lock = repository / "requirements" / "lock.txt"
    return {
        "git_revision": revision.stdout.strip() or None,
        "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "source_tree_sha256": _source_tree_hash(repository),
        "lock_sha256": _sha256_file(lock) if lock.is_file() else None,
    }


def _safe_sequence_id(value: str) -> str:
    if not SAFE_SEQUENCE_ID.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"unsafe sequence_id: {value!r}")
    return value


def _file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing input file: {path}")
    return {"size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _validated_marker(scene_dir: Path, audit_entry: dict[str, Any] | None) -> dict[str, Any]:
    marker_path = scene_dir / TOPSPIN_MARKER_NAME
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"missing/invalid TOPSPIN provenance marker: {marker_path}") from error
    if (not isinstance(marker, dict)
            or marker.get("schema_version") != TOPSPIN_CONVERTER_SCHEMA
            or marker.get("converter_schema") != TOPSPIN_CONVERTER_SCHEMA
            or marker.get("record_id") != TOPSPIN_RECORD_ID
            or marker.get("scene") != scene_dir.name
            or marker.get("archive_filename") != f"{scene_dir.name}.tar.gz"
            or not isinstance(marker.get("archive_size_bytes"), int)
            or marker.get("archive_size_bytes", 0) <= 0
            or not isinstance(marker.get("event_count"), int)
            or isinstance(marker.get("event_count"), bool)
            or marker.get("verified") is not True
            or marker.get("expected_md5") != marker.get("actual_md5")):
        raise ValueError(f"unverified TOPSPIN provenance marker: {marker_path}")
    outputs = marker.get("output_files")
    if not isinstance(outputs, dict):
        raise ValueError(f"TOPSPIN marker has no converted-file hashes: {marker_path}")
    for filename in ("events.npz", "calib.txt", "meta.json"):
        actual = _file_identity(scene_dir / filename)
        if outputs.get(filename) != actual:
            raise ValueError(f"TOPSPIN converted file failed hash verification: {scene_dir / filename}")
    try:
        with np.load(scene_dir / "events.npz", allow_pickle=False) as events:
            if (set(events.files) != {"t", "x", "y", "p"}
                    or events["t"].dtype != np.dtype(np.int64)
                    or events["x"].dtype != np.dtype(np.int16)
                    or events["y"].dtype != np.dtype(np.int16)
                    or events["p"].dtype != np.dtype(np.uint8)
                    or any(events[name].ndim != 1 for name in ("t", "x", "y", "p"))
                    or len({events[name].size for name in ("t", "x", "y", "p")}) != 1
                    or int(events["t"].size) != marker.get("event_count")):
                raise ValueError(f"TOPSPIN converted NPZ schema/count mismatch: {scene_dir}")
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"invalid TOPSPIN converted NPZ: {scene_dir}") from error
    if audit_entry is not None:
        for key in ("scene", "archive_filename", "archive_size_bytes",
                    "expected_md5", "actual_md5", "event_count",
                    "converter_schema", "output_files"):
            if audit_entry.get(key) != marker.get(key):
                raise ValueError(
                    f"TOPSPIN acquisition audit disagrees with marker for {scene_dir.name}: {key}"
                )
    return marker


def _validate_verified_manifest_row(row: dict[str, str], scene_dir: Path,
                                    *, require_fixed_split: bool) -> None:
    """Tie report-affecting manifest fields to the verified public files/name."""

    name = row["sequence_id"]
    object_token, perspective, speed_token = name.split("-")
    expected_speed = "medium" if speed_token == "med" else speed_token
    expected = {
        "object": TOPSPIN_OBJECTS[object_token],
        "perspective": perspective,
        "speed": expected_speed,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(f"{name}: manifest {field} must be {value!r}")
    if require_fixed_split:
        wanted = "dev" if name in TOPSPIN_DEV_SCENES else "test"
        if row.get("split") != wanted:
            raise ValueError(f"{name}: paper comparison requires fixed split={wanted!r}")
    calibration = (scene_dir / "calib.txt").read_text(encoding="utf-8").split()
    columns = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
    if len(calibration) < 4 or len(calibration) > len(columns):
        raise ValueError(f"{name}: invalid verified calibration")
    for field, raw in zip(columns, calibration):
        try:
            actual = float(row.get(field, ""))
            wanted = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name}: manifest {field} is missing/non-numeric") from error
        if not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{name}: manifest {field} disagrees with verified calib.txt")
    try:
        metadata = json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))
        frame = metadata["event-camera"]["frame"]
        width, height = int(frame["width"]), int(frame["height"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{name}: invalid verified meta.json frame") from error
    if row.get("width") != str(width) or row.get("height") != str(height):
        raise ValueError(f"{name}: manifest sensor size disagrees with verified meta.json")
    time_unit = row.get("time_unit", "")
    if require_fixed_split and time_unit != "us":
        raise ValueError(
            f"{name}: paper comparison requires explicit time_unit='us'"
        )
    if time_unit not in {"", "auto", "us"}:
        raise ValueError(f"{name}: verified TOPSPIN timestamps must be interpreted as microseconds")
    if _is_authoritative(row.get("frequency_source")) or _is_authoritative(row.get("cad_source")):
        raise ValueError(
            f"{name}: public TOPSPIN audit cannot authenticate frequency GT or author CAD"
        )


def _validate_topspin_acquisition(
    data_root: Path, rows: list[dict[str, str]], *, require_full72: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    audit_path = data_root / TOPSPIN_ACQUISITION_AUDIT
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"missing/invalid TOPSPIN acquisition audit: {audit_path}") from error
    if (not isinstance(audit, dict)
            or audit.get("schema_version") != TOPSPIN_CONVERTER_SCHEMA
            or audit.get("converter_schema") != TOPSPIN_CONVERTER_SCHEMA
            or audit.get("record_id") != TOPSPIN_RECORD_ID
            or audit.get("record_fingerprint_sha256")
            != TOPSPIN_RECORD_FINGERPRINT_SHA256
            or not isinstance(audit.get("scenes"), dict)):
        raise ValueError("TOPSPIN acquisition audit has the wrong record/schema")
    selected = {row["sequence_id"] for row in rows}
    if len(selected) != len(rows):
        raise ValueError("TOPSPIN manifest/selection contains duplicate sequence_id rows")
    if require_full72 and (len(rows) != len(TOPSPIN_SCENES)
                           or selected != TOPSPIN_SCENES):
        missing = sorted(TOPSPIN_SCENES - selected)
        extra = sorted(selected - TOPSPIN_SCENES)
        raise ValueError(
            f"paper comparison requires exactly all 72 TOPSPIN scenes; "
            f"missing={missing}, extra={extra}"
        )
    if require_full72 and set(audit["scenes"]) != TOPSPIN_SCENES:
        missing = sorted(TOPSPIN_SCENES - set(audit["scenes"]))
        extra = sorted(set(audit["scenes"]) - TOPSPIN_SCENES)
        raise ValueError(
            "paper comparison requires an exact 72-scene acquisition audit; "
            f"missing={missing}, extra={extra}"
        )
    identities: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = _safe_sequence_id(row["sequence_id"])
        event_path = (data_root / row["event_path"]).resolve()
        if not event_path.is_relative_to(data_root) or event_path.name != "events.npz":
            raise ValueError(f"{name}: verified TOPSPIN input must be a scene events.npz")
        if event_path.parent.name != name:
            raise ValueError(f"{name}: event directory does not match sequence_id")
        entry = audit["scenes"].get(name)
        if not isinstance(entry, dict) or entry.get("verified") is not True:
            raise ValueError(f"{name}: missing verified acquisition-audit entry")
        marker = _validated_marker(event_path.parent, entry)
        if (entry.get("events_npz") != f"{name}/events.npz"
                or entry.get("archive_filename") != f"{name}.tar.gz"):
            raise ValueError(f"{name}: acquisition-audit path/name mismatch")
        _validate_verified_manifest_row(
            row, event_path.parent, require_fixed_split=require_full72,
        )
        identities[name] = {
            "event_path": row["event_path"],
            "events": marker["output_files"]["events.npz"],
            "archive_md5": marker["actual_md5"],
            "converter_schema": marker["converter_schema"],
        }
    return audit, identities


def _is_authoritative(source: str | None) -> bool:
    return str(source or "").strip().lower() in AUTHORITATIVE_SOURCES


def _reference_mismatches(config: ESpinSLAMConfig,
                          setting_name: str = "eSpinSLAM") -> list[str]:
    """Return every departure from the frozen reference protocol.

    A Table-3 ablation is reference-comparable only when the *sole* departure
    is its preregistered switch.  This prevents a paper-labelled report from
    silently including a changed Table-1 value, eSfO front-end setting,
    observation cap, frequency optimisation, or report-affecting option.
    """

    expected = dict(REFERENCE_PROTOCOL)
    expected.update(REFERENCE_ABLATION_OVERRIDES.get(setting_name, {}))
    values = config.as_dict()
    mismatches = []
    if setting_name not in REFERENCE_ABLATION_OVERRIDES:
        mismatches.append(f"setting_name={setting_name!r} is not a registered reference setting")
    for path, wanted in expected.items():
        section, key = path.split(".", 1)
        actual = values[section][key]
        if actual != wanted:
            mismatches.append(f"{path}: expected {wanted!r}, got {actual!r}")
    return mismatches


def _reference_mode(config: ESpinSLAMConfig, setting_name: str = "eSpinSLAM") -> bool:
    """Whether outputs may be described as the frozen paper-reference path."""

    return not _reference_mismatches(config, setting_name)


def _method_profile(reference_mode: bool, setting_name: str) -> str:
    """Classify executable fidelity separately from config conformance."""

    return (
        "paper_reference_config"
        if reference_mode and setting_name in PAPER_COMPARABLE_SETTINGS
        else "experimental_interpretation"
    )


def _runtime_provenance(code_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    packages = {}
    for name in (
        "numpy", "scipy", "scikit-learn", "numba", "opencv-python-headless",
        "matplotlib", "h5py", "trimesh", "gtsam",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    total_memory = None
    if hasattr(os, "sysconf"):
        try:
            total_memory = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (OSError, ValueError, TypeError):
            pass
    gpu = None
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=False, capture_output=True, text=True, timeout=5,
        )
        gpu = completed.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    code_identity = code_identity or _code_identity()
    return {
        "hostname": platform.node() or None,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python": sys.version,
        "python_executable": sys.executable,
        "command": list(sys.argv),
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": total_memory,
        "gpu": gpu,
        **code_identity,
        "packages": packages,
    }


def _checkpoint_fingerprint(config: ESpinSLAMConfig, row: dict[str, str],
                            setting_name: str, *,
                            input_identity: dict[str, Any] | None = None,
                            code_identity: dict[str, Any] | None = None) -> str:
    payload = {
        "config": config.as_dict(),
        "manifest_row": row,
        "setting": setting_name,
        "input_identity": input_identity,
        "code_identity": code_identity,
        "schema": PROTOCOL_SCHEMA,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(name for name in row if name not in fieldnames)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    temporary.replace(path)


def _prepare_run_directory(output_dir: Path, run_identity: dict[str, Any]) -> str:
    """Refuse to mix artifacts belonging to a different run identity."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _canonical_hash(run_identity)
    manifest_path = output_dir / "run_manifest.json"
    existing_files = [path for path in output_dir.iterdir() if path.name != manifest_path.name]
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid run manifest: {manifest_path}") from error
        if previous.get("run_fingerprint") != fingerprint:
            raise ValueError(
                f"output directory belongs to a different run; use a new directory: {output_dir}"
            )
    elif existing_files:
        raise ValueError(
            f"non-empty output directory has no matching run manifest: {output_dir}"
        )
    _write_text(manifest_path, json.dumps(
        {"schema": PROTOCOL_SCHEMA, "run_fingerprint": fingerprint, **run_identity},
        ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False,
        default=_json_default,
    ) + "\n")
    return fingerprint


def _artifact_identities(output_dir: Path, paths: list[Path]) -> dict[str, dict[str, Any]]:
    identities = {}
    for path in paths:
        if not path.is_file():
            raise ValueError(f"required checkpoint artifact was not created: {path}")
        identities[str(path.relative_to(output_dir))] = _file_identity(path)
    return identities


def _validate_checkpoint_artifacts(output_dir: Path, payload: dict[str, Any]) -> None:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("checkpoint has no artifact inventory")
    for relative, expected in artifacts.items():
        path = (output_dir / relative).resolve()
        if not path.is_relative_to(output_dir.resolve()) or _file_identity(path) != expected:
            raise ValueError(f"checkpoint artifact failed verification: {relative}")


def _validate_checkpoint_record(payload: dict[str, Any], *, sequence_id: str,
                                run_fingerprint: str, setting_name: str) -> dict[str, Any]:
    """Reject truncated/edited checkpoint records before aggregation."""

    record = payload.get("record")
    if not isinstance(record, dict):
        raise ValueError(f"{sequence_id}: checkpoint has no result record")
    if payload.get("record_sha256") != _canonical_hash(record):
        raise ValueError(f"{sequence_id}: checkpoint result record failed verification")
    if (record.get("sequence_id") != sequence_id
            or record.get("run_fingerprint") != run_fingerprint
            or record.get("setting") != setting_name):
        raise ValueError(f"{sequence_id}: checkpoint result belongs to a different run")
    return record


def read_topspin_manifest(path: str | Path, data_root: str | Path) -> list[dict[str, str]]:
    return read_manifest(path, data_root, required=("sequence_id", "event_path", "object"))


def camera_for_row(base: CameraConfig, row: dict[str, str]) -> CameraConfig:
    """Per-scene calibration from the manifest, falling back to the config."""

    values = {name: _float_or_none(row.get(name)) for name in ("fx", "fy", "cx", "cy")}
    distortion = [_float_or_none(row.get(name)) for name in ("k1", "k2", "p1", "p2", "k3")]
    camera = CameraConfig(
        fx=values["fx"] if values["fx"] is not None else base.fx,
        fy=values["fy"] if values["fy"] is not None else base.fy,
        cx=values["cx"] if values["cx"] is not None else base.cx,
        cy=values["cy"] if values["cy"] is not None else base.cy,
        distortion=([value or 0.0 for value in distortion]
                    if any(value is not None for value in distortion) else list(base.distortion)),
    )
    if not camera.complete:
        raise ValueError(
            f"{row['sequence_id']}: camera intrinsics are missing; add fx, fy, cx, cy "
            "(TOPSPIN calibration) to the manifest or to the config",
        )
    return camera


def load_reference_cloud(path: Path, count: int, unit_mm: float = 1.0,
                         seed: int = 2701) -> np.ndarray:
    """CAD reference in millimetres: a point cloud or an area-sampled mesh."""

    suffix = path.suffix.lower()
    if suffix in (".npz", ".npy"):
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                points = np.asarray(archive["points"], dtype=np.float64)
        else:
            points = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
        if points.shape[0] > count:
            points = points[np.linspace(0, points.shape[0] - 1, count).round().astype(int)]
    else:
        try:
            import trimesh
        except ImportError as error:
            raise ImportError("loading CAD meshes requires trimesh (pip install trimesh)") from error
        mesh = trimesh.load(path, force="mesh")
        if mesh.faces.shape[0] == 0:
            points = np.asarray(mesh.vertices, dtype=np.float64)
        else:
            points, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
            points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 4:
        raise ValueError(f"{path}: reference must contain at least four 3D points")
    return points * float(unit_mm)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _sequence_row(row: dict[str, str], result: ESpinSLAMResult, *,
                  icp_rmse_mm: float | None, chamfer_mm: float | None,
                  icp_reason: str, scale_source: str) -> dict[str, Any]:
    declared_reference = _float_or_none(row.get("frequency_hz"))
    frequency_source = str(row.get("frequency_source") or "").strip()
    frequency_authoritative = _is_authoritative(frequency_source)
    truth = declared_reference if frequency_authoritative else None
    estimate = result.frequency_hz
    reconstruction = result.reconstruction
    observation_mean = _nullable(reconstruction.mean_observation_error_px)
    landmark_mean = _nullable(reconstruction.mean_reprojection_error_px)
    last_optimization_status = (
        str(result.optimizations[-1].termination_status) if result.optimizations else None)
    # A point cloud can be geometrically non-empty even when the last solver
    # call merely exhausted its iteration budget.  Keep that raw result, but
    # do not let it enter paper-comparison aggregate metrics as a success.
    paper_metrics_eligible = bool(
        reconstruction.valid and last_optimization_status != "max_iterations")
    return {
        "sequence_id": result.sequence_id,
        "object": row["object"],
        "split": row.get("split", ""),
        "speed": row.get("speed", ""),
        "frequency_reference_hz": declared_reference,
        "frequency_reference_source": frequency_source,
        "frequency_reference_authoritative": frequency_authoritative,
        "frequency_true_hz": truth,
        "frequency_est_hz": _nullable(estimate),
        "frequency_abs_error_hz": _nullable(abs(estimate - truth)) if truth is not None else None,
        "frequency_rel_error_percent": (
            _nullable(abs(estimate - truth) / truth * 100.0) if truth else None),
        "loop_closed": result.loop_closed,
        "loop_closure_time_s": _nullable(result.loop_closure_time),
        "landmarks": int(reconstruction.points.shape[0]),
        "observations": int(reconstruction.observation_count),
        # Paper Eq. (12)/Table 2 primary: one mean over every inlier
        # observation.  The equally weighted mean of per-landmark means is a
        # useful diagnostic but is not the headline paper-comparison metric.
        "reprojection_error_px": observation_mean,
        "reprojection_error_observation_mean_px": observation_mean,
        "reprojection_error_landmark_mean_px": landmark_mean,
        # Compatibility alias for older readers; identical to the explicitly
        # named observation-mean field above.
        "reprojection_error_obs_px": observation_mean,
        "icp_rmse_mm": icp_rmse_mm,
        "chamfer_rmse_mm": chamfer_mm,
        "icp_scale_source": scale_source,
        "icp_failure_reason": icp_reason,
        "reconstruction_valid": reconstruction.valid,
        "last_optimization_status": last_optimization_status,
        "paper_metrics_eligible": paper_metrics_eligible,
        "reconstruction_reason": reconstruction.reason,
        "events": result.event_count,
        "duration_s": result.observation_seconds,
        "scene_event_rate_ev_s": _nullable(
            result.event_count / result.observation_seconds if result.observation_seconds > 0
            else math.nan),
        "runtime_s": result.runtime_seconds,
        "throughput_ev_s": _nullable(result.throughput_events_per_second),
        # Fig. 7: real time when the system outpaces the scene's own event rate.
        "real_time": bool(result.observation_seconds > 0 and result.runtime_seconds > 0
                          and result.throughput_events_per_second
                          >= result.event_count / result.observation_seconds),
        "frontend_throughput_ev_s": _nullable(result.frontend_throughput_events_per_second),
        "frontend_s": result.frontend_seconds,
        "frequency_s": result.frequency_seconds,
        "backend_s": result.backend_seconds,
        "optimizations": len(result.optimizations),
    }


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    finite = np.asarray([value for value in values if value is not None and math.isfinite(value)])
    if not finite.size:
        return None, None
    return float(finite.mean()), float(finite.std())


def table2(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    objects = [name for name in TABLE2_OBJECTS if any(r["object"] == name for r in rows)]
    objects += sorted({r["object"] for r in rows} - set(objects))
    table = []
    for name in objects:
        subset = [r for r in rows if r["object"] == name]
        eligible = [r for r in subset if bool(r.get("paper_metrics_eligible"))]
        reprojection = _mean_std([r["reprojection_error_px"] for r in eligible])
        landmark_reprojection = _mean_std(
            [r["reprojection_error_landmark_mean_px"] for r in eligible])
        icp = _mean_std([r["icp_rmse_mm"] for r in eligible])
        table.append({
            "object": name,
            "scenes": len(subset),
            # Backward-compatible success count now follows paper-metric
            # eligibility; geometric validity remains available separately.
            "valid_reconstructions": len(eligible),
            "paper_metric_eligible_reconstructions": len(eligible),
            "geometrically_valid_reconstructions": sum(
                bool(r["reconstruction_valid"]) for r in subset),
            "reprojection_mean_px": reprojection[0], "reprojection_std_px": reprojection[1],
            "landmark_reprojection_mean_px": landmark_reprojection[0],
            "landmark_reprojection_std_px": landmark_reprojection[1],
            "icp_mean_mm": icp[0], "icp_std_mm": icp[1],
        })
    return table


def _format(mean: float | None, std: float | None) -> str:
    return "-" if mean is None else f"{mean:.2f} ± {std:.2f}"


def table2_markdown(table: list[dict[str, Any]], title: str = "eSpinSLAM") -> str:
    header = "| Metric | " + " | ".join(row["object"] for row in table) + " |"
    rule = "|---|" + "---|" * len(table)
    reprojection = f"| Reprojection error (px) ↓ [{title}] | " + " | ".join(
        _format(row["reprojection_mean_px"], row["reprojection_std_px"]) for row in table) + " |"
    landmark_reprojection = (
        f"| Landmark-mean reprojection diagnostic (px) [{title}] | " + " | ".join(
            _format(row["landmark_reprojection_mean_px"],
                    row["landmark_reprojection_std_px"]) for row in table) + " |"
    )
    icp = f"| ICP alignment RMSE (mm) ↓ [{title}] | " + " | ".join(
        _format(row["icp_mean_mm"], row["icp_std_mm"]) for row in table) + " |"
    return "\n".join((header, rule, reprojection, landmark_reprojection, icp)) + "\n"


def comparison_markdown(table: list[dict[str, Any]], paper: dict[str, dict], label: str) -> str:
    """Side-by-side "reproduction / paper" cells for one setting."""

    objects = [row["object"] for row in table]
    lines = [f"| {label} | " + " | ".join(objects) + " |", "|---|" + "---|" * len(objects)]
    for metric, mean_key, std_key in (("reprojection_px", "reprojection_mean_px",
                                       "reprojection_std_px"),
                                      ("icp_mm", "icp_mean_mm", "icp_std_mm")):
        ours = [_format(row[mean_key], row[std_key]) for row in table]
        reference = [
            _format(*paper[metric][name]) if name in paper.get(metric, {}) else "-"
            for name in objects
        ]
        title = "Reprojection error (px)" if metric == "reprojection_px" else "ICP alignment RMSE (mm)"
        lines.append(f"| {title}: this run | " + " | ".join(ours) + " |")
        lines.append(f"| {title}: paper | " + " | ".join(reference) + " |")
    return "\n".join(lines) + "\n"


def _speeds(rows: list[dict[str, Any]]) -> list[str]:
    speeds = [speed for speed in ("slow", "medium", "fast")
              if any(str(r["speed"]).lower() == speed for r in rows)]
    return speeds + sorted({str(r["speed"]).lower() for r in rows} - set(speeds))


def plot_frequency_cdf_comparison(series: dict[str, list[dict[str, Any]]], output: Path) -> None:
    """Fig. 6: CDF of f per speed for several estimators, with ground truth."""

    import matplotlib.pyplot as plt
    all_rows = [row for rows in series.values() for row in rows]
    speeds = _speeds(all_rows) or [""]
    colours = {"eSpinSLAM": "#1f77b4", "FFT f estimation": "#ff7f0e"}
    fig, axes = plt.subplots(1, len(speeds), figsize=(3.6 * len(speeds), 3.6), squeeze=False)
    for axis, speed in zip(axes[0], speeds):
        for label, rows in series.items():
            estimates = np.sort([r["frequency_est_hz"] for r in rows
                                 if str(r["speed"]).lower() == speed
                                 and r["frequency_est_hz"] is not None])
            if estimates.size:
                cumulative = np.arange(1, estimates.size + 1) / estimates.size
                axis.step(np.r_[estimates[0], estimates], np.r_[0.0, cumulative], where="post",
                          linewidth=2, color=colours.get(label), label=label)
        for truth in sorted({r["frequency_true_hz"] for r in all_rows
                             if str(r["speed"]).lower() == speed
                             and r["frequency_true_hz"] is not None}):
            axis.axvline(truth, color="#d62728", linestyle="--", linewidth=1.2)
        axis.set_title(f"{speed.capitalize() or 'All'} speed")
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.3)
    axes[0][0].set_ylabel("CDF")
    axes[0][0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_throughput_summary(rows: list[dict[str, Any]], output: Path) -> None:
    """Fig. 5 analogue: front-end, full system, and the scenes' own event rate."""

    import matplotlib.pyplot as plt
    values = {
        "Front-end (ETC analogue)": _mean_std([r["frontend_throughput_ev_s"] for r in rows])[0],
        "eSpinSLAM (system)": _mean_std([r["throughput_ev_s"] for r in rows])[0],
        "TOPSPIN scenes (event rate)": _mean_std([r["scene_event_rate_ev_s"] for r in rows])[0],
    }
    labels = [label for label, value in values.items() if value]
    fig, axis = plt.subplots(figsize=(7.0, 2.8))
    axis.barh(labels, [values[label] for label in labels],
              color=["#1f77b4", "#8b0000", "#d2691e"][:len(labels)])
    axis.set_xscale("log")
    axis.set_xlabel("Throughput (events/s)")
    axis.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_frequency_cdf(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt
    speeds = [speed for speed in ("slow", "medium", "fast")
              if any(str(r["speed"]).lower() == speed for r in rows)]
    speeds += sorted({str(r["speed"]).lower() for r in rows} - set(speeds))
    fig, axes = plt.subplots(1, max(1, len(speeds)), figsize=(3.6 * max(1, len(speeds)), 3.6),
                             squeeze=False)
    for axis, speed in zip(axes[0], speeds or [""]):
        subset = [r for r in rows if str(r["speed"]).lower() == speed]
        estimates = np.sort([r["frequency_est_hz"] for r in subset
                             if r["frequency_est_hz"] is not None])
        truths = sorted({r["frequency_true_hz"] for r in subset
                         if r["frequency_true_hz"] is not None})
        if estimates.size:
            cumulative = np.arange(1, estimates.size + 1) / estimates.size
            axis.step(np.r_[estimates[0], estimates], np.r_[0.0, cumulative],
                      where="post", color="#1f77b4", linewidth=2, label="eSpinSLAM")
        for truth in truths:
            axis.axvline(truth, color="#d62728", linestyle="--", linewidth=1.2)
        axis.set_title(f"{speed.capitalize() or 'All'} speed")
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.3)
    axes[0][0].set_ylabel("CDF")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_throughput(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt
    objects = [name for name in TABLE2_OBJECTS if any(r["object"] == name for r in rows)]
    objects += sorted({r["object"] for r in rows} - set(objects))
    data = [[r["throughput_ev_s"] / 1e6 for r in rows
             if r["object"] == name and r["throughput_ev_s"]] for name in objects]
    rates = [np.mean([r["scene_event_rate_ev_s"] / 1e6 for r in rows
                      if r["object"] == name and r["scene_event_rate_ev_s"]] or [np.nan])
             for name in objects]
    fig, axis = plt.subplots(figsize=(7.5, 4.2))
    axis.boxplot(data, showmeans=True, meanline=True)
    axis.set_xticks(range(1, len(objects) + 1), objects)
    for index, rate in enumerate(rates, 1):
        axis.hlines(rate, index - 0.35, index + 0.35, colors="green", linestyles="--",
                    label="Mean scene throughput" if index == 1 else None)
    axis.set_ylabel("Mean throughput (M events/s)")
    axis.set_xlabel("Object")
    axis.legend(loc="upper right")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_reconstruction(result: ESpinSLAMResult, orbit_radius: float, output: Path) -> None:
    """Figs. 1, 3, 4: reconstructed points inside the estimated camera orbit."""

    import matplotlib.pyplot as plt
    from .geometry import camera_pose
    reconstruction = result.reconstruction
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))
    angle = np.linspace(0.0, 2.0 * np.pi, 361)
    if reconstruction.valid:
        points = reconstruction.points
        rotation, centre = camera_pose(0.0, reconstruction.frequency_hz, orbit_radius,
                                       reconstruction.rotation)
        axis_world = rotation[2]  # optical axis in the world frame
        for axis, (i, j, name) in zip(axes, ((0, 1, "top view (x-y)"), (0, 2, "side view (x-z)"))):
            if j == 1:
                axis.plot(orbit_radius * np.cos(angle), orbit_radius * np.sin(angle), ":",
                          color="red", linewidth=1, label="camera orbit")
            else:
                axis.plot([-orbit_radius, orbit_radius], [0.0, 0.0], ":", color="red",
                          linewidth=1, label="camera orbit")
            axis.scatter(points[:, i], points[:, j], s=3, color="black", label="world points")
            axis.quiver(centre[i], centre[j], axis_world[i], axis_world[j], color="blue",
                        angles="xy", scale_units="xy", scale=0.4, label="camera at t=0")
            axis.set_aspect("equal", adjustable="datalim")
            axis.set_title(name)
            axis.grid(alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=7)
    else:
        for axis in axes:
            axis.text(0.5, 0.5, reconstruction.reason or "no reconstruction",
                      ha="center", va="center", transform=axis.transAxes)
    fig.suptitle(f"{result.sequence_id}: f* = {result.frequency_hz:.4f} Hz, "
                 f"{reconstruction.points.shape[0]} world points")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_history(result: ESpinSLAMResult, output: Path) -> None:
    import matplotlib.pyplot as plt
    reports = result.optimizations
    fig, axis = plt.subplots(figsize=(6.0, 3.6))
    if reports:
        times = [report.event_time for report in reports]
        axis.plot(times, [report.mean_observation_error_px for report in reports],
                  color="red", linewidth=2, label="all observations (primary)")
        axis.plot(times, [report.mean_reprojection_error_px for report in reports],
                  color="#ff8c00", linewidth=1.2, linestyle="--",
                  label="per-landmark mean diagnostic")
        axis.legend(loc="best", fontsize=7)
        twin = axis.twinx()
        twin.plot(times, [report.landmarks for report in reports], color="blue", linewidth=2)
        twin.set_ylabel("Number of Landmarks", color="blue")
    axis.set_xlabel("Timestamp (s)")
    axis.set_ylabel("Mean reprojection error (px)", color="red")
    axis.set_title(result.sequence_id)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def evaluate_topspin(config: ESpinSLAMConfig, data_root: Path, manifest_path: Path,
                     output_dir: Path, *, max_sequences: int = 0,
                     setting_name: str = "eSpinSLAM",
                     history_plots: bool = True,
                     sequence_ids: list[str] | tuple[str, ...] | None = None,
                     resume: bool = True,
                     require_reference: bool = False,
                     require_verified_data: bool = False,
                     paper_comparison: bool = False) -> dict[str, Any]:
    """Run eSpinSLAM and write restartable exploratory reports.

    A completed scene is checkpointed atomically. On resume, a checkpoint is
    reused only when its manifest row, setting, full configuration, input,
    source-tree/commit and dependency identities match and every inventoried
    artifact still has its recorded hash. Stale checkpoints fail instead of
    silently mixing protocols.
    """

    started = time.monotonic()
    root = Path(data_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    rows = read_topspin_manifest(manifest_path, root)
    if sequence_ids is not None:
        requested = list(sequence_ids)
        by_name = {row["sequence_id"]: row for row in rows}
        missing = [name for name in requested if name not in by_name]
        if missing:
            raise ValueError(f"requested TOPSPIN scenes are missing: {missing}")
        rows = [by_name[name] for name in requested]
    if max_sequences > 0:
        rows = rows[:max_sequences]
    sequence_names = [row["sequence_id"] for row in rows]
    if len(set(sequence_names)) != len(sequence_names):
        raise ValueError("TOPSPIN manifest/selection contains duplicate sequence_id rows")
    splits = [row.get("split", "") for row in rows]
    if any(splits) and any(value not in {"dev", "test"} for value in splits):
        raise ValueError(
            "TOPSPIN split must be blank for every row, or dev/test for every row"
        )
    for row in rows:
        _safe_sequence_id(row["sequence_id"])
    reference_mismatches = _reference_mismatches(config, setting_name)
    reference_mode = not reference_mismatches
    if require_reference and not reference_mode:
        raise ValueError(
            "the reference method profile requires the exact frozen protocol: "
            + "; ".join(reference_mismatches)
        )
    code_identity = _code_identity()
    # A frozen config alone is insufficient: FFT is not uniquely specified,
    # and the free-frequency custom solver has no direct-GTSAM equivalence
    # gate. Those executable paths remain experimental even when every config
    # value matches its registered ablation switch.
    method_profile = _method_profile(reference_mode, setting_name)
    classification = "exploratory_retrospective"
    acquisition_audit = None
    if require_verified_data or paper_comparison:
        acquisition_audit, input_identities = _validate_topspin_acquisition(
            root, rows, require_full72=paper_comparison,
        )
    else:
        input_identities = {
            row["sequence_id"]: {
                "event_path": row["event_path"],
                "events": _file_identity((root / row["event_path"]).resolve()),
                "archive_md5": None,
                "converter_schema": None,
            }
            for row in rows
        }
    comparison_ineligibility_reasons = []
    if not reference_mode:
        comparison_ineligibility_reasons.append("method config is not the frozen reference")
    if setting_name not in PAPER_COMPARABLE_SETTINGS:
        comparison_ineligibility_reasons.append(
            f"{setting_name!r} lacks a paper-faithful executable audit path"
        )
    if len(rows) != len(TOPSPIN_SCENES) or set(input_identities) != TOPSPIN_SCENES:
        comparison_ineligibility_reasons.append("selection is not exactly all 72 TOPSPIN scenes")
    if acquisition_audit is None:
        comparison_ineligibility_reasons.append("verified acquisition audit is unavailable")
    if code_identity["git_dirty"] is not False:
        comparison_ineligibility_reasons.append("source tree is dirty or git status is unavailable")
    if code_identity["git_revision"] is None:
        comparison_ineligibility_reasons.append("git revision is unavailable")
    paper_comparison_eligible = bool(
        paper_comparison and not comparison_ineligibility_reasons
    )
    if paper_comparison and not paper_comparison_eligible:
        raise ValueError(
            "paper comparison is not eligible: "
            + "; ".join(comparison_ineligibility_reasons)
        )
    run_identity = {
        "protocol_schema": PROTOCOL_SCHEMA,
        "setting": setting_name,
        "method_profile": method_profile,
        "evaluation_classification": classification,
        "paper_comparison_requested": paper_comparison,
        "sequence_ids": [row["sequence_id"] for row in rows],
        "config": config.as_dict(),
        "manifest_sha256": _sha256_file(manifest_path),
        "input_identities": input_identities,
        "code_identity": code_identity,
        "acquisition_audit_sha256": (
            _canonical_hash(acquisition_audit) if acquisition_audit is not None else None
        ),
    }
    run_fingerprint = _prepare_run_directory(output_dir, run_identity)
    checkpoint_dir = output_dir / "checkpoints"
    if not resume and checkpoint_dir.is_dir() and any(checkpoint_dir.iterdir()):
        raise ValueError(
            "--no-resume cannot reuse an output directory containing checkpoints; "
            "use a new output directory"
        )
    if acquisition_audit is not None:
        _write_text(output_dir / "acquisition_audit.snapshot.json", json.dumps(
            acquisition_audit, ensure_ascii=False, indent=2, sort_keys=True,
            allow_nan=False,
        ) + "\n")
    runtime_provenance = _runtime_provenance(code_identity)
    model = ESpinSLAM(config)
    results: list[dict[str, Any]] = []
    references: dict[tuple[str, float], np.ndarray] = {}
    for index, row in enumerate(rows, 1):
        checkpoint = output_dir / "checkpoints" / f"{row['sequence_id']}.json"
        fingerprint = _checkpoint_fingerprint(
            config, row, setting_name,
            input_identity=input_identities[row["sequence_id"]],
            code_identity=code_identity,
        )
        if resume and checkpoint.is_file():
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            if payload.get("fingerprint") != fingerprint:
                raise ValueError(
                    f"{row['sequence_id']}: checkpoint belongs to a different config/manifest"
                )
            _validate_checkpoint_artifacts(output_dir, payload)
            results.append(_validate_checkpoint_record(
                payload, sequence_id=row["sequence_id"],
                run_fingerprint=run_fingerprint, setting_name=setting_name,
            ))
            print(f"[{setting_name}] [{index}/{len(rows)}] {row['sequence_id']}: resumed", flush=True)
            continue
        stale_candidates = [
            output_dir / "history" / f"{row['sequence_id']}_optimizations.csv",
            output_dir / "history" / f"{row['sequence_id']}_frequency.csv",
            output_dir / f"fig8_{row['sequence_id']}.png",
            output_dir / f"reconstruction_{row['sequence_id']}.png",
            output_dir / "pointclouds" / f"{row['sequence_id']}.npz",
        ]
        stale = [path for path in stale_candidates if path.exists()]
        if stale:
            raise ValueError(
                f"{row['sequence_id']}: unclaimed stale scene artifacts exist; "
                "use a new output directory"
            )
        camera = camera_for_row(config.camera, row)
        width = int(row["width"]) if row.get("width") else config.data.sensor_size[0]
        height = int(row["height"]) if row.get("height") else config.data.sensor_size[1]
        sequence = load_events_as_sequence(
            root / row["event_path"], row["sequence_id"], width=width, height=height,
            time_unit=row.get("time_unit") or config.data.time_unit,
            text_columns=config.data.text_columns,
            max_duration_seconds=config.data.max_duration_seconds,
        )
        result = model.process(sequence, camera=camera)
        icp_rmse = chamfer = None
        icp_reason, scale_source = "", ""
        cad_source = str(row.get("cad_source") or "").strip()
        cad_authoritative = _is_authoritative(cad_source)
        if row.get("cad_path") and cad_authoritative:
            cad_path = (root / row["cad_path"]).resolve()
            if not cad_path.is_relative_to(root):
                raise ValueError(f"{row['sequence_id']}: cad_path resolves outside data root")
            unit = _float_or_none(row.get("cad_unit_mm")) or 1.0
            key = (row["cad_path"], unit)
            if key not in references:
                references[key] = load_reference_cloud(
                    cad_path,
                    config.evaluation.pointcloud_max_reference_points, unit,
                )
            orbit_mm = _float_or_none(row.get("orbit_radius_mm"))
            known_scale = orbit_mm / config.backend.orbit_radius if orbit_mm else None
            scale_source = "orbit_radius_mm" if known_scale else "similarity_icp"
            if result.reconstruction.valid:
                alignment = align_similarity_icp(result.reconstruction.points, references[key],
                                                 known_scale=known_scale)
                icp_rmse = _nullable(alignment.icp_rmse) if alignment.valid else None
                chamfer = _nullable(alignment.chamfer_rmse_m) if alignment.valid else None
                icp_reason = alignment.reason
                if alignment.valid:
                    (output_dir / "pointclouds").mkdir(exist_ok=True)
                    np.savez_compressed(
                        output_dir / "pointclouds" / f"{sequence.sequence_id}.npz",
                        predicted=result.reconstruction.points.astype(np.float32),
                        aligned_mm=alignment.aligned_points.astype(np.float32),
                    )
            else:
                icp_reason = result.reconstruction.reason
        elif row.get("cad_path"):
            icp_reason = "cad_path present but cad_source is not authoritative"
        else:
            icp_reason = "authoritative CAD unavailable in public TOPSPIN release"
        record = _sequence_row(row, result, icp_rmse_mm=icp_rmse, chamfer_mm=chamfer,
                               icp_reason=icp_reason, scale_source=scale_source)
        record["cad_source"] = cad_source
        record["cad_authoritative"] = cad_authoritative
        record["evaluation_classification"] = classification
        record["method_profile"] = method_profile
        record["setting"] = setting_name
        record["paper_comparison_eligible"] = paper_comparison_eligible
        record["run_fingerprint"] = run_fingerprint
        # Store the execution host with the scene, not only in the final
        # aggregate: checkpoints may be resumed on a different machine.
        record["execution_platform"] = runtime_provenance["platform"]
        record["execution_hostname"] = runtime_provenance["hostname"]
        record["execution_machine"] = runtime_provenance["machine"]
        record["execution_processor"] = runtime_provenance["processor"]
        record["execution_python"] = runtime_provenance["python"]
        record["execution_python_executable"] = runtime_provenance["python_executable"]
        record["execution_logical_cpu_count"] = runtime_provenance["logical_cpu_count"]
        record["execution_memory_bytes"] = runtime_provenance["memory_bytes"]
        record["execution_gpu"] = runtime_provenance["gpu"]
        record["execution_git_revision"] = runtime_provenance["git_revision"]
        record["execution_git_dirty"] = runtime_provenance["git_dirty"]
        record["execution_packages_json"] = json.dumps(
            runtime_provenance["packages"], sort_keys=True, allow_nan=False)
        results.append(record)
        history_dir = output_dir / "history"
        optimization_history = history_dir / f"{sequence.sequence_id}_optimizations.csv"
        frequency_history = history_dir / f"{sequence.sequence_id}_frequency.csv"
        _write_csv(optimization_history, [
            {"event_time_s": r.event_time, "frequency_hz": r.frequency_hz,
             "landmarks": r.landmarks, "observations": r.observations,
             "mean_observation_error_px": _nullable(
                 getattr(r, "mean_observation_error_px", None)),
             "mean_landmark_reprojection_error_px": _nullable(
                 r.mean_reprojection_error_px),
             "removed_short": getattr(r, "removed_short", None),
             "removed_reprojection": r.removed_reprojection, "removed_radius": r.removed_radius,
             "fused": r.fused, "iterations": r.iterations,
             "termination_status": getattr(r, "termination_status", None)}
            for r in result.optimizations
        ] or [{"event_time_s": None}])
        _write_csv(frequency_history, [
            {"event_time_s": r.event_time, "wall_s": r.wall_seconds,
             "frequency_hz": _nullable(r.frequency_hz), "loop_closed": r.loop_closed,
             "period_seconds": _nullable(r.period_seconds),
             "alignment_cost": _nullable(r.alignment_cost),
             "grid_seed_seconds": _nullable(r.grid_seed_seconds),
             "grid_seed_cost": _nullable(r.grid_seed_cost),
             "icp_iterations_used": r.icp_iterations_used,
             "icp_termination": r.icp_termination,
             "icp_clipped_to_search_interval": r.icp_clipped_to_search_interval}
            for r in result.frequency_history
        ] or [{"event_time_s": None}])
        scene_artifacts = [optimization_history, frequency_history]
        if history_plots:
            history_plot = output_dir / f"fig8_{sequence.sequence_id}.png"
            reconstruction_plot = output_dir / f"reconstruction_{sequence.sequence_id}.png"
            _plot_history(result, history_plot)
            _plot_reconstruction(result, config.backend.orbit_radius, reconstruction_plot)
            scene_artifacts.extend((history_plot, reconstruction_plot))
        pointcloud = output_dir / "pointclouds" / f"{sequence.sequence_id}.npz"
        if pointcloud.is_file():
            scene_artifacts.append(pointcloud)
        artifacts = _artifact_identities(output_dir, scene_artifacts)
        _write_text(
            checkpoint,
            json.dumps({"fingerprint": fingerprint, "record": record,
                        "record_sha256": _canonical_hash(record),
                        "artifacts": artifacts},
                       ensure_ascii=False, indent=2, allow_nan=False,
                       default=_json_default) + "\n",
        )
        # A readable partial aggregate survives interruption even before final
        # tables/figures are generated.
        _write_csv(output_dir / "per_sequence.partial.csv", results)
        print(
            f"[{setting_name}] [{index}/{len(rows)}] {sequence.sequence_id}: "
            f"f={record['frequency_est_hz']} loop={result.loop_closed} "
            f"reproj={record['reprojection_error_px']} icp_mm={icp_rmse} "
            f"throughput={record['throughput_ev_s']}",
            flush=True,
        )
    _write_csv(output_dir / "per_sequence.csv", results)
    # Split keeps dev decisions separate, but does not make previously viewed
    # public TOPSPIN scenes confirmatory or genuinely held-out.
    has_split = any(r["split"] for r in results)
    primary = [r for r in results if r["split"] == "test"] if has_split else results
    table = table2(primary)
    _write_csv(output_dir / "table2.csv", table)
    scope_suffix = "non-dev exploratory" if has_split else "all selected scenes"
    label = f"{setting_name}, {method_profile}, {classification}, {scope_suffix}"
    _write_text(output_dir / "table2.md", table2_markdown(table, label))
    extra_tables: dict[str, list[dict[str, Any]]] = {}
    if has_split:
        for name, subset in (("dev", [r for r in results if r["split"] == "dev"]),
                             ("all", results)):
            extra_tables[name] = table2(subset)
            _write_csv(output_dir / f"table2_{name}.csv", extra_tables[name])
            _write_text(output_dir / f"table2_{name}.md",
                        table2_markdown(extra_tables[name], f"{setting_name}, {name}"))
    # The full method is compared with Table 2; ablation settings with Table 3.
    paper_reference = (PAPER_TABLE2 if setting_name == "eSpinSLAM"
                       else PAPER_TABLE3.get(setting_name)) if paper_comparison_eligible else None
    comparison_table = None
    if paper_reference is not None:
        comparison_table = extra_tables.get("all", table)
        _write_text(output_dir / "table_vs_paper.md",
                    comparison_markdown(
                        comparison_table, paper_reference,
                        f"{setting_name}, all 72, exploratory/retrospective",
                    ))
    if any(r["frequency_est_hz"] is not None for r in primary):
        _plot_frequency_cdf(primary, output_dir / "fig6_frequency_cdf.png")
    if primary:
        _plot_throughput(primary, output_dir / "fig7_throughput.png")
        _plot_throughput_summary(primary, output_dir / "fig5_throughput_summary.png")
    frequency_errors = [r["frequency_rel_error_percent"] for r in primary
                        if r["frequency_rel_error_percent"] is not None]
    by_speed = {}
    consistency_deviations = []
    for speed in sorted({r["speed"] for r in primary}):
        values = [r["frequency_est_hz"] for r in primary
                  if r["speed"] == speed and r["frequency_est_hz"] is not None]
        if values:
            median = float(np.median(values))
            deviations = ([abs(value - median) / median * 100.0 for value in values]
                          if median else [])
            consistency_deviations.extend(deviations)
            by_speed[speed] = {"count": len(values), "median_hz": median,
                               "std_hz": float(np.std(values)),
                               "min_hz": float(np.min(values)), "max_hz": float(np.max(values)),
                               "median_absolute_deviation_from_speed_median_percent": (
                                   float(np.median(deviations)) if deviations else None),
                               }
    report = {
        "method": "eSpinSLAM reproduction candidate",
        "paper": "Event-based Spinning Object SLAM, ICCVW 2025",
        "setting": setting_name,
        "dataset": "TOPSPIN",
        "method_profile": method_profile,
        "evaluation_classification": classification,
        "paper_comparison_requested": paper_comparison,
        "paper_comparison_eligible": paper_comparison_eligible,
        "paper_comparison_ineligibility_reasons": comparison_ineligibility_reasons,
        "run_fingerprint": run_fingerprint,
        "reference_config_mismatches": reference_mismatches,
        "evidence_scope": "exploratory_retrospective",
        "public_data_limitations": {
            "authoritative_frequency_ground_truth_count": sum(
                bool(r["frequency_reference_authoritative"]) for r in results),
            "authoritative_frequency_ground_truth_total": len(results),
            "authoritative_frequency_ground_truth_all": bool(results) and all(
                bool(r["frequency_reference_authoritative"]) for r in results),
            "authoritative_cad_count": sum(bool(r["cad_authoritative"]) for r in results),
            "authoritative_cad_total": len(results),
            "authoritative_cad_all": bool(results) and all(
                bool(r["cad_authoritative"]) for r in results),
            "note": (
                "Public TOPSPIN lacks authoritative frequency labels and CAD. "
                "Null true-error/ICP fields are unavailable, not zero."
            ),
        },
        "sequence_count": len(results),
        "paper_metric_eligible_sequence_count": sum(
            bool(r["paper_metrics_eligible"]) for r in primary),
        "geometrically_valid_sequence_count": sum(
            bool(r["reconstruction_valid"]) for r in primary),
        "split_counts": {name: sum(r["split"] == name for r in results)
                         for name in ("dev", "test")} if has_split else None,
        "summary_scope": "test-labelled non-dev exploratory scenes" if has_split else "all scenes",
        "loop_closure_rate": (sum(bool(r["loop_closed"]) for r in primary) / len(primary)
                              if primary else None),
        "frequency_authoritative_median_relative_error_percent": (
            float(np.median(frequency_errors)) if frequency_errors else None),
        "frequency_same_speed_consistency_median_absolute_deviation_percent": (
            float(np.median(consistency_deviations)) if consistency_deviations else None),
        "frequency_by_speed_hz": by_speed,
        "mean_throughput_ev_s": _mean_std([r["throughput_ev_s"] for r in primary])[0],
        "mean_frontend_throughput_ev_s": _mean_std(
            [r["frontend_throughput_ev_s"] for r in primary])[0],
        "mean_scene_event_rate_ev_s": _mean_std([r["scene_event_rate_ev_s"] for r in primary])[0],
        "real_time_rate": (sum(bool(r["real_time"]) for r in primary) / len(primary)
                           if primary else None),
        "table2": table,
        "table2_dev": extra_tables.get("dev"),
        "table2_all": extra_tables.get("all"),
        "paper_comparison_table_all72": comparison_table,
        "runtime_seconds": float(sum(
            float(r["runtime_s"]) for r in results if r.get("runtime_s") is not None
        )),
        "scene_runtime_sum_seconds": float(sum(
            float(r["runtime_s"]) for r in results if r.get("runtime_s") is not None
        )),
        "invocation_wall_seconds": time.monotonic() - started,
        "runtime_provenance": runtime_provenance,
        "implementation_notes": [
            "The headline/Table-2 reprojection metric is the mean over every inlier "
            "observation; the equally weighted landmark mean is reported separately.",
            "A geometrically non-empty reconstruction whose last optimizer status is "
            "max_iterations is preserved per scene but excluded from Table-2 aggregates. "
            "Direct-GTSAM converged/stalled vocabulary is not fully equivalent, so this "
            "gate deliberately guarantees only the unambiguous max_iterations exclusion.",
            "Reference Eq. (9) uses every retained left event against every right event, "
            "sum Euclidean distance, with no overlap mask or subsampling.",
            "The in-tree LM passed the preregistered direct-GTSAM equivalence gate; "
            "backend.solver_backend records the selected path.",
            "No terminal offline BA or paper-unreported parallax guard is used in the "
            "paper_reference classification.",
            "True frequency error and CAD ICP millimetres require explicit authoritative "
            "manifest provenance; public TOPSPIN supplies neither.",
        ],
        "config": config.as_dict(),
    }
    _write_text(output_dir / "metrics.json",
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False,
                           default=_json_default) + "\n")
    return report


def _with_overrides(config: ESpinSLAMConfig, overrides: dict[str, dict[str, Any]]) -> ESpinSLAMConfig:
    result = copy.deepcopy(config)
    for section, values in overrides.items():
        setattr(result, section, replace(getattr(result, section), **values))
    return result


def evaluate_topspin_ablations(config: ESpinSLAMConfig, data_root: Path, manifest_path: Path,
                               output_dir: Path, *, settings: list[str] | None = None,
                               max_sequences: int = 0,
                               sequence_ids: list[str] | tuple[str, ...] | None = None,
                               resume: bool = True,
                               require_reference: bool = False,
                               require_verified_data: bool = False,
                               paper_comparison: bool = False) -> dict[str, Any]:
    """Run the Table 3 ablations and the full method, then write table3.*."""

    settings = settings or list(ABLATIONS)
    if paper_comparison and not any(
        ABLATIONS[key][0] in PAPER_COMPARABLE_SETTINGS for key in settings
    ):
        raise ValueError(
            "paper comparison was requested, but none of the selected settings has "
            "a paper-faithful executable audit path"
        )
    _prepare_run_directory(output_dir, {
        "protocol_schema": PROTOCOL_SCHEMA,
        "kind": "table3_ablations",
        "settings": settings,
        "config": config.as_dict(),
        "manifest_sha256": _sha256_file(Path(manifest_path).resolve()),
        "sequence_ids": list(sequence_ids) if sequence_ids is not None else None,
        "max_sequences": max_sequences,
        "require_verified_data": require_verified_data,
        "paper_comparison": paper_comparison,
        "code_identity": _code_identity(),
    })
    reports: dict[str, dict[str, Any]] = {}
    for key in settings:
        label, overrides = ABLATIONS[key]
        reports[key] = evaluate_topspin(
            _with_overrides(config, overrides), data_root, manifest_path,
            output_dir / key, max_sequences=max_sequences, setting_name=label,
            history_plots=False, sequence_ids=sequence_ids, resume=resume,
            require_reference=require_reference,
            require_verified_data=(require_verified_data or paper_comparison),
            paper_comparison=(paper_comparison and label in PAPER_COMPARABLE_SETTINGS),
        )
    rows = []
    nondev_rows = []
    for key in settings:
        selected_table = reports[key]["table2"]
        if paper_comparison:
            selected_table = reports[key].get("table2_all") or selected_table
        for entry in selected_table:
            rows.append({"setting": ABLATIONS[key][0], **entry})
        if paper_comparison and reports[key].get("table2_all") is not None:
            for entry in reports[key]["table2"]:
                nondev_rows.append({"setting": ABLATIONS[key][0], **entry})
    _write_csv(output_dir / "table3.csv", rows)
    objects = []
    for row in rows:
        if row["object"] not in objects:
            objects.append(row["object"])
    def table3_markdown(source_rows: list[dict[str, Any]], scope: str) -> str:
        lines = [f"# Table 3 ({scope})", "",
                 "| Setting | " + " | ".join(objects) + " |",
                 "|---|" + "---|" * len(objects)]
        for metric, mean_key, std_key in (
            ("Reprojection error (px)", "reprojection_mean_px", "reprojection_std_px"),
            ("ICP alignment RMSE (mm)", "icp_mean_mm", "icp_std_mm"),
        ):
            lines.append(f"| **{metric}** |" + " |" * len(objects))
            for key in settings:
                label = ABLATIONS[key][0]
                cells = []
                for name in objects:
                    match = [r for r in source_rows
                             if r["setting"] == label and r["object"] == name]
                    cells.append(_format(match[0][mean_key], match[0][std_key]) if match else "-")
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"

    _write_text(output_dir / "table3.md", table3_markdown(
        rows, "all 72 exploratory/retrospective scenes" if paper_comparison
        else "selected non-dev exploratory scenes",
    ))
    if nondev_rows:
        _write_csv(output_dir / "table3_nondev69.csv", nondev_rows)
        _write_text(output_dir / "table3_nondev69.md", table3_markdown(
            nondev_rows, "69 non-dev exploratory scenes",
        ))
    comparison = []
    for key in settings:
        label = ABLATIONS[key][0]
        if (label in PAPER_TABLE3 and reports[key]["paper_comparison_eligible"]):
            comparison.append(comparison_markdown(
                reports[key]["paper_comparison_table_all72"],
                PAPER_TABLE3[label], f"{label}, all 72, exploratory/retrospective",
            ))
    if comparison:
        _write_text(output_dir / "table3_vs_paper.md", "\n".join(comparison))
    if "espinslam" in settings and "fft_frequency" in settings:
        series = {}
        for key in ("espinslam", "fft_frequency"):
            with (output_dir / key / "per_sequence.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
                if any(row.get("split") for row in rows):
                    rows = [row for row in rows if row.get("split") == "test"]
                series[ABLATIONS[key][0]] = [
                    {"speed": row["speed"],
                     "frequency_est_hz": _float_or_none(row["frequency_est_hz"]),
                     "frequency_true_hz": _float_or_none(row["frequency_true_hz"])}
                    for row in rows
                ]
        plot_frequency_cdf_comparison(series, output_dir / "fig6_loop_closure_vs_fft.png")
    return {"settings": settings, "reports": reports}
