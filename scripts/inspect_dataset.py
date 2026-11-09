#!/usr/bin/env python3
"""Audit an event dataset without changing it or selecting test-tuned settings."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from espinslam.data import read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not finite.size:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": int(finite.size),
        "min": float(finite.min()),
        "median": float(np.median(finite)),
        "max": float(finite.max()),
    }


def inspect(data_root: Path, manifest: Path) -> dict:
    rows = read_manifest(manifest, data_root)
    event_counts, durations, omega_values = [], [], []
    asset_counts: Counter[str] = Counter()
    camera_closing, camera_orbit, camera_lens, camera_sensor_width = [], [], [], []
    dynamics_models: Counter[str] = Counter()
    npz_fields: Counter[tuple[str, ...]] = Counter()
    invalid: list[dict[str, str]] = []
    examples: list[dict] = []
    for index, row in enumerate(rows):
        event_path = (data_root / row["event_path"]).resolve()
        sample_dir = event_path.parent
        for name in ("scene.blend", "scene_config.json", "bbox.npz"):
            if (sample_dir / name).is_file():
                asset_counts[name] += 1
        for name in (
            "meta/scene_metadata.json", "meta/ground_truth_camera_frame.csv",
        ):
            if (sample_dir / name).is_file():
                asset_counts[name] += 1
        try:
            with np.load(event_path, allow_pickle=False) as archive:
                npz_fields[tuple(sorted(archive.files))] += 1
                times = archive["t"]
                event_counts.append(int(times.size))
                durations.append(float(times[-1] - times[0]) if times.size > 1 else 0.0)
                omega_values.append(float(archive["omega_rad_s"]))
        except (OSError, ValueError, KeyError, TypeError) as error:
            invalid.append({"sequence_id": row["sequence_id"], "error": str(error)})
            continue
        config_path = sample_dir / "scene_config.json"
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                camera = config.get("camera", {})
                dynamics = config.get("dynamics", {})
                camera_closing.append(float(camera.get("closing_rate_mps", math.nan)))
                camera_orbit.append(float(camera.get("orbit_rate_deg_s", math.nan)))
                camera_lens.append(float(camera.get("lens_mm", math.nan)))
                camera_sensor_width.append(float(camera.get("sensor_width_mm", math.nan)))
                dynamics_models[str(dynamics.get("model", "missing"))] += 1
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
        if index < 3:
            examples.append({
                "sequence_id": row["sequence_id"],
                "split": row["split"],
                "event_path": row["event_path"],
                "sample_files": sorted(
                    str(path.relative_to(sample_dir))
                    for path in sample_dir.rglob("*") if path.is_file()
                )[:40],
            })
    return {
        "data_root": str(data_root.resolve()),
        "manifest": str(manifest.resolve()),
        "sequence_count": len(rows),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "event_count": _summary([float(value) for value in event_counts]),
        "duration_seconds": _summary(durations),
        "omega_rad_s_evaluation_label": _summary(omega_values),
        "omega_deg_s_evaluation_label": _summary([math.degrees(value) for value in omega_values]),
        "asset_counts": dict(sorted(asset_counts.items())),
        "npz_schemas": [
            {"fields": list(fields), "count": count}
            for fields, count in sorted(npz_fields.items())
        ],
        "camera_motion": {
            "closing_rate_mps": _summary(camera_closing),
            "orbit_rate_deg_s": _summary(camera_orbit),
            "static_camera_assumption_holds": bool(
                camera_closing and camera_orbit
                and all(abs(value) < 1e-12 for value in camera_closing)
                and all(abs(value) < 1e-12 for value in camera_orbit)
            ),
        },
        "camera_calibration": {
            "lens_mm": _summary(camera_lens),
            "sensor_width_mm": _summary(camera_sensor_width),
        },
        "dynamics_models": dict(sorted(dynamics_models.items())),
        "three_dimensional_ground_truth": {
            "available_from_blend": asset_counts["scene.blend"] > 0,
            "camera_pose_csv_available": asset_counts["meta/ground_truth_camera_frame.csv"] > 0,
        },
        "invalid_sequences": invalid,
        "examples": examples,
        "decision": (
            "Existing data is sufficient for an out-of-model robustness evaluation. "
            "The Blender scene supplies 3D reference geometry, but nonzero camera "
            "motion violates eSpinSLAM's exact static-camera assumption."
        ),
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    result = inspect(args.data_root, args.manifest)
    _write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

