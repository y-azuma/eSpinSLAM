#!/usr/bin/env python3
"""Generate a small analytic static-camera, fixed-axis eSpinSLAM diagnostic set."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from espinslam.geometry import camera_pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sequences", type=int, default=18)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--event-rate-hz", type=float, default=400.0)
    parser.add_argument("--seed", type=int, default=3011)
    return parser.parse_args()


def _landmarks() -> np.ndarray:
    """Return one asymmetric satellite-like point set in the target frame."""

    rng = np.random.default_rng(104729)
    body = rng.uniform((-0.85, -0.55, -0.50), (0.85, 0.55, 0.50), size=(44, 3))
    left = rng.uniform((-2.55, -0.08, -0.42), (-0.90, 0.08, 0.42), size=(20, 3))
    right = rng.uniform((0.90, -0.08, -0.30), (2.10, 0.08, 0.52), size=(16, 3))
    antenna = rng.uniform((0.20, -0.18, 0.50), (0.48, 0.18, 1.25), size=(8, 3))
    return np.concatenate((body, left, right, antenna), axis=0).astype(np.float64)


def _project(points: np.ndarray, timestamp: float, frequency_hz: float,
             intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation, center = camera_pose(
        timestamp, frequency_hz, orbit_radius=10.0,
        rotation_vector=np.asarray((0.10, -0.07, 0.04)),
    )
    camera = (rotation @ (points - center).T).T
    visible = camera[:, 2] > 1e-6
    pixels = camera @ intrinsic.T
    pixels[:, :2] /= pixels[:, 2:3]
    return pixels[:, :2], visible


def _generate_sequence(points: np.ndarray, frequency_hz: float, duration: float,
                       event_rate: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    intrinsic = np.asarray(((528.6111111111, 0.0, 173.0),
                            (0.0, 528.6111111111, 130.0),
                            (0.0, 0.0, 1.0)), dtype=np.float64)
    times = np.arange(0.0, duration, 1.0 / event_rate, dtype=np.float64)
    records: list[np.ndarray] = []
    for timestamp in times:
        pixels, visible = _project(points, float(timestamp), frequency_hz, intrinsic)
        inside = (visible & (pixels[:, 0] >= 2.0) & (pixels[:, 0] < 344.0)
                  & (pixels[:, 1] >= 2.0) & (pixels[:, 1] < 258.0))
        indices = np.flatnonzero(inside)
        if not indices.size:
            continue
        keep = rng.random(indices.size) >= 0.025
        indices = indices[keep]
        if not indices.size:
            continue
        noisy = pixels[indices] + rng.normal(0.0, 0.12, size=(indices.size, 2))
        polarity = ((indices + int(timestamp * frequency_hz * 4.0)) & 1).astype(np.float64)
        records.append(np.column_stack((
            np.full(indices.size, timestamp), noisy[:, 0], noisy[:, 1], polarity,
        )))
    values = np.concatenate(records, axis=0)
    clutter_count = max(1, int(0.005 * values.shape[0]))
    clutter = np.column_stack((
        rng.uniform(0.0, duration, clutter_count),
        rng.uniform(0.0, 345.0, clutter_count),
        rng.uniform(0.0, 259.0, clutter_count),
        rng.integers(0, 2, clutter_count),
    ))
    values = np.concatenate((values, clutter), axis=0)
    values = values[np.argsort(values[:, 0], kind="stable")]
    return {
        "t": values[:, 0].astype(np.float64),
        "x": np.clip(np.rint(values[:, 1]), 0, 345).astype(np.int16),
        "y": np.clip(np.rint(values[:, 2]), 0, 259).astype(np.int16),
        "p": values[:, 3].astype(np.int8),
        "omega_rad_s": np.asarray(2.0 * math.pi * frequency_hz, dtype=np.float32),
    }


def main() -> None:
    args = parse_args()
    if args.sequences < 4 or args.duration_seconds <= 0 or args.event_rate_hz <= 0:
        raise ValueError("sequences >= 4 and positive duration/event rate are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    points = _landmarks()
    frequencies = rng.uniform(100.0 / 360.0, 200.0 / 360.0, args.sequences)
    assignment = np.asarray(["val"] * (args.sequences // 2)
                            + ["test"] * (args.sequences - args.sequences // 2))
    rng.shuffle(assignment)
    rows: list[dict[str, str]] = []
    for index, (frequency, split) in enumerate(zip(frequencies, assignment)):
        name = f"paper_static_{index:03d}"
        archive = _generate_sequence(
            points, float(frequency), args.duration_seconds, args.event_rate_hz, rng,
        )
        np.savez_compressed(args.output_dir / f"{name}.npz", **archive)
        rows.append({"sequence_id": name, "event_path": f"{name}.npz", "split": str(split)})
        print(f"[{index + 1}/{args.sequences}] {name} {split} {frequency:.6f} Hz", flush=True)
    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sequence_id", "event_path", "split"))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output_dir / "reference_points.npz", points=points.astype(np.float32))
    metadata = {
        "purpose": "paper-geometry diagnostic only; never replaces the target evaluation set",
        "camera": "static camera in the physical interpretation",
        "dual_geometry": "fixed target and circular virtual camera orbit",
        "rotation_axis": [0.0, 0.0, 1.0],
        "constant_speed": True,
        "frequency_range_hz": [100.0 / 360.0, 200.0 / 360.0],
        "duration_seconds": args.duration_seconds,
        "event_rate_hz": args.event_rate_hz,
        "sequence_count": args.sequences,
        "seed": args.seed,
        "photorealistic": False,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
