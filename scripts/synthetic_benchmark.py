#!/usr/bin/env python3
"""Verify the whole pipeline on synthetic streams that satisfy the paper's model.

A static camera (elevated, aimed at the spin axis, so R0 is a real tilt)
observes a rigid object spinning at constant f. The script reports, per scene
and per setting (full method and Table 3 ablations): estimated f, loop-closure
time, R0 error, structure error (distance from each landmark to the nearest
ground-truth patch corner, same world frame, no alignment; the object radius
is 1.5 world units with r = 10), reprojection error, and throughput.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from espinslam.config import CameraConfig, ESpinSLAMConfig, load_config
from espinslam.data import EventSequence
from espinslam.pipeline import ESpinSLAM
from espinslam.synthetic import make_scene, simulate_events
from espinslam.topspin import ABLATIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path,
                        help="optional base config (camera/sensor are replaced)")
    parser.add_argument("--output", type=Path, default=Path("runs/synthetic_benchmark.json"))
    parser.add_argument("--scenes", default="0.5:20,0.3:10,0.8:35",
                        help="comma-separated frequency_hz:elevation_deg pairs")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="seconds per scene (default: 3 revolutions + 3 s, enough for "
                             "loop detection with aleph = 20 updates)")
    parser.add_argument("--noise-rate", type=float, default=0.05,
                        help="background noise events per pixel per second")
    parser.add_argument("--ablations", action="store_true", help="also run Table 3 settings")
    return parser.parse_args()


def run_scene(config: ESpinSLAMConfig, frequency: float, elevation: float,
              duration: float, noise_rate: float, seed: int) -> dict:
    scene = make_scene(frequency_hz=frequency, elevation_deg=elevation, seed=seed)
    stream = simulate_events(scene, duration_seconds=duration,
                             noise_rate_hz_per_pixel=noise_rate, seed=seed)
    config = copy.deepcopy(config)
    k = scene.intrinsic
    config.camera = CameraConfig(fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2])
    sequence = EventSequence(f"f{frequency}_e{elevation}", stream.x, stream.y, stream.t,
                             stream.p, 2 * math.pi * frequency, scene.width, scene.height,
                             {}, Path("synthetic"))
    result = ESpinSLAM(config).process(sequence)
    record = {
        "scene": sequence.sequence_id, "frequency_true_hz": frequency,
        "elevation_deg": elevation, "events": sequence.event_count,
        "frequency_est_hz": result.frequency_hz if result.valid else None,
        "frequency_rel_error_percent": (abs(result.frequency_hz - frequency) / frequency * 100
                                        if result.valid else None),
        "loop_closure_time_s": result.loop_closure_time if result.loop_closed else None,
        "throughput_ev_s": result.throughput_events_per_second,
    }
    reconstruction = result.reconstruction
    if reconstruction.valid:
        rotation_error = np.degrees(np.linalg.norm(
            Rotation.from_matrix(reconstruction.rotation @ scene.rotation.T).as_rotvec()))
        distance, _ = cKDTree(scene.feature_points).query(reconstruction.points)
        record.update({
            "landmarks": int(reconstruction.points.shape[0]),
            "reprojection_error_px": reconstruction.mean_reprojection_error_px,
            "r0_error_deg": float(rotation_error),
            "structure_error_median": float(np.median(distance)),
            "structure_error_p90": float(np.percentile(distance, 90)),
        })
    else:
        record["failure"] = reconstruction.reason
    return record


def main() -> None:
    args = parse_args()
    base = load_config(args.config) if args.config else ESpinSLAMConfig()
    base.data = replace(base.data, max_duration_seconds=None)
    settings = {"espinslam": ("eSpinSLAM", {})}
    if args.ablations:
        settings = ABLATIONS
    records = []
    for index, item in enumerate(args.scenes.split(",")):
        frequency, elevation = (float(value) for value in item.split(":"))
        duration = args.duration or 3.0 / frequency + 3.0
        for key, (label, overrides) in settings.items():
            config = copy.deepcopy(base)
            for section, values in overrides.items():
                setattr(config, section, replace(getattr(config, section), **values))
            record = run_scene(config, frequency, elevation, duration, args.noise_rate, index)
            record["setting"] = label
            records.append(record)
            print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
