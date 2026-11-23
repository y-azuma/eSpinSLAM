#!/usr/bin/env python3
"""Set the front-end parameters the paper does not give, on the dev scenes only.

Pre-registered protocol (docs/reproduction_report.md, Sec. 13), fixed before
any tuning run:

* Scenes: the ``split=dev`` rows of the manifest (examples/splits/topspin_dev.txt).
* Parameters: only front-end settings without a value in the paper (Table 1
  values are never changed): HDBSCAN ``cluster_min_size``/``cluster_min_samples``,
  tracklet association radius and time gap, and the density radius lambda.
* Search: three stages, each starting from the previous stage's choice
  (``STAGES`` below).
* Score (no ground truth is used): the sum over dev scenes of the median
  number of landmarks that pass the paper's outlier tests (Sec. 4.2.5) over
  the optimisations after loop closure.
* Admissible: loop closure in every dev scene, and a mean reprojection error
  (averaged over dev scenes) no worse than that of the starting configuration.
* Choice: the admissible candidate with the highest score; the starting
  configuration wins ties.

Results are cached per candidate in ``runs.jsonl`` so the script can resume.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import numpy as np

from espinslam.config import ESpinSLAMConfig, load_config
from espinslam.data import load_events_as_sequence
from espinslam.pipeline import ESpinSLAM
from espinslam.topspin import camera_for_row, read_topspin_manifest

STAGES = (
    ("clustering", [
        {"cluster_min_size": 5, "cluster_min_samples": 3},
        {"cluster_min_size": 3, "cluster_min_samples": 2},
        {"cluster_min_size": 3, "cluster_min_samples": 1},
        {"cluster_min_size": 2, "cluster_min_samples": 1},
    ]),
    ("association", [
        {"association_radius_px": radius, "association_max_gap_seconds": gap}
        for radius in (5.0, 10.0, 20.0) for gap in (0.05, 0.1, 0.2)
    ]),
    ("density", [{"density_radius_px": radius} for radius in (3.0, 5.0, 8.0)]),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _with_frontend(config: ESpinSLAMConfig, values: dict) -> ESpinSLAMConfig:
    result = copy.deepcopy(config)
    result.frontend = replace(result.frontend, **values)
    return result


def run_scene(config: ESpinSLAMConfig, root: Path, row: dict) -> dict:
    sequence = load_events_as_sequence(
        root / row["event_path"], row["sequence_id"],
        width=int(row["width"]) if row.get("width") else config.data.sensor_size[0],
        height=int(row["height"]) if row.get("height") else config.data.sensor_size[1],
        time_unit=row.get("time_unit") or config.data.time_unit,
    )
    result = ESpinSLAM(config).process(sequence, camera=camera_for_row(config.camera, row))
    landmarks = [report.landmarks for report in result.optimizations]
    return {
        "scene": row["sequence_id"],
        "loop_closed": result.loop_closed,
        "frequency_hz": result.frequency_hz if result.valid else None,
        "median_landmarks": float(np.median(landmarks)) if landmarks else 0.0,
        "final_landmarks": int(result.reconstruction.points.shape[0]),
        "reprojection_px": (result.reconstruction.mean_reprojection_error_px
                            if result.reconstruction.valid else None),
        "runtime_s": result.runtime_seconds,
    }


def summarise(scenes: list[dict]) -> dict:
    errors = [scene["reprojection_px"] for scene in scenes]
    return {
        "score": float(sum(scene["median_landmarks"] for scene in scenes)),
        "all_loops": all(scene["loop_closed"] for scene in scenes),
        "mean_reprojection_px": (float(np.mean(errors)) if all(e is not None for e in errors)
                                 else math.inf),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = load_config(args.config)
    root = args.data_root.resolve()
    rows = [row for row in read_topspin_manifest(args.manifest, root) if row.get("split") == "dev"]
    if not rows:
        raise ValueError("the manifest has no split=dev rows")
    cache_path = args.output_dir / "runs.jsonl"
    cache = {}
    if cache_path.is_file():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            cache[entry["key"]] = entry

    def evaluate(frontend: dict) -> dict:
        key = json.dumps(frontend, sort_keys=True)
        if key not in cache:
            started = time.monotonic()
            config = _with_frontend(base, frontend)
            scenes = [run_scene(config, root, row) for row in rows]
            entry = {"key": key, "frontend": frontend, "scenes": scenes, **summarise(scenes),
                     "wall_s": time.monotonic() - started}
            cache[key] = entry
            with cache_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry) + "\n")
            print(f"  {frontend}: score {entry['score']:.1f}, loops {entry['all_loops']}, "
                  f"reproj {entry['mean_reprojection_px']:.2f} px "
                  f"({entry['wall_s']:.0f} s)", flush=True)
        return cache[key]

    current = {name: getattr(base.frontend, name) for _, candidates in STAGES
               for name in candidates[0]}
    reference = evaluate(dict(current))
    bound = reference["mean_reprojection_px"]
    history = []
    for stage, candidates in STAGES:
        print(f"stage {stage} (start {current})", flush=True)
        best_values, best = dict(current), evaluate(dict(current))
        for candidate in candidates:
            values = {**current, **candidate}
            entry = evaluate(values)
            admissible = entry["all_loops"] and entry["mean_reprojection_px"] <= bound
            if admissible and entry["score"] > best["score"]:
                best_values, best = values, entry
        current = best_values
        history.append({"stage": stage, "selected": dict(current), "score": best["score"],
                        "mean_reprojection_px": best["mean_reprojection_px"]})
        print(f"stage {stage} selected {current} (score {best['score']:.1f})", flush=True)
    summary = {
        "dev_scenes": [row["sequence_id"] for row in rows],
        "reprojection_bound_px": bound,
        "start": reference["frontend"], "start_score": reference["score"],
        "selected": current, "stages": history,
    }
    (args.output_dir / "selected.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                   encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
