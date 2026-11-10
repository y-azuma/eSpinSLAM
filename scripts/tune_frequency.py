#!/usr/bin/env python3
"""Tune paper-compatible loop-closure tolerances on validation rows only."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import csv
import json
import math
from pathlib import Path
import time

import numpy as np

from espinslam.config import FrequencyConfig, load_config
from espinslam.data import load_event_sequence, read_manifest
from espinslam.evaluation import rotation_metrics
from espinslam.frequency import LoopClosureFrequencyEstimator
from espinslam.frontend import EventFrontend, track_observation_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-sequences", type=int, default=60)
    return parser.parse_args()


def _training_frequency_range(rows: list[dict[str, str]], root: Path) -> tuple[float, float]:
    values = []
    resolved = root.resolve()
    for row in rows:
        if row["split"] != "train":
            continue
        path = (resolved / row["event_path"]).resolve()
        if not path.is_relative_to(resolved):
            raise ValueError("training event path escapes data root")
        with np.load(path, allow_pickle=False) as archive:
            values.append(float(archive["omega_rad_s"]) / (2.0 * math.pi))
    if not values:
        raise ValueError("manifest has no training rows for a leakage-safe search prior")
    low, high = min(values), max(values)
    span = max(high - low, 0.05 * high)
    return max(0.01, low - 0.10 * span), high + 0.10 * span


def _variants(base: FrequencyConfig, low: float, high: float) -> list[tuple[str, FrequencyConfig]]:
    values: list[tuple[str, FrequencyConfig]] = [
        ("published_baseline_range", replace(base, equivalent_score_fraction=0.01)),
    ]
    for spatial in (3.0, 6.0, 9.0):
        for temporal in (0.015, 0.030, 0.060):
            name = f"train_range_s{spatial:g}_t{int(temporal * 1000):02d}ms"
            values.append((name, replace(
                base, min_hz=low, max_hz=high, candidate_count=141,
                spatial_tolerance_px=spatial,
                temporal_tolerance_seconds=temporal,
                equivalent_score_fraction=0.0,
            )))
    return values


def _bounded_observations(sequence, frontend_config, limit: int) -> np.ndarray:
    frontend = EventFrontend(sequence.width, sequence.height, frontend_config)
    tracks = frontend.process(sequence)
    values = frontend.frequency_observation_array()
    if values.shape[0] < 20:
        values = track_observation_array(tracks)
    if values.shape[0] < 20:
        count = min(sequence.event_count, limit)
        indices = np.linspace(0, sequence.event_count - 1, count).round().astype(np.int64)
        values = np.column_stack((
            sequence.t[indices], sequence.x[indices], sequence.y[indices], sequence.p[indices],
        )).astype(np.float64, copy=False)
    elif values.shape[0] > limit:
        indices = np.linspace(0, values.shape[0] - 1, limit).round().astype(np.int64)
        values = values[indices]
    return values


def main() -> None:
    args = parse_args()
    if args.max_sequences < 1:
        raise ValueError("--max-sequences must be positive")
    config = load_config(args.config)
    rows = read_manifest(args.manifest, args.data_root)
    low, high = _training_frequency_range(rows, args.data_root)
    validation = [row for row in rows if row["split"] == "val"][:args.max_sequences]
    if not validation:
        raise ValueError("manifest has no validation rows")
    variants = _variants(config.frequency, low, high)
    predictions = {name: [] for name, _ in variants}
    truth: list[float] = []
    started = time.monotonic()
    for index, row in enumerate(validation, 1):
        sequence = load_event_sequence(
            row, args.data_root, config.data.sensor_size,
            max_duration_seconds=config.data.max_duration_seconds,
        )
        observations = _bounded_observations(
            sequence, config.frontend, config.frequency.max_alignment_samples,
        )
        truth.append(sequence.omega_rad_s)
        for name, variant in variants:
            estimate = LoopClosureFrequencyEstimator(variant).estimate(observations)
            predictions[name].append(estimate.omega_rad_s if estimate.valid else math.nan)
        print(f"[{index}/{len(validation)}] {sequence.sequence_id}", flush=True)
    actual = np.asarray(truth, dtype=np.float64)
    results = []
    for name, variant in variants:
        estimate = np.asarray(predictions[name], dtype=np.float64)
        metrics = rotation_metrics(actual, estimate, actual.size)
        results.append({"name": name, "frequency": asdict(variant), **metrics})
    # Coverage is part of correctness: never reward a candidate merely because it
    # abstains on difficult sequences.  All full-coverage candidates are ranked
    # before MAE/RMSE are considered.
    results.sort(key=lambda item: (
        -item["coverage"],
        math.inf if item["mae_rad_s"] is None else item["mae_rad_s"],
        math.inf if item["rmse_rad_s"] is None else item["rmse_rad_s"],
    ))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "tuning_candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ("name", "mae_rad_s", "rmse_rad_s", "mape_percent",
                      "correlation", "slope", "coverage")
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({name: item[name] for name in fieldnames})
    report = {
        "selection_split": "val",
        "test_rows_read": False,
        "validation_sequence_count": len(validation),
        "train_label_frequency_prior_hz": [low, high],
        "objective": "maximum validation coverage, then minimum MAE and RMSE",
        "selected": results[0],
        "candidates": results,
        "runtime_seconds": time.monotonic() - started,
    }
    (args.output_dir / "tuning_results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
