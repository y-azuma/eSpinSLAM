#!/usr/bin/env python3
"""Tune eSpinSLAM frequency parameters on train only, then audit on val.

The test split is never loaded.  The front-end configuration is the fixed
geometric rescaling of the historical debris profile; only frequency search
parameters not fixed by Table 1 are selected with rotation labels.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from espinslam.config import FrequencyConfig, load_config
from espinslam.data import load_event_sequence, read_manifest
from espinslam.evaluation import rotation_metrics
from espinslam.frequency import LoopClosureFrequencyEstimator
from espinslam.frontend import EventFrontend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-validation", type=int, default=0)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _label_period_range(rows: list[dict[str, str]], root: Path) -> tuple[float, float]:
    periods = []
    for row in rows:
        path = (root / row["event_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("training event path escapes data root")
        with np.load(path, allow_pickle=False) as archive:
            omega = float(archive["omega_rad_s"])
        if not math.isfinite(omega) or omega <= 0:
            raise ValueError(f"invalid training omega in {path}")
        periods.append(2.0 * math.pi / omega)
    low, high = min(periods), max(periods)
    margin = max(0.05, 0.10 * (high - low))
    return max(0.05, low - margin), high + margin


def _variants(base: FrequencyConfig, low: float, high: float):
    variants = []
    for factor in (0.5, 1.0, 2.0):
        for harmonic in (0.0, 0.05, 0.15):
            name = f"scale_x{factor:g}_harmonic_{harmonic:g}"
            variants.append((name, replace(
                base,
                alignment_mode="experimental",
                min_period_seconds=low,
                max_period_seconds=high,
                time_scale_px_per_second=base.time_scale_px_per_second * factor,
                harmonic_tolerance=harmonic,
            )))
    return variants


def _observations(sequence, frontend_config) -> np.ndarray:
    frontend = EventFrontend(sequence.width, sequence.height, frontend_config)
    chunk = frontend_config.chunk_events
    for start in range(0, sequence.event_count, chunk):
        stop = min(sequence.event_count, start + chunk)
        frontend.push(
            sequence.t[start:stop], sequence.x[start:stop],
            sequence.y[start:stop], sequence.p[start:stop],
        )
    values = frontend.pool.events()
    if values.shape[0] < 2 * 50:
        # A deterministic raw-event fallback is used only for tuning viability;
        # the final online test pipeline has no such fallback.
        count = min(sequence.event_count, 10_000)
        if count:
            indices = np.linspace(0, sequence.event_count - 1, count).round().astype(np.int64)
            values = np.column_stack((
                sequence.t[indices], sequence.x[indices], sequence.y[indices],
            )).astype(np.float64, copy=False)
    return values


def _load_cache(path: Path) -> dict[str, dict]:
    values = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            values[row["sequence_id"]] = row
    return values


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def _evaluate_rows(
    rows, root, config, variants, cache_path: Path, *, label: str,
) -> list[dict]:
    cache = _load_cache(cache_path)
    for index, row in enumerate(rows, 1):
        sequence_id = row["sequence_id"]
        if sequence_id in cache:
            continue
        sequence = load_event_sequence(
            row, root, config.data.sensor_size,
            max_duration_seconds=config.data.max_duration_seconds,
        )
        observations = _observations(sequence, config.frontend)
        predictions = {}
        for name, frequency in variants:
            estimate = LoopClosureFrequencyEstimator(frequency).update(observations)
            predictions[name] = estimate.omega_rad_s if estimate.valid else None
        record = {
            "sequence_id": sequence_id,
            "omega_true_rad_s": sequence.omega_rad_s,
            "event_count": sequence.event_count,
            "track_event_count": int(observations.shape[0]),
            "predictions": predictions,
        }
        _append(cache_path, record)
        cache[sequence_id] = record
        print(
            f"{label} [{index}/{len(rows)}] {sequence_id}: "
            f"events={sequence.event_count} track_events={observations.shape[0]}",
            flush=True,
        )
    return [cache[row["sequence_id"]] for row in rows]


def _metrics(records: list[dict], names: list[str]) -> list[dict]:
    truth = np.asarray([row["omega_true_rad_s"] for row in records])
    results = []
    for name in names:
        prediction = np.asarray([
            math.nan if row["predictions"][name] is None
            else row["predictions"][name]
            for row in records
        ])
        results.append({"name": name, **rotation_metrics(truth, prediction, len(records))})
    return results


def _rank(item: dict) -> tuple:
    return (
        -item["coverage"],
        math.inf if item["mae_rad_s"] is None else item["mae_rad_s"],
        math.inf if item["rmse_rad_s"] is None else item["rmse_rad_s"],
    )


def main() -> None:
    args = parse_args()
    if args.max_train < 0 or args.max_validation < 0:
        raise ValueError("maximum sequence counts must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    root = args.data_root.resolve()
    rows = read_manifest(args.manifest, root)
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "val"]
    if args.max_train:
        train = train[:args.max_train]
    if args.max_validation:
        validation = validation[:args.max_validation]
    if not train or not validation:
        raise ValueError("manifest must contain non-empty train and val splits")

    low, high = _label_period_range(train, root)
    variants = _variants(config.frequency, low, high)
    names = [name for name, _ in variants]
    fingerprint = {
        "config_sha256": _sha256(args.config),
        "manifest_sha256": _sha256(args.manifest),
        "train_sequence_ids": [row["sequence_id"] for row in train],
        "validation_sequence_ids": [row["sequence_id"] for row in validation],
        "period_range_seconds": [low, high],
        "variants": {name: asdict(value) for name, value in variants},
    }
    fingerprint_path = args.output_dir / "run_fingerprint.json"
    if fingerprint_path.is_file():
        if json.loads(fingerprint_path.read_text()) != fingerprint:
            raise RuntimeError("refusing to resume tuning with a different fingerprint")
    else:
        fingerprint_path.write_text(json.dumps(fingerprint, indent=2) + "\n")

    started = time.monotonic()
    train_records = _evaluate_rows(
        train, root, config, variants,
        args.output_dir / "train_predictions.jsonl", label="train",
    )
    train_metrics = sorted(_metrics(train_records, names), key=_rank)
    selected_name = train_metrics[0]["name"]
    selected_frequency = dict(variants)[selected_name]
    selected_only = [(selected_name, selected_frequency)]
    validation_records = _evaluate_rows(
        validation, root, config, selected_only,
        args.output_dir / "validation_predictions.jsonl", label="val",
    )
    validation_metrics = _metrics(validation_records, [selected_name])[0]

    selected_config = config.as_dict()
    selected_config["frequency"] = asdict(selected_frequency)
    (args.output_dir / "selected_config.json").write_text(
        json.dumps(selected_config, indent=2) + "\n", encoding="utf-8",
    )
    report = {
        "selection_split": "train",
        "validation_role": "audit_only_no_parameter_selection",
        "test_rows_read": False,
        "train_sequence_count": len(train),
        "validation_sequence_count": len(validation),
        "period_range_seconds_from_train_labels": [low, high],
        "objective": "maximum train coverage, then minimum train MAE and RMSE",
        "selected_name": selected_name,
        "selected_frequency": asdict(selected_frequency),
        "train_selected_metrics": train_metrics[0],
        "validation_metrics": validation_metrics,
        "train_candidates": train_metrics,
        "runtime_seconds": time.monotonic() - started,
    }
    (args.output_dir / "tuning_results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
