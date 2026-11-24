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
from espinslam.pipeline import ESpinSLAM


_CONFIRMATION_GRID = (
    (3, 0.0005), (3, 0.001), (3, 0.002),
    (5, 0.0005), (5, 0.001), (5, 0.002),
    (10, 0.0005), (10, 0.001), (10, 0.002),
    (20, 0.0005), (20, 0.001), (20, 0.002),
)


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
    return values


def _online_estimates(sequence, frontend_config, frequency) -> list[float | None]:
    """Replay the causal pre-loop path and retain every due update.

    A deliberately unreachable confirmation history keeps the estimator from
    changing the track pool through post-loop forgetting.  Invalid updates are
    retained as ``None`` because the experimental estimator clears its
    stability deque on such updates.
    """

    frontend = EventFrontend(sequence.width, sequence.height, frontend_config)
    estimator = LoopClosureFrequencyEstimator(replace(frequency, loop_history=1_000_000))
    estimated_version = -1
    next_estimate = -math.inf
    values: list[float | None] = []
    chunk = frontend_config.chunk_events
    for start in range(0, sequence.event_count, chunk):
        stop = min(sequence.event_count, start + chunk)
        frontend.push(
            sequence.t[start:stop], sequence.x[start:stop],
            sequence.y[start:stop], sequence.p[start:stop],
        )
        now = float(sequence.t[stop - 1])
        if frontend.pool.version != estimated_version and now >= next_estimate:
            estimate = estimator.update(frontend.pool.events())
            values.append(float(estimate.frequency_hz) if estimate.valid else None)
            estimated_version = frontend.pool.version
            next_estimate = now + frequency.update_interval_seconds
    return values


def _confirmation_prediction(
    values: list[float | None], history: int, threshold: float,
) -> float | None:
    window: list[float] = []
    for value in values:
        if value is None:
            window.clear()
            continue
        window.append(value)
        window = window[-history:]
        if len(window) == history and float(np.std(window, ddof=1)) < threshold:
            return float(value)
    return None


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


def _confirmation_metrics(records: list[dict]) -> list[dict]:
    truth = np.asarray([row["omega_true_rad_s"] for row in records])
    results = []
    for history, threshold in _CONFIRMATION_GRID:
        prediction = np.asarray([
            math.nan if (value := _confirmation_prediction(
                row["frequency_updates_hz"], history, threshold,
            )) is None else 2.0 * math.pi * value
            for row in records
        ])
        name = f"history_{history}_std_{threshold:g}"
        results.append({
            "name": name,
            "loop_history": history,
            "loop_std_threshold_hz": threshold,
            **rotation_metrics(truth, prediction, len(records)),
        })
    return results


def _evaluate_online_train(
    rows, root, config, frequency, cache_path: Path,
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
        record = {
            "sequence_id": sequence_id,
            "omega_true_rad_s": sequence.omega_rad_s,
            "frequency_updates_hz": _online_estimates(
                sequence, config.frontend, frequency,
            ),
        }
        _append(cache_path, record)
        cache[sequence_id] = record
        print(f"train-online [{index}/{len(rows)}] {sequence_id}", flush=True)
    return [cache[row["sequence_id"]] for row in rows]


def _evaluate_exact_validation(
    rows, root, config, cache_path: Path,
) -> list[dict]:
    cache = _load_cache(cache_path)
    model = ESpinSLAM(config)
    for index, row in enumerate(rows, 1):
        sequence_id = row["sequence_id"]
        if sequence_id in cache:
            continue
        sequence = load_event_sequence(
            row, root, config.data.sensor_size,
            max_duration_seconds=config.data.max_duration_seconds,
        )
        result = model.process(sequence, reconstruct_structure=False)
        record = {
            "sequence_id": sequence_id,
            "omega_true_rad_s": sequence.omega_rad_s,
            "omega_pred_rad_s": result.omega_rad_s if result.valid else None,
        }
        _append(cache_path, record)
        cache[sequence_id] = record
        print(
            f"val-online [{index}/{len(rows)}] {sequence_id}: valid={result.valid}",
            flush=True,
        )
    return [cache[row["sequence_id"]] for row in rows]


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
        "protocol_version": 2,
        "config_sha256": _sha256(args.config),
        "manifest_sha256": _sha256(args.manifest),
        "train_sequence_ids": [row["sequence_id"] for row in train],
        "validation_sequence_ids": [row["sequence_id"] for row in validation],
        "period_range_seconds": [low, high],
        "variants": {name: asdict(value) for name, value in variants},
        "confirmation_grid": [list(value) for value in _CONFIRMATION_GRID],
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
    alignment_name = train_metrics[0]["name"]
    alignment_frequency = dict(variants)[alignment_name]
    online_train_records = _evaluate_online_train(
        train, root, config, alignment_frequency,
        args.output_dir / "train_online_updates.jsonl",
    )
    confirmation_metrics = sorted(
        _confirmation_metrics(online_train_records), key=_rank,
    )
    confirmation = confirmation_metrics[0]
    selected_name = f"{alignment_name}__{confirmation['name']}"
    selected_frequency = replace(
        alignment_frequency,
        loop_history=confirmation["loop_history"],
        loop_std_threshold_hz=confirmation["loop_std_threshold_hz"],
    )
    selected_config_object = replace(config, frequency=selected_frequency)
    validation_records = _evaluate_exact_validation(
        validation, root, selected_config_object,
        args.output_dir / "validation_online_predictions.jsonl",
    )
    validation_truth = np.asarray([
        row["omega_true_rad_s"] for row in validation_records
    ])
    validation_prediction = np.asarray([
        math.nan if row["omega_pred_rad_s"] is None else row["omega_pred_rad_s"]
        for row in validation_records
    ])
    validation_metrics = rotation_metrics(
        validation_truth, validation_prediction, len(validation_records),
    )

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
        "train_alignment_selected_metrics": train_metrics[0],
        "train_confirmation_selected_metrics": confirmation,
        "validation_metrics": validation_metrics,
        "train_alignment_candidates": train_metrics,
        "train_confirmation_candidates": confirmation_metrics,
        "runtime_seconds": time.monotonic() - started,
    }
    (args.output_dir / "tuning_results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
