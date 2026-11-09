"""Held-out rotation and 3D point-cloud evaluation."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from .config import ESpinSLAMConfig
from .data import load_event_sequence, read_manifest
from .geometry import PointCloudAlignment, align_similarity_icp
from .pipeline import ESpinSLAM
from .plotting import create_report_plots


def _nullable(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def rotation_metrics(truth: np.ndarray, prediction: np.ndarray,
                     total_count: int) -> dict[str, float | int | None]:
    valid = np.isfinite(truth) & np.isfinite(prediction)
    coverage = float(valid.sum() / total_count) if total_count else 0.0
    if not np.any(valid):
        return {
            "sequence_count": total_count, "valid_count": 0,
            "coverage": coverage, "abstention": 1.0 - coverage,
            "mae_rad_s": None, "rmse_rad_s": None, "mape_percent": None,
            "correlation": None, "slope": None,
        }
    actual, estimate = truth[valid], prediction[valid]
    error = estimate - actual
    correlation = (
        float(np.corrcoef(actual, estimate)[0, 1])
        if actual.size > 1 and np.std(actual) > 0 and np.std(estimate) > 0 else math.nan
    )
    slope = (
        float(np.polyfit(actual, estimate, 1)[0])
        if actual.size > 1 and np.std(actual) > 0 else math.nan
    )
    return {
        "sequence_count": total_count,
        "valid_count": int(valid.sum()),
        "coverage": coverage,
        "abstention": 1.0 - coverage,
        "mae_rad_s": float(np.mean(np.abs(error))),
        "rmse_rad_s": float(np.sqrt(np.mean(error ** 2))),
        "mape_percent": float(np.mean(np.abs(error) / np.maximum(np.abs(actual), 1e-9)) * 100.0),
        "correlation": _nullable(correlation),
        "slope": _nullable(slope),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", newline="", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_reference(path: Path | None, maximum: int) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        points = np.asarray(archive["points"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("reference point cloud must have finite shape [N,3]")
    if points.shape[0] > maximum:
        indices = np.linspace(0, points.shape[0] - 1, maximum).round().astype(np.int64)
        points = points[indices]
    return points


def _pointcloud_metrics(rows: list[dict], total: int) -> dict:
    valid = [row for row in rows if row["valid"]]
    if not valid:
        return {
            "sequence_count": total, "valid_count": 0, "coverage": 0.0,
            "mean_icp_rmse_m": None, "median_icp_rmse_m": None,
            "mean_chamfer_rmse_m": None,
        }
    icp = np.asarray([row["icp_rmse_m"] for row in valid])
    chamfer = np.asarray([row["chamfer_rmse_m"] for row in valid])
    return {
        "sequence_count": total,
        "valid_count": len(valid),
        "coverage": len(valid) / total if total else 0.0,
        "mean_icp_rmse_m": float(icp.mean()),
        "median_icp_rmse_m": float(np.median(icp)),
        "mean_chamfer_rmse_m": float(chamfer.mean()),
    }


def evaluate(
    config: ESpinSLAMConfig, data_root: Path, manifest_path: Path,
    output_dir: Path, *, maximum_sequences: int = 0,
    reference_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate only the explicit held-out test rows and persist a full report."""

    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        row for row in read_manifest(manifest_path, data_root)
        if row["split"] == config.data.evaluation_split
    ]
    if not rows:
        raise ValueError("manifest contains no held-out test rows")
    if maximum_sequences > 0:
        rows = rows[:maximum_sequences]
    reference = _load_reference(
        reference_path, config.evaluation.pointcloud_max_reference_points,
    )
    model = ESpinSLAM(config)
    predictions: list[dict] = []
    time_predictions: list[dict] = []
    pointcloud_rows: list[dict] = []
    representative: tuple[float, str, PointCloudAlignment] | None = None

    for index, row in enumerate(rows, 1):
        sequence = load_event_sequence(
            row, data_root, config.data.sensor_size,
            max_duration_seconds=config.data.max_duration_seconds,
        )
        result = model.process(sequence, reconstruct_structure=True)
        prediction = result.omega_rad_s if result.valid else math.nan
        predictions.append({
            "sequence_id": sequence.sequence_id,
            "omega_true_rad_s": sequence.omega_rad_s,
            "omega_pred_rad_s": _nullable(prediction),
            "omega_true_deg_s": math.degrees(sequence.omega_rad_s),
            "omega_pred_deg_s": _nullable(math.degrees(prediction)),
            "absolute_error_rad_s": _nullable(abs(prediction - sequence.omega_rad_s)),
            "absolute_error_deg_s": _nullable(math.degrees(abs(prediction - sequence.omega_rad_s))),
            "percentage_error": _nullable(
                abs(prediction - sequence.omega_rad_s) / max(sequence.omega_rad_s, 1e-9) * 100.0,
            ),
            "frequency_hz": _nullable(result.frequency.frequency_hz),
            "frequency_confidence": result.frequency.confidence,
            "valid": result.valid,
            "failure_reason": result.frequency.reason,
            "event_count": sequence.event_count,
            "track_count": len(result.tracks),
            "landmark_count": int(result.reconstruction.points.shape[0]),
            "reprojection_error_px": _nullable(result.reconstruction.mean_reprojection_error_px),
        })
        for observation_seconds in config.evaluation.observation_times_seconds:
            prefix = sequence.before(observation_seconds)
            estimate = model.estimate_frequency(prefix)
            time_predictions.append({
                "sequence_id": sequence.sequence_id,
                "observation_seconds": float(observation_seconds),
                "available_seconds": prefix.duration_seconds,
                "omega_true_rad_s": sequence.omega_rad_s,
                "omega_pred_rad_s": _nullable(estimate.omega_rad_s),
                "confidence": estimate.confidence,
                "valid": estimate.valid,
                "failure_reason": estimate.reason,
            })
        alignment = (
            align_similarity_icp(result.reconstruction.points, reference)
            if reference is not None and result.reconstruction.valid
            else PointCloudAlignment(
                np.empty((0, 3)), math.nan, np.eye(3), np.zeros(3),
                math.nan, math.nan, False,
                "reference unavailable" if reference is None else result.reconstruction.reason,
            )
        )
        pointcloud_rows.append({
            "sequence_id": sequence.sequence_id,
            "valid": alignment.valid,
            "landmark_count": int(result.reconstruction.points.shape[0]),
            "reprojection_error_px": _nullable(result.reconstruction.mean_reprojection_error_px),
            "icp_rmse_m": _nullable(alignment.icp_rmse_m),
            "chamfer_rmse_m": _nullable(alignment.chamfer_rmse_m),
            "similarity_scale": _nullable(alignment.scale),
            "failure_reason": alignment.reason,
        })
        if alignment.valid:
            np.savez_compressed(
                output_dir / f"pointcloud_{sequence.sequence_id}.npz",
                predicted=result.reconstruction.points.astype(np.float32),
                aligned=alignment.aligned_points.astype(np.float32),
                reference=reference.astype(np.float32),
            )
            if representative is None or alignment.icp_rmse_m < representative[0]:
                representative = (alignment.icp_rmse_m, sequence.sequence_id, alignment)
        print(
            f"[{index}/{len(rows)}] {sequence.sequence_id}: "
            f"omega={prediction:.6g} tracks={len(result.tracks)} "
            f"points={result.reconstruction.points.shape[0]}",
            flush=True,
        )

    truth = np.asarray([row["omega_true_rad_s"] for row in predictions], dtype=np.float64)
    prediction = np.asarray([
        math.nan if row["omega_pred_rad_s"] is None else row["omega_pred_rad_s"]
        for row in predictions
    ], dtype=np.float64)
    final_metrics = rotation_metrics(truth, prediction, len(rows))
    time_summary: list[dict] = []
    for observation_seconds in config.evaluation.observation_times_seconds:
        subset = [
            row for row in time_predictions
            if row["observation_seconds"] == observation_seconds
        ]
        subset_truth = np.asarray([row["omega_true_rad_s"] for row in subset])
        subset_prediction = np.asarray([
            math.nan if row["omega_pred_rad_s"] is None else row["omega_pred_rad_s"]
            for row in subset
        ])
        values = rotation_metrics(subset_truth, subset_prediction, len(subset))
        time_summary.append({"observation_seconds": observation_seconds, **values})

    aligned = representative[2].aligned_points if representative else None
    pointcloud_sequence_id = representative[1] if representative else None
    plot_paths = create_report_plots(
        predictions, time_summary, output_dir,
        reference=reference, aligned=aligned,
        pointcloud_sequence_id=pointcloud_sequence_id,
    )
    _write_csv(output_dir / "predictions.csv", predictions)
    _write_csv(output_dir / "time_predictions.csv", time_predictions)
    _write_csv(output_dir / "time_accuracy.csv", time_summary)
    _write_csv(output_dir / "pointcloud_metrics.csv", pointcloud_rows)
    report = {
        "method": "Independent paper-guided eSpinSLAM reproduction",
        "paper": "Event-based Spinning Object SLAM, ICCVW 2025",
        "evaluation_split": config.data.evaluation_split,
        "data_leakage_controls": {
            "inference_inputs": ["events", "sensor geometry", "camera intrinsics"],
            "evaluation_only": ["omega_rad_s", "Blender geometry", "camera pose CSV", "bounding boxes"],
            "test_tuning": False,
        },
        "known_domain_mismatch": (
            "The debris dataset includes slow camera closing/orbit motion; the paper assumes "
            "a static event camera and constant single-axis object spin."
        ),
        "implementation_differences": [
            "OpenCV FAST on polarity time surfaces replaces the unpublished native eFAST code.",
            "SciPy robust least-squares replaces the paper's GTSAM backend.",
            "Raw-event loop alignment is a bounded fallback when Python feature tracks are sparse.",
        ],
        "rotation": final_metrics,
        "pointcloud": _pointcloud_metrics(pointcloud_rows, len(rows)),
        "runtime_seconds": time.monotonic() - started,
        "sequence_count": len(rows),
        "reference_pointcloud_available": reference is not None,
        "representative_pointcloud_sequence": pointcloud_sequence_id,
        "plots": [path.name for path in plot_paths],
        "config": config.as_dict(),
    }
    _write_json(output_dir / "metrics.json", report)
    return report

