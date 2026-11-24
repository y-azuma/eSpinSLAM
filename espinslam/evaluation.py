"""Rotation and 3D point-cloud evaluation with explicit profile labels."""

from __future__ import annotations

import csv
from collections import Counter
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


def _method_profile(config: ESpinSLAMConfig) -> tuple[str, list[str]]:
    """Classify this config using the same frozen profile as TOPSPIN reports."""

    # Local import keeps the generic evaluator independent at import time while
    # reusing the one authoritative reference-profile definition.
    from .topspin import _reference_mismatches

    mismatches = _reference_mismatches(config)
    return (
        "paper_reference_config" if not mismatches else "experimental_interpretation",
        mismatches,
    )


def _fallback_summary(reports) -> tuple[int, str]:
    reasons: Counter[str] = Counter()
    count = 0
    for report in reports:
        count += int(getattr(report, "triangulation_fallback_count", 0))
        for item in str(getattr(report, "triangulation_fallback_reasons", "")).split(";"):
            if not item:
                continue
            reason, _, amount = item.rpartition(":")
            reasons[reason or item] += int(amount) if reason else 1
    return count, ";".join(
        f"{reason}:{amount}" for reason, amount in sorted(reasons.items())
    )


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
        fallback_count, fallback_reasons = _fallback_summary(result.optimizations)
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
            "frequency_hz": _nullable(result.frequency_hz),
            "latest_unconfirmed_frequency_hz": _nullable(result.frequency.frequency_hz),
            "loop_closed": result.loop_closed,
            "loop_closure_time_s": _nullable(result.loop_closure_time),
            "valid": result.valid,
            "failure_reason": "" if result.valid else (
                result.frequency.reason or "loop closure was not detected"),
            "event_count": sequence.event_count,
            "track_count": result.track_count,
            "landmark_count": int(result.reconstruction.points.shape[0]),
            "reprojection_error_px": _nullable(result.reconstruction.mean_reprojection_error_px),
            "runtime_s": result.runtime_seconds,
            "throughput_ev_s": _nullable(result.throughput_events_per_second),
            "triangulation_fallback_count": fallback_count,
            "triangulation_fallback_reasons": fallback_reasons,
        })
        for observation_seconds in config.evaluation.observation_times_seconds:
            # The online system's causal output at this time (no re-run, no labels).
            confirmed, latest = result.frequency_at(observation_seconds)
            time_predictions.append({
                "sequence_id": sequence.sequence_id,
                "observation_seconds": float(observation_seconds),
                "available_seconds": min(float(observation_seconds), sequence.duration_seconds),
                "omega_true_rad_s": sequence.omega_rad_s,
                "omega_pred_rad_s": _nullable(2.0 * math.pi * confirmed),
                "omega_unconfirmed_rad_s": _nullable(2.0 * math.pi * latest),
                "valid": math.isfinite(confirmed),
                "failure_reason": "" if math.isfinite(confirmed) else "loop not yet detected",
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
            f"omega={prediction:.6g} tracks={result.track_count} "
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
    method_profile, reference_mismatches = _method_profile(config)
    report = {
        "method": "eSpinSLAM reproduction candidate",
        "paper": "Event-based Spinning Object SLAM, ICCVW 2025",
        "method_profile": method_profile,
        "evaluation_classification": (
            "reference_profile_domain_robustness"
            if method_profile == "paper_reference_config"
            else "experimental_interpretation_domain_robustness"
        ),
        "paper_comparison_eligible": False,
        "reference_config_mismatches": reference_mismatches,
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
        "implementation_notes": [
            "Front-end: numba eFAST (rpg_corner_events criterion), Eq. (6) density filter, "
            "HDBSCAN tracklets, tracks store events (sigma_retain circular buffer).",
            "Back-end: GTSAM replaced by Levenberg-Marquardt with Huber IRLS and analytic "
            "Jacobians; every retained track event is an observation (Eq. 11).",
            "Frequency: Eq. (9) with 1-D ICP and loop detection (aleph, sigma); time curves "
            "report the causal online output (f* only after loop closure).",
            "No frame-based or raw-event fallbacks: sequences without loop closure abstain.",
        ],
        "rotation": final_metrics,
        "pointcloud": _pointcloud_metrics(pointcloud_rows, len(rows)),
        "backend_diagnostics": {
            "triangulation_fallback_count": sum(
                int(row["triangulation_fallback_count"]) for row in predictions
            ),
            "sequences_with_triangulation_fallback": sum(
                int(row["triangulation_fallback_count"]) > 0 for row in predictions
            ),
            "reason_summaries": sorted({
                row["triangulation_fallback_reasons"] for row in predictions
                if row["triangulation_fallback_reasons"]
            }),
        },
        "runtime_seconds": time.monotonic() - started,
        "sequence_count": len(rows),
        "reference_pointcloud_available": reference is not None,
        "representative_pointcloud_sequence": pointcloud_sequence_id,
        "plots": [path.name for path in plot_paths],
        "config": config.as_dict(),
    }
    _write_json(output_dir / "metrics.json", report)
    return report
