"""Publication-style rotation, timing, and 3D reconstruction plots."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _finite_pairs(rows: list[dict], truth: str, prediction: str) -> tuple[np.ndarray, np.ndarray]:
    values = [
        (float(row[truth]), float(row[prediction]))
        for row in rows
        if row.get(truth) is not None and row.get(prediction) is not None
        and math.isfinite(float(row[truth])) and math.isfinite(float(row[prediction]))
    ]
    if not values:
        return np.empty(0), np.empty(0)
    return np.asarray(values, dtype=np.float64).T


def _empty_axis(axis, message: str = "No valid estimates") -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_xticks([])
    axis.set_yticks([])


def plot_rotation_scatter(rows: list[dict], output: Path, *, degrees: bool) -> None:
    truth, prediction = _finite_pairs(rows, "omega_true_rad_s", "omega_pred_rad_s")
    factor = 180.0 / math.pi if degrees else 1.0
    unit = "deg/s" if degrees else "rad/s"
    fig, axis = plt.subplots(figsize=(7.5, 6.5))
    if truth.size:
        truth, prediction = truth * factor, prediction * factor
        low, high = min(truth.min(), prediction.min()), max(truth.max(), prediction.max())
        margin = max((high - low) * 0.08, 1e-3)
        axis.scatter(truth, prediction, s=38, alpha=0.78, edgecolor="white", linewidth=0.5)
        axis.plot((low - margin, high + margin), (low - margin, high + margin),
                  "k--", linewidth=1.2, label="Ideal")
        axis.set_xlim(low - margin, high + margin)
        axis.set_ylim(low - margin, high + margin)
        axis.legend()
    else:
        _empty_axis(axis)
    axis.set_xlabel(f"Ground truth [{unit}]")
    axis.set_ylabel(f"Estimated [{unit}]")
    axis.set_title(f"eSpinSLAM held-out test: rotation estimate ({unit})")
    axis.grid(alpha=0.25)
    _save(fig, output)


def plot_error_distribution(rows: list[dict], output: Path, *, degrees: bool) -> None:
    truth, prediction = _finite_pairs(rows, "omega_true_rad_s", "omega_pred_rad_s")
    factor = 180.0 / math.pi if degrees else 1.0
    unit = "deg/s" if degrees else "rad/s"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    if truth.size:
        signed = (prediction - truth) * factor
        absolute = np.abs(signed)
        bins = min(20, max(5, int(math.sqrt(signed.size))))
        axes[0].hist(signed, bins=bins, color="#1f77b4", alpha=0.85)
        axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1)
        order = np.argsort(absolute)
        cumulative = np.arange(1, absolute.size + 1) / absolute.size
        axes[1].plot(absolute[order], cumulative, linewidth=2)
        axes[1].set_ylim(0, 1.02)
    else:
        _empty_axis(axes[0])
        _empty_axis(axes[1])
    axes[0].set_title("Signed error")
    axes[0].set_xlabel(f"Prediction - truth [{unit}]")
    axes[0].set_ylabel("Sequences")
    axes[1].set_title("Absolute-error CDF")
    axes[1].set_xlabel(f"Absolute error [{unit}]")
    axes[1].set_ylabel("Cumulative probability")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(f"eSpinSLAM held-out test error distribution ({unit})")
    _save(fig, output)


def plot_time_accuracy(rows: list[dict], output: Path, *, degrees: bool) -> None:
    factor = 180.0 / math.pi if degrees else 1.0
    unit = "deg/s" if degrees else "rad/s"
    times = np.asarray([row["observation_seconds"] for row in rows], dtype=np.float64)
    mae = np.asarray([
        math.nan if row["mae_rad_s"] is None else row["mae_rad_s"] * factor for row in rows
    ])
    rmse = np.asarray([
        math.nan if row["rmse_rad_s"] is None else row["rmse_rad_s"] * factor for row in rows
    ])
    mape = np.asarray([
        math.nan if row["mape_percent"] is None else row["mape_percent"] for row in rows
    ])
    coverage = np.asarray([row["coverage"] for row in rows], dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
    axes[0].plot(times, mae, marker="o", label="MAE")
    axes[0].plot(times, rmse, marker="s", linestyle="--", label="RMSE")
    axes[0].set_ylabel(f"Error [{unit}]")
    axes[0].legend()
    axes[1].plot(times, mape, marker="o", color="#1f77b4", label="MAPE")
    axes[1].set_ylabel("MAPE [%]")
    axes[1].set_xlabel("Observation time [s]")
    coverage_axis = axes[1].twinx()
    coverage_axis.plot(times, coverage * 100.0, color="#2ca02c", marker="^",
                       linestyle=":", label="Coverage")
    coverage_axis.set_ylabel("Coverage [%]")
    coverage_axis.set_ylim(0, 105)
    axes[1].legend(loc="upper left")
    coverage_axis.legend(loc="upper right")
    ticks = [value for value in (0, 0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4, 5)
             if not times.size or value <= max(5.0, float(times.max()))]
    axes[1].set_xticks(ticks)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(f"Accuracy and availability vs observation time ({unit})")
    _save(fig, output)


def plot_coverage(rows: list[dict], output: Path) -> None:
    times = np.asarray([row["observation_seconds"] for row in rows])
    coverage = np.asarray([row["coverage"] for row in rows]) * 100.0
    abstention = 100.0 - coverage
    fig, axis = plt.subplots(figsize=(9.0, 5.2))
    axis.stackplot(times, coverage, abstention, labels=("Valid estimate", "Abstention"),
                   colors=("#2ca02c", "#d9d9d9"), alpha=0.85)
    axis.set_xlabel("Observation time [s]")
    axis.set_ylabel("Share of held-out sequences [%]")
    axis.set_ylim(0, 100)
    axis.set_title("Loop-closure availability and abstention")
    axis.legend(loc="center right")
    axis.grid(alpha=0.25)
    _save(fig, output)


def plot_reprojection(rows: list[dict], output: Path) -> None:
    values = [
        (row["reprojection_error_px"], abs(row["omega_pred_rad_s"] - row["omega_true_rad_s"]))
        for row in rows
        if row.get("reprojection_error_px") is not None
        and row.get("omega_pred_rad_s") is not None
        and math.isfinite(float(row["reprojection_error_px"]))
    ]
    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    if values:
        reprojection, rotation = np.asarray(values).T
        axis.scatter(reprojection, np.degrees(rotation), s=42, alpha=0.8)
    else:
        _empty_axis(axis, "No valid 3D reconstructions")
    axis.set_xlabel("Mean reprojection error [px]")
    axis.set_ylabel("Absolute rotation error [deg/s]")
    axis.set_title("Internal consistency vs rotation accuracy")
    axis.grid(alpha=0.25)
    _save(fig, output)


def plot_pointcloud(reference: np.ndarray | None, aligned: np.ndarray | None,
                    output: Path, sequence_id: str | None = None) -> None:
    fig = plt.figure(figsize=(14.0, 4.8))
    axes = (
        fig.add_subplot(1, 3, 1, projection="3d"),
        fig.add_subplot(1, 3, 2),
        fig.add_subplot(1, 3, 3),
    )
    if reference is None or aligned is None or not aligned.size:
        for axis in axes:
            _empty_axis(axis, "No valid 3D reconstruction")
    else:
        if reference.shape[0] > 8000:
            indices = np.linspace(0, reference.shape[0] - 1, 8000).round().astype(int)
            reference = reference[indices]
        axes[0].scatter(reference[:, 0], reference[:, 1], reference[:, 2],
                        s=0.3, alpha=0.10, color="gray", label="Blender GT")
        axes[0].scatter(aligned[:, 0], aligned[:, 1], aligned[:, 2],
                        s=15, color="#d62728", label="eSpinSLAM")
        axes[0].set_xlabel("X [m]")
        axes[0].set_ylabel("Y [m]")
        axes[0].set_zlabel("Z [m]")
        axes[0].legend()
        axes[1].scatter(reference[:, 0], reference[:, 1], s=0.3, alpha=0.10, color="gray")
        axes[1].scatter(aligned[:, 0], aligned[:, 1], s=15, color="#d62728")
        axes[1].set_xlabel("X [m]")
        axes[1].set_ylabel("Y [m]")
        axes[1].set_aspect("equal", adjustable="box")
        axes[2].scatter(reference[:, 0], reference[:, 2], s=0.3, alpha=0.10, color="gray")
        axes[2].scatter(aligned[:, 0], aligned[:, 2], s=15, color="#d62728")
        axes[2].set_xlabel("X [m]")
        axes[2].set_ylabel("Z [m]")
        axes[2].set_aspect("equal", adjustable="box")
    fig.suptitle(f"Similarity-aligned sparse 3D reconstruction{': ' + sequence_id if sequence_id else ''}")
    _save(fig, output)


def create_report_plots(predictions: list[dict], time_summary: list[dict],
                        output_dir: Path, *, reference=None, aligned=None,
                        pointcloud_sequence_id: str | None = None) -> list[Path]:
    outputs = [
        output_dir / "01_rotation_scatter_rad_s.png",
        output_dir / "02_rotation_scatter_deg_s.png",
        output_dir / "03_error_distribution_rad_s.png",
        output_dir / "04_error_distribution_deg_s.png",
        output_dir / "05_accuracy_vs_observation_time_rad_s.png",
        output_dir / "06_accuracy_vs_observation_time_deg_s.png",
        output_dir / "07_coverage_vs_observation_time.png",
        output_dir / "08_reprojection_vs_rotation_error.png",
        output_dir / "09_pointcloud_3d.png",
    ]
    plot_rotation_scatter(predictions, outputs[0], degrees=False)
    plot_rotation_scatter(predictions, outputs[1], degrees=True)
    plot_error_distribution(predictions, outputs[2], degrees=False)
    plot_error_distribution(predictions, outputs[3], degrees=True)
    plot_time_accuracy(time_summary, outputs[4], degrees=False)
    plot_time_accuracy(time_summary, outputs[5], degrees=True)
    plot_coverage(time_summary, outputs[6])
    plot_reprojection(predictions, outputs[7])
    plot_pointcloud(reference, aligned, outputs[8], pointcloud_sequence_id)
    return outputs

