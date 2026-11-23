"""TOPSPIN evaluation reproducing the paper's experiments (Sec. 5, Sec. 6).

Outputs, per run directory:

* ``per_sequence.csv``  - frequency, loop closure, reprojection error, ICP
  alignment RMSE [mm], landmarks, runtime and throughput of every scene;
* ``table2.csv`` / ``table2.md`` - per-object mean +- std of reprojection
  error [px] and ICP alignment RMSE [mm] (paper Table 2);
* ``fig6_frequency_cdf.png`` - CDF of estimated f per speed with ground truth
  (paper Fig. 6; the ablation run adds ``fig6_loop_closure_vs_fft.png`` with
  both estimators as in the paper);
* ``fig5_throughput_summary.png`` - front-end (the paper's ETC tracker),
  full-system, and scene event-rate throughput (paper Fig. 5);
* ``table_vs_paper.md`` / ``table3_vs_paper.md`` - this run next to the
  values printed in the paper;
* ``fig7_throughput.png`` - per-object throughput box plot with the mean scene
  event rate (paper Fig. 7);
* ``history/<scene>_optimizations.csv`` and ``fig8_<scene>.png`` - mean
  reprojection error and number of world points over time (paper Fig. 8);
* ``reconstruction_<scene>.png`` - world points inside the estimated orbit
  (as in paper Figs. 1, 3, 4); ``per_sequence.csv`` also flags whether the
  system kept up with the scene's event rate (``real_time``, Fig. 7);
* ``metrics.json`` - everything above plus the configuration.

:func:`evaluate_topspin_ablations` runs the four settings of Table 3 and
writes ``table3.csv`` / ``table3.md``.

The manifest is a CSV with one row per scene. Required columns:
``sequence_id, event_path, object``. Optional columns: ``speed``,
``frequency_hz`` (ground truth, evaluation only), ``cad_path``,
``cad_unit_mm`` (CAD units -> mm, default 1), ``orbit_radius_mm`` (true
camera-to-spin-axis distance; fixes the monocular scale for ICP),
``fx, fy, cx, cy, k1, k2, p1, p2, k3`` (calibration overriding the config),
``width, height`` and ``time_unit``.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from .config import CameraConfig, ESpinSLAMConfig
from .data import load_events_as_sequence, read_manifest
from .geometry import align_similarity_icp
from .pipeline import ESpinSLAM, ESpinSLAMResult

TABLE2_OBJECTS = ("Hubble", "SOHO", "TDRS", "Camera", "Dualshock", "Switch")

# Values reported in the paper (mean, std per object), for side-by-side reports.
# Note: Table 3's eSpinSLAM ICP row differs from Table 2 for Hubble (1.92 vs
# 0.92) and Switch (1.83 vs 0.83); both are kept as printed.
PAPER_TABLE2 = {
    "reprojection_px": {
        "Hubble": (3.25, 3.01), "SOHO": (2.99, 3.17), "TDRS": (1.54, 0.20),
        "Camera": (1.98, 0.15), "Dualshock": (2.19, 0.91), "Switch": (1.94, 0.35),
    },
    "icp_mm": {
        "Hubble": (0.92, 0.71), "SOHO": (0.56, 0.41), "TDRS": (0.66, 0.76),
        "Camera": (0.75, 0.57), "Dualshock": (0.93, 1.02), "Switch": (0.83, 0.66),
    },
}
PAPER_TABLE3 = {
    "FFT f estimation": {
        "reprojection_px": {"Hubble": (5.61, 7.93), "SOHO": (5.81, 5.58), "TDRS": (3.41, 3.56),
                            "Camera": (2.67, 1.34), "Dualshock": (2.99, 2.35), "Switch": (4.39, 5.11)},
        "icp_mm": {"Hubble": (3.17, 1.82), "SOHO": (3.51, 2.44), "TDRS": (5.95, 1.62),
                   "Camera": (2.13, 1.05), "Dualshock": (4.21, 1.33), "Switch": (3.16, 2.03)},
    },
    "No factor graph f constraint": {
        "reprojection_px": {"Hubble": (6.01, 3.79), "SOHO": (17.16, 9.03), "TDRS": (4.29, 2.33),
                            "Camera": (2.89, 3.11), "Dualshock": (5.27, 7.75), "Switch": (12.38, 6.91)},
        "icp_mm": {"Hubble": (8.14, 7.29), "SOHO": (15.49, 3.60), "TDRS": (5.11, 2.49),
                   "Camera": (1.12, 1.27), "Dualshock": (4.35, 0.74), "Switch": (9.38, 3.29)},
    },
    "No landmark association": {
        "reprojection_px": {"Hubble": (2.11, 1.93), "SOHO": (3.68, 0.71), "TDRS": (1.18, 0.71),
                            "Camera": (0.42, 0.80), "Dualshock": (1.67, 1.86), "Switch": (1.09, 1.50)},
        "icp_mm": {"Hubble": (13.74, 3.92), "SOHO": (12.47, 11.16), "TDRS": (19.61, 8.31),
                   "Camera": (9.19, 5.10), "Dualshock": (7.72, 3.27), "Switch": (13.11, 3.30)},
    },
    "eSpinSLAM": {
        "reprojection_px": PAPER_TABLE2["reprojection_px"],
        "icp_mm": {"Hubble": (1.92, 0.71), "SOHO": (0.56, 0.41), "TDRS": (0.66, 0.76),
                   "Camera": (0.75, 0.57), "Dualshock": (0.93, 1.02), "Switch": (1.83, 0.66)},
    },
}
ABLATIONS = {
    "fft_frequency": ("FFT f estimation", {"frequency": {"method": "fft"}}),
    "free_frequency": ("No factor graph f constraint", {"backend": {"optimize_frequency": True}}),
    "no_landmark_association": ("No landmark association",
                                {"backend": {"landmark_association": False}}),
    "espinslam": ("eSpinSLAM", {}),
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _float_or_none(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _nullable(value: float | None) -> float | None:
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(name for name in row if name not in fieldnames)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    temporary.replace(path)


def read_topspin_manifest(path: str | Path, data_root: str | Path) -> list[dict[str, str]]:
    return read_manifest(path, data_root, required=("sequence_id", "event_path", "object"))


def camera_for_row(base: CameraConfig, row: dict[str, str]) -> CameraConfig:
    """Per-scene calibration from the manifest, falling back to the config."""

    values = {name: _float_or_none(row.get(name)) for name in ("fx", "fy", "cx", "cy")}
    distortion = [_float_or_none(row.get(name)) for name in ("k1", "k2", "p1", "p2", "k3")]
    camera = CameraConfig(
        fx=values["fx"] if values["fx"] is not None else base.fx,
        fy=values["fy"] if values["fy"] is not None else base.fy,
        cx=values["cx"] if values["cx"] is not None else base.cx,
        cy=values["cy"] if values["cy"] is not None else base.cy,
        distortion=([value or 0.0 for value in distortion]
                    if any(value is not None for value in distortion) else list(base.distortion)),
    )
    if not camera.complete:
        raise ValueError(
            f"{row['sequence_id']}: camera intrinsics are missing; add fx, fy, cx, cy "
            "(TOPSPIN calibration) to the manifest or to the config",
        )
    return camera


def load_reference_cloud(path: Path, count: int, unit_mm: float = 1.0,
                         seed: int = 2701) -> np.ndarray:
    """CAD reference in millimetres: a point cloud or an area-sampled mesh."""

    suffix = path.suffix.lower()
    if suffix in (".npz", ".npy"):
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                points = np.asarray(archive["points"], dtype=np.float64)
        else:
            points = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
        if points.shape[0] > count:
            points = points[np.linspace(0, points.shape[0] - 1, count).round().astype(int)]
    else:
        try:
            import trimesh
        except ImportError as error:
            raise ImportError("loading CAD meshes requires trimesh (pip install trimesh)") from error
        mesh = trimesh.load(path, force="mesh")
        if mesh.faces.shape[0] == 0:
            points = np.asarray(mesh.vertices, dtype=np.float64)
        else:
            points, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
            points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 4:
        raise ValueError(f"{path}: reference must contain at least four 3D points")
    return points * float(unit_mm)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _sequence_row(row: dict[str, str], result: ESpinSLAMResult, *,
                  icp_rmse_mm: float | None, chamfer_mm: float | None,
                  icp_reason: str, scale_source: str) -> dict[str, Any]:
    truth = _float_or_none(row.get("frequency_hz"))
    estimate = result.frequency_hz
    reconstruction = result.reconstruction
    return {
        "sequence_id": result.sequence_id,
        "object": row["object"],
        "speed": row.get("speed", ""),
        "frequency_true_hz": truth,
        "frequency_est_hz": _nullable(estimate),
        "frequency_abs_error_hz": _nullable(abs(estimate - truth)) if truth is not None else None,
        "frequency_rel_error_percent": (
            _nullable(abs(estimate - truth) / truth * 100.0) if truth else None),
        "loop_closed": result.loop_closed,
        "loop_closure_time_s": _nullable(result.loop_closure_time),
        "landmarks": int(reconstruction.points.shape[0]),
        "observations": int(reconstruction.observation_count),
        "reprojection_error_px": _nullable(reconstruction.mean_reprojection_error_px),
        "reprojection_error_obs_px": _nullable(reconstruction.mean_observation_error_px),
        "icp_rmse_mm": icp_rmse_mm,
        "chamfer_rmse_mm": chamfer_mm,
        "icp_scale_source": scale_source,
        "icp_failure_reason": icp_reason,
        "reconstruction_valid": reconstruction.valid,
        "reconstruction_reason": reconstruction.reason,
        "events": result.event_count,
        "duration_s": result.observation_seconds,
        "scene_event_rate_ev_s": _nullable(
            result.event_count / result.observation_seconds if result.observation_seconds > 0
            else math.nan),
        "runtime_s": result.runtime_seconds,
        "throughput_ev_s": _nullable(result.throughput_events_per_second),
        # Fig. 7: real time when the system outpaces the scene's own event rate.
        "real_time": bool(result.observation_seconds > 0 and result.runtime_seconds > 0
                          and result.throughput_events_per_second
                          >= result.event_count / result.observation_seconds),
        "frontend_throughput_ev_s": _nullable(result.frontend_throughput_events_per_second),
        "frontend_s": result.frontend_seconds,
        "frequency_s": result.frequency_seconds,
        "backend_s": result.backend_seconds,
        "optimizations": len(result.optimizations),
    }


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    finite = np.asarray([value for value in values if value is not None and math.isfinite(value)])
    if not finite.size:
        return None, None
    return float(finite.mean()), float(finite.std())


def table2(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    objects = [name for name in TABLE2_OBJECTS if any(r["object"] == name for r in rows)]
    objects += sorted({r["object"] for r in rows} - set(objects))
    table = []
    for name in objects:
        subset = [r for r in rows if r["object"] == name]
        reprojection = _mean_std([r["reprojection_error_px"] for r in subset])
        icp = _mean_std([r["icp_rmse_mm"] for r in subset])
        table.append({
            "object": name,
            "scenes": len(subset),
            "valid_reconstructions": sum(bool(r["reconstruction_valid"]) for r in subset),
            "reprojection_mean_px": reprojection[0], "reprojection_std_px": reprojection[1],
            "icp_mean_mm": icp[0], "icp_std_mm": icp[1],
        })
    return table


def _format(mean: float | None, std: float | None) -> str:
    return "-" if mean is None else f"{mean:.2f} ± {std:.2f}"


def table2_markdown(table: list[dict[str, Any]], title: str = "eSpinSLAM") -> str:
    header = "| Metric | " + " | ".join(row["object"] for row in table) + " |"
    rule = "|---|" + "---|" * len(table)
    reprojection = f"| Reprojection error (px) ↓ [{title}] | " + " | ".join(
        _format(row["reprojection_mean_px"], row["reprojection_std_px"]) for row in table) + " |"
    icp = f"| ICP alignment RMSE (mm) ↓ [{title}] | " + " | ".join(
        _format(row["icp_mean_mm"], row["icp_std_mm"]) for row in table) + " |"
    return "\n".join((header, rule, reprojection, icp)) + "\n"


def comparison_markdown(table: list[dict[str, Any]], paper: dict[str, dict], label: str) -> str:
    """Side-by-side "reproduction / paper" cells for one setting."""

    objects = [row["object"] for row in table]
    lines = [f"| {label} | " + " | ".join(objects) + " |", "|---|" + "---|" * len(objects)]
    for metric, mean_key, std_key in (("reprojection_px", "reprojection_mean_px",
                                       "reprojection_std_px"),
                                      ("icp_mm", "icp_mean_mm", "icp_std_mm")):
        ours = [_format(row[mean_key], row[std_key]) for row in table]
        reference = [
            _format(*paper[metric][name]) if name in paper.get(metric, {}) else "-"
            for name in objects
        ]
        title = "Reprojection error (px)" if metric == "reprojection_px" else "ICP alignment RMSE (mm)"
        lines.append(f"| {title}: this run | " + " | ".join(ours) + " |")
        lines.append(f"| {title}: paper | " + " | ".join(reference) + " |")
    return "\n".join(lines) + "\n"


def _speeds(rows: list[dict[str, Any]]) -> list[str]:
    speeds = [speed for speed in ("slow", "medium", "fast")
              if any(str(r["speed"]).lower() == speed for r in rows)]
    return speeds + sorted({str(r["speed"]).lower() for r in rows} - set(speeds))


def plot_frequency_cdf_comparison(series: dict[str, list[dict[str, Any]]], output: Path) -> None:
    """Fig. 6: CDF of f per speed for several estimators, with ground truth."""

    import matplotlib.pyplot as plt
    all_rows = [row for rows in series.values() for row in rows]
    speeds = _speeds(all_rows) or [""]
    colours = {"eSpinSLAM": "#1f77b4", "FFT f estimation": "#ff7f0e"}
    fig, axes = plt.subplots(1, len(speeds), figsize=(3.6 * len(speeds), 3.6), squeeze=False)
    for axis, speed in zip(axes[0], speeds):
        for label, rows in series.items():
            estimates = np.sort([r["frequency_est_hz"] for r in rows
                                 if str(r["speed"]).lower() == speed
                                 and r["frequency_est_hz"] is not None])
            if estimates.size:
                cumulative = np.arange(1, estimates.size + 1) / estimates.size
                axis.step(np.r_[estimates[0], estimates], np.r_[0.0, cumulative], where="post",
                          linewidth=2, color=colours.get(label), label=label)
        for truth in sorted({r["frequency_true_hz"] for r in all_rows
                             if str(r["speed"]).lower() == speed
                             and r["frequency_true_hz"] is not None}):
            axis.axvline(truth, color="#d62728", linestyle="--", linewidth=1.2)
        axis.set_title(f"{speed.capitalize() or 'All'} speed")
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.3)
    axes[0][0].set_ylabel("CDF")
    axes[0][0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_throughput_summary(rows: list[dict[str, Any]], output: Path) -> None:
    """Fig. 5 analogue: front-end, full system, and the scenes' own event rate."""

    import matplotlib.pyplot as plt
    values = {
        "Front-end (ETC analogue)": _mean_std([r["frontend_throughput_ev_s"] for r in rows])[0],
        "eSpinSLAM (system)": _mean_std([r["throughput_ev_s"] for r in rows])[0],
        "TOPSPIN scenes (event rate)": _mean_std([r["scene_event_rate_ev_s"] for r in rows])[0],
    }
    labels = [label for label, value in values.items() if value]
    fig, axis = plt.subplots(figsize=(7.0, 2.8))
    axis.barh(labels, [values[label] for label in labels],
              color=["#1f77b4", "#8b0000", "#d2691e"][:len(labels)])
    axis.set_xscale("log")
    axis.set_xlabel("Throughput (events/s)")
    axis.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_frequency_cdf(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt
    speeds = [speed for speed in ("slow", "medium", "fast")
              if any(str(r["speed"]).lower() == speed for r in rows)]
    speeds += sorted({str(r["speed"]).lower() for r in rows} - set(speeds))
    fig, axes = plt.subplots(1, max(1, len(speeds)), figsize=(3.6 * max(1, len(speeds)), 3.6),
                             squeeze=False)
    for axis, speed in zip(axes[0], speeds or [""]):
        subset = [r for r in rows if str(r["speed"]).lower() == speed]
        estimates = np.sort([r["frequency_est_hz"] for r in subset
                             if r["frequency_est_hz"] is not None])
        truths = sorted({r["frequency_true_hz"] for r in subset
                         if r["frequency_true_hz"] is not None})
        if estimates.size:
            cumulative = np.arange(1, estimates.size + 1) / estimates.size
            axis.step(np.r_[estimates[0], estimates], np.r_[0.0, cumulative],
                      where="post", color="#1f77b4", linewidth=2, label="eSpinSLAM")
        for truth in truths:
            axis.axvline(truth, color="#d62728", linestyle="--", linewidth=1.2)
        axis.set_title(f"{speed.capitalize() or 'All'} speed")
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.3)
    axes[0][0].set_ylabel("CDF")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_throughput(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt
    objects = [name for name in TABLE2_OBJECTS if any(r["object"] == name for r in rows)]
    objects += sorted({r["object"] for r in rows} - set(objects))
    data = [[r["throughput_ev_s"] / 1e6 for r in rows
             if r["object"] == name and r["throughput_ev_s"]] for name in objects]
    rates = [np.mean([r["scene_event_rate_ev_s"] / 1e6 for r in rows
                      if r["object"] == name and r["scene_event_rate_ev_s"]] or [np.nan])
             for name in objects]
    fig, axis = plt.subplots(figsize=(7.5, 4.2))
    axis.boxplot(data, showmeans=True, meanline=True)
    axis.set_xticks(range(1, len(objects) + 1), objects)
    for index, rate in enumerate(rates, 1):
        axis.hlines(rate, index - 0.35, index + 0.35, colors="green", linestyles="--",
                    label="Mean scene throughput" if index == 1 else None)
    axis.set_ylabel("Mean throughput (M events/s)")
    axis.set_xlabel("Object")
    axis.legend(loc="upper right")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_reconstruction(result: ESpinSLAMResult, orbit_radius: float, output: Path) -> None:
    """Figs. 1, 3, 4: reconstructed points inside the estimated camera orbit."""

    import matplotlib.pyplot as plt
    from .geometry import camera_pose
    reconstruction = result.reconstruction
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))
    angle = np.linspace(0.0, 2.0 * np.pi, 361)
    if reconstruction.valid:
        points = reconstruction.points
        rotation, centre = camera_pose(0.0, reconstruction.frequency_hz, orbit_radius,
                                       reconstruction.rotation)
        axis_world = rotation[2]  # optical axis in the world frame
        for axis, (i, j, name) in zip(axes, ((0, 1, "top view (x-y)"), (0, 2, "side view (x-z)"))):
            if j == 1:
                axis.plot(orbit_radius * np.cos(angle), orbit_radius * np.sin(angle), ":",
                          color="red", linewidth=1, label="camera orbit")
            else:
                axis.plot([-orbit_radius, orbit_radius], [0.0, 0.0], ":", color="red",
                          linewidth=1, label="camera orbit")
            axis.scatter(points[:, i], points[:, j], s=3, color="black", label="world points")
            axis.quiver(centre[i], centre[j], axis_world[i], axis_world[j], color="blue",
                        angles="xy", scale_units="xy", scale=0.4, label="camera at t=0")
            axis.set_aspect("equal", adjustable="datalim")
            axis.set_title(name)
            axis.grid(alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=7)
    else:
        for axis in axes:
            axis.text(0.5, 0.5, reconstruction.reason or "no reconstruction",
                      ha="center", va="center", transform=axis.transAxes)
    fig.suptitle(f"{result.sequence_id}: f* = {result.frequency_hz:.4f} Hz, "
                 f"{reconstruction.points.shape[0]} world points")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_history(result: ESpinSLAMResult, output: Path) -> None:
    import matplotlib.pyplot as plt
    reports = result.optimizations
    fig, axis = plt.subplots(figsize=(6.0, 3.6))
    if reports:
        times = [report.event_time for report in reports]
        axis.plot(times, [report.mean_reprojection_error_px for report in reports],
                  color="red", linewidth=2)
        twin = axis.twinx()
        twin.plot(times, [report.landmarks for report in reports], color="blue", linewidth=2)
        twin.set_ylabel("Number of Landmarks", color="blue")
    axis.set_xlabel("Timestamp (s)")
    axis.set_ylabel("Mean reprojection error (px)", color="red")
    axis.set_title(result.sequence_id)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def evaluate_topspin(config: ESpinSLAMConfig, data_root: Path, manifest_path: Path,
                     output_dir: Path, *, max_sequences: int = 0,
                     setting_name: str = "eSpinSLAM",
                     history_plots: bool = True) -> dict[str, Any]:
    """Run eSpinSLAM on every manifest row and write paper-style reports."""

    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_topspin_manifest(manifest_path, data_root)
    if max_sequences > 0:
        rows = rows[:max_sequences]
    model = ESpinSLAM(config)
    results: list[dict[str, Any]] = []
    references: dict[tuple[str, float], np.ndarray] = {}
    root = Path(data_root).resolve()
    for index, row in enumerate(rows, 1):
        camera = camera_for_row(config.camera, row)
        width = int(row["width"]) if row.get("width") else config.data.sensor_size[0]
        height = int(row["height"]) if row.get("height") else config.data.sensor_size[1]
        sequence = load_events_as_sequence(
            root / row["event_path"], row["sequence_id"], width=width, height=height,
            time_unit=row.get("time_unit") or config.data.time_unit,
            text_columns=config.data.text_columns,
            max_duration_seconds=config.data.max_duration_seconds,
        )
        result = model.process(sequence, camera=camera)
        icp_rmse = chamfer = None
        icp_reason, scale_source = "", ""
        if row.get("cad_path"):
            unit = _float_or_none(row.get("cad_unit_mm")) or 1.0
            key = (row["cad_path"], unit)
            if key not in references:
                references[key] = load_reference_cloud(
                    (root / row["cad_path"]).resolve(),
                    config.evaluation.pointcloud_max_reference_points, unit,
                )
            orbit_mm = _float_or_none(row.get("orbit_radius_mm"))
            known_scale = orbit_mm / config.backend.orbit_radius if orbit_mm else None
            scale_source = "orbit_radius_mm" if known_scale else "rms_radius"
            if result.reconstruction.valid:
                alignment = align_similarity_icp(result.reconstruction.points, references[key],
                                                 known_scale=known_scale)
                icp_rmse = _nullable(alignment.icp_rmse) if alignment.valid else None
                chamfer = _nullable(alignment.chamfer_rmse_m) if alignment.valid else None
                icp_reason = alignment.reason
                if alignment.valid:
                    (output_dir / "pointclouds").mkdir(exist_ok=True)
                    np.savez_compressed(
                        output_dir / "pointclouds" / f"{sequence.sequence_id}.npz",
                        predicted=result.reconstruction.points.astype(np.float32),
                        aligned_mm=alignment.aligned_points.astype(np.float32),
                    )
            else:
                icp_reason = result.reconstruction.reason
        else:
            icp_reason = "no cad_path in manifest"
        record = _sequence_row(row, result, icp_rmse_mm=icp_rmse, chamfer_mm=chamfer,
                               icp_reason=icp_reason, scale_source=scale_source)
        results.append(record)
        history_dir = output_dir / "history"
        _write_csv(history_dir / f"{sequence.sequence_id}_optimizations.csv", [
            {"event_time_s": r.event_time, "frequency_hz": r.frequency_hz,
             "landmarks": r.landmarks, "observations": r.observations,
             "mean_reprojection_error_px": _nullable(r.mean_reprojection_error_px),
             "removed_reprojection": r.removed_reprojection, "removed_radius": r.removed_radius,
             "fused": r.fused, "iterations": r.iterations}
            for r in result.optimizations
        ] or [{"event_time_s": None}])
        _write_csv(history_dir / f"{sequence.sequence_id}_frequency.csv", [
            {"event_time_s": r.event_time, "wall_s": r.wall_seconds,
             "frequency_hz": _nullable(r.frequency_hz), "loop_closed": r.loop_closed}
            for r in result.frequency_history
        ] or [{"event_time_s": None}])
        if history_plots:
            _plot_history(result, output_dir / f"fig8_{sequence.sequence_id}.png")
            _plot_reconstruction(result, config.backend.orbit_radius,
                                 output_dir / f"reconstruction_{sequence.sequence_id}.png")
        print(
            f"[{setting_name}] [{index}/{len(rows)}] {sequence.sequence_id}: "
            f"f={record['frequency_est_hz']} loop={result.loop_closed} "
            f"reproj={record['reprojection_error_px']} icp_mm={icp_rmse} "
            f"throughput={record['throughput_ev_s']}",
            flush=True,
        )
    _write_csv(output_dir / "per_sequence.csv", results)
    table = table2(results)
    _write_csv(output_dir / "table2.csv", table)
    _write_text(output_dir / "table2.md", table2_markdown(table, setting_name))
    paper_reference = PAPER_TABLE3.get(setting_name)
    if paper_reference is not None:
        _write_text(output_dir / "table_vs_paper.md",
                    comparison_markdown(table, paper_reference, setting_name))
    if any(r["frequency_est_hz"] is not None for r in results):
        _plot_frequency_cdf(results, output_dir / "fig6_frequency_cdf.png")
    _plot_throughput(results, output_dir / "fig7_throughput.png")
    _plot_throughput_summary(results, output_dir / "fig5_throughput_summary.png")
    frequency_errors = [r["frequency_rel_error_percent"] for r in results
                        if r["frequency_rel_error_percent"] is not None]
    report = {
        "method": "Paper-faithful eSpinSLAM reproduction",
        "paper": "Event-based Spinning Object SLAM, ICCVW 2025",
        "setting": setting_name,
        "dataset": "TOPSPIN",
        "sequence_count": len(results),
        "loop_closure_rate": (sum(bool(r["loop_closed"]) for r in results) / len(results)
                              if results else None),
        "frequency_median_rel_error_percent": (float(np.median(frequency_errors))
                                               if frequency_errors else None),
        "mean_throughput_ev_s": _mean_std([r["throughput_ev_s"] for r in results])[0],
        "mean_frontend_throughput_ev_s": _mean_std(
            [r["frontend_throughput_ev_s"] for r in results])[0],
        "mean_scene_event_rate_ev_s": _mean_std([r["scene_event_rate_ev_s"] for r in results])[0],
        "real_time_rate": (sum(bool(r["real_time"]) for r in results) / len(results)
                           if results else None),
        "table2": table,
        "runtime_seconds": time.monotonic() - started,
        "implementation_notes": [
            "GTSAM is replaced by a Levenberg-Marquardt solver with Huber IRLS weights "
            "(GTSAM's robust noise model strategy) and analytic Jacobians.",
            "Eq. (9) is evaluated over left-half events whose shifted time falls inside "
            "the right half; a time scale lambda makes pixels and seconds commensurate.",
            "Tracks spanning less than backend.min_triangulation_angle_deg of orbit are "
            "under-constrained and wait until they grow (triangulation parallax check).",
            "ICP scale: orbit_radius_mm / r when provided, otherwise the RMS-radius ratio.",
        ],
        "config": config.as_dict(),
    }
    _write_text(output_dir / "metrics.json",
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def _with_overrides(config: ESpinSLAMConfig, overrides: dict[str, dict[str, Any]]) -> ESpinSLAMConfig:
    result = copy.deepcopy(config)
    for section, values in overrides.items():
        setattr(result, section, replace(getattr(result, section), **values))
    return result


def evaluate_topspin_ablations(config: ESpinSLAMConfig, data_root: Path, manifest_path: Path,
                               output_dir: Path, *, settings: list[str] | None = None,
                               max_sequences: int = 0) -> dict[str, Any]:
    """Run the Table 3 ablations and the full method, then write table3.*."""

    settings = settings or list(ABLATIONS)
    reports: dict[str, dict[str, Any]] = {}
    for key in settings:
        label, overrides = ABLATIONS[key]
        reports[key] = evaluate_topspin(
            _with_overrides(config, overrides), data_root, manifest_path,
            output_dir / key, max_sequences=max_sequences, setting_name=label,
            history_plots=False,
        )
    rows = []
    for key in settings:
        for entry in reports[key]["table2"]:
            rows.append({"setting": ABLATIONS[key][0], **entry})
    _write_csv(output_dir / "table3.csv", rows)
    objects = []
    for row in rows:
        if row["object"] not in objects:
            objects.append(row["object"])
    lines = ["| Setting | " + " | ".join(objects) + " |", "|---|" + "---|" * len(objects)]
    for metric, mean_key, std_key in (("Reprojection error (px)", "reprojection_mean_px",
                                       "reprojection_std_px"),
                                      ("ICP alignment RMSE (mm)", "icp_mean_mm", "icp_std_mm")):
        lines.append(f"| **{metric}** |" + " |" * len(objects))
        for key in settings:
            label = ABLATIONS[key][0]
            cells = []
            for name in objects:
                match = [r for r in rows if r["setting"] == label and r["object"] == name]
                cells.append(_format(match[0][mean_key], match[0][std_key]) if match else "-")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    _write_text(output_dir / "table3.md", "\n".join(lines) + "\n")
    comparison = []
    for key in settings:
        label = ABLATIONS[key][0]
        if label in PAPER_TABLE3:
            comparison.append(comparison_markdown(reports[key]["table2"], PAPER_TABLE3[label], label))
    _write_text(output_dir / "table3_vs_paper.md", "\n".join(comparison))
    if "espinslam" in settings and "fft_frequency" in settings:
        series = {}
        for key in ("espinslam", "fft_frequency"):
            with (output_dir / key / "per_sequence.csv").open(newline="", encoding="utf-8") as handle:
                series[ABLATIONS[key][0]] = [
                    {"speed": row["speed"],
                     "frequency_est_hz": _float_or_none(row["frequency_est_hz"]),
                     "frequency_true_hz": _float_or_none(row["frequency_true_hz"])}
                    for row in csv.DictReader(handle)
                ]
        plot_frequency_cdf_comparison(series, output_dir / "fig6_loop_closure_vs_fft.png")
    return {"settings": settings, "reports": reports}
