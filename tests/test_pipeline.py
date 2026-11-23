"""End-to-end checks on a synthetic stream that satisfies the paper's model."""

import csv
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from espinslam.config import CameraConfig, ESpinSLAMConfig, FrequencyConfig
from espinslam.data import EventSequence
from espinslam.pipeline import ESpinSLAM
from espinslam.synthetic import make_scene, simulate_events
from espinslam.topspin import evaluate_topspin, evaluate_topspin_ablations


@pytest.fixture(scope="module")
def synthetic():
    scene = make_scene(frequency_hz=1.0, elevation_deg=25.0, point_count=25, width=200,
                       height=150, focal=300.0, patch_half_size_px=(5.0, 8.0), seed=4)
    stream = simulate_events(scene, duration_seconds=3.5, noise_rate_hz_per_pixel=0.05)
    return scene, stream


def _config(scene) -> ESpinSLAMConfig:
    k = scene.intrinsic
    return ESpinSLAMConfig(
        camera=CameraConfig(fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2]),
        frequency=FrequencyConfig(update_interval_seconds=0.05, min_period_seconds=0.4,
                                  max_period_seconds=10.0, time_scale_px_per_second=150.0),
    )


def test_online_pipeline_recovers_frequency_orientation_and_structure(synthetic):
    scene, stream = synthetic
    sequence = EventSequence("synthetic", stream.x, stream.y, stream.t, stream.p,
                             2 * math.pi, scene.width, scene.height, {}, Path("synthetic"))
    result = ESpinSLAM(_config(scene)).process(sequence)
    assert result.loop_closed
    assert abs(result.frequency_hz - 1.0) < 2e-3
    reconstruction = result.reconstruction
    assert reconstruction.valid and len(result.optimizations) >= 3
    rotation_error = np.degrees(np.linalg.norm(
        Rotation.from_matrix(reconstruction.rotation @ scene.rotation.T).as_rotvec()))
    assert rotation_error < 2.5  # low-resolution 200x150 test scene
    distance, _ = cKDTree(scene.feature_points).query(reconstruction.points)
    assert np.median(distance) < 0.2  # world units; the object radius is 1.5
    assert reconstruction.mean_reprojection_error_px < 3.0
    # Causal output: no confirmed frequency before the loop is detected.
    before, _ = result.frequency_at(result.loop_closure_time - 0.06)
    after, _ = result.frequency_at(result.loop_closure_time)
    assert math.isnan(before) and math.isfinite(after)


def _write_topspin_scene(synthetic, tmp_path):
    scene, stream = synthetic
    root = tmp_path / "topspin"
    root.mkdir()
    np.savez(root / "hubble-test-fast.npz", t=(stream.t * 1e6).astype(np.int64),
             x=stream.x, y=stream.y, p=stream.p)
    np.savez(root / "hubble_cad.npz", points=scene.feature_points * 30.0)  # CAD in mm
    k = scene.intrinsic
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "sequence_id", "event_path", "object", "speed", "frequency_hz", "cad_path",
            "orbit_radius_mm", "fx", "fy", "cx", "cy", "width", "height"))
        writer.writeheader()
        writer.writerow({
            "sequence_id": "hubble-test-fast", "event_path": "hubble-test-fast.npz",
            "object": "Hubble", "speed": "fast", "frequency_hz": "1.0",
            "cad_path": "hubble_cad.npz", "orbit_radius_mm": "300",
            "fx": k[0, 0], "fy": k[1, 1], "cx": k[0, 2], "cy": k[1, 2],
            "width": scene.width, "height": scene.height,
        })
    config = _config(scene)
    config.camera = CameraConfig()  # calibration must come from the manifest
    return root, config


def test_topspin_evaluation_writes_paper_tables(synthetic, tmp_path):
    root, config = _write_topspin_scene(synthetic, tmp_path)
    report = evaluate_topspin(config, root, root / "manifest.csv", tmp_path / "run")
    row = report["table2"][0]
    assert row["object"] == "Hubble" and row["valid_reconstructions"] == 1
    assert row["icp_mean_mm"] is not None and row["icp_mean_mm"] < 6.0  # object ~90 mm
    for name in ("per_sequence.csv", "table2.md", "table_vs_paper.md", "metrics.json",
                 "fig5_throughput_summary.png", "fig6_frequency_cdf.png",
                 "fig7_throughput.png", "fig8_hubble-test-fast.png",
                 "reconstruction_hubble-test-fast.png"):
        assert (tmp_path / "run" / name).is_file()
    assert "paper" in (tmp_path / "run" / "table_vs_paper.md").read_text(encoding="utf-8")
    assert report["mean_frontend_throughput_ev_s"] > report["mean_throughput_ev_s"]
    assert report["real_time_rate"] in (0.0, 1.0)


def test_topspin_ablations_write_table3_and_fig6_comparison(synthetic, tmp_path):
    root, config = _write_topspin_scene(synthetic, tmp_path)
    evaluate_topspin_ablations(config, root, root / "manifest.csv", tmp_path / "ablations",
                               settings=["espinslam", "fft_frequency"])
    for name in ("table3.md", "table3_vs_paper.md", "fig6_loop_closure_vs_fft.png"):
        assert (tmp_path / "ablations" / name).is_file()
    text = (tmp_path / "ablations" / "table3_vs_paper.md").read_text(encoding="utf-8")
    assert "FFT f estimation" in text and "5.61 ± 7.93" in text


def test_split_column_reports_held_out_and_dev_separately(synthetic, tmp_path):
    root, config = _write_topspin_scene(synthetic, tmp_path)
    manifest = root / "manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    dev = dict(rows[0], sequence_id="hubble-dev-fast", split="dev")
    test = dict(rows[0], split="test")
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dev))
        writer.writeheader()
        writer.writerows([dev, test])
    report = evaluate_topspin(config, root, manifest, tmp_path / "split_run", history_plots=False)
    assert report["split_counts"] == {"dev": 1, "test": 1}
    assert report["table2"][0]["scenes"] == 1          # held-out only
    assert report["table2_all"][0]["scenes"] == 2
    assert report["table2_dev"][0]["scenes"] == 1
    for name in ("table2_dev.md", "table2_all.md"):
        assert (tmp_path / "split_run" / name).is_file()
    assert "held-out" in (tmp_path / "split_run" / "table2.md").read_text(encoding="utf-8")
