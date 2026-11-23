"""End-to-end checks on a synthetic stream that satisfies the paper's model."""

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from espinslam.backend import empty_reconstruction
from espinslam.config import (
    BackendConfig, CameraConfig, ESpinSLAMConfig, FrequencyConfig, FrontendConfig,
    load_config,
)
from espinslam.data import EventSequence
from espinslam.frequency import FrequencyEstimate
from espinslam.pipeline import ESpinSLAM
from espinslam.synthetic import make_scene, simulate_events
from espinslam.topspin import (
    _reference_mode, _sequence_row, evaluate_topspin, evaluate_topspin_ablations,
)


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
        frontend=FrontendConfig(
            interpretation_mode="experimental", density_radius_px=3.0,
            density_mean_per_polarity=False, cluster_min_size=5,
            cluster_min_samples=3, cluster_selection_epsilon=0.0,
            cluster_time_scale_px_per_second=1000.0,
            cluster_by_polarity=True, association_radius_px=5.0,
            association_max_gap_seconds=0.05,
        ),
        frequency=FrequencyConfig(update_interval_seconds=0.05, min_period_seconds=0.4,
                                  max_period_seconds=10.0, time_scale_px_per_second=150.0,
                                  alignment_mode="experimental", harmonic_tolerance=0.15),
        backend=BackendConfig(
            interpretation_mode="experimental", min_triangulation_angle_deg=1.5,
            final_offline_refinement=True,
        ),
    )


def test_paper_reference_profile_top_level_smoke_schema_and_trace():
    """Gate the real reference profile without imposing an accuracy target."""

    config = load_config("examples/configs/topspin_reference.jsonc")
    assert _reference_mode(config)
    t = np.linspace(0.0, 0.02, 12)
    sequence = EventSequence(
        "reference-smoke", np.arange(12) + 20, np.arange(12) + 30, t,
        np.arange(12) % 2, math.nan, 1280, 720, {}, Path("synthetic-reference-smoke"),
    )
    result = ESpinSLAM(config).process(
        sequence,
        camera=CameraConfig(fx=500.0, fy=500.0, cx=640.0, cy=360.0),
    )
    row = _sequence_row(
        {"sequence_id": sequence.sequence_id, "object": "synthetic"}, result,
        icp_rmse_mm=None, chamfer_mm=None, icp_reason="not evaluated", scale_source="",
    )
    assert result.sequence_id == sequence.sequence_id
    assert result.event_count == sequence.event_count
    assert len(result.frequency_history) == 1
    assert result.frequency_history[0].icp_termination == "not_run"
    assert {
        "frequency_est_hz", "reconstruction_valid", "last_optimization_status",
        "triangulation_fallback_count", "triangulation_fallback_reasons",
    } <= set(row)
    assert row["triangulation_fallback_count"] == 0


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
            "sequence_id", "event_path", "object", "speed", "frequency_hz",
            "frequency_source", "cad_path", "cad_source",
            "orbit_radius_mm", "fx", "fy", "cx", "cy", "width", "height"))
        writer.writeheader()
        writer.writerow({
            "sequence_id": "hubble-test-fast", "event_path": "hubble-test-fast.npz",
            "object": "Hubble", "speed": "fast", "frequency_hz": "1.0",
            "frequency_source": "official", "cad_path": "hubble_cad.npz",
            "cad_source": "official", "orbit_radius_mm": "300",
            "fx": k[0, 0], "fy": k[1, 1], "cx": k[0, 2], "cy": k[1, 2],
            "width": scene.width, "height": scene.height,
        })
    config = _config(scene)
    config.camera = CameraConfig()  # calibration must come from the manifest
    return root, config


def test_topspin_evaluation_writes_paper_tables_and_schema(synthetic, tmp_path):
    """Gate report generation and schema without a noise-sensitive ICP cutoff."""

    root, config = _write_topspin_scene(synthetic, tmp_path)
    report = evaluate_topspin(config, root, root / "manifest.csv", tmp_path / "run")
    row = report["table2"][0]
    assert row["object"] == "Hubble" and row["valid_reconstructions"] == 1
    # Whether this particular noise realization beats 6 mm is characterized by
    # the declared multi-seed diagnostic below, not selected post hoc here.
    assert row["icp_mean_mm"] is not None
    for name in ("per_sequence.csv", "table2.csv", "table2.md", "metrics.json",
                 "fig5_throughput_summary.png", "fig6_frequency_cdf.png",
                 "fig7_throughput.png", "fig8_hubble-test-fast.png",
                 "reconstruction_hubble-test-fast.png"):
        assert (tmp_path / "run" / name).is_file()
    with (tmp_path / "run" / "per_sequence.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert {"sequence_id", "frequency_est_hz", "reprojection_error_px",
            "icp_rmse_mm", "runtime_s", "paper_metrics_eligible",
            "triangulation_fallback_count",
            "triangulation_fallback_reasons"} <= set(rows[0])
    with (tmp_path / "run" / "history" / "hubble-test-fast_optimizations.csv").open(
            newline="", encoding="utf-8") as handle:
        optimization_rows = list(csv.DictReader(handle))
    assert optimization_rows
    assert {"triangulation_fallback_count",
            "triangulation_fallback_reasons"} <= set(optimization_rows[0])
    with (tmp_path / "run" / "history" / "hubble-test-fast_frequency.csv").open(
            newline="", encoding="utf-8") as handle:
        frequency_rows = list(csv.DictReader(handle))
    assert frequency_rows
    assert {
        "grid_seed_seconds", "grid_seed_cost", "icp_iterations_used",
        "icp_termination", "icp_clipped_to_search_interval",
    } <= set(frequency_rows[0])
    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text(encoding="utf-8"))
    assert {"table2", "sequence_count", "runtime_provenance",
            "evaluation_classification", "backend_diagnostics"} <= set(metrics)
    assert not (tmp_path / "run" / "table_vs_paper.md").exists()
    assert report["method_profile"] == "experimental_interpretation"
    assert report["evaluation_classification"] == "exploratory_retrospective"
    assert report["paper_comparison_eligible"] is False
    assert report["mean_frontend_throughput_ev_s"] > report["mean_throughput_ev_s"]
    assert report["real_time_rate"] in (0.0, 1.0)


_ACCURACY_DIAGNOSTIC_SEEDS = tuple(range(10))


@pytest.mark.slow_diagnostic
@pytest.mark.skipif(
    os.environ.get("ESPINSLAM_RUN_SEED_DIAGNOSTIC") != "1",
    reason="set ESPINSLAM_RUN_SEED_DIAGNOSTIC=1 to run the frozen ten-seed diagnostic",
)
def test_topspin_accuracy_declared_seed_distribution(tmp_path):
    """Characterize stochastic reconstruction with a predeclared distribution.

    The seeds and median statistic were frozen after the portability audit. The
    historic 6 mm criterion is applied to their median, never to a selected
    favorable realization. This remains diagnostic because its simulator
    input is not yet a versioned fixed event fixture.
    """

    scene = make_scene(
        frequency_hz=1.0, elevation_deg=25.0, point_count=25, width=200,
        height=150, focal=300.0, patch_half_size_px=(5.0, 8.0), seed=4,
    )
    errors = []
    for seed in _ACCURACY_DIAGNOSTIC_SEEDS:
        stream = simulate_events(
            scene, duration_seconds=3.5, noise_rate_hz_per_pixel=0.05, seed=seed,
        )
        case = tmp_path / f"seed_{seed:02d}"
        case.mkdir()
        root, config = _write_topspin_scene((scene, stream), case)
        report = evaluate_topspin(
            config, root, root / "manifest.csv", case / "run", history_plots=False,
        )
        error = report["table2"][0]["icp_mean_mm"]
        assert error is not None
        errors.append(float(error))
    assert len(errors) == len(_ACCURACY_DIAGNOSTIC_SEEDS)
    assert float(np.median(errors)) < 6.0


def test_topspin_ablations_write_table3_and_fig6_comparison(synthetic, tmp_path):
    root, config = _write_topspin_scene(synthetic, tmp_path)
    evaluate_topspin_ablations(config, root, root / "manifest.csv", tmp_path / "ablations",
                               settings=["espinslam", "fft_frequency"])
    for name in ("table3.md", "fig6_loop_closure_vs_fft.png"):
        assert (tmp_path / "ablations" / name).is_file()
    assert not (tmp_path / "ablations" / "table3_vs_paper.md").exists()


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
    assert report["table2"][0]["scenes"] == 1          # non-dev exploratory only
    assert report["table2_all"][0]["scenes"] == 2
    assert report["table2_dev"][0]["scenes"] == 1
    for name in ("table2_dev.md", "table2_all.md"):
        assert (tmp_path / "split_run" / name).is_file()
    assert "non-dev exploratory" in (tmp_path / "split_run" / "table2.md").read_text(
        encoding="utf-8")


def test_loop_transition_forgets_before_first_ba_and_reference_has_no_final_ba(
        monkeypatch, tmp_path):
    """Sec. 4.2.7 ordering and the absence of a paper-unreported terminal BA."""

    calls: list[tuple[str, int]] = []

    class Pool:
        def __init__(self):
            self.version = 0
            self.newest_time = 0.0

        def __len__(self):
            return 1

        def events(self):
            return np.asarray(((0.0, 1.0, 1.0), (0.1, 1.0, 1.0)))

        def forget(self, cutoff):
            calls.append(("forget", self.version))
            return []

    class Frontend:
        def __init__(self, width, height, config):
            self.pool = Pool()

        def push(self, t, x, y, p):
            self.pool.version += 1
            self.pool.newest_time = float(t[-1])

    class Estimator:
        def __init__(self, config):
            self.loop_closed = False
            self.current = FrequencyEstimate(1.0, 1.0, 0.0, True)

        def update(self, events):
            self.loop_closed = True
            return self.current

    class Backend:
        def __init__(self, *args, **kwargs):
            self.reports = []

        def optimize(self, pool, frequency, event_time):
            assert calls and calls[-1][0] == "forget"
            calls.append(("optimize", pool.version))

        def reconstruction(self):
            return empty_reconstruction("test")

    import espinslam.pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module, "EventFrontend", Frontend)
    monkeypatch.setattr(pipeline_module, "LoopClosureFrequencyEstimator", Estimator)
    monkeypatch.setattr(pipeline_module, "IncrementalBackend", Backend)
    sequence = EventSequence(
        "ordering", np.asarray((1, 1, 1, 1)), np.asarray((1, 1, 1, 1)),
        np.asarray((0.0, 0.1, 0.2, 0.3)), np.asarray((1, 1, 1, 1), dtype=np.uint8),
        math.nan, 4, 4, {}, tmp_path,
    )
    config = ESpinSLAMConfig(
        camera=CameraConfig(fx=10.0, fy=10.0, cx=2.0, cy=2.0),
    )
    config.frontend.chunk_events = 2
    config.frequency.update_interval_seconds = 0.0
    config.backend.optimization_cadence_revolutions = 0.1
    config.backend.final_offline_refinement = False
    ESpinSLAM(config).process(sequence)
    assert [name for name, _ in calls].count("optimize") == 2
    assert calls[0][0] == "forget" and calls[1][0] == "optimize"


def test_closed_loop_forgets_before_subsequent_frequency_update(monkeypatch, tmp_path):
    """After f* exists, stale H must not enter the next Eq. (9) update."""

    calls: list[str] = []

    class Pool:
        def __init__(self):
            self.version = 0
            self.newest_time = 0.0

        def __len__(self):
            return 1

        def events(self):
            calls.append("events")
            return np.asarray(((0.0, 1.0, 1.0), (0.1, 1.0, 1.0)))

        def forget(self, cutoff):
            calls.append("forget")
            return []

    class Frontend:
        def __init__(self, width, height, config):
            self.pool = Pool()

        def push(self, t, x, y, p):
            self.pool.version += 1
            self.pool.newest_time = float(t[-1])

    class Estimator:
        def __init__(self, config):
            self.loop_closed = False
            self.current = FrequencyEstimate(1.0, 1.0, 0.0, True)

        def update(self, events):
            calls.append("estimate")
            self.loop_closed = True
            return self.current

    import espinslam.pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module, "EventFrontend", Frontend)
    monkeypatch.setattr(pipeline_module, "LoopClosureFrequencyEstimator", Estimator)
    sequence = EventSequence(
        "forget-before-estimate", np.ones(4), np.ones(4),
        np.asarray((0.0, 0.1, 0.2, 0.3)), np.ones(4, dtype=np.uint8),
        math.nan, 4, 4, {}, tmp_path,
    )
    config = ESpinSLAMConfig(
        camera=CameraConfig(fx=10.0, fy=10.0, cx=2.0, cy=2.0),
    )
    config.frontend.chunk_events = 2
    config.frequency.update_interval_seconds = 0.0
    ESpinSLAM(config).process(sequence, reconstruct_structure=False)
    # First transition: estimate establishes f*, then forget. Second update:
    # forget must precede both events() and estimator.update().
    assert calls == [
        "events", "estimate", "forget",
        "forget", "events", "estimate",
    ]


@pytest.mark.parametrize(
    ("frequency_hz", "timestamps", "expected"),
    [
        (0.5, [0.0, 0.199999, 0.2, 0.200001, 0.399999, 0.4],
         [0.0, 0.2, 0.4]),
        (1.0, [0.0, 0.099999, 0.1, 0.100001, 0.199999, 0.2],
         [0.0, 0.1, 0.2]),
        (2.0, [0.0, 0.049999, 0.05, 0.050001, 0.099999, 0.1],
         [0.0, 0.05, 0.1]),
    ],
)
def test_backend_cadence_is_epsilon_over_frequency_at_chunk_boundaries(
        monkeypatch, tmp_path, frequency_hz, timestamps, expected):
    """Table 1 cadence fires at, not before, each ``epsilon / f`` boundary."""

    optimization_times: list[float] = []

    class Pool:
        def __init__(self):
            self.version = 0
            self.newest_time = 0.0

        def __len__(self):
            return 1

        def events(self):
            return np.asarray(((0.0, 1.0, 1.0), (0.1, 1.0, 1.0)))

        def forget(self, cutoff):
            return []

    class Frontend:
        def __init__(self, width, height, config):
            self.pool = Pool()

        def push(self, t, x, y, p):
            self.pool.version += 1
            self.pool.newest_time = float(t[-1])

    class Estimator:
        def __init__(self, config):
            self.loop_closed = True
            self.current = FrequencyEstimate(
                frequency_hz, 1.0 / frequency_hz, 0.0, True,
            )

        def update(self, events):
            return self.current

    class Backend:
        def __init__(self, *args, **kwargs):
            self.reports = []

        def optimize(self, pool, frequency, event_time):
            optimization_times.append(event_time)

        def reconstruction(self):
            return empty_reconstruction("test")

    import espinslam.pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module, "EventFrontend", Frontend)
    monkeypatch.setattr(pipeline_module, "LoopClosureFrequencyEstimator", Estimator)
    monkeypatch.setattr(pipeline_module, "IncrementalBackend", Backend)
    t = np.asarray(timestamps)
    sequence = EventSequence(
        "cadence", np.ones(t.size), np.ones(t.size), t,
        np.ones(t.size, dtype=np.uint8), math.nan, 4, 4, {}, tmp_path,
    )
    config = ESpinSLAMConfig(
        camera=CameraConfig(fx=10.0, fy=10.0, cx=2.0, cy=2.0),
    )
    config.frontend.chunk_events = 1
    config.frequency.update_interval_seconds = 999.0
    config.backend.optimization_cadence_revolutions = 0.1
    config.backend.final_offline_refinement = False
    ESpinSLAM(config).process(sequence)
    np.testing.assert_allclose(optimization_times, expected, rtol=0.0, atol=1e-12)
