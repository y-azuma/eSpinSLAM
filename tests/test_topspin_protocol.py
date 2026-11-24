import csv
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest

from espinslam.backend import OptimizationReport, empty_reconstruction
from espinslam.config import CameraConfig, ESpinSLAMConfig, load_config
from espinslam.data import EventSequence
from espinslam.evaluation import _method_profile as _debris_method_profile
from espinslam.frequency import FrequencyEstimate
from espinslam.pipeline import ESpinSLAMResult
from espinslam.topspin import (
    REVIEW18, TOPSPIN_SCENES, _checkpoint_fingerprint, _prepare_run_directory,
    _method_profile, _reference_mismatches, _reference_mode, _safe_sequence_id,
    _same_speed_consistency, _sequence_row, _validate_verified_manifest_row,
    evaluate_topspin, evaluate_topspin_ablations, table2,
)


def test_same_speed_consistency_uses_loop_closure_not_post_ba_estimate():
    rows = [
        {"speed": "slow", "frequency_loop_closure_hz": 0.1,
         "frequency_est_hz": 10.0},
        {"speed": "slow", "frequency_loop_closure_hz": 0.2,
         "frequency_est_hz": 10.0},
    ]
    by_speed, deviations = _same_speed_consistency(rows)

    assert by_speed["slow"]["median_hz"] == pytest.approx(0.15)
    assert by_speed["slow"]["min_hz"] == 0.1
    assert by_speed["slow"]["max_hz"] == 0.2
    assert np.allclose(deviations, [100.0 / 3.0, 100.0 / 3.0])


def _result(sequence_id="scene"):
    return ESpinSLAMResult(
        sequence_id=sequence_id,
        observation_seconds=1.0,
        event_count=10,
        frequency=FrequencyEstimate(0.5, 2.0, 1.0, True),
        loop_closed=True,
        loop_closure_time=0.8,
        reconstruction=empty_reconstruction("test fixture"),
        runtime_seconds=0.5,
        frontend_seconds=0.2,
        frequency_seconds=0.2,
        backend_seconds=0.1,
    )


def test_frozen_review18_file_matches_code_constant():
    path = Path("examples/splits/topspin_review18.txt")
    names = tuple(
        line.split("#", 1)[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    )
    assert names == REVIEW18
    assert len(names) == 18 and len(set(names)) == 18


def test_frequency_error_requires_explicit_authoritative_provenance():
    common = {"sequence_id": "scene", "object": "Hubble", "speed": "fast",
              "frequency_hz": "0.4"}
    unavailable = _sequence_row(
        common, _result(), icp_rmse_mm=None, chamfer_mm=None,
        icp_reason="unavailable", scale_source="",
    )
    assert unavailable["frequency_reference_hz"] == 0.4
    assert unavailable["frequency_true_hz"] is None
    assert unavailable["frequency_abs_error_hz"] is None
    authoritative = _sequence_row(
        dict(common, frequency_source="author_provided"), _result(),
        icp_rmse_mm=None, chamfer_mm=None, icp_reason="unavailable", scale_source="",
    )
    assert authoritative["frequency_true_hz"] == 0.4
    assert math.isclose(authoritative["frequency_abs_error_hz"], 0.1)


def test_observation_weighted_reprojection_is_primary_and_landmark_mean_is_retained():
    reconstruction = replace(
        empty_reconstruction("fixture"),
        points=np.zeros((2, 3)), track_ids=np.asarray((1, 2)),
        observation_count=10, valid=True, reason="",
        mean_observation_error_px=2.0, mean_reprojection_error_px=4.0,
    )
    result = replace(_result(), reconstruction=reconstruction)
    row = _sequence_row(
        {"sequence_id": "scene", "object": "Hubble"}, result,
        icp_rmse_mm=None, chamfer_mm=None, icp_reason="unavailable", scale_source="",
    )
    assert row["reprojection_error_px"] == 2.0
    assert row["reprojection_error_observation_mean_px"] == 2.0
    assert row["reprojection_error_landmark_mean_px"] == 4.0
    aggregate = table2([row])[0]
    assert aggregate["reprojection_mean_px"] == 2.0
    assert aggregate["landmark_reprojection_mean_px"] == 4.0


def test_max_iterations_geometry_is_preserved_but_excluded_from_paper_metrics():
    reconstruction = replace(
        empty_reconstruction("fixture"),
        points=np.zeros((2, 3)), track_ids=np.asarray((1, 2)),
        observation_count=10, valid=True, reason="",
        mean_observation_error_px=2.0, mean_reprojection_error_px=4.0,
    )
    optimization = OptimizationReport(
        event_time=1.0, frequency_hz=0.5, landmarks=2, observations=10,
        mean_reprojection_error_px=4.0, removed_short=0,
        removed_reprojection=0, removed_radius=0, fused=0, iterations=100,
        termination_status="max_iterations", mean_observation_error_px=2.0,
        triangulation_fallback_count=2,
        triangulation_fallback_reasons="triangulation_failed_used_origin:2",
    )
    result = replace(_result(), reconstruction=reconstruction, optimizations=[optimization])
    row = _sequence_row(
        {"sequence_id": "scene", "object": "Hubble"}, result,
        icp_rmse_mm=1.0, chamfer_mm=1.0, icp_reason="", scale_source="author",
    )
    assert row["reconstruction_valid"] is True
    assert row["last_optimization_status"] == "max_iterations"
    assert row["triangulation_fallback_count"] == 2
    assert row["triangulation_fallback_reasons"] == "triangulation_failed_used_origin:2"
    assert row["paper_metrics_eligible"] is False
    aggregate = table2([row])[0]
    assert aggregate["geometrically_valid_reconstructions"] == 1
    assert aggregate["paper_metric_eligible_reconstructions"] == 0
    assert aggregate["valid_reconstructions"] == 0
    assert aggregate["reprojection_mean_px"] is None
    assert aggregate["icp_mean_mm"] is None


def test_reference_classification_and_checkpoint_hash_are_protocol_sensitive():
    config = load_config("examples/configs/topspin_reference.jsonc")
    assert _reference_mode(config)
    row = {"sequence_id": "a", "event_path": "a.npz", "object": "Hubble"}
    first = _checkpoint_fingerprint(config, row, "eSpinSLAM")
    assert first == _checkpoint_fingerprint(config, dict(row), "eSpinSLAM")
    config.frequency.alignment_mode = "experimental"
    assert not _reference_mode(config)
    assert first != _checkpoint_fingerprint(config, row, "eSpinSLAM")


def test_checkpoint_hash_changes_with_input_and_executable_source():
    config = load_config("examples/configs/topspin_reference.jsonc")
    row = {"sequence_id": "a", "event_path": "a.npz", "object": "Hubble"}
    common = dict(config=config, row=row, setting_name="eSpinSLAM")
    first = _checkpoint_fingerprint(
        **common, input_identity={"sha256": "a"},
        code_identity={"git_revision": "1", "source_tree_sha256": "x"},
    )
    assert first != _checkpoint_fingerprint(
        **common, input_identity={"sha256": "b"},
        code_identity={"git_revision": "1", "source_tree_sha256": "x"},
    )
    assert first != _checkpoint_fingerprint(
        **common, input_identity={"sha256": "a"},
        code_identity={"git_revision": "2", "source_tree_sha256": "y"},
    )


def test_run_directory_refuses_mixed_identity_without_deleting(tmp_path):
    output = tmp_path / "run"
    _prepare_run_directory(output, {"name": "first"})
    sentinel = output / "keep.txt"
    sentinel.write_text("do not delete", encoding="utf-8")
    with pytest.raises(ValueError, match="different run"):
        _prepare_run_directory(output, {"name": "second"})
    assert sentinel.read_text(encoding="utf-8") == "do not delete"


@pytest.mark.parametrize("value", ("../escape", "a/b", "", ".", ".."))
def test_unsafe_sequence_ids_are_rejected(value):
    with pytest.raises(ValueError, match="unsafe sequence_id"):
        _safe_sequence_id(value)


def test_formal_manifest_requires_explicit_microsecond_unit(tmp_path):
    scene = tmp_path / "camera-diagonal-slow"
    scene.mkdir()
    (scene / "calib.txt").write_text("10 10 2 2 0 0 0 0 0\n", encoding="utf-8")
    (scene / "meta.json").write_text(
        '{"event-camera":{"frame":{"width":4,"height":4}}}\n',
        encoding="utf-8",
    )
    row = {
        "sequence_id": scene.name, "object": "Camera", "perspective": "diagonal",
        "speed": "slow", "split": "test", "fx": "10.0", "fy": "10.0",
        "cx": "2.0", "cy": "2.0", "k1": "0.0", "k2": "0.0",
        "p1": "0.0", "p2": "0.0", "k3": "0.0", "width": "4",
        "height": "4", "time_unit": "auto",
    }
    _validate_verified_manifest_row(row, scene, require_fixed_split=False)
    with pytest.raises(ValueError, match="explicit time_unit='us'"):
        _validate_verified_manifest_row(row, scene, require_fixed_split=True)
    row["time_unit"] = "us"
    _validate_verified_manifest_row(row, scene, require_fixed_split=True)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("frontend", "density_radius_px"), 7.1),
        (("frequency", "loop_history"), 19),
        (("backend", "max_observations_per_track"), 1),
        (("backend", "optimize_frequency"), True),
    ),
)
def test_reference_validator_rejects_single_protocol_change(path, value):
    config = load_config("examples/configs/topspin_reference.jsonc")
    setattr(getattr(config, path[0]), path[1], value)
    mismatches = _reference_mismatches(config)
    assert len(mismatches) == 1
    assert ".".join(path) in mismatches[0]


def test_reference_validator_allows_only_the_registered_ablation_switch():
    config = load_config("examples/configs/topspin_reference.jsonc")
    config.backend.optimize_frequency = True
    assert _reference_mode(config, "No factor graph f constraint")
    config.frontend.cluster_min_size = 11
    assert not _reference_mode(config, "No factor graph f constraint")


def test_config_match_does_not_promote_unaudited_executable_paths():
    assert _method_profile(True, "eSpinSLAM") == "paper_reference_config"
    assert _method_profile(True, "No landmark association") == "paper_reference_config"
    assert _method_profile(True, "FFT f estimation") == "experimental_interpretation"
    assert _method_profile(True, "No factor graph f constraint") == "experimental_interpretation"


def test_generic_evaluation_labels_reference_and_experimental_profiles():
    config = load_config("examples/configs/topspin_reference.jsonc")
    assert _debris_method_profile(config) == ("paper_reference_config", [])
    config.frontend.chunk_events = 999
    profile, mismatches = _debris_method_profile(config)
    assert profile == "experimental_interpretation"
    assert any("frontend.chunk_events" in mismatch for mismatch in mismatches)


def test_scene_checkpoint_resumes_without_reprocessing(monkeypatch, tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    event_path = root / "scene.npz"
    np.savez(event_path, t=np.asarray((0, 1)), x=np.asarray((1, 1)),
             y=np.asarray((1, 1)), p=np.asarray((0, 1)))
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "sequence_id", "event_path", "object", "speed",
            "fx", "fy", "cx", "cy", "width", "height", "time_unit",
        ))
        writer.writeheader()
        writer.writerow({
            "sequence_id": "scene", "event_path": "scene.npz", "object": "Hubble",
            "speed": "slow", "fx": 10, "fy": 10, "cx": 2, "cy": 2,
            "width": 4, "height": 4, "time_unit": "s",
        })
    calls = []

    class FakeModel:
        def __init__(self, config):
            pass

        def process(self, sequence, camera=None):
            calls.append(sequence.sequence_id)
            return _result(sequence.sequence_id)

    monkeypatch.setattr("espinslam.topspin.ESpinSLAM", FakeModel)
    monkeypatch.setattr("espinslam.topspin._plot_frequency_cdf", lambda *args: None)
    monkeypatch.setattr("espinslam.topspin._plot_throughput", lambda *args: None)
    monkeypatch.setattr("espinslam.topspin._plot_throughput_summary", lambda *args: None)
    output = tmp_path / "run"
    config = ESpinSLAMConfig(camera=CameraConfig(fx=10, fy=10, cx=2, cy=2))
    first = evaluate_topspin(config, root, manifest, output, history_plots=False)
    second = evaluate_topspin(config, root, manifest, output, history_plots=False)
    assert calls == ["scene"]
    assert first["sequence_count"] == second["sequence_count"] == 1
    with pytest.raises(ValueError, match="--no-resume"):
        evaluate_topspin(
            config, root, manifest, output, history_plots=False, resume=False,
        )
    checkpoint = output / "checkpoints" / "scene.json"
    assert checkpoint.is_file()
    original_checkpoint = checkpoint.read_text(encoding="utf-8")
    payload = json.loads(original_checkpoint)
    payload["record"]["runtime_s"] = 999
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="record failed verification"):
        evaluate_topspin(config, root, manifest, output, history_plots=False)
    checkpoint.write_text(original_checkpoint, encoding="utf-8")
    (output / "history" / "scene_frequency.csv").write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact failed verification"):
        evaluate_topspin(config, root, manifest, output, history_plots=False)


def test_duplicate_manifest_rows_are_rejected_before_aggregation(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    np.savez(root / "scene.npz", t=np.asarray((0,)), x=np.asarray((1,)),
             y=np.asarray((1,)), p=np.asarray((0,)))
    manifest = root / "manifest.csv"
    manifest.write_text(
        "sequence_id,event_path,object,fx,fy,cx,cy,width,height,time_unit\n"
        "scene,scene.npz,Hubble,10,10,2,2,4,4,s\n"
        "scene,scene.npz,Hubble,10,10,2,2,4,4,s\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate sequence_id"):
        evaluate_topspin(ESpinSLAMConfig(), root, manifest, tmp_path / "run")


def test_review_subset_cannot_be_labeled_as_full_paper_comparison(monkeypatch, tmp_path):
    config = load_config("examples/configs/topspin_reference.jsonc")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "sequence_id", "event_path", "object", "split",
        ))
        writer.writeheader()
        writer.writerow({
            "sequence_id": next(iter(TOPSPIN_SCENES)), "event_path": "events.npz",
            "object": "Hubble", "split": "test",
        })
    monkeypatch.setattr(
        "espinslam.topspin._validate_topspin_acquisition",
        lambda *args, **kwargs: ({"record_id": 10884694}, {
            next(iter(TOPSPIN_SCENES)): {"events": {"sha256": "x"}},
        }),
    )
    monkeypatch.setattr(
        "espinslam.topspin._code_identity",
        lambda: {"git_revision": "abc", "git_dirty": False,
                 "source_tree_sha256": "tree", "lock_sha256": "lock"},
    )
    with pytest.raises(ValueError, match="not exactly all 72"):
        evaluate_topspin(
            config, tmp_path, manifest, tmp_path / "out",
            require_reference=True, paper_comparison=True,
        )


def test_require_reference_rejects_experimental_config(tmp_path):
    config = ESpinSLAMConfig()
    config.frequency.alignment_mode = "experimental"
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("sequence_id,event_path,object\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reference method profile"):
        evaluate_topspin(config, tmp_path, manifest, tmp_path / "out",
                         require_reference=True)


def test_formal_table3_uses_all72_and_keeps_nondev69_separate(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("sequence_id,event_path,object\n", encoding="utf-8")
    common = {
        "object": "Hubble", "sequences": 1, "geometrically_valid_reconstructions": 1,
        "paper_metric_eligible_reconstructions": 1, "valid_reconstructions": 1,
        "landmark_reprojection_mean_px": 1.0, "landmark_reprojection_std_px": 0.0,
        "icp_mean_mm": None, "icp_std_mm": None,
    }
    nondev = [{**common, "reprojection_mean_px": 69.0, "reprojection_std_px": 0.0}]
    all72 = [{**common, "reprojection_mean_px": 72.0, "reprojection_std_px": 0.0}]

    monkeypatch.setattr("espinslam.topspin.evaluate_topspin", lambda *args, **kwargs: {
        "table2": nondev, "table2_all": all72,
        "paper_comparison_eligible": True,
        "paper_comparison_table_all72": all72,
    })
    evaluate_topspin_ablations(
        ESpinSLAMConfig(), tmp_path, manifest, tmp_path / "out",
        settings=["espinslam"], paper_comparison=True,
    )
    with (tmp_path / "out" / "table3.csv").open(newline="", encoding="utf-8") as stream:
        formal = next(csv.DictReader(stream))
    with (tmp_path / "out" / "table3_nondev69.csv").open(
        newline="", encoding="utf-8",
    ) as stream:
        held_apart = next(csv.DictReader(stream))
    assert float(formal["reprojection_mean_px"]) == 72.0
    assert float(held_apart["reprojection_mean_px"]) == 69.0
