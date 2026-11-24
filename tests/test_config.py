from pathlib import Path

import pytest

from espinslam.config import CameraConfig, config_from_dict, load_config


def test_example_configs_are_valid():
    debris = load_config(Path("examples/configs/debris_davis346.jsonc"))
    assert debris.data.evaluation_split == "test"
    assert debris.camera.complete
    topspin = load_config(Path("examples/configs/topspin_evk4.jsonc"))
    assert topspin.data.sensor_size == [1280, 720]
    assert not topspin.camera.complete
    assert topspin.frontend.interpretation_mode == "experimental"
    assert topspin.frequency.alignment_mode == "experimental"
    reference = load_config(Path("examples/configs/topspin_reference.jsonc"))
    assert reference.frontend.interpretation_mode == "paper"
    assert reference.frequency.alignment_mode == "paper"
    assert reference.backend.final_offline_refinement is False
    assert reference.backend.solver_backend == "equivalent_custom"


def test_paper_table1_defaults():
    config = config_from_dict({})
    assert config.frequency.loop_std_threshold_hz == 1e-3        # sigma
    assert config.frequency.loop_history == 20                   # aleph
    assert config.backend.huber_threshold_px == 2.0              # rho inlier threshold
    assert config.backend.forget_revolutions == 3.0              # sigma_forget
    assert config.frontend.retain_events_per_track == 1000       # sigma_retain
    assert config.backend.min_track_events == 3                  # N_match
    assert config.backend.orbit_radius == 10.0                   # r
    assert config.backend.max_mean_reprojection_error_px == 10.0  # gamma
    assert config.backend.landmark_merge_radius == 0.01          # delta_fuse
    assert config.backend.optimization_cadence_revolutions == 0.1  # epsilon


def test_cited_esfo_frontend_defaults_are_separate_from_interpreted_values():
    config = config_from_dict({})
    assert config.frontend.density_radius_px == 7.0
    assert config.frontend.density_mean_per_polarity is True
    assert config.frontend.cluster_min_size == 10
    assert config.frontend.cluster_min_samples is None
    assert config.frontend.cluster_selection_epsilon == 5.0
    assert config.frontend.cluster_by_polarity is False
    assert config.frontend.association_radius_px == 30.0
    assert config.frontend.association_head_events == 5
    # These are explicit interpretations, not claims that the paper stated them.
    assert config.frontend.cluster_time_scale_px_per_second == 1.0
    assert config.frequency.time_scale_px_per_second == 1.0
    assert config.frequency.min_period_seconds == 0.1
    assert config.frequency.max_period_seconds is None
    assert config.frequency.alignment_mode == "paper"
    assert config.frequency.coarse_relative_step == 0.01
    assert config.frequency.icp_iterations == 30
    assert config.frequency.icp_tolerance_seconds == 1e-5
    assert config.frequency.harmonic_tolerance == 0.0


def test_finite_period_ceiling_is_experimental_only():
    with pytest.raises(ValueError, match="must be null in paper mode"):
        config_from_dict({"frequency": {"max_period_seconds": 12.0}})
    experimental = config_from_dict({"frequency": {
        "alignment_mode": "experimental", "max_period_seconds": 12.0,
    }})
    assert experimental.frequency.max_period_seconds == 12.0


def test_missing_intrinsics_are_rejected():
    with pytest.raises(ValueError, match="intrinsics are missing"):
        CameraConfig().matrix()


def test_non_test_evaluation_split_is_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"data":{"evaluation_split":"train"}}', encoding="utf-8")
    with pytest.raises(ValueError, match="remain 'test'"):
        load_config(path)


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown backend keys"):
        config_from_dict({"backend": {"not_a_key": 1}})
