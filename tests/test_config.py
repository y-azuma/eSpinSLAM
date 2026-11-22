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
