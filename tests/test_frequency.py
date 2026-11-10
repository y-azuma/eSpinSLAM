import math

import numpy as np

from espinslam.config import FrequencyConfig
from espinslam.frequency import LoopClosureFrequencyEstimator
from espinslam.frontend import _cluster_candidates


def test_loop_closure_recovers_periodic_spatial_tracks():
    frequency_hz = 0.5
    times = np.arange(0.0, 5.0, 0.01)
    phase = 2.0 * math.pi * frequency_hz * times
    x = np.rint(173.0 + 65.0 * np.cos(phase) + 8.0 * np.cos(3.0 * phase))
    y = np.rint(130.0 + 42.0 * np.sin(phase) + 5.0 * np.sin(2.0 * phase))
    polarity = ((np.arange(times.size) // 5) % 2).astype(np.float64)
    observations = np.column_stack((times, x, y, polarity))
    estimator = LoopClosureFrequencyEstimator(FrequencyConfig(
        min_hz=0.2,
        max_hz=0.8,
        candidate_count=121,
        temporal_tolerance_seconds=0.02,
        spatial_tolerance_px=3.0,
    ))
    result = estimator.estimate(observations)
    assert result.valid
    assert abs(result.frequency_hz - frequency_hz) < 0.02
    assert abs(result.omega_rad_s - math.pi) < 0.13


def test_short_observation_abstains():
    observations = np.column_stack((
        np.linspace(0.0, 0.2, 30),
        np.linspace(0.0, 10.0, 30),
        np.linspace(0.0, 5.0, 30),
        np.zeros(30),
    ))
    result = LoopClosureFrequencyEstimator(FrequencyConfig()).estimate(observations)
    assert not result.valid


def test_equivalence_band_is_disabled_by_default():
    config = FrequencyConfig()
    assert config.equivalent_score_fraction == 0.0


def test_equivalence_band_must_be_non_negative():
    with np.testing.assert_raises(ValueError):
        FrequencyConfig(equivalent_score_fraction=-0.01)


def test_minimal_candidate_cluster_is_handled_without_hdbscan():
    from espinslam.config import FrontendConfig

    values = np.asarray([
        [0.00, 10.0, 20.0, 1.0],
        [0.01, 11.0, 21.0, 1.0],
        [0.02, 12.0, 22.0, 1.0],
    ])
    observations = _cluster_candidates(values, FrontendConfig(cluster_min_samples=3))
    assert len(observations) == 1
    assert observations[0].x == 11.0
