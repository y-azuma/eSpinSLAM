import math

import numpy as np

from espinslam.config import FrequencyConfig
from espinslam.frequency import LoopClosureFrequencyEstimator


def test_loop_closure_recovers_periodic_spatial_tracks():
    frequency_hz = 0.5
    times = np.arange(0.0, 5.0, 0.01)
    phase = 2.0 * math.pi * frequency_hz * times
    x = 173.0 + 65.0 * np.cos(phase) + 8.0 * np.cos(3.0 * phase)
    y = 130.0 + 42.0 * np.sin(phase) + 5.0 * np.sin(2.0 * phase)
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

