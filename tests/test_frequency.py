import math

import numpy as np

from espinslam.config import FrequencyConfig
from espinslam.frequency import LoopClosureFrequencyEstimator
from espinslam.geometry import project_points
from espinslam.synthetic import make_scene

INTRINSIC = np.asarray(((528.6, 0.0, 173.0), (0.0, 528.6, 130.0), (0.0, 0.0, 1.0)))


def _track_events(frequency, duration, seed=0, rate=60.0):
    """Events of ideal feature tracks (each point visible on part of the orbit)."""

    scene = make_scene(frequency_hz=frequency, elevation_deg=20.0, point_count=40, seed=seed)
    rng = np.random.default_rng(seed)
    # Asynchronous timestamps (a regular grid would alias with the period).
    times = np.sort(rng.uniform(0.0, duration, int(duration * rate)))
    rows = []
    for point in scene.points:
        pixels = project_points(np.repeat(point[None], times.size, 0), times, frequency,
                                10.0, scene.rotation, INTRINSIC)
        keep = ((frequency * times + rng.uniform()) % 1.0) < 0.4
        keep &= np.isfinite(pixels).all(axis=1)
        noisy = pixels[keep] + rng.normal(0.0, 0.7, (keep.sum(), 2))
        rows.append(np.column_stack((times[keep], noisy)))
    return np.concatenate(rows)


def _online(events, config, step=0.1):
    estimator = LoopClosureFrequencyEstimator(config)
    closed_at = None
    for now in np.arange(step, events[:, 0].max() + 1e-9, step):
        estimator.update(events[events[:, 0] <= now])
        if estimator.loop_closed and closed_at is None:
            closed_at = now
    return estimator, closed_at


def test_online_loop_closure_recovers_frequency_and_detects_loop():
    events = _track_events(0.5, 6.0)
    estimator, closed_at = _online(events, FrequencyConfig(min_period_seconds=0.5,
                                                           time_scale_px_per_second=150.0,
                                                           alignment_mode="experimental",
                                                           harmonic_tolerance=0.15))
    assert estimator.loop_closed and closed_at is not None and closed_at < 5.0
    assert abs(estimator.current.frequency_hz - 0.5) < 2e-3


def test_experimental_mode_prefers_fundamental_over_subharmonics():
    events = _track_events(0.9, 6.0, seed=3)
    estimator = LoopClosureFrequencyEstimator(FrequencyConfig(min_period_seconds=0.5,
                                                              time_scale_px_per_second=150.0,
                                                              alignment_mode="experimental",
                                                              harmonic_tolerance=0.15))
    estimate = estimator.update(events)
    assert estimate.valid and abs(estimate.frequency_hz - 0.9) < 0.01


def test_less_than_one_revolution_does_not_close_a_loop():
    events = _track_events(0.2, 4.0)  # 0.8 revolutions
    estimator, closed_at = _online(events, FrequencyConfig(min_period_seconds=0.5))
    assert not estimator.loop_closed and closed_at is None


def test_fft_ablation_finds_event_rate_modulation():
    rng = np.random.default_rng(0)
    times = np.sort(rng.uniform(0.0, 20.0, 40000))
    keep = rng.random(times.size) < 0.5 + 0.4 * np.sin(2 * math.pi * 0.35 * times)
    events = np.column_stack((times[keep], np.zeros(keep.sum()), np.zeros(keep.sum())))
    estimator = LoopClosureFrequencyEstimator(FrequencyConfig(method="fft"))
    estimate = estimator.update(events)
    assert estimate.valid and abs(estimate.frequency_hz - 0.35) < 0.02


def test_short_observation_abstains():
    events = np.column_stack((np.linspace(0.0, 0.2, 300), np.linspace(0.0, 10.0, 300),
                              np.linspace(0.0, 5.0, 300)))
    estimate = LoopClosureFrequencyEstimator(
        FrequencyConfig(min_period_seconds=1.0),
    ).update(events)
    assert not estimate.valid
