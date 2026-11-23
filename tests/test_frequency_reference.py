import numpy as np

from espinslam.config import FrequencyConfig
from espinslam.frequency import FrequencyEstimate, LoopClosureFrequencyEstimator, _PaperAlignment
from espinslam.nn import GridIndex


def _brute_force_cost(left, right, shift, scale):
    query = np.column_stack((left[:, 1], left[:, 2], scale * (left[:, 0] + shift)))
    target = np.column_stack((right[:, 1], right[:, 2], scale * right[:, 0]))
    distances = np.linalg.norm(query[:, None, :] - target[None, :, :], axis=2)
    return float(np.min(distances, axis=1).sum())


def test_paper_alignment_matches_literal_eq9_for_every_shift():
    left = np.asarray([
        [0.0, 0.0, 0.0],
        [0.7, 8.0, -2.0],
        [1.4, 17.0, 4.0],
        [2.0, -3.0, 11.0],
    ])
    right = np.asarray([
        [2.2, 0.5, -0.2],
        [3.1, 8.5, -1.0],
        [4.0, 16.0, 5.0],
        [4.8, -2.0, 10.0],
    ])
    shifts = np.asarray((0.05, 1.2, 2.4, 7.0))
    scale = 0.75
    alignment = _PaperAlignment(left, right, scale)

    expected = np.asarray([
        _brute_force_cost(left, right, shift, scale) for shift in shifts
    ])
    # The final shift puts every shifted left timestamp beyond A_r.  Eq. (9)
    # still includes all left events rather than applying an overlap mask.
    assert np.all(left[:, 0] + shifts[-1] > right[-1, 0])
    np.testing.assert_allclose(alignment.costs(shifts), expected, rtol=1e-12, atol=1e-12)
    for shift, cost in zip(shifts, expected):
        assert np.isclose(alignment.cost(float(shift)), cost, rtol=1e-12, atol=1e-12)


def test_exact_batched_shift_queries_equal_independent_queries():
    rng = np.random.default_rng(9)
    target = rng.normal(size=(37, 3))
    times = rng.normal(size=23)
    x = rng.normal(size=23)
    y = rng.normal(size=23)
    shifts = np.asarray((-1.3, -0.1, 0.0, 0.6, 2.1))
    scale = 3.25
    index = GridIndex(target)

    batched = index.query_all_shifts(times, x, y, shifts, scale)
    expected = np.vstack([
        index.query_points(x, y, scale * (times + shift))[0]
        for shift in shifts
    ])
    np.testing.assert_allclose(batched, expected, rtol=1e-12, atol=1e-12)


def test_paper_mode_never_subsamples_alignment_pool():
    times = np.linspace(0.0, 4.0, 40)
    events = np.column_stack((times, 5.0 * np.sin(times), 3.0 * np.cos(times)))
    config = FrequencyConfig(
        min_period_seconds=0.5,
        max_period_seconds=5.0,
        min_samples_per_half=3,
        max_alignment_samples=4,
        coarse_alignment_samples=2,
        coarse_relative_step=0.2,
        time_scale_px_per_second=1.0,
    )

    paper = LoopClosureFrequencyEstimator(config, alignment_mode="paper").update(events)
    approximate = LoopClosureFrequencyEstimator(
        config, alignment_mode="experimental",
    ).update(events)
    assert paper.valid
    assert not approximate.valid


def test_paper_mode_ignores_experimental_harmonic_preference():
    times = np.linspace(0.0, 4.0, 80)
    events = np.column_stack((times, np.sin(2.0 * np.pi * times),
                              np.cos(2.0 * np.pi * times)))
    common = dict(
        min_period_seconds=0.4,
        max_period_seconds=3.0,
        min_samples_per_half=5,
        coarse_relative_step=0.08,
        time_scale_px_per_second=1.0,
    )
    no_tolerance = LoopClosureFrequencyEstimator(
        FrequencyConfig(**common, harmonic_tolerance=0.0), alignment_mode="paper",
    ).update(events)
    huge_tolerance = LoopClosureFrequencyEstimator(
        FrequencyConfig(**common, harmonic_tolerance=100.0), alignment_mode="paper",
    ).update(events)

    assert no_tolerance.valid and huge_tolerance.valid
    assert no_tolerance.period_seconds == huge_tolerance.period_seconds
    assert no_tolerance.alignment_cost == huge_tolerance.alignment_cost


def test_invalid_update_is_not_counted_or_used_to_clear_paper_history():
    config = FrequencyConfig(loop_history=3, loop_std_threshold_hz=1e-3)
    estimator = LoopClosureFrequencyEstimator(config, alignment_mode="paper")
    values = iter((
        FrequencyEstimate(0.5000, 2.0, 1.0, True),
        FrequencyEstimate(np.nan, np.nan, np.inf, False, "no estimate"),
        FrequencyEstimate(0.5001, 1.9996, 1.0, True),
        FrequencyEstimate(0.4999, 2.0004, 1.0, True),
    ))
    estimator._loop_closure = lambda _: next(values)
    events = np.zeros((2, 3))

    estimator.update(events)
    estimator.update(events)
    assert list(estimator.history) == [0.5]
    estimator.update(events)
    estimator.update(events)
    assert estimator.loop_closed

