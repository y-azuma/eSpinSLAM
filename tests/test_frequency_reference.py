import numpy as np

import espinslam.frequency as frequency_module
from espinslam.config import FrequencyConfig
from espinslam.frequency import FrequencyEstimate, LoopClosureFrequencyEstimator, _PaperAlignment
from espinslam.nn import GridIndex


def _brute_force_cost(left, right, shift, scale):
    query = np.column_stack((left[:, 1], left[:, 2], scale * (left[:, 0] + shift)))
    target = np.column_stack((right[:, 1], right[:, 2], scale * right[:, 0]))
    distances = np.linalg.norm(query[:, None, :] - target[None, :, :], axis=2)
    return float(np.min(distances, axis=1).sum())


def _brute_force_icp(left, right, shift, scale, iterations, tolerance,
                     minimum, maximum):
    """Independent fixed-correspondence IRLS reference for paper Eq. (9)."""

    shift = float(np.clip(shift, minimum, maximum))
    for _ in range(iterations):
        query = np.column_stack((
            left[:, 1], left[:, 2], scale * (left[:, 0] + shift),
        ))
        target = np.column_stack((
            right[:, 1], right[:, 2], scale * right[:, 0],
        ))
        distances = np.linalg.norm(query[:, None, :] - target[None, :, :], axis=2)
        neighbours = np.argmin(distances, axis=1)
        nearest = distances[np.arange(left.shape[0]), neighbours]
        weights = 1.0 / np.maximum(nearest, 1e-12)
        shifts = right[neighbours, 0] - left[:, 0]
        updated = float(np.sum(weights * shifts) / np.sum(weights))
        updated = float(np.clip(updated, minimum, maximum))
        converged = abs(updated - shift) < tolerance
        shift = updated
        if converged:
            break
    return shift, _brute_force_cost(left, right, shift, scale)


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


def test_paper_icp_matches_independent_fixed_correspondence_reference():
    """The exact acceleration must not alter the post-grid 1-D ICP result."""

    left = np.asarray([
        [0.00, 0.0, 0.0],
        [0.37, 3.0, 1.0],
        [0.91, 8.0, -2.0],
        [1.42, 12.0, 4.0],
        [1.88, -4.0, 7.0],
    ])
    right = np.asarray([
        [1.23, 0.2, -0.1],
        [1.61, 3.4, 1.2],
        [2.14, 7.7, -1.8],
        [2.67, 12.2, 3.7],
        [3.12, -3.8, 7.3],
        [3.45, 20.0, -8.0],
    ])
    scale = 1.7
    minimum, maximum = 0.2, 2.5
    iterations, tolerance = 30, 1e-10
    seed = 1.05
    expected_shift, expected_cost = _brute_force_icp(
        left, right, seed, scale, iterations, tolerance, minimum, maximum,
    )
    alignment = _PaperAlignment(left, right, scale)
    actual_shift, actual_cost = alignment.icp(
        seed, iterations, tolerance, minimum, maximum,
    )
    assert np.isclose(actual_shift, expected_shift, rtol=1e-12, atol=1e-12)
    assert np.isclose(actual_cost, expected_cost, rtol=1e-12, atol=1e-12)
    assert alignment.last_icp_iterations <= iterations
    assert alignment.last_icp_termination in {"converged", "max_iterations"}


def test_one_paper_irls_step_matches_hand_calculation_and_records_clipping():
    """One update is independently calculable, including the 1e-12 guard."""

    left = np.asarray([
        [0.0, 0.0, 0.0],
        [0.4, 2.0, 0.0],
        [0.9, 5.0, 1.0],
    ])
    right = np.asarray([
        [1.3, 0.1, 0.0],
        [1.8, 2.2, -0.1],
        [2.4, 5.2, 1.1],
    ])
    scale = 1.0
    seed = 0.8
    query = np.column_stack((left[:, 1], left[:, 2], left[:, 0] + seed))
    target = np.column_stack((right[:, 1], right[:, 2], right[:, 0]))
    distances = np.linalg.norm(query[:, None, :] - target[None, :, :], axis=2)
    indices = np.argmin(distances, axis=1)
    nearest = distances[np.arange(left.shape[0]), indices]
    weights = 1.0 / np.maximum(nearest, 1e-12)
    expected = float(np.sum(weights * (right[indices, 0] - left[:, 0]))
                     / np.sum(weights))

    alignment = _PaperAlignment(left, right, scale)
    actual, _ = alignment.icp(seed, iterations=1, tolerance=1e-20,
                              minimum=0.1, maximum=5.0)
    assert np.isclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert alignment.last_icp_iterations == 1
    assert alignment.last_icp_termination == "max_iterations"
    assert alignment.last_icp_clipped is False

    clipped = _PaperAlignment(left, right, scale)
    bounded, _ = clipped.icp(seed, iterations=1, tolerance=1e-20,
                             minimum=0.1, maximum=0.9)
    assert bounded == 0.9
    assert clipped.last_icp_clipped is True


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


def test_paper_cost_batch_bound_does_not_change_any_candidate(monkeypatch):
    """Exercise more than one temporary-matrix batch with identical support."""

    rng = np.random.default_rng(910)
    left = np.column_stack((
        np.sort(rng.uniform(0.0, 2.0, 19)),
        rng.normal(size=19), rng.normal(size=19),
    ))
    right = np.column_stack((
        np.sort(rng.uniform(2.0, 5.0, 31)),
        rng.normal(size=31), rng.normal(size=31),
    ))
    shifts = np.geomspace(0.1, 4.0, 11)
    scale = 0.8
    expected = np.asarray([
        _brute_force_cost(left, right, shift, scale) for shift in shifts
    ])
    # At most two shifts fit, so this test necessarily crosses batch bounds.
    monkeypatch.setattr(
        frequency_module, "_PAPER_ALIGNMENT_BATCH_BYTES", 2 * left.shape[0] * 8,
    )
    actual = _PaperAlignment(left, right, scale).costs(shifts)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_paper_grid_has_declared_endpoints_and_global_bruteforce_argmin():
    rng = np.random.default_rng(412)
    times = np.sort(rng.uniform(0.0, 3.0, 160))
    events = np.column_stack((times, np.sin(2.0 * times), np.cos(3.0 * times)))
    config = FrequencyConfig(
        min_period_seconds=0.1,
        min_samples_per_half=10,
        coarse_relative_step=0.2,
        time_scale_px_per_second=1.0,
    )
    estimate = LoopClosureFrequencyEstimator(config).update(events)
    assert estimate.valid
    assert estimate.candidates_seconds[0] == config.min_period_seconds
    # The observed span, rather than the configured safety ceiling, is the
    # paper-reference upper endpoint for this update.
    assert np.isclose(estimate.candidates_seconds[-1], times[-1] - times[0])
    expected_count = max(3, int(np.ceil(
        np.log((times[-1] - times[0]) / config.min_period_seconds)
        / np.log1p(config.coarse_relative_step),
    )) + 1)
    assert estimate.candidates_seconds.size == expected_count
    brute_costs = np.asarray([
        _brute_force_cost(
            events[events[:, 0] <= 0.5 * (times[0] + times[-1])],
            events[events[:, 0] > 0.5 * (times[0] + times[-1])],
            shift, config.time_scale_px_per_second,
        )
        for shift in estimate.candidates_seconds
    ])
    expected_index = int(np.argmin(brute_costs))
    np.testing.assert_allclose(estimate.candidate_costs, brute_costs,
                               rtol=1e-12, atol=1e-12)
    assert estimate.grid_seed_seconds == estimate.candidates_seconds[expected_index]
    assert estimate.grid_seed_cost == brute_costs[expected_index]
    assert estimate.alignment_cost <= estimate.grid_seed_cost + 1e-12
    assert estimate.icp_termination in {"converged", "max_iterations"}
    assert 1 <= estimate.icp_iterations_used <= config.icp_iterations


def test_paper_repeats_full_observed_range_after_loop_was_already_closed():
    config = FrequencyConfig(
        min_period_seconds=0.1,
        min_samples_per_half=5,
        coarse_relative_step=0.25,
        time_scale_px_per_second=1.0,
    )
    estimator = LoopClosureFrequencyEstimator(config)
    first_t = np.linspace(0.0, 1.8, 80)
    first = np.column_stack((first_t, np.sin(first_t), np.cos(first_t)))
    first_estimate = estimator.update(first)
    assert first_estimate.valid

    # A prior estimate and a closed loop must not turn the paper path into a
    # local warm start; Eq. (9)'s entire current interval is searched again.
    estimator.loop_closed = True
    second_t = np.linspace(0.0, 3.7, 160)
    second = np.column_stack((second_t, np.sin(second_t), np.cos(second_t)))
    second_estimate = estimator.update(second)
    assert second_estimate.valid
    assert second_estimate.candidates_seconds[0] == config.min_period_seconds
    assert np.isclose(second_estimate.candidates_seconds[-1], 3.7)
    assert second_estimate.candidates_seconds.size > first_estimate.candidates_seconds.size
    assert second_estimate.grid_seed_seconds == second_estimate.candidates_seconds[
        int(np.argmin(second_estimate.candidate_costs))
    ]


def test_paper_mode_never_subsamples_alignment_pool():
    times = np.linspace(0.0, 4.0, 40)
    events = np.column_stack((times, 5.0 * np.sin(times), 3.0 * np.cos(times)))
    config = FrequencyConfig(
        min_period_seconds=0.5,
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

