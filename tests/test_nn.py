"""Cross-platform correctness tests for :class:`espinslam.nn.GridIndex`.

Nearest-neighbour distances are floating-point reductions.  Different valid
implementations (and different CPU architectures) need not round the final bit
identically, so correctness is defined against a small brute-force reference
with a tight numerical tolerance.  A different index is accepted only when it
is at the same minimum distance, which also handles exact ties explicitly.
"""

import numpy as np
import pytest

from espinslam.nn import GridIndex


_RTOL = 1e-12
_ATOL = 1e-12


def _brute_force_nearest(points, queries, chunk_size=64):
    """Return exact-search nearest distances and indices without a tree.

    Chunking keeps the temporary ``queries x points x dimensions`` array small
    enough for the larger randomized tests while retaining a simple independent
    oracle.
    """
    points = np.asarray(points, dtype=float)
    queries = np.asarray(queries, dtype=float)
    distance = np.empty(len(queries), dtype=float)
    index = np.empty(len(queries), dtype=np.intp)
    for start in range(0, len(queries), chunk_size):
        stop = min(start + chunk_size, len(queries))
        delta = queries[start:stop, None, :] - points[None, :, :]
        squared = np.einsum("qpd,qpd->qp", delta, delta)
        nearest = np.argmin(squared, axis=1)
        index[start:stop] = nearest
        distance[start:stop] = np.sqrt(squared[np.arange(stop - start), nearest])
    return distance, index


def _check(points, queries, hint=None):
    expected_distance, _ = _brute_force_nearest(points, queries)
    grid = GridIndex(points)
    distance, index = grid.query_points(queries[:, 0], queries[:, 1], queries[:, 2], hint)
    assert np.issubdtype(index.dtype, np.integer)
    assert np.all((0 <= index) & (index < len(points)))

    # First verify that the reported distance belongs to the reported point.
    selected_distance = np.linalg.norm(points[index] - queries, axis=1)
    np.testing.assert_allclose(distance, selected_distance, rtol=_RTOL, atol=_ATOL)
    # Then verify global optimality against an implementation-independent oracle.
    # This accepts a different index only for a numerically equal nearest point.
    np.testing.assert_allclose(selected_distance, expected_distance, rtol=_RTOL, atol=_ATOL)
    return index


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_float_points_match_brute_force(seed):
    rng = np.random.default_rng(seed)
    points = np.column_stack((rng.uniform(0, 400, 3000), rng.uniform(0, 300, 3000),
                              100.0 * np.sort(rng.uniform(10, 25, 3000))))
    queries = np.column_stack((rng.uniform(-50, 450, 4000), rng.uniform(-50, 350, 4000),
                               100.0 * rng.uniform(0, 35, 4000)))
    index = _check(points, queries)
    jitter = queries + rng.normal(0.0, 2.0, queries.shape)
    _check(points, jitter, hint=index)  # warm start from a nearby point


def test_integer_pixels_with_ties_and_far_queries():
    rng = np.random.default_rng(3)
    points = np.column_stack((rng.integers(0, 64, 2000), rng.integers(0, 48, 2000),
                              np.repeat(np.arange(200.0), 10))).astype(float)
    queries = np.column_stack((rng.integers(-500, 600, 3000), rng.integers(-400, 500, 3000),
                               rng.integers(-50, 250, 3000))).astype(float)
    _check(points, queries)


def test_wrong_hint_does_not_change_the_result():
    rng = np.random.default_rng(4)
    points = rng.uniform(0, 100, (500, 3))
    queries = rng.uniform(0, 100, (300, 3))
    _check(points, queries, hint=rng.integers(0, 500, 300))


def test_query_shifts_matches_per_shift_queries():
    rng = np.random.default_rng(5)
    right_t = np.sort(rng.uniform(10.0, 20.0, 2500))
    points = np.column_stack((rng.integers(0, 300, 2500), rng.integers(0, 200, 2500),
                              100.0 * right_t)).astype(float)
    left_t = np.sort(rng.uniform(0.0, 10.0, 800))
    left_x = rng.integers(0, 300, 800).astype(float)
    left_y = rng.integers(0, 200, 800).astype(float)
    shifts = np.geomspace(2.0, 15.0, 120)
    start, stop = float(right_t[0]), float(right_t[-1])
    out = GridIndex(points).query_shifts(left_t, left_x, left_y, shifts, 100.0, start, stop)
    for k, shift in enumerate(shifts):
        shifted = left_t + shift
        mask = (shifted >= start) & (shifted <= stop)
        assert np.array_equal(np.isnan(out[k]), ~mask)
        if mask.any():
            expected, _ = _brute_force_nearest(
                points,
                np.column_stack((left_x[mask], left_y[mask],
                                 100.0 * (left_t[mask] + shift))),
            )
            np.testing.assert_allclose(out[k, mask], expected, rtol=_RTOL, atol=_ATOL)
