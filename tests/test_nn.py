"""GridIndex must return exactly cKDTree's nearest distances (bit for bit)."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from espinslam.nn import GridIndex


def _check(points, queries, hint=None):
    tree_distance, tree_index = cKDTree(points).query(queries, k=1)
    grid = GridIndex(points)
    distance, index = grid.query_points(queries[:, 0], queries[:, 1], queries[:, 2], hint)
    assert np.array_equal(distance, tree_distance)
    differ = index != tree_index
    # Only exact ties may pick a different (equally near) point.
    assert np.array_equal(np.linalg.norm(points[index[differ]] - queries[differ], axis=1),
                          np.linalg.norm(points[tree_index[differ]] - queries[differ], axis=1))
    return index


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_float_points_match_kdtree(seed):
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
    tree = cKDTree(points)
    for k, shift in enumerate(shifts):
        shifted = left_t + shift
        mask = (shifted >= start) & (shifted <= stop)
        assert np.array_equal(np.isnan(out[k]), ~mask)
        if mask.any():
            expected, _ = tree.query(np.column_stack((left_x[mask], left_y[mask],
                                                      100.0 * (left_t[mask] + shift))), k=1)
            assert np.array_equal(out[k, mask], expected)
