"""Frozen equivalence gate for replacing paper-specified GTSAM."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

gtsam = pytest.importorskip("gtsam")

from espinslam.geometry import bundle_adjust, project_points, so3_exp
from espinslam.gtsam_backend import (
    _projection_residual_and_jacobians,
    bundle_adjust_gtsam,
)


INTRINSIC = np.asarray(((528.6, 0.0, 173.0),
                        (0.0, 528.6, 130.0),
                        (0.0, 0.0, 1.0)))


def _problem():
    rng = np.random.default_rng(7305)
    truth_points = rng.uniform((-1.0, -1.0, -0.8), (1.0, 1.0, 0.8), (8, 3))
    truth_rotation = so3_exp(np.asarray((0.10, -0.18, 0.04)))
    frequency = 0.47
    times, pixels, owners = [], [], []
    for owner, point in enumerate(truth_points):
        t = np.linspace(0.0, 2.4, 40) + owner * 2e-4
        xy = project_points(np.repeat(point[None], t.size, axis=0), t,
                            frequency, 10.0, truth_rotation, INTRINSIC)
        times.append(t)
        pixels.append(xy + rng.normal(0.0, 0.15, xy.shape))
        owners.append(np.full(t.size, owner, dtype=np.int64))
    return (np.concatenate(times), np.concatenate(pixels), np.concatenate(owners),
            np.zeros_like(truth_points), np.eye(3), frequency)


def test_custom_factor_analytic_jacobians_match_gtsam_tangents():
    """Numerically verify right-local Rot3 and additive Point3 derivatives."""

    point = np.asarray((0.3, -0.5, 0.2))
    rotation = so3_exp(np.asarray((0.08, -0.13, 0.03)))
    timestamp, frequency = 0.73, 0.47
    observed = np.asarray((181.0, 127.0))
    residual, j_rotation, j_point = _projection_residual_and_jacobians(
        timestamp, observed, point, rotation, frequency, 10.0, INTRINSIC,
    )
    step = 1e-7
    numeric_rotation = np.empty((2, 3))
    numeric_point = np.empty((2, 3))
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = step
        plus = rotation @ Rotation.from_rotvec(delta).as_matrix()
        minus = rotation @ Rotation.from_rotvec(-delta).as_matrix()
        r_plus = _projection_residual_and_jacobians(
            timestamp, observed, point, plus, frequency, 10.0, INTRINSIC,
        )[0]
        r_minus = _projection_residual_and_jacobians(
            timestamp, observed, point, minus, frequency, 10.0, INTRINSIC,
        )[0]
        numeric_rotation[:, axis] = (r_plus - r_minus) / (2.0 * step)
        moved = point.copy()
        moved[axis] += step
        back = point.copy()
        back[axis] -= step
        p_plus = _projection_residual_and_jacobians(
            timestamp, observed, moved, rotation, frequency, 10.0, INTRINSIC,
        )[0]
        p_minus = _projection_residual_and_jacobians(
            timestamp, observed, back, rotation, frequency, 10.0, INTRINSIC,
        )[0]
        numeric_point[:, axis] = (p_plus - p_minus) / (2.0 * step)
    assert residual.shape == (2,)
    assert np.allclose(j_rotation, numeric_rotation, rtol=1e-7, atol=1e-6)
    assert np.allclose(j_point, numeric_point, rtol=1e-7, atol=1e-6)


def test_in_tree_solver_meets_frozen_gtsam_equivalence_thresholds():
    """Gate agreed before evaluation; thresholds must not be relaxed post hoc."""

    times, pixels, owners, points, rotation, frequency = _problem()
    kwargs = dict(
        frequency_hz=frequency, orbit_radius=10.0, intrinsic=INTRINSIC,
        huber_threshold_px=2.0, max_iterations=100,
        relative_tolerance=1e-10,
    )
    reference = bundle_adjust_gtsam(
        times, pixels, owners, points, rotation, **kwargs,
    )
    candidate = bundle_adjust(
        times, pixels, owners, points, rotation, **kwargs,
    )
    cost_relative = abs(candidate.final_cost - reference.final_cost) / max(
        abs(reference.final_cost), 1e-300,
    )
    rotation_error = np.linalg.norm(Rotation.from_matrix(
        candidate.rotation @ reference.rotation.T,
    ).as_rotvec())
    point_relative = np.linalg.norm(candidate.points - reference.points) / max(
        np.linalg.norm(reference.points), 1e-300,
    )
    assert cost_relative <= 1e-8
    assert rotation_error <= 1e-6
    assert point_relative <= 1e-6
