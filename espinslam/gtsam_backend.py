"""Direct GTSAM reference solver for paper Eq. (11).

The eSpinSLAM paper explicitly says that Eq. (11) is solved with GTSAM.  This
module is deliberately optional: importing :mod:`espinslam` does not require
the ``gtsam`` wheel, while :func:`bundle_adjust_gtsam` fails clearly when it is
not installed.  The in-tree Schur/LM implementation may replace this reference
path only after passing the frozen numerical-equivalence test.

Frequency is fixed to ``f*`` exactly as in the main method.  No priors, gauge
anchors, parallax terms, or other paper-unreported residuals are added.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np


_INVISIBLE_ERROR_PX = 1.0e4


@dataclass(frozen=True)
class GTSAMBundleAdjustmentResult:
    """Result of the direct GTSAM formulation of Eq. (11)."""

    rotation: np.ndarray
    frequency_hz: float
    points: np.ndarray
    per_point_error_px: np.ndarray
    per_observation_error_px: np.ndarray
    initial_cost: float
    final_cost: float
    iterations: int
    termination_status: str

    @property
    def converged(self) -> bool:
        # GTSAM's Python wrapper exposes the iteration count but does not
        # distinguish a tolerance stop from a stalled optimizer.  Do not
        # silently label the combined state as convergence.
        return self.termination_status == "converged"


def _import_gtsam():
    try:
        import gtsam  # type: ignore
    except ImportError as error:  # pragma: no cover - exercised without optional wheel
        raise RuntimeError(
            "the paper-reference solver requires the optional 'gtsam' package"
        ) from error
    return gtsam


def _as_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape == (3, 3):
        return value
    if value.shape == (3,):
        from scipy.spatial.transform import Rotation
        return Rotation.from_rotvec(value).as_matrix()
    raise ValueError("rotation must be a 3x3 matrix or a rotation vector")


def _projection_residual_and_jacobians(
    timestamp: float,
    observed: np.ndarray,
    point: np.ndarray,
    rotation: np.ndarray,
    frequency_hz: float,
    orbit_radius: float,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return residual, d residual / d Rot3 tangent, and d residual / d point.

    GTSAM's ``Rot3`` retract convention is right-local,
    ``R.retract(delta) = R Exp(delta)``.  The natural projection derivative in
    this project is for a left perturbation ``Exp(delta_left) R``.  Since
    ``delta_left = R delta_right`` to first order, the callback supplies
    ``J_right = J_left R``.  Landmark coordinates use the ordinary additive
    tangent convention.
    """

    phase = 2.0 * math.pi * frequency_hz * timestamp
    cosine, sine = math.cos(phase), math.sin(phase)
    center = orbit_radius * np.asarray((cosine, sine, 0.0))
    base = np.asarray(((-sine, cosine, 0.0),
                       (0.0, 0.0, -1.0),
                       (-cosine, -sine, 0.0)))
    q = base @ (point - center)
    camera = rotation @ q
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    if camera[2] <= 1e-6 * orbit_radius:
        return (np.asarray((_INVISIBLE_ERROR_PX, 0.0)),
                np.zeros((2, 3)), np.zeros((2, 3)))
    x, y, z = camera
    projection = np.asarray((fx * x / z + cx, fy * y / z + cy))
    residual = projection - observed
    pixel_camera = np.asarray(((fx / z, 0.0, -fx * x / (z * z)),
                               (0.0, fy / z, -fy * y / (z * z))))
    camera_skew = np.asarray(((0.0, -z, y),
                              (z, 0.0, -x),
                              (-y, x, 0.0)))
    jacobian_left = pixel_camera @ (-camera_skew)
    jacobian_right = jacobian_left @ rotation
    jacobian_point = pixel_camera @ rotation @ base
    return residual, jacobian_right, jacobian_point


def _raw_errors(
    times: np.ndarray,
    pixels: np.ndarray,
    owners: np.ndarray,
    points: np.ndarray,
    rotation: np.ndarray,
    frequency_hz: float,
    orbit_radius: float,
    intrinsic: np.ndarray,
) -> np.ndarray:
    result = np.empty(times.size, dtype=np.float64)
    for index, (timestamp, observed, owner) in enumerate(zip(times, pixels, owners)):
        residual, _, _ = _projection_residual_and_jacobians(
            float(timestamp), observed, points[int(owner)], rotation,
            frequency_hz, orbit_radius, intrinsic,
        )
        result[index] = float(np.linalg.norm(residual))
    return result


def _factor_callback(
    rotation_key: int,
    point_key: int,
    timestamp: float,
    observed: np.ndarray,
    frequency_hz: float,
    orbit_radius: float,
    intrinsic: np.ndarray,
) -> Callable[[Any, Any, list[np.ndarray] | None], np.ndarray]:
    observed = np.asarray(observed, dtype=np.float64).copy()

    def error(_factor, values, jacobians):
        rotation = values.atRot3(rotation_key).matrix()
        point = np.asarray(values.atPoint3(point_key), dtype=np.float64)
        residual, jacobian_rotation, jacobian_point = _projection_residual_and_jacobians(
            timestamp, observed, point, rotation, frequency_hz,
            orbit_radius, intrinsic,
        )
        if jacobians is not None:
            jacobians[0] = jacobian_rotation
            jacobians[1] = jacobian_point
        return residual

    return error


def bundle_adjust_gtsam(
    times: np.ndarray,
    pixels: np.ndarray,
    owners: np.ndarray,
    points: np.ndarray,
    rotation: np.ndarray,
    frequency_hz: float,
    orbit_radius: float,
    intrinsic: np.ndarray,
    *,
    huber_threshold_px: float,
    max_iterations: int = 50,
    optimize_frequency: bool = False,
    relative_tolerance: float = 1e-9,
) -> GTSAMBundleAdjustmentResult:
    """Solve paper Eq. (11) directly with GTSAM.

    The main method fixes frequency.  The paper's free-frequency ablation is a
    different factor graph and is intentionally rejected rather than silently
    approximated by this reference entry point.
    """

    if optimize_frequency:
        raise NotImplementedError(
            "direct GTSAM reference implements paper Eq. (11) with f fixed"
        )
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not math.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive")
    if not math.isfinite(huber_threshold_px) or huber_threshold_px <= 0.0:
        raise ValueError("huber_threshold_px must be positive")

    times = np.ascontiguousarray(times, dtype=np.float64)
    pixels = np.ascontiguousarray(pixels, dtype=np.float64)
    owners = np.ascontiguousarray(owners, dtype=np.int64)
    points = np.array(points, dtype=np.float64, copy=True)
    rotation = _as_rotation_matrix(rotation).copy()
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if times.ndim != 1 or pixels.shape != (times.size, 2) or owners.shape != times.shape:
        raise ValueError("times, pixels, and owners have incompatible shapes")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N,3]")
    counts = np.bincount(owners, minlength=points.shape[0])
    if (owners.size == 0 or np.any(owners < 0) or np.any(owners >= points.shape[0])
            or counts.size != points.shape[0] or np.any(counts == 0)):
        raise ValueError("every landmark needs at least one observation")

    gtsam = _import_gtsam()
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    rotation_key = gtsam.symbol("r", 0)
    point_keys = [gtsam.symbol("l", index) for index in range(points.shape[0])]
    initial.insert(rotation_key, gtsam.Rot3(rotation))
    for key, point in zip(point_keys, points):
        initial.insert(key, point)

    base = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
    robust = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(float(huber_threshold_px)), base,
    )
    for timestamp, observed, owner in zip(times, pixels, owners):
        point_key = point_keys[int(owner)]
        graph.add(gtsam.CustomFactor(
            robust, [rotation_key, point_key],
            _factor_callback(
                rotation_key, point_key, float(timestamp), observed,
                float(frequency_hz), float(orbit_radius), intrinsic,
            ),
        ))

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(max_iterations)
    params.setRelativeErrorTol(relative_tolerance)
    params.setAbsoluteErrorTol(1e-12)
    params.setlambdaInitial(1e-4)
    # Retain GTSAM's ordinary constructor defaults beyond the explicitly
    # declared stopping conditions.  The paper names GTSAM but does not report
    # an alternative damping convention.
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    initial_cost = float(graph.error(initial))
    solution = optimizer.optimize()
    iterations = int(optimizer.iterations())
    final_rotation = solution.atRot3(rotation_key).matrix()
    final_points = np.asarray([solution.atPoint3(key) for key in point_keys],
                              dtype=np.float64)
    errors = _raw_errors(
        times, pixels, owners, final_points, final_rotation,
        float(frequency_hz), float(orbit_radius), intrinsic,
    )
    per_point = np.bincount(owners, weights=errors,
                            minlength=points.shape[0]) / counts
    return GTSAMBundleAdjustmentResult(
        rotation=final_rotation,
        frequency_hz=float(frequency_hz),
        points=final_points,
        per_point_error_px=per_point,
        per_observation_error_px=errors,
        initial_cost=initial_cost,
        final_cost=float(graph.error(solution)),
        iterations=iterations,
        termination_status=("max_iterations" if iterations >= max_iterations
                            else "converged_or_stalled"),
    )
