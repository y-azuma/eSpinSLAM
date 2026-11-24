"""Continuous-time circular-orbit camera model, robust bundle adjustment,
triangulation, and point-cloud alignment.

Orbit model (paper Sec. 3.1.2, Eqs. 2-5)
-----------------------------------------
The camera centre at time ``t`` is ``c(t) = r [cos(2 pi f t), sin(2 pi f t), 0]``.
The base frame looks at the orbit centre: ``r_z = -c / |c|``,
``r_x = r_z x k`` and ``r_y = r_z x r_x``. The world-to-camera rotation is
``R(t) = R0 B(t)`` where the rows of ``B(t)`` are ``r_x, r_y, r_z`` and ``R0``
is the constant camera orientation relative to the orbital plane. This is the
inverse of the paper's camera-to-world composition ``T_b^t T_w^b`` up to the
(free) re-parameterisation ``R0 <- R0^T``.

Bundle adjustment (paper Eq. 11)
--------------------------------
``min_{R0, X} sum_u sum_v rho(|| pi(K R(t_v) (X_u - c(t_v))) - x_v ||)`` with
``f`` fixed to ``f*`` (optionally free for the Table 3 ablation). The paper
solves it with GTSAM; here a Levenberg-Marquardt solver with Huber IRLS
weights (the robust-noise-model strategy GTSAM uses) and analytic Jacobians
is used. The landmark blocks are eliminated with a Schur complement so each
iteration costs O(#observations) and only a 3x3 (or 4x4) system is solved.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

from numba import njit
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

_TWO_PI = 2.0 * math.pi
_INVISIBLE_ERROR_PX = 1.0e4


def _as_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape == (3, 3):
        return rotation
    if rotation.shape == (3,):
        return Rotation.from_rotvec(rotation).as_matrix()
    raise ValueError("rotation must be a 3x3 matrix or a rotation vector")


def so3_exp(vector: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(vector, dtype=np.float64)).as_matrix()


def orbit_camera_frames(times: np.ndarray, frequency_hz: float, orbit_radius: float,
                        rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return world-to-camera rotations (N,3,3) and centres (N,3) for ``times``."""

    times = np.asarray(times, dtype=np.float64)
    phase = _TWO_PI * frequency_hz * times
    cosine, sine = np.cos(phase), np.sin(phase)
    centers = orbit_radius * np.column_stack((cosine, sine, np.zeros_like(cosine)))
    base = np.zeros((times.size, 3, 3), dtype=np.float64)
    base[:, 0, 0], base[:, 0, 1] = -sine, cosine
    base[:, 1, 2] = -1.0
    base[:, 2, 0], base[:, 2, 1] = -cosine, -sine
    rotations = np.einsum("ij,njk->nik", _as_rotation_matrix(rotation), base)
    return rotations, centers


def camera_pose(time_seconds: float, frequency_hz: float, orbit_radius: float,
                rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the world-to-camera rotation and camera centre at one timestamp."""

    rotations, centers = orbit_camera_frames(
        np.asarray([time_seconds]), frequency_hz, orbit_radius, rotation,
    )
    return rotations[0], centers[0]


def projection_matrix(time_seconds: float, frequency_hz: float,
                      orbit_radius: float, rotation: np.ndarray,
                      intrinsic: np.ndarray) -> np.ndarray:
    rotation_wc, center = camera_pose(time_seconds, frequency_hz, orbit_radius, rotation)
    return intrinsic @ np.column_stack((rotation_wc, -rotation_wc @ center))


def project_points(points: np.ndarray, times: np.ndarray, frequency_hz: float,
                   orbit_radius: float, rotation: np.ndarray,
                   intrinsic: np.ndarray) -> np.ndarray:
    """Project one point per timestamp into its virtual camera (NaN if behind)."""

    points = np.asarray(points, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if points.shape[0] != times.shape[0]:
        raise ValueError("points and times must have equal lengths")
    rotations, centers = orbit_camera_frames(times, frequency_hz, orbit_radius, rotation)
    camera = np.einsum("nij,nj->ni", rotations, points - centers)
    pixel = camera @ intrinsic.T
    result = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    visible = camera[:, 2] > 1e-9
    result[visible] = pixel[visible, :2] / pixel[visible, 2, None]
    return result


# ---------------------------------------------------------------------------
# Triangulation (paper Sec. 4.2.4)
# ---------------------------------------------------------------------------

def triangulate(times: np.ndarray, pixels: np.ndarray, frequency_hz: float,
                orbit_radius: float, rotation: np.ndarray, intrinsic: np.ndarray,
                *, max_views: int | None = None,
                min_positive_depth_fraction: float | None = None) -> np.ndarray | None:
    """Linear multi-view (DLT) triangulation from virtual orbit cameras.

    The paper's Sec. 4.2.4 sends the retained observations of a track to
    multi-view triangulation and does not describe view subsampling.  The
    reference path therefore leaves ``max_views`` unset.  It also returns a
    finite DLT solution without an unreported cheirality threshold.  Finite
    values of ``max_views`` and ``min_positive_depth_fraction`` are retained
    only for replaying the earlier experimental implementation.
    """

    times = np.asarray(times, dtype=np.float64)
    pixels = np.asarray(pixels, dtype=np.float64)
    if (min_positive_depth_fraction is not None
            and not 0.0 <= min_positive_depth_fraction <= 1.0):
        raise ValueError("min_positive_depth_fraction must lie in [0, 1]")
    if times.size < 2:
        return None
    if max_views is not None and times.size > max_views:
        indices = np.unique(np.linspace(0, times.size - 1, max_views).round().astype(int))
        times, pixels = times[indices], pixels[indices]
    rotations, centers = orbit_camera_frames(times, frequency_hz, orbit_radius, rotation)
    translations = -np.einsum("nij,nj->ni", rotations, centers)
    matrices = np.einsum("ij,njk->nik", intrinsic,
                         np.concatenate((rotations, translations[:, :, None]), axis=2))
    equations = np.concatenate((
        pixels[:, 0, None] * matrices[:, 2] - matrices[:, 0],
        pixels[:, 1, None] * matrices[:, 2] - matrices[:, 1],
    ), axis=0)
    scale = np.linalg.norm(equations, axis=1, keepdims=True)
    equations = equations / np.maximum(scale, 1e-12)
    try:
        _, _, right = np.linalg.svd(equations, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    homogeneous = right[-1]
    if abs(homogeneous[3]) < 1e-12:
        return None
    point = homogeneous[:3] / homogeneous[3]
    if not np.isfinite(point).all():
        return None
    if min_positive_depth_fraction is not None:
        depth = np.einsum("nj,nj->n", rotations[:, 2], point - centers)
        if np.mean(depth > 0) < min_positive_depth_fraction:
            return None
    return point


# ---------------------------------------------------------------------------
# Robust Levenberg-Marquardt bundle adjustment (paper Sec. 4.2.2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BundleAdjustmentResult:
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
        """True only for an actual tolerance-based convergence decision."""

        return self.termination_status == "converged"


def _huber_cost(errors: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(
        errors <= threshold,
        0.5 * errors ** 2,
        threshold * errors - 0.5 * threshold ** 2,
    )


@njit(cache=True)
def _project_one(t, point, rotation, frequency, radius, fx, fy, cx, cy):
    phase = 2.0 * np.pi * frequency * t
    cosine = np.cos(phase)
    sine = np.sin(phase)
    dx = point[0] - radius * cosine
    dy = point[1] - radius * sine
    q0 = -sine * dx + cosine * dy
    q1 = -point[2]
    q2 = -cosine * dx - sine * dy
    c0 = rotation[0, 0] * q0 + rotation[0, 1] * q1 + rotation[0, 2] * q2
    c1 = rotation[1, 0] * q0 + rotation[1, 1] * q1 + rotation[1, 2] * q2
    c2 = rotation[2, 0] * q0 + rotation[2, 1] * q1 + rotation[2, 2] * q2
    return cosine, sine, c0, c1, c2


@njit(cache=True)
def _huber_scalar(error, threshold):
    if error <= threshold:
        return 0.5 * error * error
    return threshold * error - 0.5 * threshold * threshold


@njit(cache=True)
def _errors_kernel(times, pixels, owners, points, rotation, frequency, radius,
                   fx, fy, cx, cy, huber, invisible_error):
    count = times.shape[0]
    errors = np.empty(count)
    cost = 0.0
    for index in range(count):
        point = points[owners[index]]
        _, _, c0, c1, c2 = _project_one(times[index], point, rotation, frequency,
                                        radius, fx, fy, cx, cy)
        if c2 <= 1e-6 * radius:
            error = invisible_error
        else:
            du = fx * c0 / c2 + cx - pixels[index, 0]
            dv = fy * c1 / c2 + cy - pixels[index, 1]
            error = np.sqrt(du * du + dv * dv)
        errors[index] = error
        cost += _huber_scalar(error, huber)
    return cost, errors


@njit(cache=True)
def _normal_equations_kernel(times, pixels, owners, points, rotation, frequency,
                             radius, fx, fy, cx, cy, huber, invisible_error,
                             global_dim):
    """Accumulate the Huber-IRLS weighted Gauss-Newton system in one pass.

    Global parameters: left SO(3) perturbation of R0 (3) and optionally f (1).
    """

    point_count = points.shape[0]
    h_pp = np.zeros((point_count, 3, 3))
    h_gp = np.zeros((point_count, global_dim, 3))
    g_p = np.zeros((point_count, 3))
    h_gg = np.zeros((global_dim, global_dim))
    g_g = np.zeros(global_dim)
    j_point = np.zeros((2, 3))
    j_global = np.zeros((2, 4))
    residual = np.zeros(2)
    a = np.zeros((2, 3))
    cost = 0.0
    for index in range(times.shape[0]):
        owner = owners[index]
        point = points[owner]
        t = times[index]
        cosine, sine, c0, c1, c2 = _project_one(t, point, rotation, frequency,
                                                radius, fx, fy, cx, cy)
        if c2 <= 1e-6 * radius:
            cost += _huber_scalar(invisible_error, huber)
            continue
        inv_z = 1.0 / c2
        residual[0] = fx * c0 * inv_z + cx - pixels[index, 0]
        residual[1] = fy * c1 * inv_z + cy - pixels[index, 1]
        error = np.sqrt(residual[0] ** 2 + residual[1] ** 2)
        cost += _huber_scalar(error, huber)
        weight = 1.0 if error <= huber else huber / error
        # d pixel / d camera point
        p00 = fx * inv_z
        p02 = -fx * c0 * inv_z * inv_z
        p11 = fy * inv_z
        p12 = -fy * c1 * inv_z * inv_z
        for column in range(3):
            a[0, column] = p00 * rotation[0, column] + p02 * rotation[2, column]
            a[1, column] = p11 * rotation[1, column] + p12 * rotation[2, column]
        for row in range(2):
            j_point[row, 0] = -a[row, 0] * sine - a[row, 2] * cosine
            j_point[row, 1] = a[row, 0] * cosine - a[row, 2] * sine
            j_point[row, 2] = -a[row, 1]
        # Rotation: d camera / d delta = -[camera]_x
        j_global[0, 0] = p02 * c1
        j_global[0, 1] = p00 * c2 - p02 * c0
        j_global[0, 2] = -p00 * c1
        j_global[1, 0] = p12 * c1 - p11 * c2
        j_global[1, 1] = -p12 * c0
        j_global[1, 2] = p11 * c0
        if global_dim == 4:
            dq0 = -(cosine * point[0] + sine * point[1])
            dq2 = sine * point[0] - cosine * point[1]
            scale = 2.0 * np.pi * t
            j_global[0, 3] = (a[0, 0] * dq0 + a[0, 2] * dq2) * scale
            j_global[1, 3] = (a[1, 0] * dq0 + a[1, 2] * dq2) * scale
        for i in range(3):
            for k in range(2):
                g_p[owner, i] += weight * j_point[k, i] * residual[k]
            for j in range(3):
                value = 0.0
                for k in range(2):
                    value += j_point[k, i] * j_point[k, j]
                h_pp[owner, i, j] += weight * value
        for i in range(global_dim):
            for k in range(2):
                g_g[i] += weight * j_global[k, i] * residual[k]
            for j in range(3):
                value = 0.0
                for k in range(2):
                    value += j_global[k, i] * j_point[k, j]
                h_gp[owner, i, j] += weight * value
            for j in range(global_dim):
                value = 0.0
                for k in range(2):
                    value += j_global[k, i] * j_global[k, j]
                h_gg[i, j] += weight * value
    return cost, h_pp, h_gp, h_gg, g_p, g_g


class _Problem:
    """Residuals and Huber-IRLS normal equations of Eq. (11)."""

    def __init__(self, times, pixels, owners, intrinsic, orbit_radius, huber):
        self.times = np.ascontiguousarray(times, dtype=np.float64)
        self.pixels = np.ascontiguousarray(pixels, dtype=np.float64)
        self.owners = np.ascontiguousarray(owners, dtype=np.int64)
        self.fx, self.fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        self.cx, self.cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
        self.radius = float(orbit_radius)
        self.huber = float(huber)

    def errors(self, points, rotation, frequency):
        return _errors_kernel(
            self.times, self.pixels, self.owners, np.ascontiguousarray(points),
            np.ascontiguousarray(rotation), float(frequency), self.radius,
            self.fx, self.fy, self.cx, self.cy, self.huber, _INVISIBLE_ERROR_PX,
        )

    def normal_equations(self, points, rotation, frequency, global_dim):
        return _normal_equations_kernel(
            self.times, self.pixels, self.owners, np.ascontiguousarray(points),
            np.ascontiguousarray(rotation), float(frequency), self.radius,
            self.fx, self.fy, self.cx, self.cy, self.huber, _INVISIBLE_ERROR_PX,
            int(global_dim),
        )


def bundle_adjust(times: np.ndarray, pixels: np.ndarray, owners: np.ndarray,
                  points: np.ndarray, rotation: np.ndarray, frequency_hz: float,
                  orbit_radius: float, intrinsic: np.ndarray, *,
                  huber_threshold_px: float, max_iterations: int = 50,
                  optimize_frequency: bool = False,
                  relative_tolerance: float = 1e-9) -> BundleAdjustmentResult:
    """Robust LM over R0 (and optionally f) and every landmark.

    ``owners`` must index ``points`` and every landmark needs >= 1 observation.
    """

    owners = np.asarray(owners, dtype=np.int64)
    points = np.array(points, dtype=np.float64, copy=True)
    counts = np.bincount(owners, minlength=points.shape[0])
    if counts.size != points.shape[0] or np.any(counts == 0):
        raise ValueError("every landmark needs at least one observation")
    rotation = _as_rotation_matrix(rotation).copy()
    frequency = float(frequency_hz)
    problem = _Problem(times, pixels, owners, intrinsic, orbit_radius, huber_threshold_px)
    global_dim = 4 if optimize_frequency else 3
    lam = 1e-4
    system = problem.normal_equations(points, rotation, frequency, global_dim)
    cost = initial_cost = system[0]
    termination_status = "max_iterations"
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        _, h_pp, h_gp, h_gg, g_p, g_g = system
        improved = False
        relative = 0.0
        for _ in range(12):
            damped_pp = h_pp.copy()
            diagonal = np.einsum("nii->ni", damped_pp)
            diagonal += lam * diagonal + 1e-9
            damped_gg = h_gg + np.diag(lam * np.diag(h_gg) + 1e-9)
            try:
                inv_pp = np.linalg.inv(damped_pp)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            coupling = np.einsum("nij,njk->nik", h_gp, inv_pp)
            schur = damped_gg - np.einsum("nij,nkj->ik", coupling, h_gp)
            rhs = -g_g + np.einsum("nij,nj->i", coupling, g_p)
            try:
                step_global = np.linalg.solve(schur, rhs)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            step_points = np.einsum(
                "nij,nj->ni", inv_pp,
                -g_p - np.einsum("nji,j->ni", h_gp, step_global),
            )
            new_rotation = so3_exp(step_global[:3]) @ rotation
            new_frequency = frequency + (step_global[3] if optimize_frequency else 0.0)
            new_points = points + step_points
            new_cost, _ = problem.errors(new_points, new_rotation, new_frequency)
            if np.isfinite(new_cost) and new_cost < cost:
                relative = (cost - new_cost) / max(cost, 1e-300)
                rotation, frequency, points = new_rotation, new_frequency, new_points
                cost = new_cost
                lam = max(lam / 3.0, 1e-12)
                improved = True
                break
            lam *= 4.0
        if not improved:
            # Failure to find a decreasing LM step is not evidence that a
            # convergence tolerance was met.  Keeping it separate is required
            # by the predeclared viability criterion used for this reproduction.
            termination_status = "stalled"
            break
        if relative < relative_tolerance:
            termination_status = "converged"
            break
        system = problem.normal_equations(points, rotation, frequency, global_dim)
    final_cost, errors = problem.errors(points, rotation, frequency)
    per_point = np.bincount(owners, weights=errors, minlength=points.shape[0]) / counts
    return BundleAdjustmentResult(
        rotation=rotation, frequency_hz=frequency, points=points,
        per_point_error_px=per_point, per_observation_error_px=errors,
        initial_cost=float(initial_cost), final_cost=float(final_cost),
        iterations=iteration, termination_status=termination_status,
    )


# ---------------------------------------------------------------------------
# Point-cloud alignment for the CAD-model metric (paper Sec. 5.5)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PointCloudAlignment:
    """Similarity-ICP result for monocular-scale point clouds."""

    aligned_points: np.ndarray
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    icp_rmse_m: float
    chamfer_rmse_m: float
    valid: bool
    reason: str = ""

    @property
    def icp_rmse(self) -> float:
        """RMSE in the reference cloud's unit (``icp_rmse_m`` kept for compatibility)."""

        return self.icp_rmse_m


def _umeyama(source: np.ndarray, target: np.ndarray,
             *, estimate_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_centered, target_centered = source - source_mean, target - target_mean
    covariance = target_centered.T @ source_centered / source.shape[0]
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vt
    variance = float(np.mean(np.sum(source_centered ** 2, axis=1)))
    scale = (
        float(np.sum(singular * sign) / max(variance, 1e-12))
        if estimate_scale else 1.0
    )
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _pca_axes(points: np.ndarray) -> np.ndarray:
    _, vectors = np.linalg.eigh(np.cov((points - points.mean(axis=0)).T))
    return vectors[:, ::-1]


def _rms_radius(points: np.ndarray) -> float:
    centered = points - points.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))


def align_similarity_icp(predicted: np.ndarray, reference: np.ndarray,
                         *, iterations: int = 40,
                         known_scale: float | None = None) -> PointCloudAlignment:
    """Align a monocular reconstruction to a reference (CAD) surface sample.

    The monocular scale is unobservable (``r`` is fixed arbitrarily). When the
    metric scale is known (e.g. the true camera-to-spin-axis distance divided
    by ``r``) pass ``known_scale``; otherwise it is fixed once from the ratio
    of RMS radii, because re-estimating it from one-way nearest neighbours lets
    ICP collapse the cloud onto a surface patch. Orientation is initialised
    from the 24 proper PCA-axis hypotheses and refined by rigid ICP.
    """

    if predicted.ndim != 2 or predicted.shape[1:] != (3,) or predicted.shape[0] < 4:
        return _invalid_alignment("predicted point cloud has fewer than four points")
    if reference.ndim != 2 or reference.shape[1:] != (3,) or reference.shape[0] < 4:
        return _invalid_alignment("reference point cloud has fewer than four points")
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    predicted_radius = _rms_radius(predicted)
    reference_radius = _rms_radius(reference)
    if predicted_radius <= 1e-9:
        return _invalid_alignment("predicted point cloud has degenerate extent")
    if reference_radius <= 1e-9:
        return _invalid_alignment("reference point cloud has degenerate extent")
    if known_scale is not None and not (math.isfinite(known_scale) and known_scale > 0):
        return _invalid_alignment("known_scale must be positive and finite")
    reference_tree = cKDTree(reference)
    source_axes, target_axes = _pca_axes(predicted), _pca_axes(reference)
    base_scale = known_scale if known_scale is not None else reference_radius / predicted_radius
    best: tuple[float, np.ndarray, float, np.ndarray, np.ndarray] | None = None
    for permutation in itertools.permutations(range(3)):
        permutation_matrix = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            orientation = target_axes @ permutation_matrix @ np.diag(signs) @ source_axes.T
            if np.linalg.det(orientation) < 0:
                continue
            scale = base_scale
            translation = reference.mean(axis=0) - scale * (orientation @ predicted.mean(axis=0))
            aligned = scale * (predicted @ orientation.T) + translation
            for _ in range(iterations):
                _, nearest = reference_tree.query(aligned, k=1)
                _, step_rotation, step_translation = _umeyama(
                    aligned, reference[nearest], estimate_scale=False,
                )
                updated = aligned @ step_rotation.T + step_translation
                orientation = step_rotation @ orientation
                translation = step_rotation @ translation + step_translation
                moved = np.max(np.linalg.norm(updated - aligned, axis=1))
                aligned = updated
                if moved < 1e-9 * reference_radius:
                    break
            distances, _ = reference_tree.query(aligned, k=1)
            rmse = float(np.sqrt(np.mean(distances ** 2)))
            if best is None or rmse < best[0]:
                best = (rmse, aligned, scale, orientation, translation)
    if best is None:
        return _invalid_alignment("similarity alignment failed")
    rmse, aligned, scale, orientation, translation = best
    predicted_tree = cKDTree(aligned)
    backward, _ = predicted_tree.query(reference, k=1)
    forward, _ = reference_tree.query(aligned, k=1)
    chamfer = float(np.sqrt(0.5 * (np.mean(forward ** 2) + np.mean(backward ** 2))))
    return PointCloudAlignment(
        aligned, scale, orientation, translation, rmse, chamfer, True,
    )


def _invalid_alignment(reason: str) -> PointCloudAlignment:
    return PointCloudAlignment(
        np.empty((0, 3)), math.nan, np.eye(3), np.zeros(3),
        math.nan, math.nan, False, reason,
    )
