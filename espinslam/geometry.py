"""Continuous-time circular-orbit reconstruction and point-cloud evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .config import BackendConfig
from .frontend import FeatureTrack


@dataclass(frozen=True)
class Reconstruction:
    """Sparse reconstruction with per-landmark reprojection diagnostics."""

    points: np.ndarray
    track_ids: np.ndarray
    rotation_vector: np.ndarray
    mean_reprojection_error_px: float
    per_point_reprojection_error_px: np.ndarray
    valid: bool
    reason: str = ""


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


def camera_pose(time_seconds: float, frequency_hz: float, orbit_radius: float,
                rotation_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return conventional world-to-camera rotation and camera center."""

    phase = 2.0 * math.pi * frequency_hz * time_seconds
    center = orbit_radius * np.asarray((math.cos(phase), math.sin(phase), 0.0))
    z_axis = -center / np.linalg.norm(center)
    world_z = np.asarray((0.0, 0.0, 1.0))
    x_axis = np.cross(z_axis, world_z)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-9:
        raise ValueError("degenerate orbit view")
    x_axis /= x_norm
    y_axis = np.cross(z_axis, x_axis)
    base = np.stack((x_axis, y_axis, z_axis), axis=0)
    offset = Rotation.from_rotvec(np.asarray(rotation_vector)).as_matrix()
    return offset @ base, center


def projection_matrix(time_seconds: float, frequency_hz: float,
                      orbit_radius: float, rotation_vector: np.ndarray,
                      intrinsic: np.ndarray) -> np.ndarray:
    rotation, center = camera_pose(
        time_seconds, frequency_hz, orbit_radius, rotation_vector,
    )
    return intrinsic @ np.column_stack((rotation, -rotation @ center))


def project_points(points: np.ndarray, times: np.ndarray, frequency_hz: float,
                   orbit_radius: float, rotation_vector: np.ndarray,
                   intrinsic: np.ndarray) -> np.ndarray:
    """Project one point per timestamp into its virtual camera."""

    if points.shape[0] != times.shape[0]:
        raise ValueError("points and times must have equal lengths")
    phases = 2.0 * math.pi * frequency_hz * np.asarray(times, dtype=np.float64)
    cosine, sine = np.cos(phases), np.sin(phases)
    centers = orbit_radius * np.column_stack((cosine, sine, np.zeros_like(cosine)))
    base = np.zeros((times.size, 3, 3), dtype=np.float64)
    base[:, 0, 0], base[:, 0, 1] = -sine, cosine
    base[:, 1, 2] = -1.0
    base[:, 2, 0], base[:, 2, 1] = -cosine, -sine
    offset = Rotation.from_rotvec(np.asarray(rotation_vector)).as_matrix()
    rotations = np.einsum("ij,njk->nik", offset, base)
    camera = np.einsum("nij,nj->ni", rotations, points - centers)
    pixel = camera @ intrinsic.T
    result = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    visible = camera[:, 2] > 1e-9
    result[visible] = pixel[visible, :2] / pixel[visible, 2, None]
    return result


def _bundle_adjustment_sparsity(owners: np.ndarray, landmark_count: int):
    """Encode which pose/landmark variables affect each image residual."""

    pattern = lil_matrix((2 * owners.size, 3 + 3 * landmark_count), dtype=np.int8)
    pattern[:, :3] = 1
    for observation, owner in enumerate(owners):
        pattern[2 * observation:2 * observation + 2,
                3 + 3 * owner:3 + 3 * owner + 3] = 1
    return pattern.tocsr()


def triangulate_track(track: FeatureTrack, frequency_hz: float, orbit_radius: float,
                      rotation_vector: np.ndarray, intrinsic: np.ndarray) -> np.ndarray | None:
    """Linear multi-view DLT using the virtual pose of every track observation."""

    if len(track.observations) < 2:
        return None
    observations = track.observations
    if len(observations) > 32:
        indices = np.linspace(0, len(observations) - 1, 32).round().astype(int)
        observations = [observations[index] for index in np.unique(indices)]
    equations = []
    for value in observations:
        matrix = projection_matrix(
            value.t, frequency_hz, orbit_radius, rotation_vector, intrinsic,
        )
        equations.append(value.x * matrix[2] - matrix[0])
        equations.append(value.y * matrix[2] - matrix[1])
    try:
        _, _, right = np.linalg.svd(np.asarray(equations), full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    homogeneous = right[-1]
    if abs(homogeneous[3]) < 1e-10:
        return None
    point = homogeneous[:3] / homogeneous[3]
    return point if np.isfinite(point).all() else None


def _observations_for_tracks(tracks: list[FeatureTrack]) -> tuple[np.ndarray, np.ndarray,
                                                                  np.ndarray, np.ndarray]:
    times, pixels, owners, track_ids = [], [], [], []
    for owner, track in enumerate(tracks):
        values = track.observations
        if len(values) > 48:
            indices = np.linspace(0, len(values) - 1, 48).round().astype(int)
            values = [values[index] for index in np.unique(indices)]
        for value in values:
            times.append(value.t)
            pixels.append((value.x, value.y))
            owners.append(owner)
        track_ids.append(track.track_id)
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(pixels, dtype=np.float64),
        np.asarray(owners, dtype=np.int64),
        np.asarray(track_ids, dtype=np.int64),
    )


def reconstruct(
    tracks: list[FeatureTrack], frequency_hz: float, intrinsic: np.ndarray,
    config: BackendConfig,
) -> Reconstruction:
    """Triangulate and robustly refine R0 and sparse landmarks."""

    selected = [
        track for track in tracks
        if len(track.observations) >= config.min_track_observations
        and track.latest.t > track.earliest.t
    ]
    selected.sort(key=lambda track: len(track.observations), reverse=True)
    selected = selected[:config.max_landmarks]
    initial_rotation = np.zeros(3, dtype=np.float64)
    valid_tracks, initial_points = [], []
    for track in selected:
        point = triangulate_track(
            track, frequency_hz, config.orbit_radius, initial_rotation, intrinsic,
        )
        if point is not None and np.linalg.norm(point) <= config.orbit_radius:
            valid_tracks.append(track)
            initial_points.append(point)
    if len(valid_tracks) < 4:
        return Reconstruction(
            np.empty((0, 3)), np.empty(0, dtype=np.int64), initial_rotation,
            math.nan, np.empty(0), False, "fewer than four triangulated tracks",
        )
    points = np.asarray(initial_points, dtype=np.float64)
    times, pixels, owners, track_ids = _observations_for_tracks(valid_tracks)

    def residual(parameters: np.ndarray) -> np.ndarray:
        rotation_vector = parameters[:3]
        landmarks = parameters[3:].reshape((-1, 3))
        projected = project_points(
            landmarks[owners], times, frequency_hz, config.orbit_radius,
            rotation_vector, intrinsic,
        )
        values = projected - pixels
        values[~np.isfinite(values)] = 1e3
        return values.ravel()

    initial = np.concatenate((initial_rotation, points.ravel()))
    optimized = least_squares(
        residual, initial, loss="huber", f_scale=config.huber_scale_px,
        max_nfev=config.optimization_iterations, method="trf",
        jac_sparsity=_bundle_adjustment_sparsity(owners, points.shape[0]),
    )
    rotation_vector = optimized.x[:3]
    points = optimized.x[3:].reshape((-1, 3))
    errors = np.linalg.norm(residual(optimized.x).reshape((-1, 2)), axis=1)
    per_point = np.asarray([
        float(errors[owners == index].mean())
        for index in range(points.shape[0])
    ])
    keep = (
        np.isfinite(points).all(axis=1)
        & (np.linalg.norm(points, axis=1) <= config.orbit_radius)
        & (per_point <= config.max_mean_reprojection_error_px)
    )
    points, track_ids, per_point = points[keep], track_ids[keep], per_point[keep]
    if points.shape[0] < 4:
        return Reconstruction(
            points, track_ids, rotation_vector,
            float(np.mean(per_point)) if per_point.size else math.nan,
            per_point, False, "fewer than four inlier landmarks",
        )
    points, track_ids, per_point = _fuse_landmarks(
        points, track_ids, per_point, config.landmark_merge_radius,
    )
    return Reconstruction(
        points, track_ids, rotation_vector, float(np.mean(per_point)),
        per_point, True,
    )


def _fuse_landmarks(points: np.ndarray, track_ids: np.ndarray,
                    errors: np.ndarray, radius: float):
    if points.shape[0] < 2:
        return points, track_ids, errors
    tree = cKDTree(points)
    groups = tree.query_ball_tree(tree, radius)
    used: set[int] = set()
    fused_points, fused_ids, fused_errors = [], [], []
    for index, neighbors in enumerate(groups):
        if index in used:
            continue
        group = [value for value in neighbors if value not in used]
        used.update(group)
        weights = 1.0 / np.maximum(errors[group], 1e-6)
        fused_points.append(np.average(points[group], axis=0, weights=weights))
        best = group[int(np.argmin(errors[group]))]
        fused_ids.append(track_ids[best])
        fused_errors.append(float(np.average(errors[group], weights=weights)))
    return (
        np.asarray(fused_points), np.asarray(fused_ids, dtype=np.int64),
        np.asarray(fused_errors),
    )


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
                         *, iterations: int = 40) -> PointCloudAlignment:
    """Resolve monocular scale, rotation, and translation before ICP metrics."""

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
    reference_tree = cKDTree(reference)
    source_axes, target_axes = _pca_axes(predicted), _pca_axes(reference)
    best: tuple[float, np.ndarray, float, np.ndarray, np.ndarray] | None = None
    for permutation in itertools.permutations(range(3)):
        permutation_matrix = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            orientation = target_axes @ permutation_matrix @ np.diag(signs) @ source_axes.T
            if np.linalg.det(orientation) < 0:
                continue
            scale = reference_radius / predicted_radius
            translation = reference.mean(axis=0) - scale * (orientation @ predicted.mean(axis=0))
            aligned = scale * (predicted @ orientation.T) + translation
            for _ in range(iterations):
                _, nearest = reference_tree.query(aligned, k=1)
                # Monocular scale is fixed once from the global cloud extent.
                # Re-estimating it from one-way nearest neighbors lets ICP
                # collapse every point onto one reference point and report a
                # spuriously perfect RMSE.
                step_scale, step_rotation, step_translation = _umeyama(
                    aligned, reference[nearest], estimate_scale=False,
                )
                updated = step_scale * (aligned @ step_rotation.T) + step_translation
                if np.max(np.linalg.norm(updated - aligned, axis=1)) < 1e-7:
                    aligned = updated
                    scale = step_scale * scale
                    orientation = step_rotation @ orientation
                    translation = step_scale * (step_rotation @ translation) + step_translation
                    break
                aligned = updated
                scale = step_scale * scale
                orientation = step_rotation @ orientation
                translation = step_scale * (step_rotation @ translation) + step_translation
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
