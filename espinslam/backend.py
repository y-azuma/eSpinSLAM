"""Incremental eSpinSLAM back-end (paper Secs. 4.2.2-4.2.7).

Once a loop has been detected the back-end

* initialises every world point at the origin with ``R0 = I`` and ``f = f*``
  for the very first optimisation (Sec. 4.2.3),
* afterwards initialises the world points of new or updated tracks by
  multi-view triangulation with the latest ``R0`` and ``f*`` (Sec. 4.2.4),
* minimises Eq. (11) over ``R0`` and all world points with ``f`` fixed
  (Sec. 4.2.2; ``optimize_frequency`` is the Table 3 ablation),
* skips tracks without triangulation parallax (orbit arc below
  ``min_triangulation_angle_deg``) until they grow,
* eliminates tracks with fewer than ``N_match`` events, landmarks whose mean
  reprojection error exceeds ``gamma`` and landmarks farther than ``r`` from
  the orbit centre (Sec. 4.2.5),
* fuses tracks whose world points are closer than ``delta_fuse``
  (Sec. 4.2.6, disabled by the "No landmark association" ablation).

Forgetting of stale tracks (Sec. 4.2.7) is applied by the pipeline on the
shared track pool.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .config import BackendConfig
from .frontend import TrackPool
from .geometry import bundle_adjust, triangulate


@dataclass(frozen=True)
class Reconstruction:
    """Sparse map with per-landmark reprojection diagnostics."""

    points: np.ndarray
    track_ids: np.ndarray
    rotation: np.ndarray
    frequency_hz: float
    mean_reprojection_error_px: float
    per_point_reprojection_error_px: np.ndarray
    observation_count: int
    valid: bool
    reason: str = ""

    @property
    def rotation_vector(self) -> np.ndarray:
        return Rotation.from_matrix(self.rotation).as_rotvec()


def empty_reconstruction(reason: str) -> Reconstruction:
    return Reconstruction(
        np.empty((0, 3)), np.empty(0, dtype=np.int64), np.eye(3), math.nan,
        math.nan, np.empty(0), 0, False, reason,
    )


@dataclass(frozen=True)
class OptimizationReport:
    event_time: float
    frequency_hz: float
    landmarks: int
    observations: int
    mean_reprojection_error_px: float
    removed_reprojection: int
    removed_radius: int
    fused: int
    iterations: int
    converged: bool


class UndistortionMap:
    """Per-pixel undistorted coordinates for integer event locations."""

    def __init__(self, width: int, height: int, intrinsic: np.ndarray,
                 distortion: list[float]):
        import cv2
        grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float64),
                                     np.arange(height, dtype=np.float64))
        points = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)[:, None, :]
        undistorted = cv2.undistortPoints(
            points, intrinsic, np.asarray(distortion, dtype=np.float64), P=intrinsic,
        ).reshape(height, width, 2)
        self.map = undistorted

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xi = np.clip(np.rint(x).astype(np.int64), 0, self.map.shape[1] - 1)
        yi = np.clip(np.rint(y).astype(np.int64), 0, self.map.shape[0] - 1)
        values = self.map[yi, xi]
        return values[:, 0], values[:, 1]


class IncrementalBackend:
    """Continuous-time structure and camera-orientation optimisation."""

    def __init__(self, config: BackendConfig, intrinsic: np.ndarray, *,
                 time_origin: float = 0.0, undistort: UndistortionMap | None = None):
        self.config = config
        self.intrinsic = np.asarray(intrinsic, dtype=np.float64)
        self.time_origin = float(time_origin)
        self.undistort = undistort
        self.rotation = np.eye(3)
        self.frequency_hz = math.nan
        self.landmarks: dict[int, np.ndarray] = {}
        self.errors: dict[int, float] = {}
        self.revisions: dict[int, int] = {}
        self.initialized = False
        self.reports: list[OptimizationReport] = []

    # -- helpers ----------------------------------------------------------------
    def _track_observations(self, track) -> tuple[np.ndarray, np.ndarray]:
        t = track.t - self.time_origin
        x, y = track.x.astype(np.float64), track.y.astype(np.float64)
        limit = self.config.max_observations_per_track
        if limit and t.size > limit:
            indices = np.unique(np.linspace(0, t.size - 1, limit).round().astype(np.int64))
            t, x, y = t[indices], x[indices], y[indices]
        if self.undistort is not None:
            x, y = self.undistort(x, y)
        return t, np.column_stack((x, y))

    def _initial_point(self, track_id: int, times: np.ndarray, pixels: np.ndarray,
                       frequency: float, updated: bool) -> np.ndarray:
        if not self.initialized:
            return np.zeros(3)  # Sec. 4.2.3: first run starts from the origin.
        previous = self.landmarks.get(track_id)
        if previous is not None and not updated:
            return previous
        point = triangulate(times, pixels, frequency, self.config.orbit_radius,
                            self.rotation, self.intrinsic)
        if point is not None and np.linalg.norm(point) <= self.config.orbit_radius:
            return point
        return previous if previous is not None else np.zeros(3)

    # -- main entry point --------------------------------------------------------
    def optimize(self, pool: TrackPool, frequency_hz: float, event_time: float) -> OptimizationReport:
        config = self.config
        frequency = float(self.frequency_hz) if (
            config.optimize_frequency and self.initialized and math.isfinite(self.frequency_hz)
        ) else float(frequency_hz)
        min_span = config.min_triangulation_angle_deg / (360.0 * frequency)
        tracks = [
            track for track in pool.snapshot()
            if track.size >= config.min_track_events
            and track.latest_t - track.earliest_t >= max(min_span, 1e-9)
        ]
        for stale in set(self.landmarks) - set(pool.tracks):
            self.landmarks.pop(stale, None)
            self.errors.pop(stale, None)
            self.revisions.pop(stale, None)
        if not tracks:
            report = OptimizationReport(event_time, frequency, 0, 0, math.nan, 0, 0, 0, 0, False)
            self.reports.append(report)
            return report
        times, pixels, owners, initial = [], [], [], []
        for owner, track in enumerate(tracks):
            t, xy = self._track_observations(track)
            updated = self.revisions.get(track.track_id) != track.revision
            initial.append(self._initial_point(track.track_id, t, xy, frequency, updated))
            times.append(t)
            pixels.append(xy)
            owners.append(np.full(t.size, owner, dtype=np.int64))
        times_array = np.concatenate(times)
        result = bundle_adjust(
            times_array, np.concatenate(pixels), np.concatenate(owners),
            np.asarray(initial), np.eye(3) if not self.initialized else self.rotation,
            frequency, config.orbit_radius, self.intrinsic,
            huber_threshold_px=config.huber_threshold_px,
            max_iterations=config.max_iterations,
            optimize_frequency=config.optimize_frequency,
        )
        self.initialized = True
        self.rotation = result.rotation
        self.frequency_hz = result.frequency_hz
        ids = np.asarray([track.track_id for track in tracks], dtype=np.int64)
        points = result.points
        errors = result.per_point_error_px
        radius_bad = ~np.isfinite(points).all(axis=1) | (
            np.linalg.norm(points, axis=1) > config.orbit_radius)
        error_bad = ~radius_bad & ~(errors <= config.max_mean_reprojection_error_px)
        removed = ids[radius_bad | error_bad]
        pool.remove(removed)
        keep = ~(radius_bad | error_bad)
        self.landmarks = {int(i): p for i, p in zip(ids[keep], points[keep])}
        self.errors = {int(i): float(e) for i, e in zip(ids[keep], errors[keep])}
        revision = {track.track_id: track.revision for track in tracks}
        self.revisions = {int(i): revision[int(i)] for i in ids[keep]}
        fused = self._associate_landmarks(pool) if config.landmark_association else 0
        mean_error = float(np.mean(list(self.errors.values()))) if self.errors else math.nan
        report = OptimizationReport(
            event_time, result.frequency_hz, len(self.landmarks), int(times_array.size),
            mean_error, int(error_bad.sum()), int(radius_bad.sum()), fused,
            result.iterations, result.converged,
        )
        self.reports.append(report)
        return report

    def _associate_landmarks(self, pool: TrackPool) -> int:
        """Sec. 4.2.6: merge tracks whose world points are within delta_fuse."""

        if len(self.landmarks) < 2:
            return 0
        ids = np.fromiter(self.landmarks, dtype=np.int64)
        points = np.asarray([self.landmarks[int(i)] for i in ids])
        pairs = cKDTree(points).query_pairs(self.config.landmark_merge_radius, output_type="ndarray")
        if pairs.size == 0:
            return 0
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                           shape=(ids.size, ids.size))
        count, labels = connected_components(graph, directed=False)
        fused = 0
        for label in range(count):
            members = np.flatnonzero(labels == label)
            if members.size < 2:
                continue
            keep = pool.merge(ids[members])
            point = points[members].mean(axis=0)
            error = float(np.mean([self.errors[int(i)] for i in ids[members]]))
            for member in ids[members]:
                self.landmarks.pop(int(member), None)
                self.errors.pop(int(member), None)
                self.revisions.pop(int(member), None)
            # The merged track keeps a fused initial point; it is re-optimised
            # (and re-triangulated, since its revision changed) next time.
            self.landmarks[keep] = point
            self.errors[keep] = error
            fused += members.size - 1
        return fused

    def reconstruction(self) -> Reconstruction:
        if not self.initialized or not self.landmarks:
            return empty_reconstruction("back-end not initialised" if not self.initialized
                                        else "no inlier landmarks")
        ids = np.fromiter(self.landmarks, dtype=np.int64)
        points = np.asarray([self.landmarks[int(i)] for i in ids])
        errors = np.asarray([self.errors[int(i)] for i in ids])
        observations = self.reports[-1].observations if self.reports else 0
        return Reconstruction(
            points, ids, self.rotation.copy(), float(self.frequency_hz),
            float(errors.mean()), errors, observations, points.shape[0] >= 4,
            "" if points.shape[0] >= 4 else "fewer than four inlier landmarks",
        )
