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
* fuses tracks whose world points are closer than ``delta_fuse`` before the
  optimisation, following the order of Fig. 3 (Sec. 4.2.6, disabled by the
  "No landmark association" ablation),
* eliminates tracks with fewer than ``N_match`` events, landmarks whose mean
  reprojection error exceeds ``gamma`` and landmarks farther than ``r`` from
  the orbit centre after the optimisation (Sec. 4.2.5).

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
    # Mean over every inlier observation (the headline value averages landmarks).
    mean_observation_error_px: float = math.nan

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
    removed_short: int
    removed_reprojection: int
    removed_radius: int
    fused: int
    iterations: int
    termination_status: str
    mean_observation_error_px: float = math.nan
    triangulation_fallback_count: int = 0
    triangulation_fallback_reasons: str = ""

    @property
    def converged(self) -> bool:
        return self.termination_status == "converged"


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
        self.observation_error = math.nan
        self.observation_count = 0
        self.reports: list[OptimizationReport] = []
        self._current_triangulation_fallbacks: dict[str, int] = {}

    # -- helpers ----------------------------------------------------------------
    def _track_observations(self, track) -> tuple[np.ndarray, np.ndarray]:
        t = track.t - self.time_origin
        x, y = track.x.astype(np.float64), track.y.astype(np.float64)
        # Eq. (11) and Sec. 4.2.4 use every retained track event.  A finite
        # cap is an explicitly experimental replay option, never a paper-mode
        # behavior even if a caller accidentally carries that old value over.
        limit = (self.config.max_observations_per_track
                 if self.config.interpretation_mode == "experimental" else 0)
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
        # Sec. 4.2.4 states that all retained observations enter the linear
        # multi-view triangulation.  The former 64-view cap is kept only for
        # replaying the explicitly experimental implementation.
        max_views = 64 if self.config.interpretation_mode == "experimental" else None
        min_depth = 0.9 if self.config.interpretation_mode == "experimental" else None
        point = triangulate(
            times, pixels, frequency, self.config.orbit_radius,
            self.rotation, self.intrinsic, max_views=max_views,
            min_positive_depth_fraction=min_depth,
        )
        if point is not None and (
            self.config.interpretation_mode == "paper"
            or np.linalg.norm(point) <= self.config.orbit_radius
        ):
            return point
        cause = "triangulation_failed" if point is None else "triangulation_outside_orbit"
        disposition = "reused_previous" if previous is not None else "used_origin"
        reason = f"{cause}_{disposition}"
        self._current_triangulation_fallbacks[reason] = (
            self._current_triangulation_fallbacks.get(reason, 0) + 1
        )
        return previous if previous is not None else np.zeros(3)

    # -- main entry point --------------------------------------------------------
    def optimize(self, pool: TrackPool, frequency_hz: float, event_time: float) -> OptimizationReport:
        config = self.config
        self._current_triangulation_fallbacks = {}
        frequency = float(self.frequency_hz) if (
            config.optimize_frequency and self.initialized and math.isfinite(self.frequency_hz)
        ) else float(frequency_hz)
        snapshot = pool.snapshot()
        preoptimization_short = [
            track.track_id for track in snapshot
            if track.size < config.min_track_events
        ]
        # The historical experimental implementation removed short tracks
        # before triangulation.  Fig. 3 and Sec. 4.2.5 place all three outlier
        # tests after association and optimisation, so the paper path delays
        # this test; fragments may consequently acquire N_match observations
        # when landmark association merges them.
        if config.interpretation_mode == "experimental":
            pool.remove(preoptimization_short)
        min_span = config.min_triangulation_angle_deg / (360.0 * frequency)
        tracks = [
            track for track in snapshot
            if (config.interpretation_mode != "experimental"
                or track.track_id not in preoptimization_short)
            and track.latest_t - track.earliest_t >= min_span
        ]
        for stale in set(self.landmarks) - set(pool.tracks):
            self.landmarks.pop(stale, None)
            self.errors.pop(stale, None)
            self.revisions.pop(stale, None)
        if not tracks:
            report = OptimizationReport(
                event_time=event_time, frequency_hz=frequency, landmarks=0,
                observations=0, mean_reprojection_error_px=math.nan,
                removed_short=(len(preoptimization_short)
                               if config.interpretation_mode == "experimental" else 0),
                removed_reprojection=0,
                removed_radius=0, fused=0, iterations=0,
                termination_status="not_run",
            )
            self.reports.append(report)
            return report
        # Sec. 4.2.3 / 4.2.4: origin on the first run, otherwise triangulation
        # of new or updated tracks with the latest R0 and f*.
        initial: dict[int, np.ndarray] = {}
        for track in tracks:
            t, xy = self._track_observations(track)
            updated = self.revisions.get(track.track_id) != track.revision
            initial[track.track_id] = self._initial_point(track.track_id, t, xy, frequency,
                                                          updated)
        # Sec. 4.2.6 (Fig. 3 order: triangulation -> association -> optimisation
        # -> outlier elimination). Skipped on the first run, where every point
        # starts at the origin and would trivially coincide.
        fused = 0
        if self.initialized and config.landmark_association:
            tracks, initial, fused = self._associate_landmarks(pool, tracks, initial)
        times, pixels, owners = [], [], []
        for owner, track in enumerate(tracks):
            t, xy = self._track_observations(track)
            times.append(t)
            pixels.append(xy)
            owners.append(np.full(t.size, owner, dtype=np.int64))
        times_array = np.concatenate(times)
        owners_array = np.concatenate(owners)
        solver = bundle_adjust
        if config.solver_backend == "gtsam":
            from .gtsam_backend import bundle_adjust_gtsam
            solver = bundle_adjust_gtsam
        result = solver(
            times_array, np.concatenate(pixels), owners_array,
            np.asarray([initial[track.track_id] for track in tracks]),
            np.eye(3) if not self.initialized else self.rotation,
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
        # Sec. 4.2.5: outlier elimination after the optimisation.
        short_bad = np.asarray([
            track.size < config.min_track_events for track in tracks
        ], dtype=bool)
        radius_bad = ~np.isfinite(points).all(axis=1) | (
            np.linalg.norm(points, axis=1) > config.orbit_radius)
        error_bad = ~(short_bad | radius_bad) & ~(
            errors <= config.max_mean_reprojection_error_px)
        pool.remove(ids[short_bad | radius_bad | error_bad])
        keep = ~(short_bad | radius_bad | error_bad)
        self.landmarks = {int(i): p for i, p in zip(ids[keep], points[keep])}
        self.errors = {int(i): float(e) for i, e in zip(ids[keep], errors[keep])}
        revision = {track.track_id: track.revision for track in tracks}
        self.revisions = {int(i): revision[int(i)] for i in ids[keep]}
        inlier_observations = keep[owners_array]
        self.observation_error = (float(np.mean(result.per_observation_error_px[inlier_observations]))
                                  if inlier_observations.any() else math.nan)
        self.observation_count = int(inlier_observations.sum())
        mean_error = float(np.mean(list(self.errors.values()))) if self.errors else math.nan
        report = OptimizationReport(
            event_time=event_time, frequency_hz=result.frequency_hz,
            landmarks=len(self.landmarks), observations=int(times_array.size),
            mean_reprojection_error_px=mean_error,
            removed_short=(len(preoptimization_short)
                           if config.interpretation_mode == "experimental"
                           else int(short_bad.sum())),
            removed_reprojection=int(error_bad.sum()),
            removed_radius=int(radius_bad.sum()), fused=fused,
            iterations=result.iterations,
            termination_status=result.termination_status,
            mean_observation_error_px=self.observation_error,
            triangulation_fallback_count=sum(self._current_triangulation_fallbacks.values()),
            triangulation_fallback_reasons=";".join(
                f"{reason}:{count}"
                for reason, count in sorted(self._current_triangulation_fallbacks.items())
            ),
        )
        self.reports.append(report)
        return report

    def _associate_landmarks(self, pool: TrackPool, tracks: list, initial: dict[int, np.ndarray]):
        """Sec. 4.2.6: merge tracks whose world points are within delta_fuse."""

        if len(tracks) < 2:
            return tracks, initial, 0
        ids = np.asarray([track.track_id for track in tracks], dtype=np.int64)
        points = np.asarray([initial[int(i)] for i in ids])
        pairs = cKDTree(points).query_pairs(self.config.landmark_merge_radius, output_type="ndarray")
        if pairs.size == 0:
            return tracks, initial, 0
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                           shape=(ids.size, ids.size))
        count, labels = connected_components(graph, directed=False)
        fused = 0
        removed: set[int] = set()
        for label in range(count):
            members = np.flatnonzero(labels == label)
            if members.size < 2:
                continue
            keep = pool.merge(ids[members])  # H_u2 = H_u1 U H_u2
            initial[keep] = points[members].mean(axis=0)
            for member in ids[members]:
                if int(member) != keep:
                    removed.add(int(member))
                    initial.pop(int(member), None)
                    self.landmarks.pop(int(member), None)
                    self.errors.pop(int(member), None)
                    self.revisions.pop(int(member), None)
            fused += members.size - 1
        tracks = [pool.tracks[track.track_id] for track in tracks
                  if track.track_id not in removed]
        return tracks, initial, fused

    def reconstruction(self) -> Reconstruction:
        if not self.initialized or not self.landmarks:
            return empty_reconstruction("back-end not initialised" if not self.initialized
                                        else "no inlier landmarks")
        ids = np.fromiter(self.landmarks, dtype=np.int64)
        points = np.asarray([self.landmarks[int(i)] for i in ids])
        errors = np.asarray([self.errors[int(i)] for i in ids])
        return Reconstruction(
            points, ids, self.rotation.copy(), float(self.frequency_hz),
            float(errors.mean()), errors, self.observation_count, points.shape[0] >= 4,
            "" if points.shape[0] >= 4 else "fewer than four inlier landmarks",
            self.observation_error,
        )
