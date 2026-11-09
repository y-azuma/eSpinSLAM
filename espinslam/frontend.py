"""Online event-only feature detection, clustering, and track management."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import cv2
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import HDBSCAN

from .config import FrontendConfig
from .data import EventSequence


@dataclass(frozen=True)
class Observation:
    """One subpixel feature observation at a continuous timestamp."""

    t: float
    x: float
    y: float
    polarity: int


@dataclass
class FeatureTrack:
    """Bounded sequence of spatio-temporally associated observations."""

    track_id: int
    observations: list[Observation] = field(default_factory=list)

    @property
    def latest(self) -> Observation:
        return self.observations[-1]

    @property
    def earliest(self) -> Observation:
        return self.observations[0]

    def append(self, observation: Observation, retain: int) -> None:
        self.observations.append(observation)
        if len(self.observations) > retain:
            del self.observations[:-retain]


def _assign_latest(surface: np.ndarray, x: np.ndarray, y: np.ndarray,
                   t: np.ndarray, p: np.ndarray) -> None:
    """Update polarity surfaces with each pixel's most recent timestamp."""

    for polarity in (0, 1):
        mask = p == polarity
        if np.any(mask):
            np.maximum.at(surface[polarity], (y[mask], x[mask]), t[mask])


def _time_surface_image(surface: np.ndarray, now: float, decay: float) -> np.ndarray:
    age = now - np.max(surface, axis=0)
    value = np.exp(-np.clip(age, 0.0, 50.0 * decay) / decay)
    value[~np.isfinite(age)] = 0.0
    return np.rint(value * 255.0).astype(np.uint8)


def _nearest_events_to_keypoints(
    keypoints: list[cv2.KeyPoint], x: np.ndarray, y: np.ndarray,
    t: np.ndarray, p: np.ndarray, limit: int,
) -> np.ndarray:
    if not keypoints or not x.size:
        return np.empty((0, 4), dtype=np.float64)
    points = np.column_stack((x, y)).astype(np.float64, copy=False)
    tree = cKDTree(points)
    query = np.asarray([item.pt for item in keypoints], dtype=np.float64)
    distance, indices = tree.query(query, k=1)
    valid = distance <= 3.0
    if not np.any(valid):
        return np.empty((0, 4), dtype=np.float64)
    indices = np.unique(indices[valid])
    if indices.size > limit:
        strengths = np.asarray([item.response for item in keypoints], dtype=np.float64)[valid]
        # Candidate/keypoint indices are no longer one-to-one after unique(),
        # so keep the most recent matched events as a deterministic bound.
        del strengths
        indices = indices[np.argsort(t[indices])[-limit:]]
    return np.column_stack((t[indices], x[indices], y[indices], p[indices])).astype(
        np.float64, copy=False,
    )


def _density_filter(candidates: np.ndarray, radius: float) -> np.ndarray:
    """Retain same-polarity corner clusters at or above mean density."""

    if candidates.shape[0] < 2:
        return candidates
    scores = np.zeros(candidates.shape[0], dtype=np.float64)
    for polarity in (0.0, 1.0):
        indices = np.flatnonzero(candidates[:, 3] == polarity)
        if not indices.size:
            continue
        tree = cKDTree(candidates[indices, 1:3])
        scores[indices] = np.fromiter(
            (len(values) / radius for values in tree.query_ball_point(
                candidates[indices, 1:3], radius,
            )),
            dtype=np.float64,
            count=indices.size,
        )
    threshold = float(scores.mean())
    return candidates[scores >= threshold]


def _cluster_candidates(candidates: np.ndarray, config: FrontendConfig) -> list[Observation]:
    """Create one robust centroid observation for each HDBSCAN tracklet."""

    if candidates.shape[0] < config.cluster_min_samples:
        return []
    scaled = np.column_stack((
        candidates[:, 1] / config.density_radius_px,
        candidates[:, 2] / config.density_radius_px,
        (candidates[:, 0] - candidates[:, 0].min()) / config.track_max_gap_seconds,
    ))
    labels = HDBSCAN(
        min_cluster_size=config.cluster_min_samples,
        min_samples=max(1, config.cluster_min_samples - 1),
        cluster_selection_epsilon=1.0,
        allow_single_cluster=True,
    ).fit_predict(scaled)
    observations: list[Observation] = []
    for label in sorted(set(labels) - {-1}):
        values = candidates[labels == label]
        observations.append(Observation(
            t=float(np.median(values[:, 0])),
            x=float(np.median(values[:, 1])),
            y=float(np.median(values[:, 2])),
            polarity=int(np.rint(np.median(values[:, 3]))),
        ))
    return observations


class EventFrontend:
    """Paper-structured front-end with bounded global track storage."""

    def __init__(self, width: int, height: int, config: FrontendConfig):
        self.width = int(width)
        self.height = int(height)
        self.config = config
        self.detector = cv2.FastFeatureDetector_create(
            threshold=config.fast_threshold,
            nonmaxSuppression=True,
            type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16,
        )

    def _associate(self, tracks: list[FeatureTrack], values: list[Observation],
                   next_track_id: int) -> int:
        available = set(range(len(values)))
        active = [
            track for track in tracks
            if values and 0.0 <= values[0].t - track.latest.t <= self.config.track_max_gap_seconds
        ]
        pairs: list[tuple[float, FeatureTrack, int]] = []
        for track in active:
            for index, value in enumerate(values):
                if value.polarity != track.latest.polarity:
                    continue
                distance = math.hypot(value.x - track.latest.x, value.y - track.latest.y)
                if distance <= self.config.track_merge_radius_px:
                    pairs.append((distance, track, index))
        used_tracks: set[int] = set()
        for _, track, index in sorted(pairs, key=lambda item: item[0]):
            if track.track_id in used_tracks or index not in available:
                continue
            track.append(values[index], self.config.retain_observations_per_track)
            used_tracks.add(track.track_id)
            available.remove(index)
        for index in sorted(available):
            tracks.append(FeatureTrack(next_track_id, [values[index]]))
            next_track_id += 1
        return next_track_id

    def process(self, sequence: EventSequence) -> list[FeatureTrack]:
        """Process chunks causally; no label or bounding box enters the front-end."""

        surface = np.full((2, self.height, self.width), -np.inf, dtype=np.float64)
        tracks: list[FeatureTrack] = []
        next_track_id = 0
        size = self.config.chunk_events
        for start in range(0, sequence.event_count, size):
            stop = min(sequence.event_count, start + size)
            x, y = sequence.x[start:stop], sequence.y[start:stop]
            t, p = sequence.t[start:stop], sequence.p[start:stop]
            if not t.size:
                continue
            _assign_latest(surface, x, y, t, p)
            image = _time_surface_image(
                surface, float(t[-1]), self.config.surface_decay_seconds,
            )
            keypoints = self.detector.detect(image)
            if len(keypoints) > self.config.max_corners_per_chunk:
                keypoints = sorted(keypoints, key=lambda item: item.response, reverse=True)[
                    :self.config.max_corners_per_chunk
                ]
            candidates = _nearest_events_to_keypoints(
                keypoints, x, y, t, p, self.config.max_corners_per_chunk,
            )
            candidates = _density_filter(candidates, self.config.density_radius_px)
            observations = _cluster_candidates(candidates, self.config)
            if observations:
                next_track_id = self._associate(tracks, observations, next_track_id)
            if len(tracks) > self.config.max_tracks:
                tracks.sort(key=lambda track: (
                    track.latest.t, len(track.observations), -track.track_id,
                ), reverse=True)
                del tracks[self.config.max_tracks:]
        tracks.sort(key=lambda track: track.track_id)
        return tracks


def track_observation_array(tracks: list[FeatureTrack]) -> np.ndarray:
    """Flatten tracks to t/x/y/p rows for loop-closure alignment."""

    values = [
        (value.t, value.x, value.y, value.polarity)
        for track in tracks for value in track.observations
    ]
    if not values:
        return np.empty((0, 4), dtype=np.float64)
    return np.asarray(sorted(values), dtype=np.float64)

