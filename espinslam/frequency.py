"""Loop-closure frequency estimation via temporally shifted spatial alignment."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree

from .config import FrequencyConfig


@dataclass(frozen=True)
class FrequencyEstimate:
    """Frequency estimate and diagnostics independent of ground truth."""

    frequency_hz: float
    period_seconds: float
    confidence: float
    alignment_score: float
    candidates_hz: np.ndarray
    candidate_scores: np.ndarray
    valid: bool
    reason: str = ""

    @property
    def omega_rad_s(self) -> float:
        return 2.0 * math.pi * self.frequency_hz


def _alignment_score(left: np.ndarray, right_tree: cKDTree,
                     period: float, spatial_scale: float,
                     temporal_scale: float) -> float:
    """Paper Eq. 9 analogue using polarity-independent event alignment."""

    if not left.size:
        return math.inf
    query = np.column_stack((
        left[:, 1] / spatial_scale,
        left[:, 2] / spatial_scale,
        (left[:, 0] + period) / temporal_scale,
    ))
    distance, _ = right_tree.query(query, k=1)
    return float(np.mean(np.minimum(distance, 4.0) ** 2))


class LoopClosureFrequencyEstimator:
    """Estimate the revolution period from recurrent event-feature activity."""

    def __init__(self, config: FrequencyConfig):
        self.config = config

    def estimate(self, observations: np.ndarray) -> FrequencyEstimate:
        """Score the paper's bounded period bank without consulting labels."""

        if observations.ndim != 2 or observations.shape[1:] != (4,):
            raise ValueError("observations must have shape [N,4] as t/x/y/p")
        if observations.shape[0] < 20:
            return self._invalid("fewer than 20 feature observations")
        observations = observations[np.argsort(observations[:, 0])]
        duration = float(observations[-1, 0] - observations[0, 0])
        if duration <= 0:
            return self._invalid("zero observation duration")
        midpoint = 0.5 * (float(observations[0, 0]) + float(observations[-1, 0]))
        left = observations[observations[:, 0] <= midpoint]
        right = observations[observations[:, 0] > midpoint]
        if left.shape[0] < 10 or right.shape[0] < 10:
            return self._invalid("insufficient observations on either side of track midlife")
        right_tree = cKDTree(np.column_stack((
            right[:, 1] / self.config.spatial_tolerance_px,
            right[:, 2] / self.config.spatial_tolerance_px,
            right[:, 0] / self.config.temporal_tolerance_seconds,
        )))
        frequencies = np.linspace(
            self.config.min_hz, self.config.max_hz, self.config.candidate_count,
            dtype=np.float64,
        )
        periods = 1.0 / frequencies
        scores = np.full(frequencies.shape, np.inf, dtype=np.float64)
        for index, period in enumerate(periods):
            eligible = left[
                (left[:, 0] + period >= midpoint)
                & (left[:, 0] + period <= observations[-1, 0])
            ]
            if eligible.shape[0] < 10:
                continue
            scores[index] = _alignment_score(
                eligible, right_tree, float(period),
                self.config.spatial_tolerance_px,
                self.config.temporal_tolerance_seconds,
            )
        valid = np.isfinite(scores)
        if not np.any(valid):
            return self._invalid("observation is shorter than every candidate period")

        finite_indices = np.flatnonzero(valid)
        minimum_score = float(np.min(scores[valid]))
        median_score = float(np.median(scores[valid]))
        tie_tolerance = max(
            1e-12,
            self.config.equivalent_score_fraction
            * max(median_score - minimum_score, 0.0),
        )
        equivalent = finite_indices[scores[valid] <= minimum_score + tie_tolerance]
        # Prefer the first recurrence among numerically equivalent candidates.
        # A non-zero equivalence band is optional because a broad band can turn
        # a merely similar high-frequency harmonic into the selected period.
        best_index = int(equivalent[np.argmax(frequencies[equivalent])])
        # Quadratic refinement reduces bank discretization without a new label
        # dependency. The bank remains the bounded initialization mechanism.
        best_frequency = float(frequencies[best_index])
        if 0 < best_index < frequencies.size - 1 and np.isfinite(
                scores[best_index - 1:best_index + 2]).all():
            x = frequencies[best_index - 1:best_index + 2]
            y = scores[best_index - 1:best_index + 2]
            coefficients = np.polyfit(x, y, 2)
            if coefficients[0] > 0:
                vertex = float(-coefficients[1] / (2.0 * coefficients[0]))
                if x[0] <= vertex <= x[-1]:
                    best_frequency = vertex
        best_period = 1.0 / best_frequency
        best_score = float(scores[best_index])
        confidence = max(0.0, min(1.0, (median_score - best_score) / (abs(median_score) + 1e-9)))
        return FrequencyEstimate(
            best_frequency, best_period, confidence, best_score,
            frequencies, scores, True,
        )

    def _invalid(self, reason: str) -> FrequencyEstimate:
        empty = np.empty(0, dtype=np.float64)
        return FrequencyEstimate(
            math.nan, math.nan, 0.0, math.inf, empty, empty, False, reason,
        )
