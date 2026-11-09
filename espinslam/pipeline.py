"""End-to-end eSpinSLAM inference without training labels."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import ESpinSLAMConfig
from .data import EventSequence
from .frequency import FrequencyEstimate, LoopClosureFrequencyEstimator
from .frontend import EventFrontend, FeatureTrack, track_observation_array
from .geometry import Reconstruction, reconstruct


@dataclass(frozen=True)
class ESpinSLAMResult:
    """Motion, structure, and causal inference diagnostics."""

    sequence_id: str
    observation_seconds: float
    event_count: int
    tracks: list[FeatureTrack]
    frequency: FrequencyEstimate
    reconstruction: Reconstruction

    @property
    def omega_rad_s(self) -> float:
        return self.frequency.omega_rad_s if self.frequency.valid else math.nan

    @property
    def valid(self) -> bool:
        return self.frequency.valid


def _sample_raw_observations(sequence: EventSequence, limit: int) -> np.ndarray:
    """Bounded event-only fallback when the Python eFAST front-end is sparse."""

    if not sequence.event_count:
        return np.empty((0, 4), dtype=np.float64)
    if sequence.event_count <= limit:
        indices = np.arange(sequence.event_count)
    else:
        indices = np.linspace(0, sequence.event_count - 1, limit).round().astype(np.int64)
    return np.column_stack((
        sequence.t[indices], sequence.x[indices], sequence.y[indices], sequence.p[indices],
    )).astype(np.float64, copy=False)


class ESpinSLAM:
    """Paper-structured independent reproduction."""

    def __init__(self, config: ESpinSLAMConfig):
        self.config = config
        self.frequency_estimator = LoopClosureFrequencyEstimator(config.frequency)

    def estimate_frequency(self, sequence: EventSequence,
                           *, tracks: list[FeatureTrack] | None = None) -> FrequencyEstimate:
        observations = track_observation_array(tracks or [])
        if observations.shape[0] < 20:
            observations = _sample_raw_observations(
                sequence, self.config.frequency.max_alignment_samples,
            )
        elif observations.shape[0] > self.config.frequency.max_alignment_samples:
            indices = np.linspace(
                0, observations.shape[0] - 1,
                self.config.frequency.max_alignment_samples,
            ).round().astype(np.int64)
            observations = observations[indices]
        return self.frequency_estimator.estimate(observations)

    def process(self, sequence: EventSequence, *, reconstruct_structure: bool = True) -> ESpinSLAMResult:
        """Infer spin and optional point cloud without reading omega_rad_s."""

        frontend = EventFrontend(sequence.width, sequence.height, self.config.frontend)
        tracks = frontend.process(sequence)
        frequency = self.estimate_frequency(sequence, tracks=tracks)
        if reconstruct_structure and frequency.valid:
            reconstruction = reconstruct(
                tracks, frequency.frequency_hz, self.config.camera.matrix(),
                self.config.backend,
            )
        else:
            reconstruction = Reconstruction(
                np.empty((0, 3)), np.empty(0, dtype=np.int64), np.zeros(3),
                math.nan, np.empty(0), False,
                "frequency unavailable" if reconstruct_structure else "structure disabled",
            )
        return ESpinSLAMResult(
            sequence.sequence_id, sequence.duration_seconds, sequence.event_count,
            tracks, frequency, reconstruction,
        )

