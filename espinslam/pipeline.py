"""Online, chunk-driven eSpinSLAM pipeline (paper Fig. 3).

The camera driver pushes event chunks to the front-end, which updates the
global track pool ``H``. Whenever ``H`` changes the frequency estimator runs
(rate-limited by ``frequency.update_interval_seconds``). After a loop has been
detected, stale tracks are forgotten (Sec. 4.2.7) and the back-end optimises
the structure and camera orientation every ``epsilon / f`` seconds of event
time (Sec. 4.2.2). No ground-truth label is read during inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

import numpy as np

from .backend import (
    IncrementalBackend, OptimizationReport, Reconstruction, UndistortionMap,
    empty_reconstruction,
)
from .config import CameraConfig, ESpinSLAMConfig
from .data import EventSequence
from .frequency import FrequencyEstimate, LoopClosureFrequencyEstimator
from .frontend import EventFrontend


@dataclass(frozen=True)
class FrequencyRecord:
    """Frequency state after one estimator update (times relative to start)."""

    event_time: float
    wall_seconds: float
    frequency_hz: float
    loop_closed: bool


@dataclass(frozen=True)
class ESpinSLAMResult:
    """Motion, structure, and online diagnostics of one sequence."""

    sequence_id: str
    observation_seconds: float
    event_count: int
    frequency: FrequencyEstimate
    loop_closed: bool
    loop_closure_time: float
    reconstruction: Reconstruction
    frequency_history: list[FrequencyRecord] = field(default_factory=list)
    optimizations: list[OptimizationReport] = field(default_factory=list)
    track_count: int = 0
    runtime_seconds: float = math.nan

    @property
    def frequency_hz(self) -> float:
        """System output ``f*`` (only available after loop closure)."""

        if not self.loop_closed or not self.frequency.valid:
            return math.nan
        if self.reconstruction.valid and math.isfinite(self.reconstruction.frequency_hz):
            return self.reconstruction.frequency_hz  # differs only if f is optimised
        return self.frequency.frequency_hz

    @property
    def omega_rad_s(self) -> float:
        return 2.0 * math.pi * self.frequency_hz

    @property
    def valid(self) -> bool:
        return math.isfinite(self.frequency_hz)

    @property
    def throughput_events_per_second(self) -> float:
        return self.event_count / self.runtime_seconds if self.runtime_seconds > 0 else math.nan

    def frequency_at(self, seconds: float) -> tuple[float, float]:
        """(confirmed f*, latest unconfirmed estimate) available at ``seconds``."""

        confirmed = latest = math.nan
        for record in self.frequency_history:
            if record.event_time > seconds:
                break
            if math.isfinite(record.frequency_hz):
                latest = record.frequency_hz
                confirmed = record.frequency_hz if record.loop_closed else math.nan
        return confirmed, latest


class ESpinSLAM:
    """Paper-structured online eSpinSLAM."""

    def __init__(self, config: ESpinSLAMConfig):
        self.config = config

    def process(self, sequence: EventSequence, *, camera: CameraConfig | None = None,
                reconstruct_structure: bool = True) -> ESpinSLAMResult:
        config = self.config
        camera = camera or config.camera
        intrinsic = camera.matrix()
        started = time.perf_counter()
        if sequence.event_count == 0:
            return ESpinSLAMResult(
                sequence.sequence_id, 0.0, 0, LoopClosureFrequencyEstimator(config.frequency).current,
                False, math.nan, empty_reconstruction("empty sequence"),
                runtime_seconds=time.perf_counter() - started,
            )
        origin = float(sequence.t[0])
        frontend = EventFrontend(sequence.width, sequence.height, config.frontend)
        pool = frontend.pool
        estimator = LoopClosureFrequencyEstimator(config.frequency)
        undistort = (UndistortionMap(sequence.width, sequence.height, intrinsic, camera.distortion)
                     if camera.distortion else None)
        backend = IncrementalBackend(config.backend, intrinsic, time_origin=origin,
                                     undistort=undistort)
        history: list[FrequencyRecord] = []
        loop_time = math.nan
        next_estimate = -math.inf
        next_optimization = -math.inf
        estimated_version = -1
        chunk = config.frontend.chunk_events
        now = origin
        for start in range(0, sequence.event_count, chunk):
            stop = min(start + chunk, sequence.event_count)
            frontend.push(sequence.t[start:stop], sequence.x[start:stop],
                          sequence.y[start:stop], sequence.p[start:stop])
            now = float(sequence.t[stop - 1])
            due_estimate = pool.version != estimated_version and now >= next_estimate
            due_optimization = (estimator.loop_closed and reconstruct_structure
                                and now >= next_optimization)
            if estimator.loop_closed and (due_estimate or due_optimization):
                # Sec. 4.2.7: discard tracks older than sigma_forget revolutions
                # (applied whenever the back-end consumes H).
                pool.forget(pool.newest_time - config.backend.forget_revolutions
                            / estimator.current.frequency_hz)
            if due_estimate:
                estimate = estimator.update(pool.events())
                estimated_version = pool.version
                next_estimate = now + config.frequency.update_interval_seconds
                history.append(FrequencyRecord(
                    now - origin, time.perf_counter() - started,
                    estimate.frequency_hz if estimate.valid else math.nan,
                    estimator.loop_closed,
                ))
                if estimator.loop_closed and not math.isfinite(loop_time):
                    loop_time = now - origin
            if not estimator.loop_closed:
                continue
            frequency = estimator.current.frequency_hz
            if reconstruct_structure and now >= next_optimization:
                backend.optimize(pool, frequency, now - origin)
                next_optimization = now + config.backend.optimization_cadence_revolutions / frequency
        if reconstruct_structure and estimator.loop_closed:
            # Final refinement with the complete pool at the end of the stream.
            backend.optimize(pool, estimator.current.frequency_hz, now - origin)
        reconstruction = backend.reconstruction() if reconstruct_structure else (
            empty_reconstruction("structure disabled"))
        if reconstruct_structure and not estimator.loop_closed:
            reconstruction = empty_reconstruction("loop closure was not detected")
        return ESpinSLAMResult(
            sequence.sequence_id, sequence.duration_seconds, sequence.event_count,
            estimator.current, estimator.loop_closed, loop_time, reconstruction,
            history, list(backend.reports), len(pool), time.perf_counter() - started,
        )
