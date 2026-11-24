"""Online, chunk-driven eSpinSLAM pipeline (paper Fig. 3).

The camera driver pushes event chunks to the front-end, which updates the
global track pool ``H``. Whenever ``H`` changes the frequency estimator runs
(rate-limited by ``frequency.update_interval_seconds``). After a loop has been
detected, stale tracks are forgotten (Sec. 4.2.7) and the back-end optimises
the structure and camera orientation every ``epsilon / f`` seconds of event
time (Sec. 4.2.2). No ground-truth label is read during inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import time


from .backend import (
    IncrementalBackend, OptimizationReport, Reconstruction, UndistortionMap,
    empty_reconstruction,
)
from .config import CameraConfig, ESpinSLAMConfig, FrequencyConfig
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
    period_seconds: float = math.nan
    alignment_cost: float = math.nan
    grid_seed_seconds: float = math.nan
    grid_seed_cost: float = math.nan
    icp_iterations_used: int = 0
    icp_termination: str = "not_run"
    icp_clipped_to_search_interval: bool = False
    # Ground-truth-free diagnostics for the predeclared D-003 viability audit.
    # They do not alter Eq. (9), the stability test, or loop closure itself.
    observation_span_seconds: float = math.nan
    period_over_span: float = math.nan
    at_search_boundary: bool = False
    near_half_span: bool = False
    loop_artefact_suspected: bool = False
    loop_artefact_reason: str = ""


def _frequency_record_diagnostics(
    estimate: FrequencyEstimate,
    observation_span_seconds: float,
    config: FrequencyConfig,
) -> tuple[float, bool, bool]:
    """Return ``T*/span`` and two frozen, ground-truth-free audit flags.

    The flags implement only the D-003 viability diagnostic.  In particular,
    they are not fed back into :class:`LoopClosureFrequencyEstimator`.
    """

    period = float(estimate.period_seconds)
    span = float(observation_span_seconds)
    if not estimate.valid or not (math.isfinite(period) and math.isfinite(span) and span > 0):
        return math.nan, False, False
    ratio = period / span
    maximum = (span if config.max_period_seconds is None
               else min(float(config.max_period_seconds), span))
    tolerance = float(config.icp_tolerance_seconds)
    at_boundary = (
        math.isclose(period, float(config.min_period_seconds), rel_tol=0.0,
                     abs_tol=tolerance)
        or math.isclose(period, maximum, rel_tol=0.0, abs_tol=tolerance)
    )
    half_span = 0.5 * span
    # "One coarse-grid step" is multiplicative because D-004 froze a
    # geometric grid with nominal ratio 1 + coarse_relative_step.
    multiplicative_distance = max(period / half_span, half_span / period)
    near_half_span = (
        multiplicative_distance <= 1.0 + float(config.coarse_relative_step) + 1e-12
    )
    return ratio, at_boundary, near_half_span


def _loop_artefact_diagnostic(
    records: list[FrequencyRecord], loop_history: int,
) -> tuple[bool, str]:
    """Classify the *confirmed* window without changing loop detection.

    D-003 excludes a loop from the two-of-three viability count when every
    estimate in the confirmation window is either at a search boundary, or
    every estimate follows the known literal-Eq.-(9) ``span / 2`` attractor.
    """

    # The paper estimator retains only valid estimates in its stability deque;
    # invalid updates neither append nor clear it.  Mirror that exact window
    # here rather than accidentally counting diagnostic records for invalid
    # updates.
    valid_records = [
        record for record in records
        if math.isfinite(record.frequency_hz) and math.isfinite(record.period_seconds)
    ]
    window = valid_records[-loop_history:]
    if len(window) != loop_history:
        return False, ""
    if all(record.at_search_boundary for record in window):
        return True, "all_confirming_periods_at_search_boundary"
    if all(record.near_half_span for record in window):
        return True, "all_confirming_periods_within_one_grid_step_of_half_span"
    return False, ""


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
    # Wall-time breakdown (front-end = the paper's ETC tracker, Fig. 5).
    frontend_seconds: float = math.nan
    frequency_seconds: float = math.nan
    backend_seconds: float = math.nan
    # Raw loop closure remains available above.  This separate flag is used
    # only by the frozen D-003 dev-scene viability audit.
    loop_artefact_suspected: bool = False
    loop_artefact_reason: str = ""

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
        """Events processed per second of wall time by the whole system (Fig. 7)."""

        return self.event_count / self.runtime_seconds if self.runtime_seconds > 0 else math.nan

    @property
    def frontend_throughput_events_per_second(self) -> float:
        """Events per second of front-end (feature tracking) time (Fig. 5, "ETC")."""

        return (self.event_count / self.frontend_seconds
                if self.frontend_seconds > 0 else math.nan)

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
        timers = {"frontend": 0.0, "frequency": 0.0, "backend": 0.0}
        for start in range(0, sequence.event_count, chunk):
            stop = min(start + chunk, sequence.event_count)
            tick = time.perf_counter()
            frontend.push(sequence.t[start:stop], sequence.x[start:stop],
                          sequence.y[start:stop], sequence.p[start:stop])
            timers["frontend"] += time.perf_counter() - tick
            now = float(sequence.t[stop - 1])
            due_estimate = pool.version != estimated_version and now >= next_estimate
            loop_was_closed = estimator.loop_closed
            due_existing_optimization = (
                reconstruct_structure and loop_was_closed and now >= next_optimization
            )
            if loop_was_closed and (due_estimate or due_existing_optimization):
                # Once f* exists, Eq. (12) precedes every subsequent use of H.
                # In particular, frequency re-estimation must not see stale
                # tracks that the paper says have already been forgotten.
                pool.forget(
                    pool.newest_time - config.backend.forget_revolutions
                    / estimator.current.frequency_hz
                )
            if due_estimate:
                tick = time.perf_counter()
                pool_events = pool.events()
                estimate = estimator.update(pool_events)
                timers["frequency"] += time.perf_counter() - tick
                estimated_version = pool.version
                next_estimate = now + config.frequency.update_interval_seconds
                observation_span = (
                    float(pool_events[:, 0].max() - pool_events[:, 0].min())
                    if pool_events.shape[0] else math.nan
                )
                period_over_span, at_boundary, near_half_span = (
                    _frequency_record_diagnostics(
                        estimate, observation_span, config.frequency,
                    )
                )
                record = FrequencyRecord(
                    now - origin, time.perf_counter() - started,
                    estimate.frequency_hz if estimate.valid else math.nan,
                    estimator.loop_closed,
                    estimate.period_seconds,
                    estimate.alignment_cost,
                    estimate.grid_seed_seconds,
                    estimate.grid_seed_cost,
                    estimate.icp_iterations_used,
                    estimate.icp_termination,
                    estimate.icp_clipped_to_search_interval,
                    observation_span,
                    period_over_span,
                    at_boundary,
                    near_half_span,
                )
                if (config.frequency.method == "loop_closure"
                        and config.frequency.alignment_mode == "paper"
                        and estimator.loop_closed and not loop_was_closed):
                    suspected, reason = _loop_artefact_diagnostic(
                        [*history, record], config.frequency.loop_history,
                    )
                    record = replace(
                        record,
                        loop_artefact_suspected=suspected,
                        loop_artefact_reason=reason,
                    )
                history.append(record)
                if estimator.loop_closed and not math.isfinite(loop_time):
                    loop_time = now - origin
            if not estimator.loop_closed:
                continue
            frequency = estimator.current.frequency_hz
            due_optimization = reconstruct_structure and now >= next_optimization
            if not loop_was_closed and (due_estimate or due_optimization):
                # On the transition itself f* did not exist before estimation,
                # so forgetting happens immediately after f* is established
                # and before the first bundle adjustment.
                pool.forget(pool.newest_time - config.backend.forget_revolutions
                            / frequency)
            if due_optimization:
                tick = time.perf_counter()
                backend.optimize(pool, frequency, now - origin)
                timers["backend"] += time.perf_counter() - tick
                next_optimization = now + config.backend.optimization_cadence_revolutions / frequency
        if (reconstruct_structure and estimator.loop_closed
                and config.backend.final_offline_refinement):
            # Not described by the paper. It remains available solely for
            # retrospective replay of the earlier exploratory implementation.
            tick = time.perf_counter()
            backend.optimize(pool, estimator.current.frequency_hz, now - origin)
            timers["backend"] += time.perf_counter() - tick
        reconstruction = backend.reconstruction() if reconstruct_structure else (
            empty_reconstruction("structure disabled"))
        if reconstruct_structure and not estimator.loop_closed:
            reconstruction = empty_reconstruction("loop closure was not detected")
        return ESpinSLAMResult(
            sequence.sequence_id, sequence.duration_seconds, sequence.event_count,
            estimator.current, estimator.loop_closed, loop_time, reconstruction,
            history, list(backend.reports), len(pool), time.perf_counter() - started,
            timers["frontend"], timers["frequency"], timers["backend"],
            any(record.loop_artefact_suspected for record in history),
            next((record.loop_artefact_reason for record in history
                  if record.loop_artefact_suspected), ""),
        )
