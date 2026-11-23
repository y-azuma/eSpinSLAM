"""Frequency estimation and loop-closure detection (paper Sec. 4.2.1).

The cumulative events ``A`` of the global track pool (Eq. 7) are split at the
pool's "midlife" ``tau`` (Eq. 8) into ``A_l`` (t <= tau) and ``A_r`` (t > tau).
The revolution period is the temporal shift that best aligns ``A_l`` with
``A_r`` (Eq. 9)::

    t_rev* = argmin_s  sum_{m in A_l} min_{n in A_r} || [x_m, y_m, lambda (t_m + s)]
                                                       - [x_n, y_n, lambda t_n] ||_2

(restricted to the left events whose shifted time falls inside ``A_r``; see
:class:`_Alignment`) and is solved with a 1-D variant of ICP: alternate nearest-neighbour
correspondences with the closed-form (IRLS) update of the shift. Pixels and
seconds are made commensurate by ``lambda`` (``time_scale_px_per_second``),
which the paper leaves implicit.

Until a loop has been detected the shift is initialised by a bounded coarse
search (local minima within ``harmonic_tolerance`` of the best are treated as
equivalent and the shortest period, i.e. the fundamental, is chosen); after
detection ICP is warm-started from the current estimate, which is how a local
1-D ICP tracks the period online. A loop is detected when the last ``aleph``
estimates have a sample standard deviation below ``sigma`` (Table 1).

``method="fft"`` implements the Table 3 ablation ("FFT f estimation"): the
dominant peak of the spectrum of the track-event rate, as in eSfO.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np

from .config import FrequencyConfig
from .nn import GridIndex


@dataclass(frozen=True)
class FrequencyEstimate:
    """One frequency estimate with diagnostics independent of ground truth."""

    frequency_hz: float
    period_seconds: float
    alignment_cost: float
    valid: bool
    reason: str = ""
    candidates_seconds: np.ndarray = field(default_factory=lambda: np.empty(0))
    candidate_costs: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def omega_rad_s(self) -> float:
        return 2.0 * math.pi * self.frequency_hz if self.valid else math.nan


def _invalid(reason: str) -> FrequencyEstimate:
    return FrequencyEstimate(math.nan, math.nan, math.inf, False, reason)


def _uniform_subsample(values: np.ndarray, limit: int) -> np.ndarray:
    if values.shape[0] <= limit:
        return values
    indices = np.linspace(0, values.shape[0] - 1, limit).round().astype(np.int64)
    return values[indices]


class _Alignment:
    """Eq. (9) cost and 1-D ICP for a fixed split of ``A``.

    Only left events whose shifted timestamp ``t_m + s`` falls inside the time
    range of ``A_r`` take part: the others cannot be aligned by construction,
    and letting them snap to the boundary of ``A_r`` biases ``s`` towards the
    midlife-to-end span (observed as a slow drift of ``f`` once the pool spans
    more than two revolutions).
    """

    def __init__(self, left: np.ndarray, right: np.ndarray, scale: float,
                 minimum_overlap: int):
        self.left = left
        self.left_t = np.ascontiguousarray(left[:, 0])
        self.left_x = np.ascontiguousarray(left[:, 1])
        self.left_y = np.ascontiguousarray(left[:, 2])
        self.scale = scale
        self.right_t = right[:, 0]
        self.start = float(right[0, 0])
        self.stop = float(right[-1, 0])
        self.minimum_overlap = minimum_overlap
        # Exact nearest neighbours in (x, y, scale * t); distances are identical
        # to scipy's cKDTree (see espinslam.nn), only computed faster.
        self.grid = GridIndex(np.column_stack((right[:, 1], right[:, 2], scale * right[:, 0])))

    def _overlap(self, shift: float) -> np.ndarray:
        shifted = self.left_t + shift
        return (shifted >= self.start) & (shifted <= self.stop)

    def _query(self, mask: np.ndarray, shift: float, hint: np.ndarray | None = None):
        times = self.left_t[mask]
        distance, index = self.grid.query_points(
            self.left_x[mask], self.left_y[mask], self.scale * (times + shift),
            None if hint is None else hint[mask],
        )
        return times, distance, index

    def cost(self, shift: float, hint: np.ndarray | None = None) -> float:
        mask = self._overlap(shift)
        if np.count_nonzero(mask) < self.minimum_overlap:
            return math.inf
        _, distance, _ = self._query(mask, shift, hint)
        return float(np.mean(distance))

    def costs(self, shifts: np.ndarray, left: np.ndarray) -> np.ndarray:
        """``cost(shift, left)`` for many shifts in one parallel pass.

        Uses the same overlap test, query coordinates and per-shift means as
        :meth:`cost`, with exact nearest distances, so the result is identical.
        """

        minimum = max(10, self.minimum_overlap * left.shape[0] // max(self.left.shape[0], 1))
        distances = self.grid.query_shifts(left[:, 0], left[:, 1], left[:, 2], shifts,
                                           self.scale, self.start, self.stop)
        result = np.full(len(shifts), math.inf)
        for index in range(len(shifts)):
            row = distances[index]
            overlap = row[~np.isnan(row)]
            if overlap.shape[0] >= minimum:
                result[index] = float(np.mean(overlap))
        return result

    def icp(self, shift: float, iterations: int, tolerance: float) -> tuple[float, float]:
        # Each point's neighbour from the previous iteration warm-starts its
        # search (an exact upper bound), which leaves the result unchanged.
        hint = np.full(self.left_t.shape[0], -1, dtype=np.int64)
        for _ in range(iterations):
            mask = self._overlap(shift)
            if np.count_nonzero(mask) < self.minimum_overlap:
                return shift, math.inf
            times, distance, index = self._query(mask, shift, hint)
            hint[mask] = index
            # Minimise sum ||.||_2 for fixed correspondences by IRLS (weights 1/d).
            weights = 1.0 / np.maximum(distance, 1e-6)
            target = self.right_t[index] - times
            updated = float(np.sum(weights * target) / np.sum(weights))
            converged = abs(updated - shift) < tolerance
            shift = updated
            if converged:
                break
        return shift, self.cost(shift, hint)


class LoopClosureFrequencyEstimator:
    """Online period estimator with loop-closure detection."""

    def __init__(self, config: FrequencyConfig):
        self.config = config
        self.history: deque[float] = deque(maxlen=config.loop_history)
        self.loop_closed = False
        self.current: FrequencyEstimate = _invalid("no estimate yet")
        self.updates = 0

    def update(self, events: np.ndarray) -> FrequencyEstimate:
        """Re-estimate ``f`` from ``A`` ([N,3] t/x/y rows) and test for a loop."""

        self.updates += 1
        if self.config.method == "fft":
            estimate = self._fft(events)
        else:
            estimate = self._loop_closure(events)
        if estimate.valid:
            self.current = estimate
            self.history.append(estimate.frequency_hz)
            if self.loop_closed:
                pass
            elif self.config.method == "fft":
                # FFT bins are 1/span wide, so the sigma test of Sec. 4.2.1 can
                # never pass; the ablation initialises once two estimated
                # revolutions are observed (when loop closure typically fires).
                span = float(events[:, 0].max() - events[:, 0].min())
                self.loop_closed = span >= 2.0 * estimate.period_seconds
            elif (len(self.history) == self.history.maxlen
                    and float(np.std(np.asarray(self.history), ddof=1))
                    < self.config.loop_std_threshold_hz):
                self.loop_closed = True
        elif not self.loop_closed:
            # Detection needs aleph *consecutive* consistent estimates.
            self.history.clear()
        return estimate

    # -- Eq. (9) ---------------------------------------------------------------
    def _loop_closure(self, events: np.ndarray) -> FrequencyEstimate:
        config = self.config
        if events.ndim != 2 or events.shape[1] < 3:
            raise ValueError("events must have shape [N,3] as t/x/y")
        if events.shape[0] < 2 * config.min_samples_per_half:
            return _invalid("too few track events")
        events = events[np.argsort(events[:, 0], kind="stable")]
        events = _uniform_subsample(events, config.max_alignment_samples)
        first, last = float(events[0, 0]), float(events[-1, 0])
        span = last - first
        midlife = 0.5 * (first + last)
        left = events[events[:, 0] <= midlife]
        right = events[events[:, 0] > midlife]
        if left.shape[0] < config.min_samples_per_half or right.shape[0] < config.min_samples_per_half:
            return _invalid("insufficient events on either side of the midlife")
        maximum = min(config.max_period_seconds, span)
        if maximum <= config.min_period_seconds:
            return _invalid("observation span is shorter than the minimum period")
        alignment = _Alignment(left, right, config.time_scale_px_per_second,
                               config.min_samples_per_half)

        if self.loop_closed and self.current.valid and self.current.period_seconds <= span:
            shift, cost = alignment.icp(
                self.current.period_seconds, config.icp_iterations,
                config.icp_tolerance_seconds,
            )
            if config.min_period_seconds <= shift <= maximum:
                return FrequencyEstimate(1.0 / shift, shift, cost, True)

        # Bounded coarse initialisation followed by ICP refinement.
        count = max(3, int(math.ceil(
            math.log(maximum / config.min_period_seconds) / math.log1p(config.coarse_relative_step),
        )) + 1)
        candidates = np.geomspace(config.min_period_seconds, maximum, count)
        coarse_left = _uniform_subsample(left, config.coarse_alignment_samples)
        costs = alignment.costs(candidates, coarse_left)
        interior = np.flatnonzero(
            (costs[1:-1] <= costs[:-2]) & (costs[1:-1] <= costs[2:]),
        ) + 1
        seeds = list(interior) or [int(np.argmin(costs))]
        # Refine every coarse minimum that could be equivalent to the best one
        # (the fundamental and its multiples), plus the best few overall.
        finite = costs[np.isfinite(costs)]
        threshold = (float(finite.min()) * (1.0 + 2.0 * config.harmonic_tolerance)
                     if finite.size else math.inf)
        ranked = sorted(seeds, key=lambda index: costs[index])
        seeds = sorted(set(ranked[:3]) | {index for index in ranked[:16] if costs[index] <= threshold})
        refined: list[tuple[float, float]] = []
        for index in seeds:
            shift, cost = alignment.icp(
                float(candidates[index]), config.icp_iterations,
                config.icp_tolerance_seconds,
            )
            if config.min_period_seconds <= shift <= maximum and math.isfinite(cost):
                refined.append((shift, cost))
        if not refined:
            return _invalid("no admissible alignment minimum")
        best_cost = min(cost for _, cost in refined)
        equivalent = [shift for shift, cost in refined
                      if cost <= best_cost * (1.0 + config.harmonic_tolerance)]
        shift = min(equivalent)
        cost = min(cost for value, cost in refined if value == shift)
        return FrequencyEstimate(1.0 / shift, shift, cost, True, "",
                                 candidates, costs)

    # -- Table 3 ablation --------------------------------------------------------
    def _fft(self, events: np.ndarray) -> FrequencyEstimate:
        config = self.config
        if events.shape[0] < 2 * config.min_samples_per_half:
            return _invalid("too few track events")
        times = np.sort(events[:, 0])
        span = float(times[-1] - times[0])
        if span <= config.min_period_seconds:
            return _invalid("observation span is shorter than the minimum period")
        bins = max(8, int(math.ceil(span / config.fft_bin_seconds)))
        counts, _ = np.histogram(times, bins=bins)
        signal = counts - counts.mean()
        spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
        frequencies = np.fft.rfftfreq(signal.size, d=span / bins)
        band = (frequencies >= 1.0 / min(config.max_period_seconds, span)) & (
            frequencies <= 1.0 / config.min_period_seconds)
        if not np.any(band):
            return _invalid("no FFT bin inside the admissible band")
        indices = np.flatnonzero(band)
        peak = int(indices[np.argmax(spectrum[indices])])
        frequency = float(frequencies[peak])
        if 0 < peak < spectrum.size - 1:
            a, b, c = np.log(spectrum[peak - 1:peak + 2] + 1e-12)
            denominator = a - 2.0 * b + c
            if abs(denominator) > 1e-12:
                offset = 0.5 * (a - c) / denominator
                frequency += float(np.clip(offset, -0.5, 0.5)) * (frequencies[1] - frequencies[0])
        if frequency <= 0:
            return _invalid("non-positive FFT peak")
        return FrequencyEstimate(frequency, 1.0 / frequency, float(-spectrum[peak]), True)
