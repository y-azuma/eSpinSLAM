"""Asynchronous event-based FAST corner detector (eFAST, Mueggler et al., BMVC 2017).

The detector keeps one Surface of Active Events (SAE) per polarity. For every
incoming event the SAE is updated and the event is classified as a corner when
the most recent timestamps on the radius-3 circle (16 pixels) form a
contiguous arc of 3-6 pixels *and* those on the radius-4 circle (20 pixels)
form a contiguous arc of 4-8 pixels; every timestamp outside the arc must be
strictly older than every timestamp inside it. This follows the reference
implementation (rpg_corner_events, ``FastDetector::isFeature``).
"""

from __future__ import annotations

import numpy as np

try:  # numba is a hard dependency, but keep a clear error if it is missing.
    from numba import njit
except ImportError as error:  # pragma: no cover
    raise ImportError("eSpinSLAM's eFAST detector requires numba") from error

CIRCLE3 = np.asarray((
    (0, 3), (1, 3), (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3),
    (0, -3), (-1, -3), (-2, -2), (-3, -1), (-3, 0), (-3, 1), (-2, 2), (-1, 3),
), dtype=np.int64)
CIRCLE4 = np.asarray((
    (0, 4), (1, 4), (2, 3), (3, 2), (4, 1), (4, 0), (4, -1), (3, -2), (2, -3), (1, -4),
    (0, -4), (-1, -4), (-2, -3), (-3, -2), (-4, -1), (-4, 0), (-4, 1), (-3, 2), (-2, 3), (-1, 4),
), dtype=np.int64)


@njit(cache=True)
def _has_streak(surface, x, y, circle, minimum, maximum):
    size = circle.shape[0]
    values = np.empty(size, dtype=np.float64)
    for index in range(size):
        values[index] = surface[y + circle[index, 1], x + circle[index, 0]]
    for start in range(size):
        for length in range(minimum, maximum + 1):
            if values[start] < values[(start - 1 + size) % size]:
                continue
            if values[(start + length - 1) % size] < values[(start + length) % size]:
                continue
            newest_minimum = values[start]
            for offset in range(1, length):
                value = values[(start + offset) % size]
                if value < newest_minimum:
                    newest_minimum = value
            valid = True
            for offset in range(length, size):
                if values[(start + offset) % size] >= newest_minimum:
                    valid = False
                    break
            if valid:
                return True
    return False


@njit(cache=True)
def _detect(sae, x, y, t, p, circle3, circle4, output):
    height = sae.shape[1]
    width = sae.shape[2]
    for index in range(x.shape[0]):
        polarity = p[index]
        xi = x[index]
        yi = y[index]
        sae[polarity, yi, xi] = t[index]
        if xi < 4 or xi >= width - 4 or yi < 4 or yi >= height - 4:
            output[index] = False
            continue
        surface = sae[polarity]
        output[index] = (
            _has_streak(surface, xi, yi, circle3, 3, 6)
            and _has_streak(surface, xi, yi, circle4, 4, 8)
        )


class EFastDetector:
    """Stateful eFAST detector operating on arbitrary event chunks."""

    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        self.sae = np.full((2, self.height, self.width), -np.inf, dtype=np.float64)

    def detect(self, x: np.ndarray, y: np.ndarray, t: np.ndarray,
               p: np.ndarray) -> np.ndarray:
        """Update the SAE with the chunk and return a boolean corner mask."""

        output = np.zeros(x.shape[0], dtype=np.bool_)
        if x.shape[0]:
            _detect(
                self.sae,
                np.ascontiguousarray(x, dtype=np.int64),
                np.ascontiguousarray(y, dtype=np.int64),
                np.ascontiguousarray(t, dtype=np.float64),
                np.ascontiguousarray(p, dtype=np.int64),
                CIRCLE3, CIRCLE4, output,
            )
        return output
