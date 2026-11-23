"""Exact Euclidean nearest-neighbour search for the Eq. (9) alignment.

The frequency estimator queries, thousands of times per update, the nearest
right-half event to every shifted left-half event in ``(x, y, lambda * t)``.
:class:`GridIndex` answers these queries exactly and returns the same distances
as :class:`scipy.spatial.cKDTree` bit for bit, only faster:

* points are bucketed into square ``(x, y)`` cells and sorted by ``z`` inside
  each cell; cells are visited in rings around the query and, inside a cell,
  outwards from the query's ``z`` by binary search;
* a cell or point is skipped only when a lower bound of its squared distance is
  already ``>=`` the best squared distance found. Every bound is formed with
  operations that are monotone under IEEE rounding (differences to a cell's
  actual bounding box, sums of non-negative squares), so the computed bound
  never exceeds the computed distance and pruning cannot change the result;
* a distance is computed as ``((dx*dx + dy*dy) + dz*dz)`` followed by
  ``sqrt``, the operation order of cKDTree's Minkowski p = 2 kernel, so the
  returned distances are identical. A query batch runs in parallel (numba);
* :meth:`GridIndex.query_shifts` evaluates one point at many temporal shifts
  (the coarse search of Eq. 9) and warm-starts each search with the previous
  shift's neighbour, whose distance is an exact upper bound.

With exact distance ties the returned index may differ from cKDTree's (both
are nearest neighbours). ``tests/test_nn.py`` checks the equivalence.
"""

from __future__ import annotations

import math

from numba import njit, prange
import numpy as np


@njit(cache=True, inline="always")
def _scan_cell(xq, yq, zq, start, stop, xs, ys, zs, lb2, best2, best):
    """Scan one cell (points sorted by z) outwards from the query's z."""

    low, high = start, stop
    while low < high:  # first index with zs >= zq
        middle = (low + high) // 2
        if zs[middle] < zq:
            low = middle + 1
        else:
            high = middle
    k = low
    while k < stop:
        dz = zs[k] - zq
        if lb2 + dz * dz >= best2:
            break
        dx = xs[k] - xq
        dy = ys[k] - yq
        d2 = dx * dx
        d2 = d2 + dy * dy
        d2 = d2 + dz * dz
        if d2 < best2:
            best2 = d2
            best = k
        k += 1
    k = low - 1
    while k >= start:
        dz = zs[k] - zq
        if lb2 + dz * dz >= best2:
            break
        dx = xs[k] - xq
        dy = ys[k] - yq
        d2 = dx * dx
        d2 = d2 + dy * dy
        d2 = d2 + dz * dz
        if d2 < best2:
            best2 = d2
            best = k
        k -= 1
    return best2, best


@njit(cache=True, inline="always")
def _visit(i, j, xq, yq, zq, gx, cell_start, bx0, bx1, by0, by1, xs, ys, zs, best2, best):
    c = j * gx + i
    start = cell_start[c]
    stop = cell_start[c + 1]
    if start == stop:
        return best2, best
    gap_x = max(0.0, bx0[c] - xq, xq - bx1[c])
    gap_y = max(0.0, by0[c] - yq, yq - by1[c])
    lb2 = gap_x * gap_x + gap_y * gap_y
    if lb2 >= best2:
        return best2, best
    return _scan_cell(xq, yq, zq, start, stop, xs, ys, zs, lb2, best2, best)


@njit(cache=True, inline="always")
def _nearest(xq, yq, zq, best2, best, xs, ys, zs, cell_start, bx0, bx1, by0, by1,
             gx, gy, xmin, ymin, cell):
    """Ring search; ``best2``/``best`` may carry an exact upper bound (warm start)."""

    slack = 1e-6 * cell
    iq = int(math.floor((xq - xmin) / cell))
    jq = int(math.floor((yq - ymin) / cell))
    reach = max(abs(iq), abs(iq - (gx - 1)), abs(jq), abs(jq - (gy - 1)))
    # Rings closer than this contain no grid cell (query outside the grid).
    first = max(0, -iq, iq - (gx - 1), -jq, jq - (gy - 1))
    for r in range(first, reach + 1):
        if r >= 2:
            # Every cell on ring r is at least (r - 1) cells away in x or y;
            # the slack absorbs rounding in the cell assignment.
            bound = (r - 1) * cell - slack
            if bound * bound >= best2:
                break
        for i in range(iq - r, iq + r + 1):
            if i < 0 or i >= gx:
                continue
            if i == iq - r or i == iq + r:
                for j in range(max(jq - r, 0), min(jq + r, gy - 1) + 1):
                    best2, best = _visit(i, j, xq, yq, zq, gx, cell_start, bx0, bx1,
                                         by0, by1, xs, ys, zs, best2, best)
            else:
                for j in (jq - r, jq + r):
                    if 0 <= j < gy:
                        best2, best = _visit(i, j, xq, yq, zq, gx, cell_start, bx0, bx1,
                                             by0, by1, xs, ys, zs, best2, best)
    return best2, best


@njit(cache=True, parallel=True)
def _query(qx, qy, qz, hint, xs, ys, zs, order, cell_start, bx0, bx1, by0, by1,
           gx, gy, xmin, ymin, cell, out_distance, out_index):
    """Nearest neighbours; ``hint[q] >= 0`` is a sorted-space warm-start point."""

    for q in prange(qx.shape[0]):
        best2 = np.inf
        best = -1
        start = hint[q]
        if start >= 0:
            dx = xs[start] - qx[q]
            dy = ys[start] - qy[q]
            dz = zs[start] - qz[q]
            best2 = dx * dx
            best2 = best2 + dy * dy
            best2 = best2 + dz * dz
            best = start
        best2, best = _nearest(qx[q], qy[q], qz[q], best2, best, xs, ys, zs, cell_start,
                               bx0, bx1, by0, by1, gx, gy, xmin, ymin, cell)
        out_distance[q] = math.sqrt(best2)
        out_index[q] = order[best]


@njit(cache=True, parallel=True)
def _query_shifts(times, xq_all, yq_all, shifts, scale, start, stop, xs, ys, zs,
                  cell_start, bx0, bx1, by0, by1, gx, gy, xmin, ymin, cell, out):
    """Distances for every (shift, point) pair whose shifted time lies in [start, stop].

    Shifts are processed in order for each point; the previous shift's nearest
    neighbour, re-evaluated at the new shift, is an exact upper bound.
    """

    for m in prange(times.shape[0]):
        xq = xq_all[m]
        yq = yq_all[m]
        previous = -1
        for k in range(shifts.shape[0]):
            shifted = times[m] + shifts[k]
            if shifted < start or shifted > stop:
                out[k, m] = np.nan
                continue
            zq = scale * (times[m] + shifts[k])
            best2 = np.inf
            best = -1
            if previous >= 0:
                dx = xs[previous] - xq
                dy = ys[previous] - yq
                dz = zs[previous] - zq
                best2 = dx * dx
                best2 = best2 + dy * dy
                best2 = best2 + dz * dz
                best = previous
            best2, best = _nearest(xq, yq, zq, best2, best, xs, ys, zs, cell_start,
                                   bx0, bx1, by0, by1, gx, gy, xmin, ymin, cell)
            previous = best
            out[k, m] = math.sqrt(best2)


@njit(cache=True, parallel=True)
def _query_all_shifts(times, xq_all, yq_all, shifts, scale, xs, ys, zs,
                      cell_start, bx0, bx1, by0, by1, gx, gy, xmin, ymin, cell, out):
    """Exact distances for every ``(shift, point)`` pair.

    Unlike :func:`_query_shifts`, this routine deliberately performs no
    timestamp-overlap filtering.  It is the batched form required by the
    literal support of paper Eq. (9), where every event in ``A_l`` contributes
    to the objective for every candidate shift.
    """

    for m in prange(times.shape[0]):
        xq = xq_all[m]
        yq = yq_all[m]
        previous = -1
        for k in range(shifts.shape[0]):
            zq = scale * (times[m] + shifts[k])
            best2 = np.inf
            best = -1
            if previous >= 0:
                dx = xs[previous] - xq
                dy = ys[previous] - yq
                dz = zs[previous] - zq
                best2 = dx * dx
                best2 = best2 + dy * dy
                best2 = best2 + dz * dz
                best = previous
            best2, best = _nearest(xq, yq, zq, best2, best, xs, ys, zs, cell_start,
                                   bx0, bx1, by0, by1, gx, gy, xmin, ymin, cell)
            previous = best
            out[k, m] = math.sqrt(best2)


class GridIndex:
    """Exact nearest neighbour in 3-D with a uniform (x, y) grid."""

    def __init__(self, points: np.ndarray, cell: float = 16.0):
        points = np.ascontiguousarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError("points must be a non-empty [N,3] array")
        if not np.isfinite(points).all():
            raise ValueError("points must be finite")
        self.cell = float(cell)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        self.xmin, self.ymin = float(x.min()), float(y.min())
        ix = np.floor((x - self.xmin) / self.cell).astype(np.int64)
        iy = np.floor((y - self.ymin) / self.cell).astype(np.int64)
        self.gx, self.gy = int(ix.max()) + 1, int(iy.max()) + 1
        cells = iy * self.gx + ix
        order = np.lexsort((z, cells))  # by cell, then by z
        self.order = order.astype(np.int64)
        self.inverse = np.empty_like(self.order)
        self.inverse[self.order] = np.arange(self.order.size)
        self.xs = np.ascontiguousarray(x[order])
        self.ys = np.ascontiguousarray(y[order])
        self.zs = np.ascontiguousarray(z[order])
        sorted_cells = cells[order]
        count = self.gx * self.gy
        self.cell_start = np.searchsorted(sorted_cells, np.arange(count + 1)).astype(np.int64)
        # Actual bounding box of the points of each cell (exact pruning bounds).
        self.bx0 = np.full(count, np.inf)
        self.bx1 = np.full(count, -np.inf)
        self.by0 = np.full(count, np.inf)
        self.by1 = np.full(count, -np.inf)
        np.minimum.at(self.bx0, sorted_cells, self.xs)
        np.maximum.at(self.bx1, sorted_cells, self.xs)
        np.minimum.at(self.by0, sorted_cells, self.ys)
        np.maximum.at(self.by1, sorted_cells, self.ys)

    def query(self, queries: np.ndarray, hint: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
        """Distances and indices (original point order) of the nearest points."""

        queries = np.asarray(queries, dtype=np.float64)
        return self.query_points(queries[:, 0], queries[:, 1], queries[:, 2], hint)

    def query_points(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     hint: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Like :meth:`query` with separate coordinate arrays.

        ``hint`` (original indices, ``-1`` for none) seeds each search with a
        known point; its distance is an exact upper bound, so the result is
        unchanged and the search is shorter when the hint is close (ICP).
        """

        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        z = np.ascontiguousarray(z, dtype=np.float64)
        count = x.shape[0]
        positions = np.full(count, -1, dtype=np.int64)
        if hint is not None:
            hint = np.asarray(hint, dtype=np.int64)
            known = hint >= 0
            positions[known] = self.inverse[hint[known]]
        distance = np.empty(count, dtype=np.float64)
        index = np.empty(count, dtype=np.int64)
        if count:
            _query(x, y, z, positions, self.xs, self.ys, self.zs, self.order,
                   self.cell_start, self.bx0, self.bx1, self.by0, self.by1, self.gx,
                   self.gy, self.xmin, self.ymin, self.cell, distance, index)
        return distance, index

    def query_shifts(self, times: np.ndarray, x: np.ndarray, y: np.ndarray,
                     shifts: np.ndarray, scale: float, start: float, stop: float) -> np.ndarray:
        """Nearest distance of ``(x, y, scale * (t + s))`` for every shift ``s``.

        Returns ``[len(shifts), len(times)]``; entries whose shifted time
        ``t + s`` falls outside ``[start, stop]`` are NaN (not queried). The
        query coordinates and the in-range test use the same floating-point
        expressions as the estimator's per-shift queries.
        """

        out = np.empty((len(shifts), len(times)), dtype=np.float64)
        if len(shifts) and len(times):
            _query_shifts(np.ascontiguousarray(times, dtype=np.float64),
                          np.ascontiguousarray(x, dtype=np.float64),
                          np.ascontiguousarray(y, dtype=np.float64),
                          np.ascontiguousarray(shifts, dtype=np.float64), float(scale),
                          float(start), float(stop), self.xs, self.ys, self.zs,
                          self.cell_start, self.bx0, self.bx1, self.by0, self.by1,
                          self.gx, self.gy, self.xmin, self.ymin, self.cell, out)
        return out

    def query_all_shifts(self, times: np.ndarray, x: np.ndarray, y: np.ndarray,
                         shifts: np.ndarray, scale: float) -> np.ndarray:
        """Nearest distances for all points and all temporal shifts.

        This is an exact batched equivalent of calling :meth:`query_points`
        once per shift with queries ``(x, y, scale * (t + shift))``.  No event
        is removed based on the shifted timestamp.  The previous shift's
        nearest neighbour is only an upper-bound warm start, so it cannot
        alter the selected distance.
        """

        times = np.ascontiguousarray(times, dtype=np.float64)
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        shifts = np.ascontiguousarray(shifts, dtype=np.float64)
        out = np.empty((len(shifts), len(times)), dtype=np.float64)
        if len(shifts) and len(times):
            _query_all_shifts(times, x, y, shifts, float(scale), self.xs, self.ys,
                              self.zs, self.cell_start, self.bx0, self.bx1,
                              self.by0, self.by1, self.gx, self.gy, self.xmin,
                              self.ymin, self.cell, out)
        return out
