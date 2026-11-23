import numpy as np

from espinslam.config import FrontendConfig
from espinslam.efast import EFastDetector
from espinslam.frontend import EventFrontend, TrackPool, density_filter
from espinslam.synthetic import make_scene, simulate_events


def _moving_square_events(steps=40, size=12, start=30):
    """ON events of the leading edges of a square translating along +x."""

    t, x, y = [], [], []
    for step in range(steps):
        column = start + size + step
        for row in range(start, start + size):
            t.append(step * 1e-3 + row * 1e-6)
            x.append(column)
            y.append(row)
        for col in range(start + step, start + size + step + 1):
            t.append(step * 1e-3 + 5e-4 + col * 1e-6)
            x.append(col)
            y.append(start)
    order = np.argsort(t)
    return (np.asarray(t)[order], np.asarray(x)[order], np.asarray(y)[order],
            np.ones(len(t), dtype=np.uint8))


def test_efast_fires_near_corners_not_along_edges():
    t, x, y, p = _moving_square_events()
    detector = EFastDetector(120, 90)
    corners = detector.detect(x, y, t, p)
    assert corners.any()
    corner_y = y[corners]
    # Corner events cluster at the square's top-right corner, which moves along
    # the top edge; events in the middle of the leading edge are rejected.
    middle = (y >= 34) & (y <= 38)
    assert corners[middle].mean() < 0.2
    assert np.mean(corner_y <= 33) > 0.5


def test_density_filter_removes_isolated_corners():
    rng = np.random.default_rng(0)
    cluster = rng.normal((50.0, 50.0), 0.8, (30, 2))
    isolated = np.asarray(((5.0, 5.0), (90.0, 10.0), (10.0, 80.0)))
    xy = np.vstack((cluster, isolated))
    keep = density_filter(np.zeros(len(xy)), xy[:, 0], xy[:, 1], np.zeros(len(xy), int), 3.0)
    assert keep[:30].mean() > 0.6 and not keep[30:].any()


def test_density_threshold_global_or_per_polarity():
    """Eq. (6) uses one mean mu_D; the per-polarity variant keeps the rarer polarity."""

    rng = np.random.default_rng(1)
    on = rng.normal((30.0, 30.0), 0.5, (60, 2))     # sensitive polarity: very dense
    off = rng.normal((70.0, 70.0), 1.5, (15, 2))    # rarer polarity: sparser corner
    xy = np.vstack((on, off))
    p = np.r_[np.ones(60, int), np.zeros(15, int)]
    global_mean = density_filter(np.zeros(len(xy)), xy[:, 0], xy[:, 1], p, 3.0)
    per_polarity = density_filter(np.zeros(len(xy)), xy[:, 0], xy[:, 1], p, 3.0,
                                  per_polarity_mean=True)
    assert not global_mean[60:].any()       # the paper's single mean drops it
    assert per_polarity[60:].any()          # a per-polarity mean keeps it


def test_track_pool_buffer_merge_and_forget():
    pool = TrackPool(retain=5)
    first = pool.add(1, np.arange(4.0), np.zeros(4), np.zeros(4))
    pool.extend(first, np.arange(4.0, 8.0), np.ones(4), np.ones(4))
    assert pool.tracks[first].size == 5 and pool.tracks[first].earliest_t == 3.0
    second = pool.add(0, np.asarray([8.5, 9.0]), np.ones(2), np.ones(2))
    merged = pool.merge([first, second])
    assert merged == first and second not in pool.tracks
    assert pool.tracks[first].latest_t == 9.0 and pool.tracks[first].size == 5
    third = pool.add(1, np.asarray([20.0]), np.zeros(1), np.zeros(1))
    assert pool.forget(15.0) == [first] and list(pool.tracks) == [third]


def test_frontend_builds_long_tracks_on_a_spinning_object():
    scene = make_scene(frequency_hz=0.5, elevation_deg=20.0, point_count=30, seed=1)
    stream = simulate_events(scene, duration_seconds=0.6)
    frontend = EventFrontend(scene.width, scene.height, FrontendConfig())
    for start in range(0, stream.t.size, 1000):
        frontend.push(stream.t[start:start + 1000], stream.x[start:start + 1000],
                      stream.y[start:start + 1000], stream.p[start:start + 1000])
    tracks = list(frontend.pool.tracks.values())
    sizes = np.asarray([track.size for track in tracks], dtype=float)
    spans = np.asarray([track.latest_t - track.earliest_t for track in tracks])
    order = np.argsort(spans)
    weighted_median = spans[order][np.searchsorted(np.cumsum(sizes[order]) / sizes.sum(), 0.5)]
    assert weighted_median > 0.05  # most track events belong to >50 ms tracks
