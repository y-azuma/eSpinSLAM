import numpy as np

import espinslam.frontend as frontend_module
from espinslam.config import FrontendConfig
from espinslam.efast import EFastDetector
from espinslam.frontend import (
    EventFrontend, TrackPool, Tracklet, cluster_tracklets, density_filter,
)
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
    """The cited eSfO per-polarity mean keeps a sparse polarity independently."""

    rng = np.random.default_rng(1)
    on = rng.normal((30.0, 30.0), 0.5, (60, 2))     # sensitive polarity: very dense
    off = rng.normal((70.0, 70.0), 1.5, (15, 2))    # rarer polarity: sparser corner
    xy = np.vstack((on, off))
    p = np.r_[np.ones(60, int), np.zeros(15, int)]
    global_mean = density_filter(np.zeros(len(xy)), xy[:, 0], xy[:, 1], p, 3.0)
    per_polarity = density_filter(np.zeros(len(xy)), xy[:, 0], xy[:, 1], p, 3.0,
                                  per_polarity_mean=True)
    assert not global_mean[60:].any()       # historical global mean drops it
    assert per_polarity[60:].any()          # cited eSfO reference keeps it


def test_reference_hdbscan_uses_cited_settings_and_mixes_polarity(monkeypatch):
    """eSfO reports minPts=10, epsilon=5 and no polarity dimension."""

    observed: dict[str, object] = {}

    class RecordingHDBSCAN:
        def __init__(self, **kwargs):
            observed["kwargs"] = kwargs

        def fit_predict(self, features):
            observed["features"] = np.asarray(features)
            return np.zeros(features.shape[0], dtype=int)

    monkeypatch.setattr(frontend_module, "HDBSCAN", RecordingHDBSCAN)
    count = 12
    t = np.linspace(0.0, 0.01, count)
    x = np.linspace(10.0, 11.0, count)
    y = np.linspace(20.0, 21.0, count)
    p = np.arange(count) % 2
    tracklets = cluster_tracklets(t, x, y, p, FrontendConfig())
    assert observed["kwargs"] == {
        "min_cluster_size": 10,
        "min_samples": None,
        "cluster_selection_epsilon": 5.0,
        "copy": True,
    }
    # (x, y, t) only: alternating polarities coexist in one cited-source
    # cluster instead of becoming a fourth, separating feature dimension.
    assert observed["features"].shape == (count, 3)
    assert len(tracklets) == 1 and tracklets[0].t.size == count


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


def test_removing_newest_track_recomputes_eq12_time_reference():
    pool = TrackPool(retain=5)
    old = pool.add(1, np.asarray((1.0, 2.0)), np.zeros(2), np.zeros(2))
    newest = pool.add(1, np.asarray((9.0, 10.0)), np.zeros(2), np.zeros(2))
    pool.remove([newest])
    assert list(pool.tracks) == [old]
    assert pool.newest_time == 2.0
    pool.remove([old])
    assert pool.newest_time == -np.inf


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


def test_paper_association_uses_forward_spatiotemporal_mean_and_ignores_polarity():
    config = FrontendConfig(
        interpretation_mode="paper", cluster_time_scale_px_per_second=1000.0,
        association_radius_px=30.0, association_head_events=2,
    )
    frontend = EventFrontend(120, 90, config)
    prior = frontend.pool.add(
        0, np.asarray((0.000, 0.001)), np.asarray((10.0, 10.0)),
        np.asarray((20.0, 20.0)),
    )
    close = Tracklet(
        1, np.asarray((0.010, 0.011)), np.asarray((12.0, 12.0)),
        np.asarray((20.0, 20.0)),
    )
    far_in_time = Tracklet(
        0, np.asarray((0.050, 0.051)), np.asarray((10.0, 10.0)),
        np.asarray((20.0, 20.0)),
    )
    associated = frontend._associate([close, far_in_time], 0.010)
    assert associated == 1
    assert frontend.pool.tracks[prior].size == 4
    assert len(frontend.pool) == 2


def test_paper_association_frozen_collision_and_hemisphere_boundaries():
    """Freeze online ambiguities not resolved by the cited offline eSfO text."""

    config = FrontendConfig(
        interpretation_mode="paper", cluster_time_scale_px_per_second=1.0,
        association_radius_px=30.0, association_head_events=2,
    )
    frontend = EventFrontend(120, 90, config)
    prior = frontend.pool.add(
        0, np.asarray((1.0, 1.0)), np.asarray((0.0, 0.0)),
        np.asarray((0.0, 0.0)),
    )
    # Equal descriptor time is admitted as the equatorial plane of the
    # forward hemisphere.  Two current heads may choose the same earlier tail
    # (frozen many-to-one online adaptation), regardless of polarity.
    first = Tracklet(
        1, np.asarray((1.0, 1.0)), np.asarray((1.0, 1.0)), np.zeros(2),
    )
    second = Tracklet(
        0, np.asarray((1.0, 1.0)), np.asarray((2.0, 2.0)), np.zeros(2),
    )
    # The cited source does not specify an open/closed phi boundary.  The
    # predeclared reference interpretation is strict distance < phi.
    boundary = Tracklet(
        0, np.asarray((1.0, 1.0)), np.asarray((30.0, 30.0)), np.zeros(2),
    )
    associated = frontend._associate([first, second, boundary], 1.0)
    assert associated == 2
    assert frontend.pool.tracks[prior].size == 6
    assert len(frontend.pool) == 2
