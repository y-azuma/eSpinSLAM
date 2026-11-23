import numpy as np

from espinslam.backend import IncrementalBackend
from espinslam.config import BackendConfig
from espinslam.frontend import TrackPool
from espinslam.geometry import project_points
from espinslam.synthetic import make_scene

INTRINSIC = np.asarray(((528.6, 0.0, 173.0), (0.0, 528.6, 130.0), (0.0, 0.0, 1.0)))


def _pool_with_split_track(frequency=0.5):
    """Ideal tracks; point 0 is observed by two disconnected tracks (Sec. 4.2.6)."""

    scene = make_scene(frequency_hz=frequency, elevation_deg=20.0, point_count=12, seed=3)
    pool = TrackPool(retain=1000)
    times = np.linspace(0.0, 4.0, 400)
    owners = {}
    for index, point in enumerate(scene.points):
        pixels = project_points(np.repeat(point[None], times.size, 0), times, frequency,
                                10.0, scene.rotation, INTRINSIC)
        segments = [(0, 120), (200, 320)] if index == 0 else [(0, 300)]
        for start, stop in segments:
            track_id = pool.add(1, times[start:stop], pixels[start:stop, 0], pixels[start:stop, 1])
            owners[track_id] = index
    return scene, pool, owners


def test_first_run_starts_from_origin_and_association_fuses_split_tracks():
    scene, pool, owners = _pool_with_split_track()
    backend = IncrementalBackend(BackendConfig(), INTRINSIC)
    first = backend.optimize(pool, 0.5, 4.0)
    assert first.fused == 0  # no association on the zero-initialised first run
    assert first.landmarks == len(owners)
    second = backend.optimize(pool, 0.5, 4.0)
    assert second.fused == 1 and second.landmarks == len(owners) - 1
    reconstruction = backend.reconstruction()
    for track_id, point in zip(reconstruction.track_ids, reconstruction.points):
        assert np.linalg.norm(point - scene.points[owners[int(track_id)]]) < 1e-3
    assert reconstruction.mean_reprojection_error_px < 0.05


def test_outliers_are_removed_from_the_pool():
    scene, pool, owners = _pool_with_split_track()
    bad = pool.add(1, np.linspace(0.0, 2.0, 50), np.full(50, 20.0), np.linspace(5.0, 250.0, 50))
    backend = IncrementalBackend(BackendConfig(), INTRINSIC)
    report = backend.optimize(pool, 0.5, 4.0)
    assert bad not in pool.tracks
    assert report.removed_reprojection + report.removed_radius >= 1


def test_tracks_shorter_than_nmatch_are_discarded_not_only_skipped():
    _, pool, _ = _pool_with_split_track()
    short = pool.add(1, np.asarray((0.0, 0.1)), np.asarray((10.0, 11.0)),
                     np.asarray((20.0, 21.0)))
    backend = IncrementalBackend(BackendConfig(), INTRINSIC)
    report = backend.optimize(pool, 0.5, 4.0)
    assert short not in pool.tracks
    assert report.removed_short == 1


def test_paper_path_merges_short_fragments_before_nmatch_elimination():
    """Fig. 3 association precedes the Sec. 4.2.5 N_match test."""

    scene = make_scene(frequency_hz=0.5, elevation_deg=20.0, point_count=4, seed=4)
    point = scene.points[0]
    times = np.asarray((0.0, 0.5, 1.0, 1.5))
    pixels = project_points(np.repeat(point[None], times.size, 0), times, 0.5,
                            10.0, scene.rotation, INTRINSIC)
    pool = TrackPool(retain=100)
    first = pool.add(1, times[:2], pixels[:2, 0], pixels[:2, 1])
    second = pool.add(1, times[2:], pixels[2:, 0], pixels[2:, 1])
    backend = IncrementalBackend(
        BackendConfig(interpretation_mode="paper", min_track_events=3), INTRINSIC,
    )
    backend.initialized = True
    backend.rotation = scene.rotation.copy()
    backend.frequency_hz = 0.5
    backend.landmarks = {first: point.copy(), second: point.copy()}
    backend.errors = {first: 0.0, second: 0.0}
    backend.revisions = {
        first: pool.tracks[first].revision,
        second: pool.tracks[second].revision,
    }
    report = backend.optimize(pool, 0.5, 1.5)
    assert report.fused == 1
    assert report.removed_short == 0
    assert len(pool) == 1
    assert next(iter(pool.tracks.values())).size == 4


def test_reference_initialisation_keeps_radius_outlier_until_post_ba(monkeypatch):
    """Sec. 4.2.5 applies the radius test after, not before, optimisation."""

    import espinslam.backend as backend_module

    outside = np.asarray((12.0, -1.0, 0.5))
    seen: list[tuple[int | None, float | None]] = []

    def fake_triangulate(*args, max_views=None,
                         min_positive_depth_fraction=None, **kwargs):
        seen.append((max_views, min_positive_depth_fraction))
        return outside.copy()

    monkeypatch.setattr(backend_module, "triangulate", fake_triangulate)
    paper = IncrementalBackend(BackendConfig(interpretation_mode="paper"), INTRINSIC)
    paper.initialized = True
    experimental = IncrementalBackend(
        BackendConfig(interpretation_mode="experimental"), INTRINSIC,
    )
    experimental.initialized = True
    times = np.asarray((0.0, 0.1, 0.2))
    pixels = np.zeros((3, 2))
    assert np.array_equal(paper._initial_point(0, times, pixels, 0.5, True), outside)
    assert np.array_equal(
        experimental._initial_point(0, times, pixels, 0.5, True), np.zeros(3),
    )
    assert seen == [(None, None), (64, 0.9)]


def test_paper_backend_never_caps_retained_track_observations():
    pool = TrackPool(retain=200)
    track_id = pool.add(
        1, np.linspace(0.0, 2.0, 100), np.arange(100.0), np.arange(100.0),
    )
    track = pool.tracks[track_id]
    paper = IncrementalBackend(
        BackendConfig(interpretation_mode="paper", max_observations_per_track=10),
        INTRINSIC,
    )
    experimental = IncrementalBackend(
        BackendConfig(interpretation_mode="experimental", max_observations_per_track=10),
        INTRINSIC,
    )
    assert paper._track_observations(track)[0].size == 100
    assert experimental._track_observations(track)[0].size == 10
