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
