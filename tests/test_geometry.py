import numpy as np
from scipy.spatial.transform import Rotation

from espinslam.geometry import (
    _Problem, align_similarity_icp, bundle_adjust, camera_pose, project_points,
    so3_exp, triangulate,
)
from espinslam.synthetic import make_scene

INTRINSIC = np.asarray(((528.6, 0.0, 173.0), (0.0, 528.6, 130.0), (0.0, 0.0, 1.0)))


def test_camera_principal_axis_points_to_orbit_center():
    rotation, center = camera_pose(0.25, 0.5, 10.0, np.zeros(3))
    camera_origin = rotation @ (np.zeros(3) - center)
    assert camera_origin[2] > 0
    assert np.allclose(camera_origin[:2], 0.0, atol=1e-10)


def test_projected_orbit_center_is_principal_point():
    pixels = project_points(np.zeros((3, 3)), np.asarray((0.0, 0.5, 1.0)), 0.5, 10.0,
                            np.zeros(3), INTRINSIC)
    assert np.allclose(pixels, np.asarray((173.0, 130.0)), atol=1e-8)


def test_vectorized_projection_matches_camera_pose_definition():
    times = np.asarray((0.1, 0.35, 0.8, 1.2))
    points = np.asarray(((0.2, -0.1, 0.4), (-0.3, 0.2, -0.5),
                         (0.1, 0.4, 0.2), (-0.2, -0.3, 0.1)))
    rotation_vector = np.asarray((0.03, -0.02, 0.01))
    actual = project_points(points, times, 0.42, 10.0, rotation_vector, INTRINSIC)
    expected = []
    for point, timestamp in zip(points, times):
        rotation, center = camera_pose(timestamp, 0.42, 10.0, rotation_vector)
        pixel = INTRINSIC @ (rotation @ (point - center))
        expected.append(pixel[:2] / pixel[2])
    assert np.allclose(actual, np.asarray(expected), atol=1e-10)


def test_analytic_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    points = rng.normal(size=(3, 3))
    owners = np.asarray([0, 0, 1, 1, 1, 2, 2])
    times = rng.uniform(0, 5, owners.size)
    pixels = rng.uniform(0, 300, (owners.size, 2))
    rotation, frequency = so3_exp([0.2, -0.1, 0.3]), 0.37
    problem = _Problem(times, pixels, owners, INTRINSIC, 10.0, 1e9)  # quadratic cost
    _, _, _, _, g_points, g_global = problem.normal_equations(points, rotation, frequency, 4)
    step = 1e-6
    numeric = np.zeros(4)
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = step
        numeric[axis] = (problem.errors(points, so3_exp(delta) @ rotation, frequency)[0]
                         - problem.errors(points, so3_exp(-delta) @ rotation, frequency)[0]) / (2 * step)
    numeric[3] = (problem.errors(points, rotation, frequency + step)[0]
                  - problem.errors(points, rotation, frequency - step)[0]) / (2 * step)
    assert np.allclose(numeric, g_global, rtol=1e-6, atol=1e-4)
    moved = points.copy()
    moved[1, 2] += step
    back = points.copy()
    back[1, 2] -= step
    numeric_point = (problem.errors(moved, rotation, frequency)[0]
                     - problem.errors(back, rotation, frequency)[0]) / (2 * step)
    assert np.isclose(numeric_point, g_points[1, 2], rtol=1e-6)


def _scene_observations(elevation, noise, frequency=0.5, seed=0):
    scene = make_scene(frequency_hz=frequency, elevation_deg=elevation, point_count=40, seed=seed)
    rng = np.random.default_rng(seed)
    times = np.arange(0.0, 4.0, 1.0 / 100.0)
    all_t, all_px, owners = [], [], []
    for index, point in enumerate(scene.points):
        pixels = project_points(np.repeat(point[None], times.size, 0), times, frequency,
                                10.0, scene.rotation, INTRINSIC)
        arc = ((frequency * times + rng.uniform()) % 1.0) < 0.35
        keep = np.isfinite(pixels).all(axis=1) & arc
        all_t.append(times[keep])
        all_px.append(pixels[keep] + rng.normal(0.0, noise, (keep.sum(), 2)))
        owners.append(np.full(keep.sum(), index))
    return scene, np.concatenate(all_t), np.concatenate(all_px), np.concatenate(owners)


def test_bundle_adjustment_converges_from_paper_initialisation():
    """Sec. 4.2.3: points at the origin, R0 = I, f = f*; camera tilted by 30 deg."""

    scene, times, pixels, owners = _scene_observations(30.0, 0.5)
    result = bundle_adjust(times, pixels, owners, np.zeros_like(scene.points), np.eye(3),
                           0.5, 10.0, INTRINSIC, huber_threshold_px=2.0, max_iterations=100)
    error = np.degrees(np.linalg.norm(
        Rotation.from_matrix(result.rotation @ scene.rotation.T).as_rotvec()))
    assert error < 0.1
    assert np.sqrt(np.mean(np.sum((result.points - scene.points) ** 2, axis=1))) < 0.01
    assert np.median(result.per_observation_error_px) < 1.0


def test_bundle_adjustment_can_also_estimate_frequency():
    scene, times, pixels, owners = _scene_observations(15.0, 0.5)
    result = bundle_adjust(times, pixels, owners, np.zeros_like(scene.points), np.eye(3),
                           0.5 * 1.004, 10.0, INTRINSIC, huber_threshold_px=2.0,
                           max_iterations=100, optimize_frequency=True)
    assert abs(result.frequency_hz - 0.5) < 1e-4


def test_iteration_limit_is_not_mislabeled_as_convergence():
    scene, times, pixels, owners = _scene_observations(30.0, 0.5)
    result = bundle_adjust(times, pixels, owners, np.zeros_like(scene.points), np.eye(3),
                           0.5, 10.0, INTRINSIC, huber_threshold_px=2.0,
                           max_iterations=1)
    assert result.termination_status == "max_iterations"
    assert not result.converged


def test_triangulation_recovers_point():
    scene, times, pixels, owners = _scene_observations(20.0, 0.0)
    mask = owners == 3
    point = triangulate(times[mask], pixels[mask], 0.5, 10.0, scene.rotation, INTRINSIC)
    assert point is not None and np.allclose(point, scene.points[3], atol=1e-6)


def test_cheirality_guard_is_experimental_and_includes_threshold_boundary(monkeypatch):
    """The paper does not state the former 90% positive-depth rejection."""

    import espinslam.geometry as geometry_module

    times = np.arange(10.0)
    pixels = np.zeros((10, 2))
    rotations = np.repeat(np.eye(3)[None], 10, axis=0)
    centres = np.zeros((10, 3))
    centres[:, 2] = 1.0

    def fake_frames(*args, **kwargs):
        return rotations, centres.copy()

    def origin_svd(matrix, *args, **kwargs):
        return np.eye(matrix.shape[0], 4), np.ones(4), np.eye(4)

    monkeypatch.setattr(geometry_module, "orbit_camera_frames", fake_frames)
    monkeypatch.setattr(np.linalg, "svd", origin_svd)

    # The finite DLT solution is behind every camera. It remains valid in the
    # paper path, whereas the old unreported guard rejects it.
    paper = triangulate(times, pixels, 0.5, 10.0, np.eye(3), INTRINSIC)
    guarded = triangulate(
        times, pixels, 0.5, 10.0, np.eye(3), INTRINSIC,
        min_positive_depth_fraction=0.9,
    )
    assert np.array_equal(paper, np.zeros(3))
    assert guarded is None

    # Nine strictly positive depths out of ten is exactly the 0.9 boundary
    # and is admitted by the historical experimental rule.
    centres[:9, 2] = -1.0
    centres[9, 2] = 1.0
    boundary = triangulate(
        times, pixels, 0.5, 10.0, np.eye(3), INTRINSIC,
        min_positive_depth_fraction=0.9,
    )
    assert np.array_equal(boundary, np.zeros(3))


def test_reference_triangulation_uses_every_retained_observation(monkeypatch):
    """Sec. 4.2.4 has no 64-view subsampling step in the reference path."""

    scene, times, pixels, owners = _scene_observations(20.0, 0.0)
    mask = owners == 3
    times, pixels = times[mask], pixels[mask]
    assert times.size > 64
    rows: list[int] = []
    original = np.linalg.svd

    def recording_svd(matrix, *args, **kwargs):
        rows.append(matrix.shape[0])
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", recording_svd)
    triangulate(times, pixels, 0.5, 10.0, scene.rotation, INTRINSIC)
    triangulate(times, pixels, 0.5, 10.0, scene.rotation, INTRINSIC,
                max_views=64)
    assert rows == [2 * times.size, 2 * 64]


def test_similarity_icp_resolves_monocular_transform():
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(300, 3)) * np.asarray((2.0, 0.7, 1.2))
    rotation = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    predicted = (reference @ rotation.T) / 2.4 + np.asarray((3.0, -2.0, 1.0))
    result = align_similarity_icp(predicted, reference, iterations=20)
    assert result.valid
    assert result.icp_rmse < 1e-5
    known = align_similarity_icp(predicted, reference, iterations=20, known_scale=2.4)
    assert known.valid and known.icp_rmse < 1e-5 and np.isclose(known.scale, 2.4)


def test_similarity_icp_rejects_collapsed_cloud():
    reference = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                            (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    result = align_similarity_icp(np.zeros((8, 3)), reference)
    assert not result.valid
    assert "degenerate extent" in result.reason
