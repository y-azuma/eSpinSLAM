import numpy as np

from espinslam.geometry import align_similarity_icp, camera_pose, project_points


def test_camera_principal_axis_points_to_orbit_center():
    rotation, center = camera_pose(0.25, 0.5, 10.0, np.zeros(3))
    camera_origin = rotation @ (np.zeros(3) - center)
    assert camera_origin[2] > 0
    assert np.allclose(camera_origin[:2], 0.0, atol=1e-10)


def test_similarity_icp_resolves_monocular_transform():
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(300, 3)) * np.asarray((2.0, 0.7, 1.2))
    rotation = np.asarray((
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ))
    predicted = 2.4 * (reference @ rotation.T) + np.asarray((3.0, -2.0, 1.0))
    result = align_similarity_icp(predicted, reference, iterations=20)
    assert result.valid
    assert result.icp_rmse_m < 1e-5


def test_similarity_icp_rejects_collapsed_cloud():
    reference = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                            (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    result = align_similarity_icp(np.zeros((8, 3)), reference)
    assert not result.valid
    assert "degenerate extent" in result.reason


def test_projected_orbit_center_is_principal_point():
    intrinsic = np.asarray(((500.0, 0.0, 173.0), (0.0, 500.0, 130.0), (0.0, 0.0, 1.0)))
    pixels = project_points(
        np.zeros((3, 3)), np.asarray((0.0, 0.5, 1.0)),
        0.5, 10.0, np.zeros(3), intrinsic,
    )
    assert np.allclose(pixels, np.asarray((173.0, 130.0)), atol=1e-8)


def test_vectorized_projection_matches_camera_pose_definition():
    intrinsic = np.asarray(((530.0, 0.0, 173.0), (0.0, 525.0, 130.0), (0.0, 0.0, 1.0)))
    times = np.asarray((0.1, 0.35, 0.8, 1.2))
    points = np.asarray(((0.2, -0.1, 0.4), (-0.3, 0.2, -0.5),
                         (0.1, 0.4, 0.2), (-0.2, -0.3, 0.1)))
    rotation_vector = np.asarray((0.03, -0.02, 0.01))
    actual = project_points(
        points, times, 0.42, 10.0, rotation_vector, intrinsic,
    )
    expected = []
    for point, timestamp in zip(points, times):
        rotation, center = camera_pose(timestamp, 0.42, 10.0, rotation_vector)
        camera = rotation @ (point - center)
        pixel = intrinsic @ camera
        expected.append(pixel[:2] / pixel[2])
    assert np.allclose(actual, np.asarray(expected), atol=1e-10)
