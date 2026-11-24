import numpy as np
import pytest

from espinslam.evaluation import _reference_for_row


def test_reference_for_row_loads_only_inside_dataset_root(tmp_path):
    cad = tmp_path / "cad" / "scene"
    cad.mkdir(parents=True)
    path = cad / "reference_points.npz"
    np.savez_compressed(path, points=np.arange(15, dtype=np.float32).reshape(5, 3))
    points, label = _reference_for_row(
        {"reference_pointcloud_path": "cad/scene/reference_points.npz"},
        tmp_path, None, 30_000,
    )
    assert label == "cad/scene/reference_points.npz"
    assert points.shape == (5, 3)
    with pytest.raises(ValueError, match="outside data root"):
        _reference_for_row(
            {"reference_pointcloud_path": "../reference_points.npz"},
            tmp_path, None, 30_000,
        )
