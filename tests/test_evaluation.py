import numpy as np
import pytest

from espinslam.data import evaluation_condition_fields
from espinslam.evaluation import _reference_for_row


def test_evaluation_conditions_are_allowlisted_and_exclude_inputs_and_labels():
    metadata = {
        "object_id": "debris_000003",
        "scene_id": "scene_000003",
        "provenance.target.periodic_order": "4",
        "provenance.camera.initial_range_m": "12.5",
        "omega_rad_s": "2.0",
        "event_path": "events/debris_000003.npz",
        "bbox_path": "bbox/debris_000003.csv",
        "reference_pointcloud_path": "cad/debris_000003/reference_points.npz",
        "arbitrary_future_field": "must-not-leak",
    }
    assert evaluation_condition_fields(metadata) == {
        "object_id": "debris_000003",
        "scene_id": "scene_000003",
        "provenance.target.periodic_order": "4",
        "provenance.camera.initial_range_m": "12.5",
    }


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
