import csv
import json
import os

import numpy as np
import pytest

from scripts.prepare_periodic_logo_dataset import prepare_dataset


def _write_source(tmp_path):
    events = tmp_path / "event_stream"
    cad = tmp_path / "raw_cad"
    (events / "packets").mkdir(parents=True)
    (cad / "sample_000003" / "meta" / "cad").mkdir(parents=True)
    event_path = events / "packets" / "debris_000003.npz"
    np.savez_compressed(
        event_path,
        x=np.asarray((1, 2), dtype=np.int16),
        y=np.asarray((3, 4), dtype=np.int16),
        t_us=np.asarray((100, 200), dtype=np.int64),
        p01=np.asarray((0, 1), dtype=np.uint8),
        omega_rad_s=np.asarray(2.0, dtype=np.float32),
    )
    points = cad / "sample_000003" / "meta" / "cad" / "reference_points.npz"
    np.savez_compressed(points, points=np.zeros((3, 3), dtype=np.float32))
    mesh = cad / "sample_000003" / "meta" / "cad" / "target_mesh.ply"
    mesh.write_text("ply\n", encoding="utf-8")

    fieldnames = [
        "sequence_id", "event_path", "split", "object_id", "scene_id",
        "reference_pointcloud_path", "reference_mesh_path", "cache_sha256",
        "cache_events", "provenance.generator.sample_index",
        "provenance.camera_intrinsics.fx_px",
        "provenance.camera_intrinsics.fy_px",
        "provenance.camera_intrinsics.cx_px",
        "provenance.camera_intrinsics.cy_px",
        "provenance.camera_intrinsics.width_px",
        "provenance.camera_intrinsics.height_px",
        "provenance.camera_intrinsics.distortion.k1",
        "provenance.camera_intrinsics.distortion.k2",
        "provenance.camera_intrinsics.distortion.p1",
        "provenance.camera_intrinsics.distortion.p2",
    ]
    row = {
        "sequence_id": "debris_000003",
        "event_path": "packets/debris_000003.npz",
        "split": "train",
        "object_id": "debris_000003",
        "scene_id": "scene_000003",
        "reference_pointcloud_path": "sample_000003/meta/cad/reference_points.npz",
        "reference_mesh_path": "sample_000003/meta/cad/target_mesh.ply",
        "cache_sha256": "abc",
        "cache_events": "2",
        "provenance.generator.sample_index": "3",
        "provenance.camera_intrinsics.fx_px": "528.0",
        "provenance.camera_intrinsics.fy_px": "526.0",
        "provenance.camera_intrinsics.cx_px": "173.0",
        "provenance.camera_intrinsics.cy_px": "130.0",
        "provenance.camera_intrinsics.width_px": "346",
        "provenance.camera_intrinsics.height_px": "260",
        "provenance.camera_intrinsics.distortion.k1": "-0.018",
        "provenance.camera_intrinsics.distortion.k2": "0.003",
        "provenance.camera_intrinsics.distortion.p1": "0.0002",
        "provenance.camera_intrinsics.distortion.p2": "-0.0001",
    }
    with (events / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    (events / "precompute_summary.json").write_text(json.dumps({
        "complete": True,
        "storage_format": "event-stream",
        "roi_policy": "fixed_global_crop_fixed_scale",
        "source_manifest_sha256": "source-hash",
        "split_counts": {"train": 1},
        "fixed_crop_fit": {
            "fit_split": "train",
            "uses_validation_or_test": False,
            "crop_xyxy": [129, 47, 213, 211],
            "fixed_scale": 0.5,
            "output_size_hw": [82, 42],
        },
    }), encoding="utf-8")
    return events, cad, event_path, points


def test_prepare_dataset_links_exact_events_and_matching_cad(tmp_path):
    events, cad, event_source, point_source = _write_source(tmp_path)
    output = tmp_path / "espinslam_dataset"
    report = prepare_dataset(events, cad, output, link_mode="hardlink")
    event_output = output / "events" / "debris_000003.npz"
    point_output = output / "cad" / "debris_000003" / "reference_points.npz"
    assert os.stat(event_source).st_ino == os.stat(event_output).st_ino
    assert os.stat(point_source).st_ino == os.stat(point_output).st_ino
    assert report["split_counts"] == {"train": 1}
    assert report["storage"]["file_counts"] == {"hardlink": 3, "copy": 0}
    assert report["espinslam_camera_intrinsics"] == {
        "fx": 264.0, "fy": 263.0, "cx": 22.0, "cy": 41.5,
        "width": 42, "height": 82,
    }
    rows = list(csv.DictReader((output / "manifest.csv").open()))
    assert rows[0]["split"] == "train"
    assert rows[0]["reference_pointcloud_path"].endswith("reference_points.npz")
    with pytest.raises(FileExistsError):
        prepare_dataset(events, cad, output, link_mode="hardlink")


def test_prepare_dataset_rejects_crop_fitted_with_test_data(tmp_path):
    events, cad, _, _ = _write_source(tmp_path)
    summary_path = events / "precompute_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["fixed_crop_fit"]["uses_validation_or_test"] = True
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="train only"):
        prepare_dataset(events, cad, tmp_path / "output")
