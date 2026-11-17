import csv
from types import SimpleNamespace

import numpy as np

from espinslam.config import DataConfig, ESpinSLAMConfig, EvaluationConfig
from espinslam.evaluation import evaluate


def test_evaluation_uses_each_manifest_cad_reference(tmp_path, monkeypatch):
    sample = tmp_path / "sample_000000"
    sample.mkdir()
    np.savez_compressed(
        sample / "events.npz",
        x=np.asarray((1, 2, 3), dtype=np.int16),
        y=np.asarray((1, 2, 3), dtype=np.int16),
        t=np.asarray((0.0, 0.05, 0.10)),
        p=np.asarray((0, 1, 0), dtype=np.uint8),
        omega_rad_s=np.asarray(2.0),
    )
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(100, 3)).astype(np.float32)
    reference_dir = sample / "meta" / "cad"
    reference_dir.mkdir(parents=True)
    np.savez_compressed(reference_dir / "reference_points.npz", points=reference)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "sequence_id", "event_path", "split", "reference_pointcloud_path",
        ))
        writer.writeheader()
        writer.writerow({
            "sequence_id": "debris_000000",
            "event_path": "sample_000000/events.npz",
            "split": "test",
            "reference_pointcloud_path": (
                "sample_000000/meta/cad/reference_points.npz"
            ),
        })

    frequency = SimpleNamespace(
        frequency_hz=2.0 / (2.0 * np.pi),
        omega_rad_s=2.0,
        confidence=1.0,
        reason="",
        valid=True,
    )
    predicted = 2.5 * reference + np.asarray((3.0, -2.0, 1.0))
    result = SimpleNamespace(
        valid=True,
        omega_rad_s=2.0,
        frequency=frequency,
        frequency_observations=np.empty((0, 4), dtype=np.float64),
        tracks=[],
        reconstruction=SimpleNamespace(
            points=predicted,
            valid=True,
            reason="",
            mean_reprojection_error_px=0.5,
        ),
    )

    class FakeESpinSLAM:
        def __init__(self, _config):
            pass

        def process(self, _sequence, *, reconstruct_structure=True):
            assert reconstruct_structure
            return result

        def estimate_frequency(self, _sequence, *, observations=None):
            return frequency

    monkeypatch.setattr("espinslam.evaluation.ESpinSLAM", FakeESpinSLAM)
    monkeypatch.setattr(
        "espinslam.evaluation.create_report_plots",
        lambda *_args, **_kwargs: [],
    )
    config = ESpinSLAMConfig(
        data=DataConfig(sensor_size=[8, 8], max_duration_seconds=1.0),
        evaluation=EvaluationConfig(
            observation_times_seconds=[0.1],
            pointcloud_max_reference_points=100,
        ),
    )
    output = tmp_path / "output"
    report = evaluate(config, tmp_path, manifest, output)

    assert report["reference_pointcloud_available"] is True
    assert report["reference_pointcloud_mode"] == "per_sequence_manifest"
    assert report["reference_pointcloud_sequence_count"] == 1
    assert report["pointcloud"]["coverage"] == 1.0
    assert report["pointcloud"]["mean_icp_rmse_m"] < 1e-5
    with (output / "pointcloud_metrics.csv").open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["reference_point_count"] == "100"
    assert row["reference_pointcloud_path"].endswith("reference_points.npz")
