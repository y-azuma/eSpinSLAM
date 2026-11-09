import csv

import numpy as np

from espinslam.data import load_event_sequence, read_manifest


def test_npz_manifest_loader_preserves_explicit_test_membership(tmp_path):
    sample = tmp_path / "sample_000000"
    sample.mkdir()
    np.savez_compressed(
        sample / "events.npz",
        x=np.asarray((1, 2, 3)),
        y=np.asarray((4, 5, 6)),
        t=np.asarray((0.0, 0.1, 0.2)),
        p=np.asarray((-1, 1, -1)),
        omega_rad_s=np.asarray(2.0),
    )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sequence_id", "event_path", "split"))
        writer.writeheader()
        writer.writerow({
            "sequence_id": "debris_000000",
            "event_path": "sample_000000/events.npz",
            "split": "test",
        })
    row = read_manifest(manifest, tmp_path)[0]
    sequence = load_event_sequence(row, tmp_path, (346, 260))
    assert sequence.sequence_id == "debris_000000"
    assert sequence.omega_rad_s == 2.0
    assert sequence.p.tolist() == [0, 1, 0]

