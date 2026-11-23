import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

from scripts import download_topspin
from espinslam.topspin import _validate_topspin_acquisition, _validate_verified_manifest_row


def _archive(tmp_path: Path, name: str = "camera-diagonal-slow"):
    source = tmp_path / "source" / name
    source.mkdir(parents=True)
    rows = "0,1,2,0\n10,3,4,1\n20,5,6,0\n"
    (source / "events.csv").write_text(rows, encoding="utf-8")
    (source / "calib.txt").write_text("1 1 0 0 0 0 0 0 0\n", encoding="utf-8")
    (source / "meta.json").write_text(
        '{"event-camera": {"frame": {"width": 1280, "height": 720}}}\n',
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    archive = cache / f"{name}.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(source, arcname=name)
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    record = {
        "id": 10884694,
        "files": [{
            "key": archive.name,
            "size": archive.stat().st_size,
            "checksum": f"md5:{digest}",
        }],
    }
    return name, archive, cache, record


def test_cached_archive_is_verified_converted_and_audited(tmp_path):
    name, archive, cache, record = _archive(tmp_path)
    output = tmp_path / "output"
    audit = download_topspin.acquire(record, output, cache, [name],
                                     record_url="https://example.invalid/record")

    scene = output / name
    assert archive.is_file()  # reusable cache is retained
    assert not (scene / "events.csv").exists()
    with np.load(scene / "events.npz", allow_pickle=False) as events:
        np.testing.assert_array_equal(events["t"], (0, 10, 20))
        np.testing.assert_array_equal(events["x"], (1, 3, 5))
        np.testing.assert_array_equal(events["y"], (2, 4, 6))
        np.testing.assert_array_equal(events["p"], (0, 1, 0))
    marker = json.loads((scene / download_topspin.MARKER_NAME).read_text(encoding="utf-8"))
    assert marker["actual_md5"] == marker["expected_md5"]
    assert marker["verified"] is True and marker["event_count"] == 3
    stored = json.loads((output / download_topspin.AUDIT_NAME).read_text(encoding="utf-8"))
    assert stored == audit
    entry = audit["scenes"][name]
    assert entry["verification_status"] == "archive_md5_verified"
    assert entry["verified"] is True and entry["event_count"] == 3


def test_only_matching_provenance_marker_allows_skip(tmp_path):
    name, archive, cache, record = _archive(tmp_path)
    output = tmp_path / "output"
    download_topspin.acquire(record, output, cache, [name])
    scene = output / name

    # A matching verified marker is sufficient for restart even if the cache
    # was moved after acquisition.
    moved = archive.with_suffix(".saved")
    archive.replace(moved)
    audit = download_topspin.acquire(record, output, cache, [name])
    assert audit["scenes"][name]["verification_status"] == "verified_marker"

    # Presence of events.npz alone is not trusted. Invalidating the marker
    # forces archive verification and a complete atomic rebuild.
    moved.replace(archive)
    marker_path = scene / download_topspin.MARKER_NAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["expected_md5"] = "0" * 32
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    (scene / "events.npz").write_bytes(b"not an npz")
    download_topspin.acquire(record, output, cache, [name])
    with np.load(scene / "events.npz", allow_pickle=False) as events:
        assert events["t"].tolist() == [0, 10, 20]


def test_marker_does_not_hide_corrupted_converted_output(tmp_path):
    name, archive, cache, record = _archive(tmp_path)
    output = tmp_path / "output"
    download_topspin.acquire(record, output, cache, [name])
    scene = output / name
    marker_before = (scene / download_topspin.MARKER_NAME).read_text(encoding="utf-8")
    (scene / "events.npz").write_bytes(b"corrupt while marker remains unchanged")

    audit = download_topspin.acquire(record, output, cache, [name])

    with np.load(scene / "events.npz", allow_pickle=False) as events:
        assert events["t"].tolist() == [0, 10, 20]
    marker_after = (scene / download_topspin.MARKER_NAME).read_text(encoding="utf-8")
    assert marker_after != marker_before
    assert audit["scenes"][name]["verification_status"] == "archive_md5_verified"


def test_evaluator_ties_verified_manifest_metadata_to_acquired_scene(tmp_path, monkeypatch):
    name, _, cache, record = _archive(tmp_path)
    output = tmp_path / "output"
    download_topspin.acquire(record, output, cache, [name])
    audit = json.loads((output / download_topspin.AUDIT_NAME).read_text(encoding="utf-8"))
    monkeypatch.setattr(
        "espinslam.topspin.TOPSPIN_RECORD_FINGERPRINT_SHA256",
        audit["record_fingerprint_sha256"],
    )
    row = {
        "sequence_id": name, "event_path": f"{name}/events.npz",
        "object": "Camera", "perspective": "diagonal", "speed": "slow",
        "split": "test", "fx": "1.0", "fy": "1.0", "cx": "0.0",
        "cy": "0.0", "k1": "0.0", "k2": "0.0", "p1": "0.0",
        "p2": "0.0", "k3": "0.0", "width": "1280", "height": "720",
        "time_unit": "us", "frequency_source": "", "cad_source": "",
    }
    _, identities = _validate_topspin_acquisition(
        output.resolve(), [row], require_full72=False,
    )
    assert identities[name]["events"]["sha256"]
    _validate_verified_manifest_row(row, output / name, require_fixed_split=True)
    with pytest.raises(ValueError, match="fixed split"):
        _validate_verified_manifest_row(
            {**row, "split": "dev"}, output / name, require_fixed_split=True,
        )
    with pytest.raises(ValueError, match="manifest object"):
        _validate_topspin_acquisition(
            output.resolve(), [{**row, "object": "Hubble"}], require_full72=False,
        )


def test_corrupt_cache_is_not_extracted(tmp_path, monkeypatch):
    name, archive, cache, record = _archive(tmp_path)
    archive.write_bytes(b"corrupt")

    def no_network(*args, **kwargs):
        raise RuntimeError("download would be required")

    monkeypatch.setattr(download_topspin, "_fetch", no_network)
    with pytest.raises(RuntimeError, match="download would be required"):
        download_topspin.acquire(record, tmp_path / "output", cache, [name])
    assert not (tmp_path / "output" / name).exists()


def test_safe_extraction_rejects_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("../escape")
        payload = b"unsafe"
        member.size = len(payload)
        stream.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="unsafe archive member"):
        download_topspin._safe_extract(archive, tmp_path / "extract")


def test_acquisition_rejects_unsafe_scene_name_before_writing(tmp_path):
    record = {
        "id": 10884694,
        "files": [{"key": "../escape.tar.gz", "size": 0,
                   "checksum": "md5:" + "0" * 32}],
    }
    with pytest.raises(ValueError, match="unsafe scene names"):
        download_topspin.acquire(
            record, tmp_path / "output", tmp_path / "cache", ["../escape"],
        )
    assert not (tmp_path / "escape").exists()
