#!/usr/bin/env python3
"""Acquire TOPSPIN from Zenodo with archive-level provenance.

Each ``<scene>.tar.gz`` of the public release (record 10884694, CC-BY-4.0)
is verified, extracted, and converted from ``events.csv`` (t [us], x, y, p)
to ``events.npz`` (int64 t, int16 x/y, uint8 p). The CSV is removed after an
atomic NPZ write. A cached archive is never trusted merely because it exists:
its byte size and published MD5 are checked before every extraction.

Converted scenes carry a per-scene provenance marker. Only a marker matching
the current Zenodo MD5 permits a restart to skip that scene.

Examples::

    python scripts/download_topspin.py --output datasets/topspin \
        --archive-cache /path/to/existing/archives
    python scripts/download_topspin.py --output datasets/topspin \
        --record-json record_10884694.json --scenes soho-perpendicular-fast
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
import time
import urllib.request

import numpy as np

RECORD = "https://zenodo.org/api/records/10884694"
AUDIT_NAME = "acquisition_audit.json"
MARKER_NAME = ".topspin_provenance.json"
CONVERTER_SCHEMA = 2
OFFICIAL_RECORD_ID = 10884694
OFFICIAL_RECORD_FINGERPRINT_SHA256 = (
    "c97ac90550208bd2225f9d116016cd188b9f9037638ee2c608a376de8f6208e7"
)
SAFE_SCENE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--archive-cache", type=Path, default=None,
        help="directory containing/reusing verified <scene>.tar.gz files "
             "(default: <output>/archives)",
    )
    parser.add_argument(
        "--record-json", type=Path, default=None,
        help="use a saved Zenodo API record instead of requesting it again",
    )
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="scene names to fetch (default: all 72)")
    return parser.parse_args()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_identity(path: Path) -> dict:
    return {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _expected_md5(checksum: str) -> str:
    if not isinstance(checksum, str) or not checksum.startswith("md5:"):
        raise ValueError(f"Zenodo file is missing an MD5 checksum: {checksum!r}")
    value = checksum.split(":", 1)[1].lower()
    if len(value) != 32 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"invalid Zenodo MD5 checksum: {checksum!r}")
    return value


def _verify_archive(path: Path, size: int, checksum: str) -> str:
    """Return measured MD5 after strict size and checksum validation."""

    expected = _expected_md5(checksum)
    if not path.is_file():
        raise OSError(f"archive is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != size:
        raise OSError(f"archive size {actual_size} != published {size}: {path}")
    actual = _md5(path)
    if actual != expected:
        raise OSError(f"archive md5 {actual} != published {expected}: {path}")
    return actual


def _fetch(url: str, destination: Path, size: int, checksum: str, attempts: int = 5) -> None:
    """Download atomically and validate against Zenodo size and MD5."""

    _expected_md5(checksum)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out, length=1 << 20)
            _verify_archive(temporary, size, checksum)
            temporary.replace(destination)
            return
        except OSError as error:
            temporary.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            print(f"  retry {attempt} for {destination.name}: {error}", flush=True)
            time.sleep(2 ** attempt)


def _safe_extract(archive: Path, root: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (root / member.name).resolve()
            if not target.is_relative_to(root.resolve()) or not (member.isfile() or member.isdir()):
                raise ValueError(f"unsafe archive member: {member.name}")
        # Python 3.10 has no ``filter=`` argument. Every member was already
        # constrained above to a regular file/directory below ``root``; links,
        # devices, FIFOs, and traversal paths are rejected before extraction.
        tar.extractall(root)


def convert(scene_dir: Path) -> int:
    """Losslessly convert integer t/x/y/p columns and remove the source CSV."""

    csv_path = scene_dir / "events.csv"
    data = np.loadtxt(csv_path, delimiter=",", dtype=np.int64, ndmin=2)
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(f"events.csv must contain exactly t,x,y,p columns: {csv_path}")
    int16 = np.iinfo(np.int16)
    uint8 = np.iinfo(np.uint8)
    if (np.any(data[:, 1:3] < int16.min) or np.any(data[:, 1:3] > int16.max)
            or np.any(data[:, 3] < uint8.min) or np.any(data[:, 3] > uint8.max)):
        raise ValueError(f"event value cannot be represented losslessly: {csv_path}")
    destination = scene_dir / "events.npz"
    temporary = scene_dir / "events.npz.part"
    with temporary.open("wb") as stream:
        np.savez(stream, t=data[:, 0], x=data[:, 1].astype(np.int16),
                 y=data[:, 2].astype(np.int16), p=data[:, 3].astype(np.uint8))
    temporary.replace(destination)
    csv_path.unlink()
    return int(data.shape[0])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _record_fingerprint(record: dict) -> str:
    payload = {
        "id": record.get("id"),
        "files": sorted(
            (item.get("key"), item.get("size"), item.get("checksum"))
            for item in record.get("files", [])
        ),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _verified_marker(scene_dir: Path, *, scene: str, record_id: int,
                     expected_md5: str, size: int) -> dict | None:
    marker = _load_json(scene_dir / MARKER_NAME)
    required = (scene_dir / "events.npz", scene_dir / "calib.txt", scene_dir / "meta.json")
    if (marker is None or not all(path.is_file() for path in required)
            or marker.get("schema_version") != CONVERTER_SCHEMA
            or marker.get("converter_schema") != CONVERTER_SCHEMA
            or marker.get("scene") != scene
            or marker.get("record_id") != record_id
            or marker.get("archive_filename") != f"{scene}.tar.gz"
            or marker.get("expected_md5") != expected_md5
            or marker.get("actual_md5") != expected_md5
            or marker.get("archive_size_bytes") != size
            or marker.get("verified") is not True
            or not isinstance(marker.get("event_count"), int)
            or isinstance(marker.get("event_count"), bool)):
        return None
    outputs = marker.get("output_files")
    if not isinstance(outputs, dict):
        return None
    for path in required:
        identity = outputs.get(path.name)
        if (not isinstance(identity, dict)
                or identity.get("size_bytes") != path.stat().st_size
                or identity.get("sha256") != _sha256(path)):
            return None
    try:
        with np.load(scene_dir / "events.npz", allow_pickle=False) as events:
            if (set(events.files) != {"t", "x", "y", "p"}
                    or events["t"].dtype != np.dtype(np.int64)
                    or events["x"].dtype != np.dtype(np.int16)
                    or events["y"].dtype != np.dtype(np.int16)
                    or events["p"].dtype != np.dtype(np.uint8)
                    or any(events[name].ndim != 1 for name in ("t", "x", "y", "p"))
                    or len({events[name].size for name in ("t", "x", "y", "p")}) != 1
                    or int(events["t"].size) != marker["event_count"]):
                return None
    except (OSError, ValueError, TypeError):
        return None
    return marker


def _ensure_archive(url: str, archive: Path, size: int, checksum: str) -> str:
    """Validate a cache hit, or atomically replace it with a verified download."""

    try:
        return _verify_archive(archive, size, checksum)
    except OSError as error:
        if archive.exists():
            print(f"  cache verification failed; downloading replacement: {error}", flush=True)
        _fetch(url, archive, size, checksum)
        return _verify_archive(archive, size, checksum)


def _scene_entry(name: str, item: dict, marker: dict, *, status: str) -> dict:
    return {
        "scene": name,
        "archive_filename": f"{name}.tar.gz",
        "archive_size_bytes": int(item["size"]),
        "expected_md5": _expected_md5(item.get("checksum", "")),
        "actual_md5": marker["actual_md5"],
        "verified": True,
        "verification_status": status,
        "events_npz": f"{name}/events.npz",
        "event_count": int(marker["event_count"]),
        "verified_at_utc": marker["verified_at_utc"],
        "converter_schema": marker["converter_schema"],
        "output_files": marker["output_files"],
    }


def acquire(record: dict, output: Path, archive_cache: Path, names: list[str],
            *, record_url: str = RECORD) -> dict:
    """Acquire selected scenes from a supplied Zenodo record (restartable)."""

    output.mkdir(parents=True, exist_ok=True)
    archive_cache.mkdir(parents=True, exist_ok=True)
    files = {item["key"][: -len(".tar.gz")]: item for item in record.get("files", [])
             if item.get("key", "").endswith(".tar.gz")}
    unsafe = sorted(name for name in files if not SAFE_SCENE.fullmatch(name)
                    or name in {".", ".."})
    if unsafe:
        raise ValueError(f"record contains unsafe scene names: {unsafe}")
    unknown = set(names) - set(files)
    if unknown:
        raise ValueError(f"unknown scenes: {sorted(unknown)}")

    audit_path = output / AUDIT_NAME
    old_audit = _load_json(audit_path) or {}
    record_fingerprint = _record_fingerprint(record)
    compatible_audit = (
        old_audit.get("schema_version") == CONVERTER_SCHEMA
        and old_audit.get("converter_schema") == CONVERTER_SCHEMA
        and old_audit.get("record_id") == record.get("id")
        and old_audit.get("record_fingerprint_sha256") == record_fingerprint
    )
    old_scenes = old_audit.get("scenes", {}) if compatible_audit else {}
    scene_audit = dict(old_scenes) if isinstance(old_scenes, dict) else {}
    audit = {
        "schema_version": CONVERTER_SCHEMA,
        "converter_schema": CONVERTER_SCHEMA,
        "record_id": record.get("id"),
        "record_fingerprint_sha256": record_fingerprint,
        "record_url": record_url,
        "generated_at_utc": _utc_now(),
        "archive_cache": str(archive_cache.resolve()),
        "scenes": scene_audit,
    }
    for index, name in enumerate(names, 1):
        item = files[name]
        size = int(item["size"])
        expected = _expected_md5(item.get("checksum", ""))
        scene_dir = output / name
        marker = _verified_marker(
            scene_dir, scene=name, record_id=record.get("id"),
            expected_md5=expected, size=size,
        )
        if marker is not None:
            scene_audit[name] = _scene_entry(name, item, marker, status="verified_marker")
            audit["generated_at_utc"] = _utc_now()
            _write_json_atomic(audit_path, audit)
            print(f"[{index}/{len(names)}] {name}: verified marker", flush=True)
            continue

        started = time.monotonic()
        archive = archive_cache / f"{name}.tar.gz"
        actual = _ensure_archive(
            f"{record_url}/files/{name}.tar.gz/content", archive, size,
            item.get("checksum", ""),
        )
        with tempfile.TemporaryDirectory(prefix=f".{name}.", dir=output) as temporary:
            staging_root = Path(temporary)
            _safe_extract(archive, staging_root)
            staged_scene = staging_root / name
            if not staged_scene.is_dir():
                raise ValueError(f"archive does not contain expected scene directory: {name}")
            if not (staged_scene / "calib.txt").is_file() or not (staged_scene / "meta.json").is_file():
                raise ValueError(f"archive is missing calibration or metadata: {name}")
            events = convert(staged_scene)
            marker = {
                "schema_version": CONVERTER_SCHEMA,
                "converter_schema": CONVERTER_SCHEMA,
                "scene": name,
                "archive_filename": archive.name,
                "archive_size_bytes": size,
                "expected_md5": expected,
                "actual_md5": actual,
                "verified": True,
                "verified_at_utc": _utc_now(),
                "event_count": events,
                "record_id": record.get("id"),
                "record_url": record_url,
                "output_files": {
                    filename: _output_identity(staged_scene / filename)
                    for filename in ("events.npz", "calib.txt", "meta.json")
                },
            }
            _write_json_atomic(staged_scene / MARKER_NAME, marker)
            previous = output / f".{name}.previous"
            if previous.exists():
                shutil.rmtree(previous)
            if scene_dir.exists():
                scene_dir.replace(previous)
            try:
                staged_scene.replace(scene_dir)
            except BaseException:
                if previous.exists() and not scene_dir.exists():
                    previous.replace(scene_dir)
                raise
            if previous.exists():
                shutil.rmtree(previous)
        scene_audit[name] = _scene_entry(name, item, marker, status="archive_md5_verified")
        audit["generated_at_utc"] = _utc_now()
        _write_json_atomic(audit_path, audit)
        print(f"[{index}/{len(names)}] {name}: {events} events, "
              f"{size / 1e6:.0f} MB in {time.monotonic() - started:.0f} s", flush=True)
    return audit


def main() -> None:
    args = parse_args()
    if args.record_json is None:
        with urllib.request.urlopen(RECORD, timeout=60) as response:
            record = json.load(response)
    else:
        record = json.loads(args.record_json.read_text(encoding="utf-8"))
    fingerprint = _record_fingerprint(record)
    if (record.get("id") != OFFICIAL_RECORD_ID
            or fingerprint != OFFICIAL_RECORD_FINGERPRINT_SHA256):
        raise ValueError(
            "TOPSPIN record metadata does not match the audited Zenodo record 10884694"
        )
    files = {item["key"][: -len(".tar.gz")]: item for item in record["files"]
             if item["key"].endswith(".tar.gz")}
    names = sorted(files) if not args.scenes else args.scenes
    archive_cache = args.archive_cache or args.output / "archives"
    acquire(record, args.output, archive_cache, names)


if __name__ == "__main__":
    main()
