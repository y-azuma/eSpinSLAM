#!/usr/bin/env python3
"""Download TOPSPIN from Zenodo and store each scene compactly.

Each ``<scene>.tar.gz`` of the public release (record 10884694, CC-BY-4.0) is
downloaded, extracted, and its ``events.csv`` (t [us], x, y, p) is converted to
``events.npz`` (int64 t in microseconds, int16 x/y, uint8 p) next to
``calib.txt`` and ``meta.json``. The CSV and the archive are then deleted
(the npz is about 3x smaller and loads much faster). Each archive is checked
against the size and MD5 published by Zenodo and downloaded again on a
mismatch. Finished scenes are skipped, so the script can be re-run after an
interruption.

Example::

    python scripts/download_topspin.py --output datasets/topspin
    python scripts/download_topspin.py --output datasets/topspin --scenes soho-perpendicular-fast
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request

import numpy as np

RECORD = "https://zenodo.org/api/records/10884694"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="scene names to fetch (default: all 72)")
    return parser.parse_args()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(url: str, destination: Path, size: int, checksum: str, attempts: int = 5) -> None:
    """Download and verify against Zenodo's size and MD5 (retry on any mismatch)."""

    expected = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None
    for attempt in range(1, attempts + 1):
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out, length=1 << 20)
            if temporary.stat().st_size != size:
                raise OSError(f"size {temporary.stat().st_size} != {size}")
            if expected is not None and _md5(temporary) != expected:
                raise OSError("md5 mismatch")
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
        tar.extractall(root, filter="data")


def convert(scene_dir: Path) -> int:
    csv_path = scene_dir / "events.csv"
    data = np.loadtxt(csv_path, delimiter=",", dtype=np.int64, ndmin=2)
    np.savez(scene_dir / "events.npz", t=data[:, 0], x=data[:, 1].astype(np.int16),
             y=data[:, 2].astype(np.int16), p=data[:, 3].astype(np.uint8))
    csv_path.unlink()
    return int(data.shape[0])


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(RECORD, timeout=60) as response:
        record = json.load(response)
    files = {item["key"][: -len(".tar.gz")]: item for item in record["files"]
             if item["key"].endswith(".tar.gz")}
    names = sorted(files) if not args.scenes else args.scenes
    unknown = set(names) - set(files)
    if unknown:
        raise ValueError(f"unknown scenes: {sorted(unknown)}")
    for index, name in enumerate(names, 1):
        scene_dir = args.output / name
        if (scene_dir / "events.npz").is_file() and (scene_dir / "calib.txt").is_file():
            print(f"[{index}/{len(names)}] {name}: present", flush=True)
            continue
        archive = args.output / f"{name}.tar.gz"
        started = time.monotonic()
        archive.unlink(missing_ok=True)  # a leftover from an interrupted run
        _fetch(f"{RECORD}/files/{name}.tar.gz/content", archive,
               int(files[name]["size"]), files[name].get("checksum", ""))
        _safe_extract(archive, args.output)
        archive.unlink()
        events = convert(scene_dir)
        print(f"[{index}/{len(names)}] {name}: {events} events, "
              f"{files[name]['size'] / 1e6:.0f} MB in {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
