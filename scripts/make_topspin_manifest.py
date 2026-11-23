#!/usr/bin/env python3
"""Build a TOPSPIN manifest from the extracted Zenodo archives.

The public TOPSPIN release (Zenodo record 10884694, 72 scenes) unpacks to one
directory per scene, named ``{object}-{perspective}-{speed}``::

    hubble-perpendicular-slow/
        events.csv   # t [us], x, y, polarity (0=off, 1=on); no header
        calib.txt    # fx fy cx cy k1 k2 p1 p2 k3
        meta.json    # {"event-camera": {"frame": {"width": 1280, "height": 720}}}

Objects are mapped to the paper's Table 2 names (``controller`` is the
Dualshock) and ``med`` to ``medium``. Calibration and sensor size are copied
from ``calib.txt`` and ``meta.json``. The release contains neither the
ground-truth rotation frequency nor the CAD models, so ``frequency_hz`` and
``frequency_source`` stay empty. ``--cad-dir`` may populate ``cad_path``, but
such assets are marked ``user_supplied_unverified`` and cannot populate the
paper-comparison ICP metric without explicit author provenance.
Other layouts (single event files named after the scene) also work.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

from espinslam.topspin import TABLE2_OBJECTS

EVENT_SUFFIXES = {".raw", ".dat", ".h5", ".hdf5", ".npz", ".npy", ".csv", ".txt"}
OBJECT_ALIASES = {"controller": "Dualshock", "dualshock": "Dualshock"}
SPEED_ALIASES = {"slow": "slow", "med": "medium", "medium": "medium", "fast": "fast"}
COLUMNS = (
    "sequence_id", "event_path", "split", "object", "perspective", "speed",
    "frequency_hz", "frequency_source", "cad_path", "cad_source", "cad_unit_mm",
    "orbit_radius_mm", "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3",
    "width", "height", "time_unit",
)
CALIBRATION_COLUMNS = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
SCENE_PATTERN = re.compile(
    r"^(hubble|soho|tdrs|camera|controller|dualshock|switch)[-_][a-z0-9]+[-_](slow|med|medium|fast)$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cad-dir", type=Path,
                        help="directory (relative to data-root) holding <object>.* CAD files")
    parser.add_argument("--dev-scenes", type=Path,
                        help="file listing the parameter-tuning scenes (split=dev); "
                             "all other scenes get split=test")
    return parser.parse_args()


def _object(name: str) -> str | None:
    lowered = name.lower()
    for alias, canonical in OBJECT_ALIASES.items():
        if lowered.startswith(alias):
            return canonical
    return next((obj for obj in TABLE2_OBJECTS if lowered.startswith(obj.lower())), None)


def _speed(name: str) -> str:
    tokens = re.split(r"[-_]", name.lower())
    return next((SPEED_ALIASES[token] for token in reversed(tokens) if token in SPEED_ALIASES), "")


def _perspective(name: str) -> str:
    tokens = re.split(r"[-_]", name.lower())
    return tokens[1] if len(tokens) >= 3 else ""


def _calibration(directory: Path) -> dict[str, str]:
    path = directory / "calib.txt"
    if not path.is_file():
        return {}
    values = path.read_text(encoding="utf-8").split()
    if len(values) < 4:
        raise ValueError(f"{path}: expected 'fx fy cx cy [k1 k2 p1 p2 k3]'")
    return {column: str(float(value)) for column, value in zip(CALIBRATION_COLUMNS, values)}


def _frame(directory: Path) -> dict[str, str]:
    path = directory / "meta.json"
    if not path.is_file():
        return {}
    frame = json.loads(path.read_text(encoding="utf-8")).get("event-camera", {}).get("frame", {})
    return {key: str(int(frame[key])) for key in ("width", "height") if key in frame}


def _dev_scenes(path: Path | None) -> set[str]:
    if path is None:
        return set()
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(line)
    return names


def main() -> None:
    args = parse_args()
    root = args.data_root.resolve()
    dev = _dev_scenes(args.dev_scenes)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in EVENT_SUFFIXES or not path.is_file():
            continue
        if path.name in ("calib.txt",):
            continue
        # TOPSPIN keeps one scene per directory with a generic events.csv.
        name = path.parent.name if path.stem.lower() == "events" else path.stem
        obj = _object(name)
        # Only full scene names count, so reports or logs that merely start
        # with an object name (e.g. hubble-..._frequency.csv) are ignored.
        if obj is None or not SCENE_PATTERN.match(name):
            continue
        cad = ""
        if args.cad_dir is not None:
            candidates = [candidate for alias in [obj.lower()] + [
                key for key, value in OBJECT_ALIASES.items() if value == obj]
                for candidate in sorted((root / args.cad_dir).glob(f"{alias}*"),
                                        key=lambda item: item.name.lower())]
            candidates += sorted((root / args.cad_dir).glob(f"{obj}*"))
            if candidates:
                cad = str(candidates[0].resolve().relative_to(root))
        row = {column: "" for column in COLUMNS}
        row.update({
            "sequence_id": name, "event_path": str(path.relative_to(root)),
            "split": "dev" if name in dev else "test", "object": obj,
            "perspective": _perspective(name), "speed": _speed(name), "cad_path": cad,
            "cad_source": "user_supplied_unverified" if cad else "",
            "time_unit": "us" if path.suffix.lower() in (".csv", ".txt") else "",
        })
        row.update(_calibration(path.parent))
        row.update(_frame(path.parent))
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    missing = dev - {row["sequence_id"] for row in rows}
    if missing:
        print(f"warning: dev scenes not found under the data root: {sorted(missing)}")
    missing_calibration = sum(not row["fx"] for row in rows)
    print(f"wrote {len(rows)} rows to {args.output}"
          + (f"; {missing_calibration} rows lack calibration" if missing_calibration else "")
          + f"; {sum(row['split'] == 'dev' for row in rows)} dev rows"
          + "; TOPSPIN does not ship ground-truth frequency or CAD models")


if __name__ == "__main__":
    main()
