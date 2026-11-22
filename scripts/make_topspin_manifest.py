#!/usr/bin/env python3
"""Build a TOPSPIN manifest skeleton from a directory of event files.

The object and speed are inferred from file names such as
``soho-perpendicular-fast.raw`` (case-insensitive). Ground-truth frequency,
CAD paths and calibration are left blank or copied from the optional
arguments; fill in whatever the dataset provides before evaluating.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

from espinslam.topspin import TABLE2_OBJECTS

EVENT_SUFFIXES = {".raw", ".dat", ".h5", ".hdf5", ".npz", ".npy", ".csv", ".txt"}
COLUMNS = ("sequence_id", "event_path", "object", "speed", "frequency_hz", "cad_path",
           "cad_unit_mm", "orbit_radius_mm", "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2",
           "k3", "width", "height", "time_unit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cad-dir", type=Path,
                        help="directory (relative to data-root) holding <object>.* CAD files")
    for name in ("fx", "fy", "cx", "cy"):
        parser.add_argument(f"--{name}", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.data_root.resolve()
    rows = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in EVENT_SUFFIXES or not path.is_file():
            continue
        name = path.stem
        lowered = name.lower()
        objects = [obj for obj in TABLE2_OBJECTS if obj.lower() in lowered]
        if not objects:
            continue
        speed = next((s for s in ("slow", "medium", "fast") if re.search(s, lowered)), "")
        cad = ""
        if args.cad_dir is not None:
            candidates = sorted((root / args.cad_dir).glob(f"{objects[0].lower()}*")) + sorted(
                (root / args.cad_dir).glob(f"{objects[0]}*"))
            if candidates:
                cad = str(candidates[0].resolve().relative_to(root))
        row = {column: "" for column in COLUMNS}
        row.update({
            "sequence_id": name, "event_path": str(path.relative_to(root)),
            "object": objects[0], "speed": speed, "cad_path": cad,
        })
        for column in ("fx", "fy", "cx", "cy"):
            value = getattr(args, column)
            if value is not None:
                row[column] = repr(value)
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}; fill frequency_hz, cad_path and "
          "calibration columns before running scripts/evaluate_topspin.py")


if __name__ == "__main__":
    main()
