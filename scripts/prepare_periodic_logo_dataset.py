#!/usr/bin/env python3
"""Create an auditable eSpinSLAM view of an event_snn_torch event stream.

The input event NPZ files are already cropped with a single crop fitted on the
training split and then scaled by a fixed factor.  This tool does not crop or
resample them again.  It links those exact files into a dataset view, links the
matching per-sequence Blender CAD assets, and records the transformed camera
intrinsics required by eSpinSLAM.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
import zipfile


EVENT_KEYS = {
    "x.npy", "y.npy", "t_us.npy", "p01.npy", "omega_rad_s.npy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-stream-root", required=True, type=Path)
    parser.add_argument("--cad-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--link-mode", choices=("auto", "hardlink", "copy"), default="auto",
        help="auto uses hard links on one filesystem and copies otherwise",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"manifest has no header: {path}")
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    return list(reader.fieldnames), rows


def _constrained(root: Path, relative: str, *, label: str) -> Path:
    if not relative:
        raise ValueError(f"missing {label}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{label} escapes its dataset root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _validate_event_archive(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            missing = EVENT_KEYS - set(archive.namelist())
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"invalid event NPZ: {path}") from error
    if missing:
        raise ValueError(f"event NPZ {path} is missing {sorted(missing)}")


def _link_or_copy(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode != "copy":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError as error:
            if mode == "hardlink" or error.errno not in {
                errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP,
            }:
                raise
    shutil.copy2(source, destination)
    return "copy"


def _float(row: dict[str, str], name: str) -> float:
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"manifest lacks numeric provenance column {name!r}") from error


def prepare_dataset(
    event_stream_root: Path,
    cad_root: Path,
    output_root: Path,
    *,
    link_mode: str = "auto",
) -> dict[str, Any]:
    event_stream_root = event_stream_root.resolve()
    cad_root = cad_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output_root}")
    partial = output_root.with_name(output_root.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"partial dataset already exists: {partial}")

    source_manifest = event_stream_root / "manifest.csv"
    summary_path = event_stream_root / "precompute_summary.json"
    summary = _load_json(summary_path)
    _, rows = _read_rows(source_manifest)
    if not rows:
        raise ValueError("source manifest is empty")
    fit = summary.get("fixed_crop_fit")
    if not isinstance(fit, dict):
        raise ValueError("precompute summary lacks fixed_crop_fit")
    if not summary.get("complete"):
        raise ValueError("precomputed dataset is not marked complete")
    if summary.get("storage_format") != "event-stream":
        raise ValueError("only the event-stream representation is accepted")
    if fit.get("fit_split") != "train" or fit.get("uses_validation_or_test"):
        raise ValueError("fixed crop must be fitted on train only")

    crop = fit.get("crop_xyxy")
    scale = float(fit.get("fixed_scale"))
    output_hw = fit.get("output_size_hw")
    if (
        not isinstance(crop, list) or len(crop) != 4
        or not isinstance(output_hw, list) or len(output_hw) != 2
        or not 0.0 < scale <= 1.0
    ):
        raise ValueError("invalid fixed crop/scale metadata")
    x0, y0, x1, y1 = (int(value) for value in crop)
    height, width = (int(value) for value in output_hw)
    if width != round((x1 - x0) * scale) or height != round((y1 - y0) * scale):
        raise ValueError("output size is inconsistent with crop and fixed scale")

    first = rows[0]
    source_intrinsics = {
        "fx": _float(first, "provenance.camera_intrinsics.fx_px"),
        "fy": _float(first, "provenance.camera_intrinsics.fy_px"),
        "cx": _float(first, "provenance.camera_intrinsics.cx_px"),
        "cy": _float(first, "provenance.camera_intrinsics.cy_px"),
        "width": int(_float(first, "provenance.camera_intrinsics.width_px")),
        "height": int(_float(first, "provenance.camera_intrinsics.height_px")),
    }
    transformed_intrinsics = {
        "fx": source_intrinsics["fx"] * scale,
        "fy": source_intrinsics["fy"] * scale,
        "cx": (source_intrinsics["cx"] - x0) * scale,
        "cy": (source_intrinsics["cy"] - y0) * scale,
        "width": width,
        "height": height,
    }

    split_counts: dict[str, int] = {}
    link_counts = {"hardlink": 0, "copy": 0}
    output_rows: list[dict[str, str | int]] = []
    partial.mkdir(parents=True)
    for index, row in enumerate(rows, 1):
        sequence_id = row.get("sequence_id", "")
        split = row.get("split", "")
        if not sequence_id or not split:
            raise ValueError(f"source row {index} lacks sequence_id or split")
        split_counts[split] = split_counts.get(split, 0) + 1

        event_source = _constrained(
            event_stream_root, row.get("event_path", ""), label="event_path"
        )
        _validate_event_archive(event_source)
        point_source = _constrained(
            cad_root,
            row.get("reference_pointcloud_path", ""),
            label="reference_pointcloud_path",
        )
        mesh_source = _constrained(
            cad_root,
            row.get("reference_mesh_path", ""),
            label="reference_mesh_path",
        )

        event_relative = Path("events") / f"{sequence_id}.npz"
        point_relative = Path("cad") / sequence_id / "reference_points.npz"
        mesh_relative = Path("cad") / sequence_id / "target_mesh.ply"
        for source, relative in (
            (event_source, event_relative),
            (point_source, point_relative),
            (mesh_source, mesh_relative),
        ):
            actual_mode = _link_or_copy(source, partial / relative, link_mode)
            link_counts[actual_mode] += 1

        output_rows.append({
            "sequence_id": sequence_id,
            "event_path": event_relative.as_posix(),
            "split": split,
            "object_id": row.get("object_id", sequence_id),
            "scene_id": row.get("scene_id", sequence_id),
            "reference_pointcloud_path": point_relative.as_posix(),
            "reference_mesh_path": mesh_relative.as_posix(),
            "event_sha256": row.get("cache_sha256", ""),
            "event_count": row.get("cache_events", ""),
            "source_sample_index": row.get(
                "provenance.generator.sample_index", ""
            ),
        })

    expected_splits = summary.get("split_counts")
    if isinstance(expected_splits, dict) and split_counts != expected_splits:
        raise ValueError(
            f"split counts disagree with precompute summary: {split_counts} != {expected_splits}"
        )

    fieldnames = list(output_rows[0])
    with (partial / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    metadata = {
        "schema_version": 1,
        "complete": True,
        "purpose": "eSpinSLAM comparison on the exact event_snn_torch ROI event stream",
        "sequence_count": len(output_rows),
        "split_counts": split_counts,
        "event_schema": "event_snn_torch_xy_t_us_p01",
        "event_loader_conversion": {
            "t_seconds": "t_us * 1e-6",
            "polarity": "p01 is already {0,1}",
            "event_order_changed": False,
            "additional_roi_or_resizing": False,
        },
        "roi": {
            "policy": summary.get("roi_policy"),
            "fit_split": fit.get("fit_split"),
            "uses_validation_or_test": fit.get("uses_validation_or_test"),
            "crop_xyxy_source_pixels": crop,
            "fixed_scale_xy": [scale, scale],
            "output_size_hw": output_hw,
        },
        "source_camera_intrinsics": source_intrinsics,
        "espinslam_camera_intrinsics": transformed_intrinsics,
        "distortion": {
            "k1": _float(first, "provenance.camera_intrinsics.distortion.k1"),
            "k2": _float(first, "provenance.camera_intrinsics.distortion.k2"),
            "p1": _float(first, "provenance.camera_intrinsics.distortion.p1"),
            "p2": _float(first, "provenance.camera_intrinsics.distortion.p2"),
        },
        "cad": {
            "usage": "evaluation_only_not_inference",
            "per_sequence_pointcloud": True,
            "per_sequence_mesh": True,
            "coordinate_frame": "TargetRig_COM body frame",
        },
        "provenance": {
            "event_stream_root": str(event_stream_root),
            "event_stream_manifest": str(source_manifest),
            "event_stream_manifest_sha256": _sha256(source_manifest),
            "precompute_summary": str(summary_path),
            "precompute_summary_sha256": _sha256(summary_path),
            "cad_root": str(cad_root),
            "source_raw_manifest_sha256": summary.get("source_manifest_sha256"),
        },
        "storage": {
            "requested_mode": link_mode,
            "file_counts": link_counts,
            "hardlinks_preserve_source_bytes": link_counts["hardlink"] > 0,
        },
    }
    (partial / "dataset_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, output_root)
    return metadata


def main() -> None:
    args = parse_args()
    report = prepare_dataset(
        args.event_stream_root,
        args.cad_root,
        args.output_root,
        link_mode=args.link_mode,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
