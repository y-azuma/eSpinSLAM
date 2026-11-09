#!/usr/bin/env python3
"""Evaluate eSpinSLAM on the explicit held-out split and create runs/1110."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from espinslam.config import load_config
from espinslam.evaluation import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--reference-pointcloud", type=Path)
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def _log_wandb(config, output_dir: Path, report: dict) -> str | None:
    if not config.logging.wandb:
        return None
    import wandb
    run = wandb.init(
        project=config.logging.wandb_project,
        entity=config.logging.wandb_entity,
        name=config.logging.wandb_run_name,
        tags=config.logging.wandb_tags,
        mode=config.logging.wandb_mode,
        job_type="espinslam-held-out-evaluation",
        config=config.as_dict(),
    )
    flat = {
        f"test/{name}": value
        for name, value in report["rotation"].items()
        if isinstance(value, (int, float)) and value is not None
    }
    flat.update({
        f"pointcloud/{name}": value
        for name, value in report["pointcloud"].items()
        if isinstance(value, (int, float)) and value is not None
    })
    flat["runtime_seconds"] = report["runtime_seconds"]
    for path in sorted(output_dir.glob("*.png")):
        flat[f"plots/{path.stem}"] = wandb.Image(str(path))
    run.log(flat)
    url = run.url
    run.finish()
    return url


def main() -> None:
    args = parse_args()
    if args.max_sequences < 0:
        raise ValueError("--max-sequences must be non-negative")
    config = load_config(args.config)
    if args.wandb:
        config.logging.wandb = True
    reference = args.reference_pointcloud
    if reference is None:
        default_reference = Path("/workspace/reference/reference_points.npz")
        reference = default_reference if default_reference.is_file() else None
    report = evaluate(
        config, args.data_root, args.manifest, args.output_dir,
        maximum_sequences=args.max_sequences,
        reference_path=reference,
    )
    wandb_url = _log_wandb(config, args.output_dir, report)
    if wandb_url:
        report["wandb_url"] = wandb_url
        metrics_path = args.output_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

