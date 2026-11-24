#!/usr/bin/env python3
"""Run train-only tuning, validation audit, and held-out test for orders 0--4.

The driver is intentionally restartable: the tuner resumes its per-sequence
JSONL caches and completed test directories are skipped only after both the
metrics file and completion marker have been written.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--orders", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/20261124_periodic_logo_espinslam_train_eval"),
    )
    return parser.parse_args()


def stamp(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run(command: list[str], *, root: Path, log_path: Path) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    printable = " ".join(command)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] $ {printable}\n")
        log.flush()
        subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    # Keep the virtual-environment entry point itself.  Resolving its symlink
    # would invoke the base interpreter without the venv's site-packages.
    python = str(args.python.expanduser().absolute())
    output_root = (root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    base_config = root / "examples/configs/periodic_logo_roi_base.jsonc"

    for order in args.orders:
        if order not in range(5):
            raise ValueError(f"unsupported periodic order: {order}")
        dataset = root / "datasets" / (
            f"20261118_periodic_logo_order{order}_fixed_global_crop_espinslam_v1"
        )
        manifest = dataset / "manifest.csv"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)

        order_dir = output_root / f"order{order}"
        tuning_dir = order_dir / "tuning_online_v2"
        test_dir = order_dir / "held_out_test"
        order_dir.mkdir(parents=True, exist_ok=True)
        log_path = order_dir / "run.log"
        completed = order_dir / "COMPLETED.json"
        metrics = test_dir / "metrics.json"
        if completed.is_file() and metrics.is_file():
            print(f"order{order}: already complete; skipping", flush=True)
            continue

        stamp(
            order_dir / "RUNNING.json",
            {
                "order": order,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": str(dataset),
                "selection_split": "train",
                "validation_role": "audit_only",
                "evaluation_split": "test",
            },
        )
        print(f"order{order}: train-only tuning and validation audit", flush=True)
        run(
            [
                python,
                "scripts/tune_periodic_logo.py",
                "--config",
                str(base_config),
                "--data-root",
                str(dataset),
                "--manifest",
                str(manifest),
                "--output-dir",
                str(tuning_dir),
            ],
            root=root,
            log_path=log_path,
        )

        print(f"order{order}: independent held-out test", flush=True)
        run(
            [
                python,
                "scripts/evaluate.py",
                "--config",
                str(tuning_dir / "selected_config.json"),
                "--data-root",
                str(dataset),
                "--manifest",
                str(manifest),
                "--output-dir",
                str(test_dir),
            ],
            root=root,
            log_path=log_path,
        )
        if not metrics.is_file():
            raise RuntimeError(f"evaluation did not create {metrics}")
        stamp(
            completed,
            {
                "order": order,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "metrics": str(metrics),
            },
        )
        (order_dir / "RUNNING.json").unlink(missing_ok=True)
        print(f"order{order}: complete", flush=True)

    stamp(
        output_root / "ALL_COMPLETED.json",
        {
            "orders": args.orders,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print("all requested orders complete", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"fatal: {error}", file=sys.stderr, flush=True)
        raise
