#!/usr/bin/env python3
"""Add selective and forced-output comparison metrics to existing evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from espinslam.evaluation import rotation_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dirs", nargs="+", type=Path)
    return parser.parse_args()


def optional_float(value: str) -> float:
    return math.nan if value.strip() == "" else float(value)


def main() -> None:
    args = parse_args()
    for output_dir in args.output_dirs:
        predictions_path = output_dir / "predictions.csv"
        metrics_path = output_dir / "metrics.json"
        with predictions_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        truth = np.asarray([float(row["omega_true_rad_s"]) for row in rows])
        accepted = np.asarray([optional_float(row["omega_pred_rad_s"]) for row in rows])
        forced = np.asarray([
            prediction if math.isfinite(prediction)
            else 2.0 * math.pi * optional_float(row["latest_unconfirmed_frequency_hz"])
            for prediction, row in zip(accepted, rows, strict=True)
        ])
        accepted_metrics = rotation_metrics(truth, accepted, len(rows))
        forced_metrics = rotation_metrics(truth, forced, len(rows))
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        report["rotation"] = accepted_metrics
        report["rotation_accepted_only"] = accepted_metrics
        report["rotation_forced_output"] = forced_metrics
        report["comparison_summary"] = {
            "accepted_sequence_count": accepted_metrics["valid_count"],
            "accepted_ratio": accepted_metrics["coverage"],
            "accepted_mae_rad_s": accepted_metrics["mae_rad_s"],
            "accepted_mae_deg_s": (
                None if accepted_metrics["mae_rad_s"] is None
                else math.degrees(accepted_metrics["mae_rad_s"])
            ),
            "forced_output_sequence_count": forced_metrics["valid_count"],
            "forced_output_ratio": forced_metrics["coverage"],
            "forced_output_mae_rad_s": forced_metrics["mae_rad_s"],
            "forced_output_mae_deg_s": (
                None if forced_metrics["mae_rad_s"] is None
                else math.degrees(forced_metrics["mae_rad_s"])
            ),
            "no_frequency_estimate_count": len(rows) - forced_metrics["valid_count"],
            "forced_output_definition": (
                "confirmed estimate when accepted, otherwise the latest valid "
                "unconfirmed frequency; sequences with no frequency estimate remain missing"
            ),
        }
        metrics_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"updated {metrics_path}")


if __name__ == "__main__":
    main()
