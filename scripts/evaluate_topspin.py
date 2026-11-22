#!/usr/bin/env python3
"""Reproduce the paper's TOPSPIN experiments (Table 2, Table 3, Figs. 6-8).

Example::

    python scripts/evaluate_topspin.py \
      --config examples/configs/topspin_evk4.jsonc \
      --data-root datasets/topspin \
      --manifest datasets/topspin/manifest.csv \
      --output-dir runs/topspin

    # Table 3 ablations (writes table3.md next to one sub-directory per setting)
    python scripts/evaluate_topspin.py ... --ablations all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from espinslam.config import load_config
from espinslam.topspin import ABLATIONS, evaluate_topspin, evaluate_topspin_ablations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument(
        "--ablations", default="",
        help="comma-separated subset of {%s} or 'all' to run Table 3" % ",".join(ABLATIONS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_sequences < 0:
        raise ValueError("--max-sequences must be non-negative")
    config = load_config(args.config)
    if args.ablations:
        settings = list(ABLATIONS) if args.ablations == "all" else [
            value.strip() for value in args.ablations.split(",") if value.strip()]
        unknown = set(settings) - set(ABLATIONS)
        if unknown:
            raise ValueError(f"unknown ablation settings: {sorted(unknown)}")
        result = evaluate_topspin_ablations(
            config, args.data_root, args.manifest, args.output_dir,
            settings=settings, max_sequences=args.max_sequences,
        )
        print((args.output_dir / "table3.md").read_text(encoding="utf-8"))
        print(json.dumps({key: value["table2"] for key, value in result["reports"].items()},
                         ensure_ascii=False, indent=2))
        return
    report = evaluate_topspin(config, args.data_root, args.manifest, args.output_dir,
                              max_sequences=args.max_sequences)
    print((args.output_dir / "table2.md").read_text(encoding="utf-8"))
    print(json.dumps({key: report[key] for key in (
        "sequence_count", "loop_closure_rate", "frequency_median_rel_error_percent",
        "mean_throughput_ev_s", "runtime_seconds")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
