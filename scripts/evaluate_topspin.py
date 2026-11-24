#!/usr/bin/env python3
"""Run the exploratory public-TOPSPIN reproduction protocol.

Public TOPSPIN has no authoritative frequency labels or author CAD, so those
paper metrics remain unavailable. User-supplied provenance may be inspected
only in separately labelled exploratory runs; the strict public-data audit
does not promote a manifest string into author ground truth. The already
inspected public scenes are retrospective, not a new held-out confirmation
set.

Example::

    python scripts/evaluate_topspin.py \
      --config examples/configs/topspin_reference.jsonc \
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
from espinslam.topspin import ABLATIONS, REVIEW18, evaluate_topspin, evaluate_topspin_ablations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-sequences", type=int, default=0)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--review-subset", action="store_true",
        help="run the frozen 18-scene implementation-review subset",
    )
    selection.add_argument(
        "--scene-list", type=Path,
        help="newline-delimited sequence IDs to run, in the listed order",
    )
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore matching per-scene checkpoints and recompute")
    parser.add_argument(
        "--require-reference", action="store_true",
        help="fail unless the method config uses every frozen reference setting",
    )
    parser.add_argument(
        "--require-verified-topspin", action="store_true",
        help="verify the Zenodo acquisition audit and converted-file hashes",
    )
    parser.add_argument(
        "--paper-comparison", action="store_true",
        help="write paper comparison only for clean, verified, exact all-72 runs; "
             "implies --require-reference and --require-verified-topspin",
    )
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
    require_reference = args.require_reference or args.paper_comparison
    require_verified_data = args.require_verified_topspin or args.paper_comparison
    if args.review_subset:
        sequence_ids = list(REVIEW18)
    elif args.scene_list:
        sequence_ids = [
            line.split("#", 1)[0].strip()
            for line in args.scene_list.read_text(encoding="utf-8").splitlines()
            if line.split("#", 1)[0].strip()
        ]
    else:
        sequence_ids = None
    if args.ablations:
        settings = list(ABLATIONS) if args.ablations == "all" else [
            value.strip() for value in args.ablations.split(",") if value.strip()]
        unknown = set(settings) - set(ABLATIONS)
        if unknown:
            raise ValueError(f"unknown ablation settings: {sorted(unknown)}")
        result = evaluate_topspin_ablations(
            config, args.data_root, args.manifest, args.output_dir,
            settings=settings, max_sequences=args.max_sequences,
            sequence_ids=sequence_ids, resume=not args.no_resume,
            require_reference=require_reference,
            require_verified_data=require_verified_data,
            paper_comparison=args.paper_comparison,
        )
        print((args.output_dir / "table3.md").read_text(encoding="utf-8"))
        print(json.dumps({key: value["table2"] for key, value in result["reports"].items()},
                         ensure_ascii=False, indent=2))
        return
    report = evaluate_topspin(config, args.data_root, args.manifest, args.output_dir,
                              max_sequences=args.max_sequences,
                              sequence_ids=sequence_ids, resume=not args.no_resume,
                              require_reference=require_reference,
                              require_verified_data=require_verified_data,
                              paper_comparison=args.paper_comparison)
    print((args.output_dir / "table2.md").read_text(encoding="utf-8"))
    print(json.dumps({key: report[key] for key in (
        "sequence_count", "loop_closure_rate",
        "frequency_authoritative_median_relative_error_percent",
        "frequency_same_speed_consistency_median_absolute_deviation_percent",
        "mean_throughput_ev_s", "runtime_seconds")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
