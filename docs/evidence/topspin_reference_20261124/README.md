# TOPSPIN reference evidence — 2026-11-24

This directory preserves the machine-readable evidence for the audited
reference-profile evaluation after Gate 0/B conditional approval and the Gate
D reporting correction.  The method path is unchanged; commit `6b4868a` adds
only explicit loop-decision/first-BA fields and corrects the consistency
aggregation source.

## Frozen execution identity

- Git commit: `6b4868a474fc2b79eed66552d11b14811a0aecd8`
- source-tree SHA-256:
  `206875ef7da611d9e614b7e73126375b42b1ab28c8b1c1e46769389150e7fcc2`
- dependency-lock SHA-256:
  `3e90af4a5ed0c2cc58f3250e2c68394fa785530d4016b017266e1b2927859d1e`
- host: `azumayudai-B660M-Pro-RS`
- platform: Linux 6.8.0-149, x86_64, Python 3.10.14
- compute: 16 logical CPUs, 33,475,567,616-byte RAM, RTX 4070 12,282 MiB
- data: verified public TOPSPIN conversion from Zenodo record 10884694
- method: exact `topspin_reference.jsonc`, verified by `--require-reference`

## Results

| Scope | Runtime | Loop | Valid reconstruction | Frozen decision |
|---|---:|---:|---:|---|
| canonical dev 3 | 120.09 s | 3/3 | 0/3 | D-003 `passed=false`, eligible 0/3 |
| fixed review 18 | 691.61 s | 12/18 | 0/18 | retrospective implementation review |

The corrected review-18 same-speed consistency median absolute deviation is
`2.9221%`, computed from the frequency that first closed each loop, not from
the last valid online estimate after BA/outlier elimination changed `H`.  The
former `20.45%` value was invalid for this purpose because it measured the
post-BA collapse.  Mean throughput is 734,302 events/s. Public TOPSPIN has no
authoritative frequency labels or author CAD/scale, so true frequency error
and CAD ICP RMSE remain unavailable. No method parameter was selected or
modified after these results.

`dev3/` and `review18/` contain the evaluator's exact `metrics.json`,
`per_sequence.csv`, and `run_manifest.json`, plus the launcher transcript and
exit code.  Their `history/` directories contain every frequency-estimation
and optimisation record needed to audit the observed failure chain. Full scene
checkpoints and figures remain on the 4070PC under the corresponding immutable
`runs/20261124_topspin_reference_*_r1r2_6b4868a` directories.
