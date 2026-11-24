# TOPSPIN reference evidence — 2026-11-24

This directory preserves the compact, machine-readable evidence for the first
audited reference-profile evaluation after Gate 0/B conditional approval.

## Frozen execution identity

- Git commit: `34f18493ba1d0f37da317445bd405ca7635b8c5a`
- source-tree SHA-256:
  `16ab88a6b085fb9aabdf3cb9cddb7273e1c6c7f297e4963789e4ad29a487894c`
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
| canonical dev 3 | 120.29 s | 3/3 | 0/3 | D-003 `passed=false`, eligible 0/3 |
| fixed review 18 | 683.24 s | 12/18 | 0/18 | retrospective implementation review |

The review-18 same-speed consistency median absolute deviation was 20.45% and
mean throughput was 745,216 events/s. Public TOPSPIN has no authoritative
frequency labels or author CAD/scale, so true frequency error and CAD ICP RMSE
remain unavailable. No parameter was selected or modified after these results.

`dev3/` and `review18/` contain the evaluator's exact `metrics.json`,
`per_sequence.csv`, and `run_manifest.json`, plus the launcher transcript and
exit code. Full scene checkpoints, histories, and figures remain on the 4070PC
under the corresponding immutable `runs/20261124_topspin_reference_*`
directories.
