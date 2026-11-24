# Exact all-72 source

Copied without modification (except for this provenance note) from
`runs/20261124_topspin_reference_all72_final_a2bdf29/` on the recorded RTX
4070 host.  The invocation was:

```text
.venv/bin/python scripts/evaluate_topspin.py \
  --config examples/configs/topspin_reference.jsonc \
  --data-root /home/azumayudai/Desktop/ws_snn/datasets/TOPSPIN_v1/raw_paper_verified \
  --manifest /home/azumayudai/Desktop/ws_snn/datasets/TOPSPIN_v1/raw_paper_verified/manifest.csv \
  --output-dir runs/20261124_topspin_reference_all72_final_a2bdf29 \
  --require-reference --require-verified-topspin --paper-comparison
```

The evaluator recorded clean Git revision
`a2bdf2985a70fc55da49ed6182104e2f10cb989a`, source-tree SHA-256
`206875ef7da611d9e614b7e73126375b42b1ab28c8b1c1e46769389150e7fcc2`,
72 verified scenes, `paper_comparison_eligible=true`, and exit status `0`.
This was the single exact all-72 run authorized after Gate D.  It is
exploratory/retrospective and cannot alter the preregistered D-003 decision.

The `checkpoints/` directory is retained on the source host for restart audit
but is not duplicated here.  All aggregate files, per-scene histories,
tables, and generated figures are preserved in this directory.
