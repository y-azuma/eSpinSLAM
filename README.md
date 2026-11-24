# eSpinSLAM reproduction

This repository is an independent reproduction of **Event-based Spinning
Object SLAM** (Elms, Latif, and Chin, ICCV Workshops 2025). The authors' code
is not public, and the paper leaves several consequential implementation
details unspecified. Accordingly, this repository does **not** describe the
whole implementation as paper-exact. It keeps five categories separate:

| Category | Meaning |
|---|---|
| **Exact** | The paper or a cited source specifies enough detail to implement the operation. |
| **Ambiguous** | The operation is described, but a consequential value or convention is omitted; the frozen primary interpretation is documented. |
| **Unavailable** | Public TOPSPIN lacks the ground truth or assets required for the paper metric. |
| **Exploratory** | Synthetic diagnostics or retrospective public-TOPSPIN analysis; not confirmatory evidence. |
| **Future Work / experimental** | Paper-unreported alternatives and proposed improvements, excluded from paper-comparison outputs. |

The frozen decisions and the paper-to-code mapping are in
[`docs/decision_log.md`](docs/decision_log.md) and
[`docs/traceability_matrix.md`](docs/traceability_matrix.md). A narrative
summary is in [`docs/reproduction_report.md`](docs/reproduction_report.md).

## Outcome on public TOPSPIN

The frozen primary interpretation did **not** reproduce the paper's reported
performance. The preregistered D-003 viability check failed with `0/3`
eligible development scenes, and the single authorised all-72 run produced
only `1/72` valid reconstructions. Consequently, Table 2 was not reproduced.
This repository records a paper-faithful reproduction **attempt that failed on
public TOPSPIN**; it is not a refutation of the paper, whose authors' code and
several consequential implementation details are unavailable. See
[`docs/reproduction_report.md` §4.4](docs/reproduction_report.md#44-新reference評価結果)
for the run identity, failure chain, and evidence boundary.

## Implemented pipeline

| Paper location | Implementation | Reproduction boundary |
|---|---|---|
| Sec. 3.1.2, Eqs. (2)-(5) | `espinslam/geometry.py` | Circular camera orbit, constant spin frequency, and constant `R0`. |
| Sec. 4.1 | `espinslam/efast.py`, `espinslam/frontend.py` | Chunked eFAST, density filtering, HDBSCAN tracklets, and online track association. |
| Sec. 4.2.1, Eqs. (7)-(9) | `espinslam/frequency.py`, `espinslam/nn.py` | Reference mode uses all `A_l` events against all `A_r`, the sum of Euclidean nearest-neighbour distances, a global grid argmin, then one-dimensional ICP. |
| Sec. 4.2.2, Eq. (11) | `espinslam/geometry.py`, `espinslam/gtsam_backend.py` | Robust bundle adjustment over `R0` and world points with `f=f*`. |
| Secs. 4.2.3-4.2.7 | `espinslam/backend.py`, `espinslam/pipeline.py` | Paper initialization, triangulation, outlier removal, landmark association, and time-based forgetting. |
| Secs. 5-6 | `espinslam/topspin.py`, `scripts/evaluate_topspin.py` | Restartable exploratory TOPSPIN protocol with provenance and unavailable-metric handling. |

The in-tree Schur/LM solver is allowed in the reference profile only because
it passed thresholds frozen before measurement against a direct GTSAM 4.2
`CustomFactor` formulation. On the frozen deterministic problem, final-cost
relative difference was `4.07e-15`, `R0` difference was `1.71e-11 rad`, and
point relative difference was `1.77e-10` (limits: `1e-8`, `1e-6 rad`, and
`1e-6`). Direct GTSAM remains available through
`backend.solver_backend="gtsam"`. This one test is evidence of numerical
equivalence for the frozen problem, not a claim of bitwise identity on every
scene.

## Frozen primary interpretations

The following values are necessary to execute the method but are not all
stated by the eSpinSLAM paper:

- eSpinSLAM says that its front-end adapts eSfO. The cited eSfO source fixes
  density radius `lambda=7`, a separate density mean per polarity,
  polarity-agnostic HDBSCAN with `minPts=10` and cluster-selection epsilon
  `5`, mean first/last `N_sigma=5` descriptors, and forward-hemisphere
  association radius `phi=30`.
- Neither paper specifies the scale between seconds and pixels in the
  spatio-temporal distances. Reference mode freezes the time multiplier to
  `1` and labels it **Ambiguous**.
- Frequency updates are batched every `0.1 s`, the period grid starts at
  `0.1 s`, and the robust loss is Huber with `k=2 px`. These are frozen
  primary interpretations, not paper-reported values.
- Simultaneous online tracklets may join the same earlier track. This
  deterministic many-to-one collision policy is an online-adaptation
  interpretation; eSfO does not settle it.

No primary interpretation was selected because it improved TOPSPIN accuracy.
The evidence hierarchy is equation/text, figure/table, cited source,
established convention, then predeclared viability only.

Reference output excludes overlap-restricted Eq. (9), event subsampling, the
15%-near-best shortest-period rule, a parallax filter, and terminal offline
bundle adjustment. Historical versions of those mechanisms are retained only
under explicit experimental profiles.

## Public TOPSPIN limitations

The public TOPSPIN release contains 72 event sequences, calibration, and
metadata. It does **not** contain authoritative rotation-frequency labels or
the authors' CAD models. Therefore public-data runs cannot reproduce:

- authoritative-frequency MAE or another true frequency-error metric;
- the paper's CAD ICP RMSE in millimetres; or
- Table 3 rows that require those unavailable assets.

Same-speed agreement is reported only as **same-speed consistency**. Values
read from Fig. 6 are reported only as a **figure-proxy comparison**. Neither
is ground-truth accuracy. Separately acquired GT/CAD may enable exploratory
metrics only with documented provenance. For a verified public-TOPSPIN run,
an `authoritative` manifest string alone is rejected because record 10884694
does not authenticate either asset.

All public TOPSPIN scenes have already been inspected during development, so
their evaluation is exploratory and retrospective. It is not described as a
new held-out or confirmatory result.

## Reproducible setup

Install the locked portable environment. The lock file is a constraints file,
so `pyproject.toml` remains the source of direct dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --constraint requirements/lock.txt -e '.[test,topspin]'
```

GTSAM 4.2 requires NumPy `<2`, whereas the portable lock uses NumPy 2.2.6.
Run the optional direct-GTSAM audit in a separate environment:

```bash
python -m venv .venv-gtsam
. .venv-gtsam/bin/activate
python -m pip install -e '.[test,gtsam]'
python -m pytest -q tests/test_gtsam_equivalence.py
```

Acquire TOPSPIN with published size/MD5 verification. Existing archives may
be reused, but they are verified before extraction. Converted events,
calibration and metadata receive SHA-256/size identities that are rechecked on
every verified resume. The strict path also pins the canonical file-list
fingerprint of Zenodo record 10884694 to
`c97ac90550208bd2225f9d116016cd188b9f9037638ee2c608a376de8f6208e7`:

```bash
python scripts/download_topspin.py \
  --output datasets/topspin \
  --archive-cache datasets/topspin_archives

python scripts/make_topspin_manifest.py \
  --data-root datasets/topspin \
  --output datasets/topspin/manifest.csv \
  --dev-scenes examples/splits/topspin_dev.txt
```

Run the frozen 18-scene implementation-review subset with strict reference
validation and scene-level resume:

```bash
python scripts/evaluate_topspin.py \
  --config examples/configs/topspin_reference.jsonc \
  --data-root datasets/topspin \
  --manifest datasets/topspin/manifest.csv \
  --output-dir runs/topspin_reference_review18 \
  --review-subset --require-reference --require-verified-topspin
```

After the code and protocol review gates have passed, omit `--review-subset`
to run all 72 scenes once. If a measured 4070PC benchmark projects more than
48 hours total, or any scene exceeds two hours, the fixed 18 scenes become the
formal exploratory scope only after the computational limit is reported; the
method is not silently replaced with an approximation.

Only that clean-commit, verified, exact-72 invocation may request the separate
paper-comparison output:

```bash
python scripts/evaluate_topspin.py \
  --config examples/configs/topspin_reference.jsonc \
  --data-root datasets/topspin \
  --manifest datasets/topspin/manifest.csv \
  --output-dir runs/topspin_reference_all72 \
  --paper-comparison
```

`--require-reference` validates the method configuration only;
`--require-verified-topspin` validates the acquired data. `--paper-comparison`
requires both, an exact all-72 selection, an eligible executable method path,
and a clean Git source tree. Review subsets and dirty working trees remain
exploratory and cannot create a paper-comparison table.
At present the eligible paths are the full `eSpinSLAM` setting and the
fixed-frequency `No landmark association` ablation. The FFT construction is
not uniquely specified by the paper, and the custom free-frequency solver has
not passed a direct-GTSAM equivalence gate; neither may be paper-compared.
For a Table-3 run, the formal aggregate uses all 72 scenes and the non-dev 69
are written separately. Public TOPSPIN remains retrospective and does not
authenticate frequency ground truth or author CAD.

`examples/configs/topspin_evk4.jsonc` and
`examples/configs/debris_davis346.jsonc` reproduce earlier historical
experimental behavior. They are not valid inputs to `--require-reference` and
must not populate paper-reference tables.

## Outputs and terminology

The TOPSPIN evaluator writes per-scene checkpoints, partial/final CSV files,
history CSV files, figures, protocol metadata, execution-hardware provenance,
and aggregate Markdown/JSON summaries. Missing authoritative metrics remain
null with a reason; they are never filled with zero or a proxy value.

The fixed review list is
[`examples/splits/topspin_review18.txt`](examples/splits/topspin_review18.txt).
Resume reuses a checkpoint only when its manifest row, setting, complete
configuration, input-file identity, source-tree/commit identity and dependency
lock match. Its required history/figure/point-cloud artifacts are hash-checked.
Each output directory has one immutable run fingerprint; a different run is
rejected rather than deleting or mixing existing results.

## Historical results

Results through commit `e4c75cd` used tuned/approximate choices that are now
classified as experimental, including restricted/subsampled Eq. (9), harmonic
preference, a parallax guard, and terminal refinement. Those results are kept
for archaeology only. They are **not** results of the current frozen reference
profile and are not evidence that the paper was reproduced. See the
"Exploratory and retrospective evidence" section of the reproduction report.

The current 4070PC reference evaluation is intentionally not summarized here
until the final code is mutually reviewed and the recorded run completes.

## Tests

```bash
python -m pytest
```

The test suite covers motion/projection Jacobians, direct-GTSAM equivalence,
Eq. (9) against brute-force nearest neighbours, frequency-history logic,
front-end and pipeline ordering, event decoders, acquisition provenance, and
restart-safe TOPSPIN protocol output. CI exercises the locked portable suite
on Linux and macOS with Python 3.10 and 3.12; the optional GTSAM gate requires
the compatible GTSAM environment.
