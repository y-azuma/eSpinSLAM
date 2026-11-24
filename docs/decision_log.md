# Paper-faithful reproduction decision log

Status: **preimplementation decisions frozen 2026-11-24; history correction and later evidence separated**
Agreed by: user direction, Codex, and Claude Code (session discussion)  
Scope: eSpinSLAM reproduction only; not a method-improvement project

Changes to a frozen primary interpretation require a dated entry that cites a
stronger paper/citation source or a reproducible code defect. Better TOPSPIN
performance is not an admissible reason.

The viability minimums in D-003 and numerical equivalence thresholds in D-006
are **provisional engineering definitions, frozen before implementation**.
They are not values claimed by the paper. "Provisional" means they still need
implementation evidence; "frozen" means they cannot be moved after seeing a
favorable or unfavorable evaluation result without a dated cross-review.

## Frozen decisions

### D-001 - Reproduction boundary

- Implement the method described by the paper as literally as the available
  information permits.
- Do not add mechanisms whose purpose is to exceed the paper's performance.
- Put harmonic disambiguation/abstention, overlap-restricted Eq. (9), the
  15%-near-best shortest-period rule, parallax guards, terminal offline BA,
  and a defensive "safe" backend in Future Work or an experimental namespace.
- Only the reference path may populate paper-comparison tables.

### D-002 - Evidence hierarchy for ambiguous details

Primary interpretations are selected in this order:

1. displayed equation or explicit paper prose;
2. paper figure/table;
3. definition in a cited source;
4. established general implementation convention;
5. frozen viability selection on the designated dev scenes.

Algorithmic interpretations are never selected by empirical performance.
A stronger cited source found before implementation may update the matrix only
after Claude/Codex re-review. After implementation begins, changes require a
new dated entry and cannot be justified by dev/test performance alone.

### D-003 - Missing mandatory numeric values

For a parameter missing from the paper and citations:

1. Before observing new results, commit its finite candidate set, evaluation
   order, and the concrete meaning of "neutral/simple."
2. The provisional minimum output is met when the algorithm is numerically
   defined and at least **two of the three** dev scenes each produce:
   - a detected loop;
   - at least four inlier landmarks after outlier removal; and
   - at least one actual BA, with the final termination status neither
     `not_run` nor `max_iterations`.
3. Before the first **audited reference-profile D-003 dev benchmark**, a
   ground-truth-free artefact rule is frozen for the `aleph=20` estimates that
   first satisfy the paper's loop-stability test.  Historical exploratory
   TOPSPIN/dev artifacts had already been viewed and remain explicitly
   excluded; this is not a claim that the public data were unseen.  Record the
   observed span and `T*/span` on every update.  The raw loop
   output is preserved but does **not** count toward the two-of-three minimum
   when either of the following holds for every estimate in that confirming
   window:
   - `T*` is at the lower or upper search endpoint (within the already frozen
     `1e-5 s` ICP tolerance); or
   - `T*` is within one nominal geometric-grid step of `span/2`, i.e. its
     multiplicative distance from `span/2` is at most `1 + 0.01`.
   Such a loop is reported as `loop_artefact_suspected`; the estimator,
   stability test, and reconstruction path are not changed.  A guard or
   harmonic correction is Future Work, not part of this reproduction.
4. `camera-topdown-fast` is marked near the Sec. 4.3 degeneracy. Failure on it
   alone does not disqualify a candidate; its outcome remains reported.
5. Do not rank viable candidates by frequency error, reprojection error,
   landmark count, CAD alignment, loop time, or throughput.
6. If more than one candidate is viable, choose by: value supported by a cited
   reference, then library default, then the predeclared neutral/simple value.
7. Freeze the selected value. Sensitivity results do not retroactively change
   the primary interpretation.

The parameter registry below is an execution gate. Values resolved by a cited
primary source must be used directly and must not be reopened as a dev sweep.
For any future mandatory value still absent from both the paper and its cited
sources, no dev selection may begin until its finite candidates and neutral
choice are appended in a dated entry. Earlier values selected by
landmark/reprojection performance (`3,2`; `5 px`; `0.1 s`; density radius
`5`) are not grandfathered in.

### D-004 - Eq. (9) and exact acceleration

- Reference support is all `A_l` against all `A_r`; the objective is the sum
  of Euclidean norms; no overlap mask, subsampling, or harmonic preference.
- Primary time interpretation is seconds without an additional scale
  (`lambda_t=1 px/s`). This is **ambiguous**, not paper-exact.
- The primary global initialization uses a geometric period grid over
  `[0.1 s, observed span]`, then 1-D ICP from the lowest-cost point. The
  `0.1 s` bound and grid are **primary interpretations**, not paper values.
- The frozen grid is geometric with a nominal `1%` multiplicative step and
  includes both interval endpoints exactly. `min_samples_per_half=50` is a
  numerical validity guard, not a paper value. The reference profile leaves
  `max_period_seconds` unset and searches through the full observed span.
  A finite maximum is permitted only in an explicitly experimental profile,
  where it intersects the observed-span upper bound.
- ICP is capped at `30` iterations with shift tolerance `1e-5 s`. Period
  clipping to the declared search interval and an IRLS weight floor of
  `1e-12` are deterministic numerical protections, **Not-stated** by the
  paper. Every estimate records convergence versus the iteration cap.
- Batching, exact grid lookup, and parallel execution are permitted only if
  brute-force tests show equivalent neighbor choices (allowing exact-distance
  ties) and numerical distances within the declared tolerance.
- Any approximation is excluded from the reproduction result.

### D-005 - Frequency scheduling

The phrase "whenever H is updated" is interpreted operationally as an `H`
update trigger batched every `Delta=0.1 s` of event time. `Delta=0.1 s` is a
primary interpretation, not an exact paper setting. Literal every-update
execution is tested on small synthetic input, and `{0.05, 0.5, 1.0}` are
sensitivity values. Sensitivity cannot change the frozen primary because it
performs better.

### D-006 - Solver fidelity gate

The paper explicitly uses GTSAM. Before measurement, the custom LM was allowed
to serve as the reference implementation only if, on a frozen small synthetic
problem, it agreed with a direct GTSAM `CustomFactor` implementation within
all of:

- final objective relative difference `<= 1e-8`;
- `R0` geodesic difference `<= 1e-6 rad`;
- point-set relative difference `<= 1e-6` under identical correspondence and
  variable ordering.

The deterministic Gate C problem is defined in
`tests/test_gtsam_equivalence.py` (random seed `7305`). The thresholds above,
not the measurements below, are the frozen preimplementation decision.

#### Gate C evidence (recorded after implementation)

The direct comparison measured:

| Quantity | Direct GTSAM | In-tree LM | Difference | Frozen threshold | Result |
|---|---:|---:|---:|---:|---|
| Final objective | `7.422517109231391` | `7.422517109231421` | relative `4.07e-15` | `<=1e-8` | Pass |
| `R0` | - | - | geodesic `1.71e-11 rad` | `<=1e-6 rad` | Pass |
| Points | - | - | relative `1.77e-10` | `<=1e-6` | Pass |
| Iterations | `5` | `5` | `0` | diagnostic only | Match |

Gate C therefore passes. The paper-reference configuration is frozen as
`backend.solver_backend="equivalent_custom"`; `"gtsam"` remains an optional
direct execution/audit path. This decision follows the predeclared thresholds,
not better TOPSPIN performance. It does not claim bitwise equality outside the
frozen deterministic problem.

The tested direct environment uses GTSAM `4.2` with NumPy `<2`. The GTSAM
equivalence module is optional: `tests/test_gtsam_equivalence.py` uses
`pytest.importorskip("gtsam")` and therefore reports a skip when that dependency
is absent. A skip preserves ordinary installation compatibility but is not new
evidence for the gate; release/reproduction environments that claim direct
equivalence must run the test with the compatible optional dependency present.

### D-007 - Fixed review subset

Implementation/review iterations use a frozen 18-scene subset, separate from
the three dev scenes and from the previously diagnosed
`soho-perpendicular-fast`. Construction is deterministic:

- objects in order: Camera, Dualshock (`controller` directory prefix), Hubble,
  SOHO, Switch, TDRS;
- speeds in order: slow, medium (`med` suffix), fast;
- perspectives in order: diagonal, perpendicular, sideon, topdown;
- initial perspective index is `(object_index + speed_index) mod 4`;
- if that sequence is a dev or diagnostic scene, advance cyclically until it
  is not.

Frozen list:

1. `camera-diagonal-slow`
2. `camera-perpendicular-med`
3. `camera-sideon-fast`
4. `controller-perpendicular-slow`
5. `controller-sideon-med`
6. `controller-topdown-fast`
7. `hubble-sideon-slow`
8. `hubble-topdown-med`
9. `hubble-diagonal-fast`
10. `soho-topdown-slow`
11. `soho-diagonal-med`
12. `soho-sideon-fast`
13. `switch-diagonal-slow`
14. `switch-perpendicular-med`
15. `switch-sideon-fast`
16. `tdrs-perpendicular-slow`
17. `tdrs-topdown-med`
18. `tdrs-topdown-fast`

These results may reveal bugs but may not be used to choose parameters or
reinterpret the method.

### D-008 - Full TOPSPIN compute rule

1. Benchmark one dev sequence on the recorded execution environment (CPU,
   core count, memory, OS, Python, and dependency lock).
2. Use only proven-equivalent exact acceleration.
3. During review iterations, run the fixed 18 scenes with checkpoint/resume.
4. After A-D receive mutual approval, run all 72 TOPSPIN scenes once on the
   final candidate, also checkpointable.
5. If the measured projection is over 48 hours for 72 scenes, or any one
   scene exceeds two hours, do not silently introduce an approximation. Report
   the measurement and estimate to the user. Only after that report may the
   fixed 18 become the formal exploratory scope, labeled a computational
   reproduction limit.

### D-009 - Evaluation terminology and unavailable ground truth

- Synthetic GT results are **synthetic verification**, not proof of paper
  performance.
- Existing and new public-TOPSPIN analysis is **exploratory / retrospective**.
- Same-speed median deviation is **consistency error**, not true frequency MAE.
- Values read from paper Fig. 6 are **figure-proxy error**, not ground truth.
- ICP RMSE in millimeters and true frequency error are **unavailable** without
  authoritative CAD, scale, and frequency annotations.
- "Confirmatory" is reserved for previously unseen preregistered data or
  author-provided GT/CAD not inspected during method decisions.

### D-010 - Review and merge gate

Codex implements stages A-D. After every stage it supplies code, tests,
matrix rows, and results to Claude Code. Claude reviews paper fidelity,
performance-motivated deviation, exact-equivalence claims, tests, and docs.
Codex revises and resubmits until both approve. Disagreement is shown to the
user with both arguments. Nothing merges to `main` until:

- A-D are mutually approved;
- supported CI jobs pass;
- the matrix and reproduction report are complete;
- results have been reviewed for code/protocol consistency; and
- the user gives final approval.

### D-011 - Adapted eSfO front-end source audit

eSpinSLAM says that its online front-end adapts eSfO. The cited primary source,
Elms et al., [*Event-based Structure-from-Orbit*](https://arxiv.org/abs/2405.06216)
(arXiv:2405.06216 / CVPR 2024), therefore has hierarchy level 3 under D-002.
Its Secs. 4.1-4.2, Eq. (3), and Table 2 supply the following cited values and
mechanics without a dev sweep. The offline-verifiable page/section/table map,
short source fragments, and PDF hash are recorded in
[`cited_source_audit.md`](cited_source_audit.md):

The papers name eFAST, but this audit has not pinned an authoritative eFAST
implementation revision/hash or a golden-output fixture. The in-tree arc and
polarity tests are behavioral evidence only. eFAST implementation fidelity
therefore remains **Ambiguous**; no behavior is changed or TOPSPIN-tuned on
that basis.

- density radius `lambda=7`, with the mean density computed separately for
  positive- and negative-polarity corners;
- polarity-agnostic HDBSCAN with reported `minPts=10` and cluster-selection
  epsilon `5`; the Python mapping is `min_cluster_size=10`,
  `cluster_selection_epsilon=5`, and no separately invented `min_samples`
  value (`None` in the reference configuration);
- cluster-tail/head descriptors are means of the last/first `N_sigma=5`
  events in `(t,x,y)`;
- association selects the nearest head in the forward spatio-temporal
  hemisphere of radius `phi=30`.

The entire transfer from offline eSfO clustering to eSpinSLAM's approximately
1,000-event online chunks is **Ambiguous (adapted)**. The constants are cited;
their chunk-local behavior is not Exact. If this literal cited configuration
does not meet D-003 on the dev scenes, record a reproduction failure and do
not search alternative front-end constants. Three additional adaptation details
remain Ambiguous, but no longer trigger parameter selection by accuracy:

1. Neither source declares how timestamp values are scaled relative to pixels
   in the Euclidean `(t,x,y)` distance. The frozen primary representation is
   timestamp seconds with multiplier `1`; alternatives are sensitivity-only.
   Under the literal `(t,x,y)` Euclidean gate and `phi=30`, a near-zero spatial
   displacement can connect events separated by almost 30 seconds and hence
   across several rotations. This is a known limitation of the frozen units.
2. Eq. (6) describes events within distance `lambda` from an event tuple, but
   neither source identifies whether that distance is spatial `(x,y)` or
   spatio-temporal `(t,x,y)`. The frozen primary uses spatial `(x,y)` with
   `lambda=7`; this is an interpretation, not an Exact source-derived fact.
3. Offline eSfO does not specify how simultaneous current-chunk tracklets
   competing for one global online track are resolved. The frozen primary is
   deterministic nearest-first, many-to-one merging. Greedy one-to-one and
   Hungarian matching are experimental alternatives.

The cited hemisphere is the reference association gate; an independently
tuned `max_gap` must not become an additional reference-path acceptance rule.
Operational pruning may exist only if it is proven not to alter the cited
head/tail matches on the retained pool.

## Dev scenes and their roles

| Scene | Role | Special handling |
|---|---|---|
| `hubble-perpendicular-slow` | missing-value viability only | Never used to optimize an accuracy metric. |
| `tdrs-sideon-med` | missing-value viability only | Never used to optimize an accuracy metric. |
| `camera-topdown-fast` | viability plus degeneracy diagnostic | Sec. 4.3 near-degeneracy; two-of-three rule prevents this one case from forcing a value. |
| `soho-perpendicular-fast` | previously diagnosed, excluded from dev/review subset | Report separately; never tune on it. |

## Resolved pre-run parameter registry

The cited-source audit resolves the prior blockers. The remaining dimensional
and collision choices are frozen interpretations, not performance-selected
candidates:

| Parameter | Candidate set | Evaluation order | Predeclared neutral tie-break | Status |
|---|---|---|---|---|
| eFAST implementation identity | In-tree published-criterion implementation | Behavioral fixtures only; no TOPSPIN sweep | Not applicable | **Ambiguous implementation fidelity - official revision/hash/golden output not pinned** |
| Eq. (6) density radius, metric, and mean | `{lambda=7}`, spatial `(x,y)`, same-polarity mean | Radius/mean are cited; spatial metric and chunk adaptation are frozen without a sweep | Not applicable | **Resolved cited values / Ambiguous metric and adapted use - eSfO Sec. 4.1/Table 2** |
| HDBSCAN parameters/polarity | `{min_cluster_size=10, cluster_selection_epsilon=5, min_samples=None}`, polarity ignored | Direct cited-source values; no sweep; chunk adaptation is not exact | Not applicable | **Resolved cited values / Ambiguous adapted use - eSfO Sec. 4.2/Table 2** |
| HDBSCAN/association time-coordinate scale | Primary `{timestamp seconds x 1}`; raw-unit or normalized alternatives are sensitivity-only | Primary fixed; any sensitivity must be preregistered and cannot replace it by performance | Multiplier `1` | **Frozen Ambiguous** |
| Track association descriptor/gate | `{N_sigma=5, phi=30}`, nearest forward hemisphere, no independent tuned max-gap | Direct cited-source values; no sweep; online chunk/global-pool adaptation remains ambiguous | Not applicable | **Resolved cited mechanics / Ambiguous adapted use - eSfO Sec. 4.2/Eq. (3)/Table 2** |
| Online association collision | Primary `{deterministic nearest-first, many-to-one}` | Primary only for reference result | Stable distance then input-order tie break | **Frozen Ambiguous** |

This table prohibits performance-driven trial and error. It is not permission
to reuse earlier empirically optimized values.

## Change log

| Date | Decision | Reason category | Agreed by |
|---|---|---|---|
| 2026-11-24 | D-001 through D-010 frozen; implementation not yet claimed complete. | Paper text and cross-review | User direction, Codex, Claude Code |
| 2026-11-24 | D-011 records the cited eSfO front-end constants/mechanics; their online chunk adaptation, time scaling, and collision handling remain frozen ambiguities. | Primary cited-source audit: eSfO Secs. 4.1-4.2, Eq. (3), Table 2 | Gate 0 review |
| 2026-11-24 | D-006 Gate C passed; `equivalent_custom` selected by the frozen thresholds, with optional direct `gtsam`. | Deterministic direct-GTSAM equivalence measurement | Gate C implementation evidence |
| 2026-11-24 | **History correction:** commit `9bb723d` bundled the already-agreed D-006 thresholds with later Gate C measurements and references that did not yet exist at that commit. The thresholds were agreed during the external Claude/Codex session before implementation, but that commit does not independently prove the ordering. History is not rewritten; measured values are now explicitly separated under “Gate C evidence.” | Audit-trail correction; no method or threshold changed | Claude review response and Codex correction |
| 2026-11-24 | Reclassified cited eSfO constants in the online chunked front end from Exact to **Ambiguous (adapted)**; froze “report reproduction failure, do not sweep” if cited values fail D-003; documented the seconds/`phi=30` long-time association limitation. | Cross-review of offline cited method versus online adaptation | Claude review response and Codex correction |
| 2026-11-24 | Recorded unresolved eFAST source identity and Eq. (6) distance-domain ambiguity; retained existing behavior without performance-motivated changes. | Independent source-fidelity audit | Codex pre-review audit |
| 2026-11-24 | Before the first audited reference-profile D-003 dev benchmark, froze the ground-truth-free search-boundary / `span/2` loop-artefact diagnostic. Historical exploratory TOPSPIN/dev artifacts were already known and stay excluded; no claim of unseen public data is made. Raw inference is unchanged; suspected loops are reported but excluded from the two-of-three viability count. | Literal Eq. (9) synthetic-path audit; no new audited dev benchmark executed | Claude re-review request B2 and Codex implementation |

