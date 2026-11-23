# eSpinSLAM

Paper-faithful reimplementation of **Event-based Spinning Object SLAM**
(Elms, Latif, and Chin, ICCV Workshops 2025). The authors' code is not public,
so this repository follows the paper section by section and records every
place where the paper leaves a detail open (see
[Interpretation choices](#interpretation-choices)).

## Pipeline and paper mapping

| Paper | Module | What is implemented |
|---|---|---|
| Sec. 3.1.2, Eqs. 2-5 | `espinslam/geometry.py` | Circular orbit `c(t) = r[cos 2πft, sin 2πft, 0]`, camera looking at the orbit centre, constant `R0` |
| Sec. 4.1 | `espinslam/efast.py`, `espinslam/frontend.py` | Event chunks (`|C| = 1000`), eFAST corners (rpg_corner_events criterion, numba), same-polarity density filter (Eq. 6), HDBSCAN spatio-temporal tracklets, association/merging into the global pool `H`; tracks store **events** in a `σ_retain` circular buffer |
| Sec. 4.2.1, Eqs. 7-9 | `espinslam/frequency.py` | Midlife split of all track events, temporal-shift alignment solved by 1-D ICP, loop detection when the last `ℵ` estimates have std `< σ` |
| (speed) | `espinslam/nn.py` | Exact nearest neighbours for Eq. 9 (numba grid with warm starts); distances are bit-identical to SciPy's k-d tree, so results do not change |
| Sec. 4.2.2, Eq. 11 | `espinslam/geometry.py` (`bundle_adjust`) | Robust (Huber, 2 px) optimisation of `R0` and all world points with `f = f*`; every retained track event is a virtual-camera observation; runs every `ε/f` seconds |
| Sec. 4.2.3 | `espinslam/backend.py` | First optimisation from world points at the origin and `R0 = I` |
| Sec. 4.2.4 | `espinslam/backend.py` | Multi-view triangulation of new/updated tracks with the latest `R0`, `f*` |
| Sec. 4.2.5 | `espinslam/backend.py` | Drop tracks with `< N_match` events, landmarks with mean reprojection error `> γ`, points farther than `r` |
| Sec. 4.2.6 | `espinslam/backend.py`, `TrackPool.merge` | Tracks whose world points are within `δ_fuse` are merged (`H_u2 = H_u1 ∪ H_u2`) and re-optimised |
| Sec. 4.2.7 | `espinslam/pipeline.py`, `TrackPool.forget` | Tracks older than `σ_forget · t_rev` (Eq. 12) are discarded |
| Fig. 3 | `espinslam/pipeline.py` | Online, chunk-driven loop: front-end → frequency/loop detection → periodic back-end |
| Table 1 | `espinslam/config.py` | Every hyperparameter has a key with the paper's default (mapping table in the module docstring) |
| Tables 2-3, Figs. 6-8 | `espinslam/topspin.py`, `scripts/evaluate_topspin.py` | TOPSPIN evaluation and ablations |

GTSAM is replaced by a Levenberg-Marquardt solver with Huber IRLS weights
(the robust-noise-model strategy GTSAM uses), analytic Jacobians, and a Schur
complement on the landmark blocks (numba kernels). It converges from the
paper's initialisation (`X = 0`, `R0 = I`) for camera tilts of at least 40°.

## Speed-up without changing results

The Eq. 9 nearest-neighbour searches dominated the run time. They now use
`espinslam/nn.py`, an exact grid search whose distances are bit-identical to
SciPy's k-d tree (same operation order, rounding-safe pruning bounds, exact
warm-start upper bounds). On a real TOPSPIN scene (hubble-perpendicular-slow)
every output, all 553 frequency updates and all coarse cost curves are
unchanged, while the run takes 37.9 s instead of 149.5 s (107k events/s,
above the scene's 68k events/s, i.e. real time in the sense of Fig. 7).
Details: [`docs/reproduction_report.md`](docs/reproduction_report.md), Sec. 12.

## Interpretation choices

The paper does not pin these details down; each choice is configurable.

1. **Eq. 9 scaling and support.** Pixels and seconds are combined with
   `frequency.time_scale_px_per_second`. Only left-half events whose shifted
   timestamp falls inside the right half are used: the others cannot be
   aligned, and letting them snap to the boundary biases `t_rev` (observed as
   a slow drift of `f` once `H` spans more than two revolutions).
2. **1-D ICP initialisation.** Before loop detection the shift is initialised
   by a bounded coarse search; minima within `harmonic_tolerance` of the best
   are treated as equal and the shortest period (the fundamental) is chosen.
   After detection ICP is warm-started from the current estimate.
3. **Estimator cadence.** The estimator runs when `H` changed and at least
   `frequency.update_interval_seconds` (0.1 s) of event time has passed, so
   `ℵ = 20` estimates span 2 s.
4. **Triangulation parallax.** Tracks covering less than
   `backend.min_triangulation_angle_deg` (1.5°, COLMAP's default minimum
   triangulation angle) of orbit are under-constrained; they stay in `H` and
   enter the optimisation once they grow. Without this, single-chunk track
   fragments (which pass `N_match = 3`) produce landmarks with arbitrary depth.
5. **Tracklets and association.** Sec. 4.1 only says HDBSCAN clusters `G`
   "spatio-temporally". By default each polarity is clustered separately
   (consistent with the per-polarity density of Eq. 6); mixing polarities
   (`cluster_by_polarity: false`) merged the ON/OFF corners of small features
   and doubled the structure error on a low-resolution synthetic scene. Each tracklet joins the nearest track whose newest
   events are within `association_radius_px` and
   `association_max_gap_seconds`; several tracklets of one chunk may join the
   same track (HDBSCAN can split one corner).
6. **FFT ablation (Table 3).** The dominant peak of the track-event rate
   spectrum. Because FFT bins are `1/span` wide the `σ` test cannot pass, so
   this setting initialises once two estimated revolutions are observed.
7. **ICP metric scale.** The monocular scale is unobservable (`r = 10` is
   arbitrary). If the manifest gives `orbit_radius_mm` (true camera-to-spin-axis
   distance) the scale is `orbit_radius_mm / r`; otherwise it is fixed from
   the RMS-radius ratio (re-estimating it inside ICP collapses the cloud).
8. **Density threshold.** Eq. 6 removes corners below "the mean `μ_D`", so one
   mean over all corners is used. A per-polarity mean
   (`density_mean_per_polarity: true`) is available; it was less accurate with
   unequal ON/OFF contrast thresholds on the synthetic benchmark.
9. **Map contents and other details.** World points of forgotten tracks leave
   the map with their track; the "no f constraint" ablation starts each run
   from the previously optimised `f`; loop detection needs `ℵ` consecutive
   valid estimates. The full list (15 items, with measurements) is in
   [`docs/reproduction_report.md`](docs/reproduction_report.md), Sec. 4.

## TOPSPIN evaluation (paper Sec. 5-6)

TOPSPIN is available from Zenodo (doi:10.5281/zenodo.10884693, current
version record 10884694): 72 scenes `{object}-{perspective}-{speed}`, each with
`events.csv` (t [us], x, y, p), `calib.txt` (fx fy cx cy k1 k2 p1 p2 k3) and
`meta.json`. The public release does **not** include ground-truth rotation
frequencies or CAD models, so frequency accuracy (Fig. 6) and ICP RMSE
(Tables 2-3) need data from elsewhere; reprojection error, loop closure,
throughput and Fig. 8 work as is. Event files
can be Prophesee `.raw` (EVT 2.0/3.0) or `.dat`, `.h5`, `.npz`, `.npy`,
`.csv`/`.txt`. The Prophesee decoders follow the OpenEB format specification;
ECF-compressed Prophesee HDF5 files must be converted (e.g. to `.npz`) first.

0. Download and convert the 72 scenes (about 10 GB of archives, 22 GB as
   `events.npz`; archives are checked against Zenodo's MD5):

   ```bash
   python scripts/download_topspin.py --output datasets/topspin
   ```

1. Build the manifest (object, perspective and speed come from the scene
   directory name, `controller` maps to Dualshock and `med` to medium;
   calibration and sensor size are read from `calib.txt` and `meta.json`):

   ```bash
   python scripts/make_topspin_manifest.py --data-root datasets/topspin \
     --output datasets/topspin/manifest.csv --dev-scenes examples/splits/topspin_dev.txt
   ```

2. Optionally add what the public release lacks: `frequency_hz` (ground
   truth, used only for metrics), `cad_path` / `cad_unit_mm`, and
   `orbit_radius_mm`. Empty columns only leave the matching metrics blank;
   rows without calibration stop the evaluation.

3. Run the full method (Table 2, Figs. 6-8) and the ablations (Table 3):

   ```bash
   python scripts/evaluate_topspin.py --config examples/configs/topspin_evk4.jsonc \
     --data-root datasets/topspin --manifest datasets/topspin/manifest.csv \
     --output-dir runs/topspin
   python scripts/evaluate_topspin.py --config examples/configs/topspin_evk4.jsonc \
     --data-root datasets/topspin --manifest datasets/topspin/manifest.csv \
     --output-dir runs/topspin_ablations --ablations all
   ```

Outputs: `per_sequence.csv` (including the Fig. 7 `real_time` flag),
`table2.md`, `table_vs_paper.md`, `fig5_throughput_summary.png`,
`fig6_frequency_cdf.png`, `fig7_throughput.png`, `fig8_<scene>.png`,
`reconstruction_<scene>.png` (as in Figs. 1, 3, 4), `history/*.csv`,
`metrics.json`; ablation runs add `table3.md`, `table3_vs_paper.md` and
`fig6_loop_closure_vs_fft.png`.

### TOPSPIN results

The paper gives no tuning procedure or data split. Front-end settings the
paper does not specify were set on three pre-registered dev scenes
(`scripts/tune_topspin.py`, `examples/splits/topspin_dev.txt`), and all 72
scenes were then evaluated once; the 69 held-out scenes are reported
separately (`docs/topspin_results/`):

- loop closure in 72/72 scenes, valid reconstruction in 68/72;
- reprojection error per object 3.9-4.8 px (paper: 1.5-3.3 px);
- frequency: 49/72 scenes within 5% of their speed's median (medians match
  the paper's ground-truth lines), but 17 scenes of symmetric objects (SOHO,
  TDRS) report about twice the frequency, unlike the paper's Fig. 6;
- ICP RMSE cannot be computed: the public release has no CAD models or
  ground-truth frequencies;
- real time (Fig. 7) in 9/72 scenes; the Python front-end is the bottleneck.

Full analysis: [`docs/reproduction_report.md`](docs/reproduction_report.md),
Sec. 7.4-7.5 and 13.

## Synthetic verification (no TOPSPIN required)

`scripts/synthetic_benchmark.py` renders a rigid object made of planar
patches spinning at constant `f` in front of an elevated static camera (so
`R0` is a genuine tilt), simulates events with the contrast-threshold model,
and compares the online output with ground truth in the same world frame:

```bash
python scripts/synthetic_benchmark.py --ablations --output runs/synthetic.json
```

Result of that command on the final code (346x260, fx = 528.6, 0.05 Hz/px
noise; structure error is the distance from each landmark to the nearest true
patch corner in world units, object radius 1.5, `r = 10`):

| Scene (f, elevation) | Setting | f error | Loop detected | R0 error | Structure median / p90 | Reproj. | Landmarks |
|---|---|---|---|---|---|---|---|
| 0.5 Hz, 20° | eSpinSLAM | <0.0001 % | 3.95 s | 0.48° | 0.090 / 0.197 | 1.59 px | 599 |
| | No landmark association | <0.0001 % | 3.95 s | 0.24° | 0.051 / 0.143 | 1.32 px | 1112 |
| | No factor graph f constraint | <0.0001 % | 3.95 s | 0.45° | 0.087 / 0.196 | 1.55 px | 623 |
| | FFT f estimation | 0.28 % | 4.25 s | 0.21° | 0.084 / 0.129 | 1.51 px | 417 |
| 0.3 Hz, 10° | eSpinSLAM | 0.0003 % | 5.39 s | 0.36° | 0.066 / 0.137 | 1.21 px | 678 |
| | No landmark association | 0.0005 % | 5.39 s | 0.28° | 0.050 / 0.094 | 1.09 px | 1301 |
| | No factor graph f constraint | 1.71 % | 5.39 s | 0.05° | 0.213 / 0.384 | 1.11 px | 1112 |
| | FFT f estimation | 122.7 % (harmonic) | 3.87 s | 6.25° | 0.465 / 0.960 | 1.82 px | 346 |
| 0.8 Hz, 35° | eSpinSLAM | <0.0001 % | 5.96 s | 0.36° | 0.079 / 0.212 | 1.22 px | 422 |
| | No landmark association | <0.0001 % | 5.96 s | 0.75° | 0.085 / 0.158 | 1.12 px | 709 |
| | No factor graph f constraint | 1.70 % | 5.96 s | 1.36° | 0.363 / 0.600 | 1.14 px | 674 |
| | FFT f estimation | 89.6 % (harmonic) | 2.63 s | - | no inlier landmarks | - | 0 |

FFT frequencies are fragile (harmonics) and freeing `f` lets it drift and
degrades the structure, as in Table 3. Without landmark association the map
has 1.7-1.9x more (duplicate) points and a slightly lower reprojection error,
also as in Table 3, but these short runs (at most 9 s) do not show the
structure degradation the paper reports for long-term operation. Python
throughput is ~0.08-0.12 M events/s, about an order of magnitude below the
paper's C++ figures (Fig. 7). The full record of the changes and results is
in [`docs/reproduction_report.md`](docs/reproduction_report.md).

## Debris evaluation (comparison with the proposed method)

```bash
python scripts/evaluate.py --config examples/configs/debris_davis346.jsonc \
  --data-root datasets/20261031_debris_rotation_v3_davis346 \
  --manifest datasets/20261031_debris_rotation_v3_davis346/manifest.csv \
  --output-dir runs/debris
```

Only `test` rows are evaluated; labels, Blender geometry and poses are used
only for metrics. Time-to-accuracy curves report the causal online output:
`f*` is available only after loop detection, otherwise the sequence abstains.
The debris data contains camera motion, which violates the paper's static
camera assumption; it is a robustness comparison, not a reproduction.

## Tests

```bash
pip install -e ".[test,topspin]"
python -m pytest
```

The suite checks analytic Jacobians against finite differences, bundle
adjustment from the paper's initialisation, Eq. 9 / loop detection, eFAST,
the Prophesee decoders, an end-to-end synthetic sequence, and the TOPSPIN
report generation.
