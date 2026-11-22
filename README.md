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
5. **Tracklet association.** Each tracklet joins the nearest same-polarity
   track whose newest events are within `association_radius_px` and
   `association_max_gap_seconds`; several tracklets of one chunk may join the
   same track (HDBSCAN can split one corner).
6. **FFT ablation (Table 3).** The dominant peak of the track-event rate
   spectrum. Because FFT bins are `1/span` wide the `σ` test cannot pass, so
   this setting initialises once two estimated revolutions are observed.
7. **ICP metric scale.** The monocular scale is unobservable (`r = 10` is
   arbitrary). If the manifest gives `orbit_radius_mm` (true camera-to-spin-axis
   distance) the scale is `orbit_radius_mm / r`; otherwise it is fixed from
   the RMS-radius ratio (re-estimating it inside ICP collapses the cloud).

## TOPSPIN evaluation (paper Sec. 5-6)

TOPSPIN is available from Zenodo (doi:10.5281/zenodo.10884693). Event files
can be Prophesee `.raw` (EVT 2.0/3.0) or `.dat`, `.h5`, `.npz`, `.npy`,
`.csv`/`.txt`. The Prophesee decoders follow the OpenEB format specification;
ECF-compressed Prophesee HDF5 files must be converted (e.g. to `.npz`) first.

1. Build a manifest skeleton (object and speed are parsed from file names
   such as `soho-perpendicular-fast.raw`):

   ```bash
   python scripts/make_topspin_manifest.py --data-root datasets/topspin \
     --output datasets/topspin/manifest.csv --cad-dir cad
   ```

2. Fill in the columns the dataset provides:
   `frequency_hz` (ground truth, used only for metrics), `cad_path`,
   `cad_unit_mm`, calibration `fx, fy, cx, cy` (and `k1, k2, p1, p2, k3`), and
   optionally `orbit_radius_mm`. The TOPSPIN config intentionally has no
   intrinsics; evaluation stops if they are missing.

3. Run the full method (Table 2, Figs. 6-8) and the ablations (Table 3):

   ```bash
   python scripts/evaluate_topspin.py --config examples/configs/topspin_evk4.jsonc \
     --data-root datasets/topspin --manifest datasets/topspin/manifest.csv \
     --output-dir runs/topspin
   python scripts/evaluate_topspin.py --config examples/configs/topspin_evk4.jsonc \
     --data-root datasets/topspin --manifest datasets/topspin/manifest.csv \
     --output-dir runs/topspin_ablations --ablations all
   ```

Outputs: `per_sequence.csv`, `table2.md`, `fig6_frequency_cdf.png`,
`fig7_throughput.png`, `fig8_<scene>.png`, `history/*.csv`, `metrics.json`,
and `table3.md` for ablations.

## Synthetic verification (no TOPSPIN required)

`scripts/synthetic_benchmark.py` renders a rigid object made of planar
patches spinning at constant `f` in front of an elevated static camera (so
`R0` is a genuine tilt), simulates events with the contrast-threshold model,
and compares the online output with ground truth in the same world frame:

```bash
python scripts/synthetic_benchmark.py --ablations --output runs/synthetic.json
```

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
