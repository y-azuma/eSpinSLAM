# eSpinSLAM

Independent, paper-guided reimplementation of **Event-based Spinning Object
SLAM** (Elms, Latif, and Chin, ICCV Workshops 2025). The authors' source code
was not publicly linked when this repository was created, so this repository
does not claim bit-for-bit equivalence with their implementation.

The code deliberately follows the structure and conventions of
[`event_snn_torch`](https://github.com/y-azuma/event_snn_torch): validated
JSONC configuration, thin scripts, explicit split manifests, atomic reports,
W&B logging, and reproducible Docker execution.

## Implemented pipeline

1. Event chunks and polarity time surfaces.
2. eFAST-style corner candidates, same-polarity density filtering, and
   HDBSCAN tracklets.
3. Global track association and bounded track retention.
4. Loop-closure frequency estimation by temporally shifted spatial alignment.
5. The paper's continuous-time circular-orbit camera model.
6. Multi-view triangulation, robust reprojection refinement, outlier removal,
   and landmark fusion.
7. Rotation-speed metrics, time-to-accuracy curves, reprojection error, and
   similarity-ICP alignment to the Blender reference point cloud.

The original paper uses GTSAM and a native asynchronous eFAST implementation.
This reproduction uses SciPy least-squares and an OpenCV time-surface FAST
front-end so that it remains reviewable and runnable from Python. These two
engineering substitutions are recorded in every evaluation report.
When the Python eFAST/HDBSCAN path yields fewer than four usable tracks, an
event-frame Lucas-Kanade tracker is used as an explicit, reported fallback so
the 3D backend can still be evaluated rather than silently producing a map.

## Evaluation

eSpinSLAM is geometric and has no learning stage. Hyperparameters are fixed
before reading the held-out test results. Only rows marked `test` in
`manifest.csv` are evaluated.

```bash
python scripts/evaluate.py \
  --config examples/configs/debris_davis346.jsonc \
  --data-root datasets/20261031_debris_rotation_v3_davis346 \
  --manifest datasets/20261031_debris_rotation_v3_davis346/manifest.csv \
  --output-dir runs/1110
```

The existing synthetic dataset contains slow camera translation/orbit in
addition to object spin. That is intentionally retained for an apples-to-apples
comparison with `event_snn_torch`, but it violates eSpinSLAM's exact static
camera and stationary circular-orbit assumptions. The report therefore labels
this as an out-of-model robustness evaluation rather than a reproduction of
the TOPSPIN benchmark.

## Data-leakage policy

- Frequency and point-cloud inference use events and camera intrinsics only.
- `omega_rad_s`, bounding boxes, Blender poses, and CAD geometry are used only
  after inference for metrics and visualization.
- No validation or test result changes the estimator's search range or
  hyperparameters.
- The manifest test partition is never replaced by train or validation rows.
