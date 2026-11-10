# eSpinSLAM paper-validation and held-out evaluation — 2026-11-10

## Evaluation contract

- The target dataset remains `20261031_debris_rotation_v3_davis346`.
- Its manifest contains 600 train, 200 validation, and 200 test sequences.
- All 1,000 target sequences use a fixed body-frame rotation axis `[0, 0, 1]`.
- Training labels determine only a padded frequency search interval. The first 60
  validation rows select paper-compatible spatial and temporal loop tolerances.
- Test labels are read only by the evaluator after the configuration is fixed.
- A separate 18-sequence analytic set with a static camera interpretation,
  constant single-axis motion, and circular dual geometry is diagnostic only. It
  verifies the implementation under the paper assumptions and never replaces the
  target test set.

eSpinSLAM is not a learned neural model. “Training” in this experiment means the
above validation-based parameter selection, not gradient optimization.

## Changes

1. Preserve the cumulative global feature pool required by the paper's loop
   closure stage instead of using only the most recent backend tracks.
2. Disable the broad near-tie rule that systematically preferred higher-frequency
   harmonics.
3. Select a train-derived frequency interval of 0.250314–0.583160 Hz, 141
   candidates, 3 px spatial tolerance, and 15 ms temporal tolerance.
4. Handle degenerate HDBSCAN candidates without aborting the front end.
5. Evaluate causal observation prefixes from the same cumulative feature pool as
   the final estimate.

## Validation selection

On 60 validation sequences, the selected configuration achieved 100% coverage,
0.379808 rad/s MAE, 0.657424 rad/s RMSE, and 15.57% MAPE. The old broad-range
configuration had 3.074615 rad/s MAE and 127.23% MAPE on the same validation rows.

## Paper-geometry diagnostic

The diagnostic test contains nine sequences and is non-photorealistic. The
configuration was selected on the target validation split, not on these nine
labels.

| Metric | Baseline | Improved |
|---|---:|---:|
| Coverage | 100% | 100% |
| MAE | 0.399987 rad/s | 0.179462 rad/s |
| MAE | 22.92 deg/s | 10.28 deg/s |
| RMSE | 0.856262 rad/s | 0.532141 rad/s |
| MAPE | 13.08% | 5.62% |
| Correlation | 0.218 | 0.536 |

This confirms that the independent reproduction runs under the paper's static
camera, fixed-axis, constant-speed geometry. The small analytic set is a software
diagnostic, not evidence of paper-level benchmark parity.

## Original target test (200 sequences)

| Metric | Old run | Improved run |
|---|---:|---:|
| Coverage | 100% | 100% |
| MAE | 2.428278 rad/s | 0.406990 rad/s |
| MAE | 139.13 deg/s | 23.32 deg/s |
| RMSE | 2.642099 rad/s | 0.670436 rad/s |
| RMSE | 151.38 deg/s | 38.41 deg/s |
| MAPE | 102.32% | 16.27% |
| Correlation | 0.052 | 0.354 |
| Slope | 0.120 | 0.504 |

The improved MAE is 83.2% lower than the old run. This is a major correction,
but not a solved estimator: the correlation and slope remain weak. Median
absolute error is 0.08264 rad/s (4.73 deg/s), while the 90th percentile is
1.35190 rad/s (77.46 deg/s). Eleven predictions hit the lower search bound and
five hit the upper bound exactly. A minority of harmonic/period failures therefore
dominates RMSE.

The scatter plot makes the failure mode explicit: 132/200 predictions are within
20% of the correct speed ratio, 19/200 lie within 20% of a half-speed alias, and
7/200 lie within 20% of a double-speed alias. In absolute terms, 102/200 are
within 5 deg/s and 121/200 are within 10 deg/s, but only 167/200 are within
60 deg/s. A robust alias detector with abstention is more important than further
small changes to the average loss.

The causal time analysis abstains through 1.5 s because a complete period is not
observable at the configured minimum frequency. It gives 100% coverage from 2 s:

| Observation | MAE | MAPE | Correlation |
|---|---:|---:|---:|
| 2 s | 0.9124 rad/s | 40.98% | -0.033 |
| 3 s | 0.3774 rad/s | 15.50% | 0.279 |
| 4 s | 0.7291 rad/s | 27.62% | 0.093 |
| 5 s | 0.4070 rad/s | 16.27% | 0.354 |

The non-monotonic curve indicates unstable loop selection, not a reason to choose
3 s after seeing test labels. Any duration change requires a new prespecified
validation experiment and a fresh holdout.

## 3D reconstruction

| Metric | Old run | Improved run |
|---|---:|---:|
| Valid reconstructions | 200/200 | 200/200 |
| Mean similarity-ICP RMSE | 0.36508 m | 0.35916 m |
| Median similarity-ICP RMSE | 0.36562 m | 0.35759 m |
| Mean Chamfer RMSE | 0.41490 m | 0.41311 m |

The small 3D change is expected because only frequency selection was tuned. These
metrics use similarity alignment and must not be interpreted as unaligned metric
scale recovery.

## Interpretation and next experiment

The reproduction is now technically functional and materially better, but the
target data violate the paper's static-camera assumption through camera orbit,
closing motion, and jitter. The target also tests only a fixed Z-axis. The next
experiment should be preregistered on validation data and target robust rejection
of harmonic aliases using multi-cycle consistency. It should report abstention
instead of clipping ambiguous sequences to the search bounds. Because this test
set has now been inspected, further algorithm selection requires a new untouched
holdout or a documented nested split.
