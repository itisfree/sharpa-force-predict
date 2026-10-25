# Step 40,000 — complete held-out episode evaluation

The checkpoint was evaluated in FP32 on 5,602,300 image–force pairs from 546 test episodes and all ten fingers. Every expected (episode, finger, frame) key was checked for uniqueness and coverage, and every target was checked against the cached labels. The evaluation completed on 2026-10-24 UTC (2026-10-25 in UTC+8).

## Protocol

- Train / validation / test: 4,372 / 546 / 546 episodes; random episode split, seed 42, stride 1 (30 Hz).
- Objects overlap across splits. There is no unseen-object claim.
- Input: original grayscale tactile image, 320 × 240, scaled by 1/255.
- Output: Fx, Fy, Fz, unstandardized using training-only mean and standard deviation.
- Joint ten-finger ResNet-50, no finger ID or force history input.
- Evaluation: global batch 384 over two GPUs (192 per GPU), FP32, TF32 disabled.
- Checkpoint: step_00040000.pt; source SHA-256 and manifest SHA-256 are in provenance.json.
- Force values remain in dataset-native units; independent SI calibration has not been verified.

## Overall

| Metric | Fx | Fy | Fz |
|---|---:|---:|---:|
| mae | 0.180040 | 0.188406 | 0.222573 |
| rmse | 0.453671 | 0.488995 | 0.555868 |
| r2 | 0.934089 | 0.955182 | 0.971158 |
| pearson | 0.966664 | 0.977394 | 0.985568 |
| bias | -0.002167 | -0.019827 | -0.034453 |

## Force subsets and baselines

| Subset / predictor | Count | Mean Euclidean force error |
|---|---:|---:|
| all | 5,602,300 | 0.392341 |
| near_zero | 3,903,723 | 0.092875 |
| above_near_zero | 1,698,577 | 1.080582 |
| high_force | 552,703 | 1.870260 |
| zero_baseline | 5,602,300 | 1.660094 |
| mean_baseline | 5,602,300 | 2.608739 |

Near-zero threshold: 0.1; high-force threshold: training force-norm 90th percentile (7.187729). Thresholds refer to ground-truth vector norm, not a calibrated binary contact label. Near-zero samples account for 69.68% of the test set.

## All fingers

| Finger | Count | MAE Fx | MAE Fy | MAE Fz | Vector error |
|---|---:|---:|---:|---:|---:|
| left_index | 560,230 | 0.194332 | 0.208058 | 0.267589 | 0.447499 |
| left_middle | 560,230 | 0.220526 | 0.212268 | 0.231075 | 0.440364 |
| left_pinky | 560,230 | 0.153552 | 0.135911 | 0.182143 | 0.314312 |
| left_ring | 560,230 | 0.140614 | 0.136833 | 0.162309 | 0.291000 |
| left_thumb | 560,230 | 0.316181 | 0.380595 | 0.409030 | 0.728220 |
| right_index | 560,230 | 0.180604 | 0.205279 | 0.238770 | 0.413456 |
| right_middle | 560,230 | 0.143791 | 0.123760 | 0.145616 | 0.277390 |
| right_pinky | 560,230 | 0.113135 | 0.120607 | 0.149941 | 0.254238 |
| right_ring | 560,230 | 0.084347 | 0.089914 | 0.100432 | 0.180382 |
| right_thumb | 560,230 | 0.253321 | 0.270838 | 0.338820 | 0.576546 |

## Definitions

For N samples and error e = prediction − target: per-axis MAE = mean(|e|), RMSE = sqrt(mean(e²)), bias = mean(e), R² = 1 − sum(e²)/sum((target − mean(target))²). Vector error = mean(sqrt(e_Fx² + e_Fy² + e_Fz²)). Pearson r is computed independently per axis. Undefined R²/correlations on constant signals are stored as null. Each image–finger pair is one sample; episode macro error gives equal weight to each episode.

## Interpretation

The model captures much of the force variation, with full-test R² from 0.9341 to 0.9712. The aggregate error is strongly influenced by the large near-zero subset. Error increases during stronger contact, and thumb errors exceed several other fingers. Good example trajectories on the homepage do not establish generalization to a new object or sensor. This checkpoint is intermediate: no claim is made that the planned 20 epochs completed or that this is the validation-selected best checkpoint.

## Included evidence

- `test.json`: complete aggregate and grouped MAE/RMSE/vector metrics, thresholds, baselines, and coverage.
- `extended_metrics.json`: R², correlation, bias and quantiles for the full set, all 10 fingers, and all 546 episodes.
- `metrics_by_finger.csv`: compact machine-readable finger comparison.
- `provenance.json`: checkpoint identity and evaluation protocol.
- `../../assets/episode_*.csv`: numeric data behind the homepage figure.

The full 5.6-million-row prediction CSV and all-episode PDFs remain in the original evaluation archive; these large artifacts and the training checkpoint are not included in this Git repository. Re-running the evaluation CLI produces the per-frame CSV from a locally trained checkpoint.
