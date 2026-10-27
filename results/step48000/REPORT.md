# Step 48,000 (replayed) — complete held-out episode evaluation

Fresh inference completed on 2026-10-27 UTC: **5,602,300 image–force pairs, 546 episodes, all ten fingers**. Every expected (episode, finger, frame) key was checked for uniqueness and coverage, and every ground-truth value was verified against the immutable cached labels.

## Checkpoint and protocol

The original run did not save step 48,000. This checkpoint was reconstructed by resuming the saved step 45,000 checkpoint and replaying training to exactly step 48,000. It is an actual saved 48,000-step checkpoint, but is not claimed to be bitwise identical to the discarded original run.

- Episode split, seed 42: train 4,372 / validation 546 / test 546; stride 1, 30 Hz. Objects may overlap across splits.
- Raw grayscale tactile image only, 320 × 240, divided by 255. No RGB, deformation map, finger ID, or force history.
- Shared ten-finger ResNet-50, trained from scratch. Force normalization uses training-only statistics.
- Replay: FP32, global batch 1,024, one GPU with two microbatches of 512; only the first microbatch persists BatchNorm running buffers, matching the primary replica behavior of the earlier two-GPU run. Data-loader cursors and sampling configuration are restored.
- AdamW learning rate 0.0003, weight decay 0.0001; standardized SmoothL1; gradient clipping 5. AMP, TF32, and activation recomputation disabled.
- Evaluation: one GPU, batch 512, FP32, TF32 disabled; complete test split, no batch limit.
- Checkpoint identity and manifest SHA-256: [provenance.json](provenance.json).
- Force units are dataset-native. Independent SI calibration has not been verified.

## Overall

| Metric | Fx | Fy | Fz |
|---|---:|---:|---:|
| mae | 0.183249 | 0.177801 | 0.210278 |
| rmse | 0.447859 | 0.458696 | 0.545102 |
| r2 | 0.935767 | 0.960564 | 0.972265 |
| pearson | 0.967776 | 0.980195 | 0.986045 |
| bias | 0.007280 | -0.022533 | -0.014500 |

## Same-test-set comparison

| Metric | Step 40,000 | Step 48,000 (replayed) | Change |
|---|---:|---:|---:|
| Mean vector error | 0.392341 | 0.380168 | -3.10% |
| MAE Fx | 0.180040 | 0.183249 | +1.78% |
| MAE Fy | 0.188406 | 0.177801 | -5.63% |
| MAE Fz | 0.222573 | 0.210278 | -5.52% |
| Vector error P50 | 0.074117 | 0.079404 | +7.13% |
| Vector error P90 | 1.221365 | 1.147964 | -6.01% |
| Vector error P95 | 1.899382 | 1.829588 | -3.67% |

Both evaluations use the same manifest and complete episode test split. The mean vector error improves by 3.10%, and all three RMSE and R² values improve. Fy and Fz MAE improve by 5.63% and 5.52%, while Fx MAE worsens by 1.78%. The median vector error also increases; improvement is not uniform across samples or fingers. These are point estimates, without confidence intervals. The earlier evaluation used two GPUs with global batch 384; this evaluation used one GPU with batch 512, both in evaluation mode and FP32.

## Force subsets and baselines

| Subset / predictor | Count | Step 40,000 vector error | Step 48,000 vector error |
|---|---:|---:|---:|
| all | 5,602,300 | 0.392341 | 0.380168 |
| near_zero | 3,903,723 | 0.092875 | 0.091471 |
| above_near_zero | 1,698,577 | 1.080582 | 1.043660 |
| high_force | 552,703 | 1.870260 | 1.835481 |
| zero_baseline | 5,602,300 | 1.660094 | 1.660094 |
| mean_baseline | 5,602,300 | 2.608739 | 2.608739 |

Near-zero: target force norm < 0.1. High force: target force norm ≥ the training 90th-percentile threshold, 7.187729. These subsets are defined by continuous ground-truth norm, not a calibrated contact label. Near-zero samples account for 69.68% of the test set; aggregate metrics therefore do not replace the contact-force subset metrics.

## All fingers

| Finger | Count | MAE Fx | MAE Fy | MAE Fz | Vector error | Change from 40,000 |
|---|---:|---:|---:|---:|---:|---:|
| left_index | 560,230 | 0.209751 | 0.211977 | 0.259938 | 0.452277 | +1.07% |
| left_middle | 560,230 | 0.192615 | 0.205741 | 0.231053 | 0.415278 | -5.70% |
| left_pinky | 560,230 | 0.135072 | 0.151255 | 0.162998 | 0.293216 | -6.71% |
| left_ring | 560,230 | 0.123760 | 0.137647 | 0.146597 | 0.272002 | -6.53% |
| left_thumb | 560,230 | 0.369157 | 0.264702 | 0.356408 | 0.668620 | -8.18% |
| right_index | 560,230 | 0.219168 | 0.188694 | 0.231599 | 0.421632 | +1.98% |
| right_middle | 560,230 | 0.130653 | 0.119420 | 0.143720 | 0.263780 | -4.91% |
| right_pinky | 560,230 | 0.123161 | 0.121772 | 0.138042 | 0.253845 | -0.15% |
| right_ring | 560,230 | 0.099287 | 0.093093 | 0.095668 | 0.191212 | +6.00% |
| right_thumb | 560,230 | 0.229867 | 0.283708 | 0.336754 | 0.569816 | -1.17% |

Thumb errors remain larger than those of several other fingers. Seven of the ten fingers improve in mean vector error; left index, right index, and right ring worsen. Strong-contact errors remain substantially larger than near-zero errors.

## Definitions

For N samples, e = prediction − target. Per-axis MAE = mean(|e|); RMSE = sqrt(mean(e²)); bias = mean(e); R² = 1 − sum(e²)/sum((target − mean(target))²). Mean vector error = mean(sqrt(e_Fx² + e_Fy² + e_Fz²)). Pearson correlation is calculated per axis. Undefined R² or correlation on constant signals is null. Each image–finger pair is one sample; the episode macro metric weights each episode equally. Percentage change = (step48000 / step40000 − 1) × 100; lower errors are better.

## Evidence and reproduction

- [test.json](test.json): all aggregate and grouped errors, force subsets, baselines, thresholds, and coverage.
- [extended_metrics.json](extended_metrics.json): full-set, 10-finger and 546-episode MAE, RMSE, bias, R², Pearson r and vector-error quantiles.
- [metrics_by_finger.csv](metrics_by_finger.csv): complete per-finger table.
- [comparison_step40000_step48000.json](comparison_step40000_step48000.json): full-precision paired aggregate comparison.
- [provenance.json](provenance.json): checkpoint, manifest and full-archive SHA-256 values.
- [Time-series figure](../../assets/step48000/force_comparison.png) and numeric traces in `assets/step48000/`.
- [Earlier step 40,000 report](../step40000/REPORT.md) remains unchanged.

Run the evaluation and summary commands in the main README with a trusted local checkpoint to recreate per-frame predictions and aggregate metrics. The original evaluation archive contains all 5,602,300 prediction rows and 546 episode PDFs, each showing all ten fingers and three force axes. Large prediction dumps, weights and source data are not bundled in Git.

This is an intermediate checkpoint, not the completion of the planned 20 epochs or a validation-selected best model. Held-out episodes do not establish unseen-object or new-sensor generalization.
