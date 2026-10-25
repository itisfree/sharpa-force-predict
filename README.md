# Sharpa Force Predict

**Predict 3D fingertip force from a single raw tactile image.**

A PyTorch ResNet-50 baseline trained jointly on all ten fingers of the [T-Rex dataset](https://huggingface.co/datasets/zekaiwang/trex_dataset), collected with Sharpa Wave dexterous hands. The model takes only tactile image pixels and predicts **Fx, Fy, Fz**. No RGB camera image, deformation map, finger ID, or force history enters the network.

[中文说明](README.zh-CN.md) · [Full evaluation](results/step40000/REPORT.md) · [Per-finger metrics](results/step40000/metrics_by_finger.csv)

## Prediction versus ground truth

Temporal force predictions from held-out test episodes, evaluated at **step 40,000**. Blue: ground truth. Orange: model prediction. Curves show complete sequences at 30 Hz without smoothing.

![Force prediction time series](assets/force_comparison.png)

## Full test-set results

**546 held-out episodes · 10 fingers · 5,602,300 image–force pairs.**

| Metric | Fx | Fy | Fz |
|---|---:|---:|---:|
| MAE | 0.1800 | 0.1884 | 0.2226 |
| RMSE | 0.4537 | 0.4890 | 0.5559 |
| R² | 0.9341 | 0.9552 | 0.9712 |
| Pearson r | 0.9667 | 0.9774 | 0.9856 |

Mean Euclidean force error: **0.3923**. Metrics use dataset-native force units; an independent SI calibration has not been verified.

Episodes are disjoint between train, validation, and test; objects may occur in multiple splits. This evaluates held-out episodes, not unseen-object generalization. About 69.68% of test samples have target force norm below 0.1. Mean vector error is 1.0806 above that threshold and 1.8703 on the high-force subset. See the [evaluation report](results/step40000/REPORT.md) for all groups and limitations.

## Installation

Linux, Python 3.10+, and a suitable PyTorch CUDA installation are recommended for training. CPU inference and fixture tests are supported.

```bash
git clone https://github.com/itisfree/sharpa-force-predict.git
cd sharpa-force-predict
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
```

## Dataset

Download the dataset from its [official Hugging Face repository](https://huggingface.co/datasets/zekaiwang/trex_dataset). For this model, only raw tactile videos, force Parquet files, and metadata are required. Each include pattern is passed separately:

```bash
python -m pip install huggingface_hub
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  hf download zekaiwang/trex_dataset --repo-type dataset \
  --local-dir datasets/trex_dataset --max-workers 4 \
  --include 'videos/observation.images.tactile_*_raw_*/*' \
  --include 'data/*' --include 'meta/*' \
  --include README.md --include LICENSE
```

The mirror is optional; omit `HF_ENDPOINT` to use huggingface.co. Preparation validates the selected files and refuses incomplete coverage. For a pilot, `--limit-per-split` explicitly limits episodes; pilot metrics must not be confused with the complete results above.

```bash
python -m trex_force prepare \
  --root datasets/trex_dataset --out artifacts/all_fingers \
  --finger all --target force --split episode --seed 42 --stride 1
```

## Training

The evaluated configuration uses FP32, ResNet-50, AdamW (lr 3e-4, weight decay 1e-4), SmoothL1 on training-standardized force labels, gradient clipping at 5, and no activation recomputation. The global training batch is 1,024 (512 per GPU on two GPUs).

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m trex_force train \
  --manifest artifacts/all_fingers/manifest.json --run-dir runs/r50 \
  --backbone resnet50 --data-parallel --device cuda:0 \
  --batch-size 1024 --workers 4 --cpu-threads 4 \
  --epochs 20 --lr 0.0003 --weight-decay 0.0001 --seed 42 \
  --shuffle-buffer 8192 --streams-per-finger 2 \
  --near-zero-keep 1 --near-zero-threshold 0.1 \
  --log-every 1 --checkpoint-every 5000
```

`--batch-size` is the **global** batch. Reduce it for smaller GPUs. FP32 is the default and disables TF32. AMP and activation recomputation are optional features and were disabled for the reported run. The command requests 20 epochs; the published results are an intermediate checkpoint at step 40,000, not a completed 20-epoch run.

To resume, repeat the same command with `--resume runs/r50/last_step.pt`. Mid-epoch resume restores the next unconsumed batch, worker cursors, shuffle buffer, optimizer, and RNG state. Keep the manifest, batch size, worker count, sampling settings, and training configuration unchanged. Sample continuity does not promise bitwise identical CUDA computation. Step checkpoints are saved every 5,000 steps, and epoch checkpoints are saved after validation.

## Evaluation and inference

Use a checkpoint created by this project. Checkpoints contain Python objects: load only trusted files. The 40,000-step checkpoint and raw dataset are not bundled in this Git repository; the reported metrics and plotted numeric traces are included.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m trex_force evaluate \
  --checkpoint runs/r50/step_00040000.pt \
  --manifest artifacts/all_fingers/manifest.json \
  --split test --output results/new_test/test.json \
  --device cuda:0 --data-parallel --batch-size 384 --workers 4

python scripts/summarize_predictions.py \
  --csv results/new_test/test.predictions.csv \
  --report results/new_test/test.json --output results/new_test/extended_metrics.json

python scripts/plot_predictions.py \
  --csv results/new_test/test.predictions.csv \
  --episode 1688 --finger right_index --output results/new_test/example.png

python -m trex_force predict \
  --checkpoint runs/r50/step_00040000.pt \
  --image tactile_raw.png --device cpu
```

Inference expects an original 320×240 uint8 grayscale image. Video preprocessing uses the raw luminance plane, scaled by 1/255, with no cropping or resizing. Evaluation uses the checkpoint's training normalization and verifies the manifest hash. Manifests contain source paths and file checks: keep dataset/cache locations stable across training, resume, and evaluation.

Recreate the README figure from the included numeric traces:

```bash
python scripts/plot_examples.py
```

## Architecture

`1 × 240 × 320 → ResNet-50 convolutional encoder → adaptive pool 4 × 5 → 40960 → 256 → 128 → 3`

ReLU activations follow the first two linear layers. The model has 34,021,059 parameters and uses no pretrained backbone weights. All fingers share one model. Training statistics are computed only on the training split.

## License and acknowledgements

Project code is available under the [MIT License](LICENSE). The T-Rex dataset is maintained by its original authors and is separately licensed; its MIT notice is retained in [third_party/TREX_DATASET_LICENSE](third_party/TREX_DATASET_LICENSE) for the included derived example traces. Dataset data are not redistributed here.

This is an independent baseline, not an official Sharpa or T-Rex release. Please credit and consult the [T-Rex project](https://tactile-rex.github.io/) and [original code](https://github.com/ZhuoyangLiu2005/T-Rex) when using their dataset.
