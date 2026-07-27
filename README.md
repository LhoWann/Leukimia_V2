# Leukemia Classification with UDA

Multi-dataset leukemia cell classification using ConvNeXtV2-Tiny with Unsupervised Domain Adaptation (UDA) via pseudo-labeling. Trained on three source datasets (C-NMC, ALL-IDB, SN-AM), adapted to an unlabeled target domain (Taleqani).

## Hardware Requirements

Tested on RTX 3050 4GB VRAM, 16GB RAM. ConvNeXtV2-Tiny with bf16-mixed and batch=32 uses approximately 1.2–1.6 GB VRAM. If OOM occurs, reduce batch size or increase `--accumulate-grad`.

## Dataset Structure

```
dataset/
  train/
    Abnormal/   # leukemia positive cells
    Normal/     # healthy cells
  val/
    Abnormal/
    Normal/
  test/
    Abnormal/
    Normal/
  metadata/
    dataset_manifest.csv   # source_dataset, split, mapped_label, output_filename
```

The manifest CSV drives the UDA split: images where `source_dataset == Taleqani` become the unlabeled target domain. All others are labeled source.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+. PyTorch 2.1 with CUDA 11.8 recommended.

## Dataset Preparation

Prepare and merge all source datasets into the unified structure:

```bash
python src/prepare_datasets.py
```

Create a source validation split (15% of source training data, patient-stratified for C-NMC):

```bash
python src/make_source_val.py
```

## Training

### Single experiment

```bash
python src/main.py --exp source_only --seeds 42
python src/main.py --exp uda_pseudolabel --seeds 42
```

### All experiments (sequential, no checkpoint handoff)

```bash
python src/main.py --all --seeds 42
```

### Dual pipeline (recommended)

Runs `source_only` first, then transfers its best checkpoint weights to initialize `uda_pseudolabel`. This ensures pseudo-label generation starts from a trained model rather than random weights.

```bash
python src/main.py --dual --seeds 42
```

Multiple seeds:

```bash
python src/main.py --dual --seeds 42 123 456
```

### CLI options

| Option | Default | Description |
|---|---|---|
| `--exp` | `source_only` | Single experiment to run |
| `--seeds` | `42` | One or more random seeds |
| `--all` | off | Run all experiments sequentially |
| `--dual` | off | Two-stage pipeline with checkpoint handoff |
| `--data-dir` | `dataset` | Dataset root directory |
| `--ckpt-root` | `checkpoints` | Checkpoint save directory |
| `--log-root` | `logs` | CSV log directory |
| `--num-workers` | `2` | DataLoader worker processes |
| `--accumulate-grad` | `1` | Gradient accumulation steps (e.g. `2` doubles effective batch) |

## Model Architecture

- Backbone: ConvNeXtV2-Tiny (`convnextv2_tiny.fcmae_ft_in22k_in1k`, pretrained ImageNet-22k)
- Optional: Multi-Head Attention (MHA) inserted at a specified backbone stage
- Loss: Focal loss (gamma=3.0) + class-weighted cross-entropy with Mixup-style targets
- Augmentation: FocusAugMix — SLIC superpixel paste guided by spectral saliency, plus Reinhard stain jitter
- Optimization: AdamW with layer-wise LR decay (LLRD=0.75), cosine annealing, 5-epoch warmup
- Precision: bf16-mixed (Ampere and newer GPUs)
- Gradient clip: 1.0

## UDA Pseudo-Labeling

UDA is a two-phase process:

1. **Burn-in** (epochs 0–19): train on source only, no pseudo-labels
2. **Pseudo-label phase** (epoch 20+, only when `val_f1 >= 0.70`): inference on unlabeled Taleqani images, filter by confidence threshold (0.95 → 0.90 → 0.85), balance classes, inject into training

The `--dual` flag starts Stage 2 from the best Stage 1 checkpoint, which means pseudo-label quality at epoch 20 benefits from 100 epochs of source training.

## Evaluation

Run evaluation on the test set:

```bash
python src/evaluate.py --exp source_only uda_pseudolabel --seed 42
```

Results are saved to `results/{exp_name}_seed{seed}_eval.json`.

Metrics reported: Accuracy, Balanced Accuracy, F1 (macro and binary), Precision, Recall, Sensitivity, Specificity, AUC-ROC, Youden-J optimal threshold. Subtype breakdown for Taleqani test images if manifest is present.

## Model Complexity

```bash
python src/complexity_benchmark.py
```

Saves a Markdown table to `results/complexity.md`.

## Memory Notes (RTX 3050 4GB)

- `batch_size=32` + bf16 + ConvNeXtV2-Tiny: ~1.4 GB peak VRAM during forward+backward
- GradCAM refresh (if using `focusmix_cam` mode) runs batched inference — no per-image loop
- Between the dual-pipeline stages, VRAM is explicitly freed via `torch.cuda.empty_cache()`
- Use `--accumulate-grad 2` for an effective batch size of 64 without extra VRAM cost
- `num_workers=2` spawns 2 persistent DataLoader workers (~200–400 MB RAM total)

## Project Structure

```
src/
  main.py               # training entry point, dual pipeline
  lightning_model.py    # ConvNeXtV2Classifier, LeukemiaLightningModel
  data_module.py        # LeukemiaDataModule, FocusAugMixDataset, augmentations
  evaluate.py           # inference, metrics, checkpoint selection
  prepare_datasets.py   # dataset preprocessing and manifest generation
  make_source_val.py    # carve validation split from source train data
  complexity_benchmark.py # FLOPs, params, latency benchmark
dataset/                # processed images + metadata CSV
checkpoints/            # Lightning checkpoints per run
logs/                   # CSVLogger outputs
results/                # JSON evaluation outputs
```

## Known Issues Fixed

- `evaluate.py`: class probability was extracted from index `[:, 0]` (Normal) instead of `[:, 1]` (Abnormal) — all previous evaluation results were inverted. Fixed.
- `evaluate.py`: checkpoint selection was alphabetical (lowest epoch wins). Fixed to sort by val_f1 descending.
- `data_module.py`: `_ExternalDataset` used outer-scope `class_to_idx` via closure with wrong attribute name. Fixed.
- `data_module.py`: `_dataset_sample_weights` not initialized before `setup()` — `AttributeError` possible. Fixed.
- `lightning_model.py`: `register_buffer(None)` can fail on checkpoint load. Fixed.
- `main.py`: GradCAM refresh iterated images one-by-one with no batching. Fixed to use batched DataLoader.
- `main.py`: no checkpoint handoff between source_only and uda_pseudolabel. Fixed via `--dual` pipeline.
