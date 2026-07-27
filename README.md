# Leukemia Classification with UDA (Domain-Adversarial Neural Networks)

Multi-dataset leukemia cell classification using ConvNeXtV2-Tiny with Unsupervised Domain Adaptation (UDA). Trained on three source datasets (C-NMC, ALL-IDB, SN-AM), adapted to an unlabeled target domain (Taleqani) using a Hybrid DANN (Domain-Adversarial Neural Network) and Pseudo-Labeling approach to resolve severe domain shifts.

## Hardware Requirements

Tested on RTX 3050 4GB VRAM, 16GB RAM. 
- ConvNeXtV2-Tiny with bf16-mixed and batch=32 uses approximately 1.2–1.6 GB VRAM. 
- If OOM occurs, reduce batch size or increase `--accumulate-grad`.

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

Prepare and merge all source datasets into the unified structure. This script automatically handles saliency segmentation, cropping, and dataset splits:

```bash
python src/prepare_datasets.py
```

## Training

### Dual pipeline (Recommended)

Runs `source_only` first, then transfers its best checkpoint weights to initialize `uda_pseudolabel`. This ensures the DANN and pseudo-label generator start from a trained model rather than random weights. At the end of the pipeline, automatic evaluation is triggered.

```bash
python src/main.py --dual --seeds 42
```

### CLI Options

| Option | Default | Description |
|---|---|---|
| `--exp` | `source_only` | Single experiment to run |
| `--seeds` | `42` | One or more random seeds |
| `--all` | off | Run all experiments sequentially |
| `--dual` | off | Two-stage pipeline with checkpoint handoff and auto-evaluation |
| `--data-dir` | `dataset` | Dataset root directory |
| `--ckpt-root` | `checkpoints` | Checkpoint save directory |
| `--log-root` | `logs` | CSV log directory |
| `--num-workers` | `2` | DataLoader worker processes |
| `--accumulate-grad` | `1` | Gradient accumulation steps |

## Hyperparameter Tuning (Optuna)

To find the optimal hyperparameters for the DANN pipeline, run the included Optuna tuning script. The results are automatically saved in `optuna_study.db` (SQLite) so you can resume tuning at any time. Unpromising trials are pruned early to save GPU time.

```bash
python src/tune.py --n-trials 10
```

## Model Architecture & UDA

- **Backbone**: ConvNeXtV2-Tiny (`convnextv2_tiny.fcmae_ft_in22k_in1k`, pretrained ImageNet-22k)
- **UDA Method**: 
  - **Domain-Adversarial Neural Network (DANN)**: Employs a Gradient Reversal Layer (GRL) and Domain Discriminator to align feature distributions between Source and Target domains.
  - **Iterative Pseudo-Labeling**: High-confidence predictions on the Target domain are injected back into the training set (starting after a 20-epoch burn-in).
- **Loss**: Focal loss + class-weighted cross-entropy.
- **Optimization**: AdamW with layer-wise LR decay (LLRD).

## Evaluation

Evaluation runs automatically at the end of `--dual`, but can be run manually:

```bash
python src/evaluate.py --exp source_only uda_pseudolabel --seed 42
```

Results are saved to `results/{exp_name}_seed{seed}_eval.json`.
Metrics reported: Accuracy, Balanced Accuracy, F1 (macro and binary), Precision, Recall, Sensitivity, Specificity, AUC-ROC, Youden-J optimal threshold.

## Project Structure

```
src/
  main.py               # training entry point, dual pipeline
  lightning_model.py    # ConvNeXtV2Classifier + DANN GRL, LeukemiaLightningModel
  data_module.py        # LeukemiaDataModule, UDA CombinedLoader
  evaluate.py           # inference, metrics, checkpoint selection
  prepare_datasets.py   # dataset preprocessing and manifest generation
  tune.py               # Optuna hyperparameter tuning
dataset/                # processed images + metadata CSV
checkpoints/            # Lightning checkpoints per run
logs/                   # CSVLogger outputs
results/                # JSON evaluation outputs
```
