import warnings
import logging

warnings.filterwarnings("ignore", message="triton not found.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lightning_model import LeukemiaLightningModel

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

LABEL_MAP = {'Abnormal': 1, 'Normal': 0}


class TestDataset(Dataset):
    def __init__(self, test_dir: str, image_size: int = 224):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        self.samples: List[Tuple[str, int]] = []

        test_path = Path(test_dir)
        for class_dir in sorted(test_path.iterdir()):
            if not class_dir.is_dir():
                continue
            label = LABEL_MAP.get(class_dir.name)
            if label is None:
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in exts:
                    self.samples.append((str(img_path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label, os.path.basename(path)


def load_model(ckpt_path: str, device: torch.device) -> LeukemiaLightningModel:
    model = LeukemiaLightningModel.load_from_checkpoint(
        ckpt_path,
        map_location=device,
    )
    model.eval()
    model.to(device)
    return model


def run_inference(
    model: LeukemiaLightningModel,
    dataset: TestDataset,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    all_preds = []
    all_probs = []
    all_labels = []
    all_names = []

    with torch.inference_mode():
        for images, labels, names in loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_names.extend(names)

    return (
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_labels),
        all_names,
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')

    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        optimal_threshold = float(thresholds[best_idx])
        optimal_sensitivity = float(tpr[best_idx])
        optimal_specificity = float(1 - fpr[best_idx])
    except Exception:
        optimal_threshold = float('nan')
        optimal_sensitivity = float('nan')
        optimal_specificity = float('nan')

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_binary': f1_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'sensitivity': recall_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0),
        'specificity': specificity,
        'balanced_accuracy': (recall_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0) + specificity) / 2,
        'auc_roc': auc,
        'optimal_threshold': optimal_threshold,
        'optimal_sensitivity': optimal_sensitivity,
        'optimal_specificity': optimal_specificity,
        'optimal_balanced_acc': (optimal_sensitivity + optimal_specificity) / 2 if not (np.isnan(optimal_sensitivity) or np.isnan(optimal_specificity)) else float('nan'),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }


def load_subtype_map(manifest_path: str) -> Dict[str, str]:
    subtype_map: Dict[str, str] = {}
    if not os.path.exists(manifest_path):
        return subtype_map
    with open(manifest_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('source_dataset') == 'Taleqani':
                subtype_map[row['output_filename']] = row.get('original_label', 'Unknown')
    return subtype_map


def subtype_breakdown(
    names: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subtype_map: Dict[str, str],
) -> Dict[str, Dict[str, int]]:
    breakdown: Dict[str, Dict[str, int]] = {}
    for name, true, pred in zip(names, y_true.tolist(), y_pred.tolist()):
        subtype = subtype_map.get(name, 'Unknown')
        if subtype not in breakdown:
            breakdown[subtype] = {'total': 0, 'correct': 0}
        breakdown[subtype]['total'] += 1
        if true == pred:
            breakdown[subtype]['correct'] += 1
    return breakdown


def format_metrics_table(name: str, metrics: Dict[str, float], threshold: float = 0.5) -> str:
    lines = [
        f"Experiment : {name}",
        f"Threshold  : {threshold:.4f}",
        f"{'Metric':<28} {'Value':>10}",
        '-' * 40,
        f"{'Accuracy':<28} {metrics['accuracy']:>10.4f}",
        f"{'Balanced Accuracy':<28} {metrics['balanced_accuracy']:>10.4f}",
        f"{'F1 (macro)':<28} {metrics['f1_macro']:>10.4f}",
        f"{'F1 (binary ALL)':<28} {metrics['f1_binary']:>10.4f}",
        f"{'Precision (macro)':<28} {metrics['precision_macro']:>10.4f}",
        f"{'Recall (macro)':<28} {metrics['recall_macro']:>10.4f}",
        f"{'Sensitivity (ALL)':<28} {metrics['sensitivity']:>10.4f}",
        f"{'Specificity (Normal)':<28} {metrics['specificity']:>10.4f}",
        f"{'AUC-ROC':<28} {metrics['auc_roc']:>10.4f}",
        '-' * 40,
        f"{'TP/TN/FP/FN':<28} {metrics['tp']}/{metrics['tn']}/{metrics['fp']}/{metrics['fn']}",
        '-' * 40,
        f"{'Optimal threshold (Youden J)':<28} {metrics['optimal_threshold']:>10.4f}",
        f"{'Optimal Sensitivity':<28} {metrics['optimal_sensitivity']:>10.4f}",
        f"{'Optimal Specificity':<28} {metrics['optimal_specificity']:>10.4f}",
        f"{'Optimal Balanced Acc':<28} {metrics['optimal_balanced_acc']:>10.4f}",
    ]
    return '\n'.join(lines)


def _val_f1_from_filename(path: Path) -> float:
    try:
        return float(path.stem.split('val_f1=')[-1])
    except Exception:
        return 0.0


def find_best_checkpoint(exp_name: str, ckpt_root: str, seed: Optional[int]) -> Optional[str]:
    if seed is not None:
        search_dirs = [Path(ckpt_root) / f'{exp_name}_seed{seed}']
    else:
        search_dirs = sorted(Path(ckpt_root).glob(f'{exp_name}*'))

    for ckpt_dir in search_dirs:
        if not ckpt_dir.is_dir():
            continue
        candidates = sorted(
            [c for c in ckpt_dir.glob('*.ckpt') if c.name != 'last.ckpt'],
            key=_val_f1_from_filename,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])
        last = ckpt_dir / 'last.ckpt'
        if last.exists():
            return str(last)
    return None


def evaluate_experiment(
    exp_name: str,
    test_dir: str,
    ckpt_root: str,
    manifest_path: str,
    results_dir: str,
    device: torch.device,
    batch_size: int,
    seed: Optional[int],
    threshold: float = 0.5,
) -> Optional[Dict]:
    ckpt_path = find_best_checkpoint(exp_name, ckpt_root, seed)
    if ckpt_path is None:
        print(f"  No checkpoint found for {exp_name}. Skipping.")
        return None

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {exp_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Test dir  : {test_dir}")

    model = load_model(ckpt_path, device)
    dataset = TestDataset(test_dir)

    if len(dataset) == 0:
        print(f"  ERROR: No images found in {test_dir}.")
        return None

    print(f"Test samples: {len(dataset)}")

    _, probs, labels, names = run_inference(model, dataset, device, batch_size)
    preds = (probs >= threshold).astype(int)
    metrics = compute_metrics(labels, preds, probs)

    print()
    print(format_metrics_table(exp_name, metrics, threshold))

    subtype_map = load_subtype_map(manifest_path)
    if subtype_map:
        breakdown = subtype_breakdown(names, labels, preds, subtype_map)
        print(f"\n{'Subtype Breakdown':}")
        print(f"{'Subtype':<15} {'Total':>7} {'Correct':>9} {'Acc':>8}")
        print('-' * 42)
        for subtype, counts in sorted(breakdown.items()):
            acc = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
            print(f"{subtype:<15} {counts['total']:>7} {counts['correct']:>9} {acc:>8.4f}")

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    seed_suffix = f'_seed{seed}' if seed is not None else ''
    result_path = Path(results_dir) / f'{exp_name}{seed_suffix}_eval.json'
    result_data = {
        'experiment': exp_name,
        'checkpoint': ckpt_path,
        'test_dir': test_dir,
        'seed': seed,
        'threshold': threshold,
        'n_samples': len(dataset),
        'metrics': {k: (v if not (isinstance(v, float) and np.isnan(v)) else None)
                    for k, v in metrics.items()},
    }
    if subtype_map:
        result_data['subtype_breakdown'] = {
            s: {'total': c['total'], 'correct': c['correct'],
                'accuracy': round(c['correct'] / c['total'], 4) if c['total'] > 0 else 0}
            for s, c in breakdown.items()
        }

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {result_path}")

    return result_data


def compare_experiments(results: List[Dict]) -> None:
    if len(results) < 2:
        return

    print(f"\n{'=' * 60}")
    print("Experiment Comparison")
    print(f"{'=' * 60}")

    metric_keys = [
        'accuracy', 'balanced_accuracy', 'f1_macro', 'f1_binary',
        'sensitivity', 'specificity', 'auc_roc',
        'optimal_threshold', 'optimal_sensitivity', 'optimal_specificity', 'optimal_balanced_acc',
    ]
    header = f"{'Metric':<22}" + ''.join(f"{r['experiment']:>18}" for r in results)
    print(header)
    print('-' * (22 + 18 * len(results)))

    for key in metric_keys:
        row = f"{key:<22}"
        values = [r['metrics'].get(key) for r in results]
        best_val = max(v for v in values if v is not None)
        for v in values:
            cell = f"{v:.4f}" if v is not None else 'N/A'
            marker = ' *' if v == best_val else '  '
            row += f"{cell + marker:>18}"
        print(row)

    print("\n  * = best value per metric")


def main():
    parser = argparse.ArgumentParser(description='UDA Leukemia Evaluation on Test Set')
    parser.add_argument('--exp', type=str, nargs='+',
                        default=['source_only', 'uda_pseudolabel'],
                        help='Experiment name(s) to evaluate')
    parser.add_argument('--test-dir', type=str, default='dataset/test')
    parser.add_argument('--ckpt-root', type=str, default='checkpoints')
    parser.add_argument('--manifest', type=str,
                        default='dataset/metadata/dataset_manifest.csv')
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=None,
                        help='Specific seed to evaluate (None = auto-find)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Decision threshold for Abnormal class (default=0.5). '
                             'Use Youden J optimal from a previous run or sweep manually.')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    all_results = []
    for exp_name in args.exp:
        result = evaluate_experiment(
            exp_name=exp_name,
            test_dir=args.test_dir,
            ckpt_root=args.ckpt_root,
            manifest_path=args.manifest,
            results_dir=args.results_dir,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
            threshold=args.threshold,
        )
        if result is not None:
            all_results.append(result)

    if len(all_results) >= 2:
        compare_experiments(all_results)


if __name__ == '__main__':
    main()
