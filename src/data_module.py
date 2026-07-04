import os
import random
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.segmentation import slic
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms

import lightning as L

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def compute_saliency_map(image_np: np.ndarray) -> np.ndarray:
    if image_np.dtype != np.uint8:
        image_np = (image_np * 255).astype(np.uint8)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0

    fft = np.fft.fft2(gray)
    log_amp = np.log(np.abs(fft) + 1e-8)
    phase = np.angle(fft)
    spectral_residual = log_amp - cv2.blur(log_amp, (3, 3))

    sal = np.abs(np.fft.ifft2(np.exp(spectral_residual + 1j * phase))) ** 2
    sal = cv2.GaussianBlur(sal, (9, 9), 2.5)

    s_min, s_max = sal.min(), sal.max()
    if s_max - s_min > 1e-8:
        sal = (sal - s_min) / (s_max - s_min)
    else:
        sal = np.zeros_like(sal)
    return sal.astype(np.float32)


def focus_aug_mix(
    image_a_np: np.ndarray,
    image_b_np: np.ndarray,
    gradcam_map: Optional[np.ndarray] = None,
    use_saliency: bool = True,
    n_segments: int = 50,
    compactness: float = 10.0,
    paste_ratio: float = 0.25,
    saliency_weight: float = 0.6,
) -> Tuple[np.ndarray, float]:
    h, w = image_a_np.shape[:2]
    if image_b_np.shape[:2] != (h, w):
        image_b_np = cv2.resize(image_b_np, (w, h), interpolation=cv2.INTER_LINEAR)

    segments = slic(
        image_a_np, n_segments=n_segments, compactness=compactness,
        start_label=0, channel_axis=2,
    )

    if use_saliency:
        score_map = compute_saliency_map(image_b_np)
    else:
        score_map = np.zeros((h, w), dtype=np.float32)

    if gradcam_map is not None:
        if gradcam_map.shape != (h, w):
            gradcam_map = cv2.resize(gradcam_map, (w, h), interpolation=cv2.INTER_LINEAR)
        if use_saliency:
            score_map = saliency_weight * score_map + (1 - saliency_weight) * gradcam_map
        else:
            score_map = gradcam_map.astype(np.float32)

    seg_ids = np.unique(segments)
    seg_scores = np.array([score_map[segments == s].mean() for s in seg_ids])
    order = np.argsort(seg_scores)[::-1]

    num_paste = max(1, int(len(seg_ids) * paste_ratio))
    paste_ids = seg_ids[order[:num_paste]]

    paste_mask = np.isin(segments, paste_ids)
    mixed = image_a_np.copy()
    mixed[paste_mask] = image_b_np[paste_mask]

    lam = 1.0 - paste_mask.mean()
    return mixed, float(lam)


def saliency_mix(
    image_a_np: np.ndarray,
    image_b_np: np.ndarray,
    patch_ratio: float = 0.25,
) -> Tuple[np.ndarray, float]:
    h, w = image_a_np.shape[:2]
    if image_b_np.shape[:2] != (h, w):
        image_b_np = cv2.resize(image_b_np, (w, h), interpolation=cv2.INTER_LINEAR)

    sal = compute_saliency_map(image_b_np)
    cy, cx = np.unravel_index(np.argmax(sal), sal.shape)

    ph = int(h * np.sqrt(patch_ratio))
    pw = int(w * np.sqrt(patch_ratio))
    x1 = max(0, cx - pw // 2); y1 = max(0, cy - ph // 2)
    x2 = min(w, x1 + pw);      y2 = min(h, y1 + ph)

    mixed = image_a_np.copy()
    mixed[y1:y2, x1:x2] = image_b_np[y1:y2, x1:x2]
    lam = 1.0 - ((y2 - y1) * (x2 - x1)) / (h * w)
    return mixed, float(lam)


class ReinhardJitter:
    def __init__(self, sigma_mean: float = 0.15, sigma_std: float = 0.10):
        self.sigma_mean = sigma_mean
        self.sigma_std  = sigma_std

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img, dtype=np.uint8)
        lab    = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        for ch in range(3):
            mu  = lab[:, :, ch].mean()
            std = lab[:, :, ch].std() + 1e-6
            lab[:, :, ch] = (lab[:, :, ch] - mu) / std
            new_mu  = mu  + np.random.normal(0, self.sigma_mean * std)
            new_std = std * np.exp(np.random.normal(0, self.sigma_std))
            lab[:, :, ch] = lab[:, :, ch] * new_std + new_mu
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


class FocusAugMixDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        classes: List[str],
        class_to_idx: Dict[str, int],
        transform: Optional[Callable] = None,
        aug_mode: str = 'focusmix',
        aug_prob: float = 0.5,
        n_segments: int = 50,
        compactness: float = 10.0,
        paste_ratio: float = 0.25,
    ):
        assert aug_mode in {'none', 'saliency', 'focusmix', 'focusmix_cam'}
        self.samples = samples
        self.transform = transform
        self.aug_mode = aug_mode
        self.aug_prob = aug_prob
        self.n_segments = n_segments
        self.compactness = compactness
        self.paste_ratio = paste_ratio
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.gradcam_maps = {}

    def set_gradcam_maps(self, maps: dict):
        self.gradcam_maps = maps

    def __len__(self):
        return len(self.samples)

    def _sample_partner(self, idx: int) -> int:
        partner = random.randint(0, len(self.samples) - 1)
        while partner == idx:
            partner = random.randint(0, len(self.samples) - 1)
        return partner

    def __getitem__(self, idx):
        path_a, label_a = self.samples[idx]
        img_a = Image.open(path_a).convert('RGB')
        img_a_np = np.array(img_a)

        if self.aug_mode == 'none' or random.random() >= self.aug_prob:
            if self.transform:
                img_a = self.transform(img_a)
            return img_a, label_a, label_a, 1.0

        idx_b = self._sample_partner(idx)
        path_b, label_b = self.samples[idx_b]
        img_b = Image.open(path_b).convert('RGB')
        img_b_np = np.array(img_b)

        if self.aug_mode == 'saliency':
            mixed_np, lam = saliency_mix(img_a_np, img_b_np, self.paste_ratio)
        else:
            cam = self.gradcam_maps.get(idx) if self.aug_mode == 'focusmix_cam' else None
            mixed_np, lam = focus_aug_mix(
                img_a_np, img_b_np,
                gradcam_map=cam,
                use_saliency=True,
                n_segments=self.n_segments,
                compactness=self.compactness,
                paste_ratio=self.paste_ratio,
            )

        mixed_pil = Image.fromarray(mixed_np)
        if self.transform:
            mixed_pil = self.transform(mixed_pil)
        return mixed_pil, label_a, label_b, lam


def focusaugmix_collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets_a = torch.tensor([b[1] for b in batch], dtype=torch.long)
    targets_b = torch.tensor([b[2] for b in batch], dtype=torch.long)
    lam = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    return images, targets_a, targets_b, lam


def build_dataset_weights(
    image_paths: List[str],
    manifest_csv: str,
    dataset_column: str = 'source_dataset',
) -> List[float]:
    try:
        df = pd.read_csv(manifest_csv)
        name_to_dataset: Dict[str, str] = dict(
            zip(df['output_filename'], df[dataset_column])
        )
    except Exception:
        return [1.0] * len(image_paths)

    dataset_labels = []
    for p in image_paths:
        fname = os.path.basename(p)
        ds = name_to_dataset.get(fname, 'unknown')
        dataset_labels.append(ds)

    unique_datasets = list(set(dataset_labels))
    dataset_counts = {ds: dataset_labels.count(ds) for ds in unique_datasets}
    n_datasets = len(unique_datasets)
    total = len(image_paths)

    dataset_weight = {
        ds: total / (n_datasets * count)
        for ds, count in dataset_counts.items()
    }

    weights = [dataset_weight[ds] for ds in dataset_labels]
    return weights


class LeukemiaDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = 'dataset',
        batch_size: int = 32,
        num_workers: int = 8,
        aug_mode: str = 'focusmix',
        aug_prob: float = 0.5,
        n_segments: int = 50,
        compactness: float = 10.0,
        paste_ratio: float = 0.25,
        image_size: int = 224,
        use_robust_aug: bool = False,
        stain_sigma_mean: float = 0.15,
        stain_sigma_std: float = 0.10,
        stain_aug_prob: float = 0.5,
        use_dataset_weighted_sampling: bool = False,
        manifest_csv: str = '',
        uda_mode: bool = False,
    ):
        super().__init__()
        self.data_dir         = data_dir
        self.batch_size       = batch_size
        self.num_workers      = num_workers
        self.aug_mode         = aug_mode
        self.aug_prob         = aug_prob
        self.n_segments       = n_segments
        self.compactness      = compactness
        self.paste_ratio      = paste_ratio
        self.image_size       = image_size
        self.use_robust_aug   = use_robust_aug
        self.stain_sigma_mean = stain_sigma_mean
        self.stain_sigma_std  = stain_sigma_std
        self.stain_aug_prob   = stain_aug_prob
        self.use_dataset_weighted_sampling = use_dataset_weighted_sampling
        self.manifest_csv     = manifest_csv or os.path.join(data_dir, 'metadata', 'dataset_manifest.csv')
        self.save_hyperparameters()

        if use_robust_aug:
            self.train_transform = transforms.Compose([
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08),
                transforms.RandomApply([ReinhardJitter(sigma_mean=self.stain_sigma_mean, sigma_std=self.stain_sigma_std)], p=self.stain_aug_prob),
                transforms.RandomGrayscale(p=0.05),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08),
                transforms.RandomGrayscale(p=0.05),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        self.train_dataset = None
        self.val_dataset = None
        self.classes = None
        self.num_classes = None

        self.uda_mode = uda_mode
        self.target_samples = []
        self.source_samples = []
        self.pseudo_labels = {}

    def setup(self, stage=None):
        train_root = os.path.join(self.data_dir, 'train')
        full_dataset = datasets.ImageFolder(root=train_root)
        
        try:
            df = pd.read_csv(self.manifest_csv)
            name_to_ds = dict(zip(df['output_filename'], df['source_dataset']))
        except Exception:
            name_to_ds = {}

        self.source_samples = []
        self.target_samples = []
        
        for path, target in full_dataset.samples:
            fname = os.path.basename(path)
            if name_to_ds.get(fname) == 'Taleqani':
                self.target_samples.append((path, target))
            else:
                self.source_samples.append((path, target))

        self.classes = full_dataset.classes
        self.num_classes = len(self.classes)
        self._class_to_idx = full_dataset.class_to_idx
        self.update_train_dataset()

        val_root = os.path.join(self.data_dir, 'val')
        if os.path.isdir(val_root):
            self.val_dataset = datasets.ImageFolder(
                root=val_root,
                transform=self.val_transform,
            )
        else:
            self.val_dataset = datasets.ImageFolder(
                root=os.path.join(self.data_dir, 'test'),
                transform=self.val_transform,
            )

        if self.use_dataset_weighted_sampling:
            image_paths = [s[0] for s in self.train_dataset.samples]
            self._dataset_sample_weights = build_dataset_weights(
                image_paths, self.manifest_csv
            )
        else:
            self._dataset_sample_weights = None

    def update_pseudo_labels(self, pseudo_labels: Dict[str, int]):
        self.pseudo_labels = pseudo_labels
        self.update_train_dataset()
        if self.use_dataset_weighted_sampling:
            image_paths = [s[0] for s in self.train_dataset.samples]
            self._dataset_sample_weights = build_dataset_weights(
                image_paths, self.manifest_csv
            )

    def update_train_dataset(self):
        active_samples = list(self.source_samples)
        
        # Inject target samples that have pseudo-labels
        for path, _ in self.target_samples:
            fname = os.path.basename(path)
            if fname in self.pseudo_labels:
                active_samples.append((path, self.pseudo_labels[fname]))

        self.train_dataset = FocusAugMixDataset(
            samples=active_samples,
            classes=self.classes,
            class_to_idx=self._class_to_idx,
            transform=self.train_transform,
            aug_mode=self.aug_mode.replace('uda_', ''),
            aug_prob=self.aug_prob,
            n_segments=self.n_segments,
            compactness=self.compactness,
            paste_ratio=self.paste_ratio,
        )

    def get_class_weights(self) -> torch.Tensor:
        targets = torch.tensor([s[1] for s in self.train_dataset.samples])
        counts = torch.bincount(targets, minlength=self.num_classes)
        # Avoid division by zero
        counts = torch.max(counts, torch.tensor(1))
        weights = targets.numel() / (len(counts) * counts.float())
        return weights

    def train_dataloader(self):
        n_workers = 0 if self.aug_mode == 'focusmix_cam' else self.num_workers

        if self._dataset_sample_weights is not None:
            sampler = WeightedRandomSampler(
                weights=self._dataset_sample_weights,
                num_samples=len(self._dataset_sample_weights),
                replacement=True,
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=n_workers,
                collate_fn=focusaugmix_collate_fn,
                pin_memory=False,
                persistent_workers=(n_workers > 0),
                drop_last=False,
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=n_workers,
            collate_fn=focusaugmix_collate_fn,
            pin_memory=False,
            persistent_workers=(n_workers > 0),
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def external_test_dataloader(
        self,
        test_dir: str,
        normalizer=None,
        num_workers: Optional[int] = None,
    ) -> DataLoader:
        CNMC_ALIAS: Dict[str, str] = {
            'all': 'Abnormal', 'ALL': 'Abnormal',
            'hem': 'Normal',   'HEM': 'Normal',
        }

        test_path = os.path.abspath(test_dir)

        class _ExternalDataset(Dataset):
            def __init__(self_, root, cls_map, transform, normalizer_):
                exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
                self_.samples: List[Tuple[str, int]] = []
                self_.transform    = transform
                self_.normalizer   = normalizer_
                for subdir in sorted(os.listdir(root)):
                    full_subdir = os.path.join(root, subdir)
                    if not os.path.isdir(full_subdir):
                        continue
                    canonical = cls_map.get(subdir, subdir)
                    label     = self.class_to_idx.get(canonical)
                    if label is None:
                        continue
                    for fname in sorted(os.listdir(full_subdir)):
                        if os.path.splitext(fname)[1].lower() in exts:
                            self_.samples.append(
                                (os.path.join(full_subdir, fname), label)
                            )

            def __len__(self_):
                return len(self_.samples)

            def __getitem__(self_, idx):
                path, label = self_.samples[idx]
                img = Image.open(path).convert('RGB')
                if self_.normalizer is not None:
                    img_np = np.array(img)
                    try:
                        img_np = self_.normalizer.transform(img_np)
                        img    = Image.fromarray(img_np)
                    except Exception:
                        pass
                return self_.transform(img), label

        cls_map = {**CNMC_ALIAS}
        dataset = _ExternalDataset(
            root        = test_path,
            cls_map     = cls_map,
            transform   = self.val_transform,
            normalizer_ = normalizer,
        )

        n_workers = num_workers if num_workers is not None else (
            0 if normalizer is not None else self.num_workers
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=n_workers,
            pin_memory=(n_workers > 0),
            persistent_workers=(n_workers > 0),
        )
