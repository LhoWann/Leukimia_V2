from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple


def _rgb_to_od(rgb: np.ndarray) -> np.ndarray:
    rgb_f = np.clip(rgb.astype(np.float64), 1, 255)
    return -np.log(rgb_f / 255.0)


def _od_to_rgb(od: np.ndarray) -> np.ndarray:
    rgb = np.exp(-od) * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


class MacenkoNormalizer:

    def __init__(
        self,
        luminosity_threshold: float = 0.15,
        angular_percentile: float = 99.0,
    ):
        self.luminosity_threshold = luminosity_threshold
        self.angular_percentile = angular_percentile
        self._stain_matrix_ref: Optional[np.ndarray] = None
        self._max_conc_ref: Optional[np.ndarray] = None

    def _get_stain_matrix(self, image: np.ndarray) -> np.ndarray:
        od = _rgb_to_od(image).reshape(-1, 3)
        od_norm = np.linalg.norm(od, axis=1)
        tissue = od[od_norm > self.luminosity_threshold]

        if len(tissue) < 10:
            return np.eye(3, 2)

        _, _, Vt = np.linalg.svd(tissue, full_matrices=False)
        V = Vt[:2].T

        proj = tissue @ V
        angles = np.arctan2(proj[:, 1], proj[:, 0])

        lo = np.percentile(angles, 100.0 - self.angular_percentile)
        hi = np.percentile(angles, self.angular_percentile)

        v1 = V @ np.array([np.cos(lo), np.sin(lo)])
        v2 = V @ np.array([np.cos(hi), np.sin(hi)])

        if v1[0] < v2[0]:
            v1, v2 = v2, v1

        HE = np.stack([v1, v2], axis=1)
        HE /= np.linalg.norm(HE, axis=0, keepdims=True) + 1e-8
        return HE

    def _get_concentrations(
        self,
        image: np.ndarray,
        stain_matrix: np.ndarray,
    ) -> np.ndarray:
        od = _rgb_to_od(image).reshape(-1, 3).T
        C, _, _, _ = np.linalg.lstsq(stain_matrix, od, rcond=None)
        return C

    def fit(self, reference_image: np.ndarray) -> "MacenkoNormalizer":
        self._stain_matrix_ref = self._get_stain_matrix(reference_image)
        C_ref = self._get_concentrations(reference_image, self._stain_matrix_ref)
        self._max_conc_ref = np.percentile(C_ref, 99, axis=1)
        return self

    def transform(self, source_image: np.ndarray) -> np.ndarray:
        if self._stain_matrix_ref is None:
            raise RuntimeError("Call .fit(reference_image) before .transform().")

        H, W = source_image.shape[:2]
        SM_src = self._get_stain_matrix(source_image)
        C_src = self._get_concentrations(source_image, SM_src)

        max_src = np.percentile(C_src, 99, axis=1, keepdims=True)
        max_src = np.maximum(max_src, 1e-6)
        C_norm = C_src / max_src * self._max_conc_ref[:, np.newaxis]

        od_norm = self._stain_matrix_ref @ C_norm
        rgb_norm = _od_to_rgb(od_norm.T.reshape(H, W, 3))
        return rgb_norm

    def fit_transform(
        self,
        reference_image: np.ndarray,
        source_image: np.ndarray,
    ) -> np.ndarray:
        return self.fit(reference_image).transform(source_image)


class ReinhardNormalizer:

    def __init__(self, clip: bool = True):
        self.clip = clip
        self._target_mean: Optional[np.ndarray] = None
        self._target_std: Optional[np.ndarray] = None

    @staticmethod
    def _lab_stats(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        mean = lab.mean(axis=(0, 1))
        std  = lab.std(axis=(0, 1)) + 1e-6
        return mean, std

    def fit(self, reference_image: np.ndarray) -> "ReinhardNormalizer":
        self._target_mean, self._target_std = self._lab_stats(reference_image)
        return self

    def fit_from_stats(
        self,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> "ReinhardNormalizer":
        self._target_mean = np.asarray(mean, dtype=np.float32)
        self._target_std  = np.asarray(std, dtype=np.float32)
        return self

    def transform(self, source_image: np.ndarray) -> np.ndarray:
        if self._target_mean is None:
            raise RuntimeError("Call .fit() or .fit_from_stats() before .transform().")

        lab = cv2.cvtColor(source_image, cv2.COLOR_RGB2LAB).astype(np.float32)
        src_mean, src_std = self._lab_stats(source_image)

        lab_norm = (lab - src_mean) / src_std * self._target_std + self._target_mean

        if self.clip:
            lab_norm = np.clip(lab_norm, 0, 255)
        lab_uint8 = lab_norm.astype(np.uint8)
        return cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2RGB)

    def fit_transform(
        self,
        reference_image: np.ndarray,
        source_image: np.ndarray,
    ) -> np.ndarray:
        return self.fit(reference_image).transform(source_image)


def compute_reference_from_dir(
    directory: str,
    n_samples: int = 100,
    image_size: int = 224,
    seed: int = 42,
    balanced: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import random as rnd
    rnd.seed(seed)

    root = Path(directory)
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    selected: list
    if balanced:
        subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
        if len(subdirs) >= 2:
            per_class = max(1, n_samples // len(subdirs))
            selected = []
            for subdir in subdirs:
                imgs = sorted([p for p in subdir.iterdir() if p.suffix.lower() in exts])
                rnd.shuffle(imgs)
                selected.extend(imgs[:per_class])
        else:
            balanced = False

    if not balanced:
        all_imgs = [p for p in root.rglob('*') if p.suffix.lower() in exts]
        rnd.shuffle(all_imgs)
        selected = all_imgs[:n_samples]

    lab_means, lab_stds, imgs_np = [], [], []

    for img_path in selected:
        try:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img.resize((image_size, image_size), Image.BILINEAR))
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
            lab_means.append(lab.mean(axis=(0, 1)))
            lab_stds.append(lab.std(axis=(0, 1)))
            imgs_np.append(img_np)
        except Exception:
            continue

    if not lab_means:
        raise RuntimeError(f"No valid images found in {directory}")

    means_arr   = np.stack(lab_means)
    mean_global = means_arr.mean(axis=0)
    std_global  = np.stack(lab_stds).mean(axis=0)

    dists = np.linalg.norm(means_arr - mean_global, axis=1)
    reference_image = imgs_np[int(np.argmin(dists))]

    return reference_image, mean_global, std_global
