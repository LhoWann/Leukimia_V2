from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


VALID_IMG_EXT = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

ABNORMAL = 'Abnormal'
NORMAL = 'Normal'

WBC_HSV_LOWER = np.array([110, 30, 20])
WBC_HSV_UPPER = np.array([175, 255, 160])
WBC_AREA_MIN = 1000
WBC_AREA_MAX = 35000
WBC_CIRCULARITY_MIN = 0.35
WBC_MIN_CELL_DIST = 200
WBC_MAX_CELLS_PER_IMG = 20

SNAM_NUCLEUS_AREA_MIN = 500
SNAM_CROP_SIZE = 512
SNAM_OUTPUT_SIZE = 257

ALLIDB_CROP_SIZE = 257

BLUR_THRESHOLD = 50
BRIGHTNESS_LOW = 30
BRIGHTNESS_HIGH = 230
EMPTY_WHITE_THRESHOLD = 0.85

SEED = 42
CNMC_TRAIN_RATIO = 0.70
JPEG_QUALITY = 95


@dataclass
class QualityResult:
    blur_score: float
    brightness: float
    is_blurry: bool
    is_overexposed: bool
    is_underexposed: bool
    is_empty: bool
    is_corrupt: bool

    @property
    def is_ok(self) -> bool:
        return not (self.is_corrupt or self.is_empty)

    @property
    def flag(self) -> str:
        if self.is_corrupt:
            return 'REJECTED'
        if self.is_empty:
            return 'REJECTED'
        if self.is_blurry or self.is_overexposed or self.is_underexposed:
            return 'FLAGGED'
        return 'OK'


@dataclass
class ManifestEntry:
    output_filename: str
    source_dataset: str
    source_path: str
    original_label: str
    mapped_label: str
    patient_id: str
    split: str
    width: int
    height: int
    blur_score: float
    brightness: float
    quality_flag: str


@dataclass
class DatasetStats:
    name: str
    raw_files: int = 0
    output_files: int = 0
    rejected: int = 0
    flagged: int = 0
    abnormal_count: int = 0
    normal_count: int = 0


def quality_check(img_np: np.ndarray) -> QualityResult:
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    white_ratio = float(np.mean(gray > 240))
    return QualityResult(
        blur_score=round(blur_score, 2),
        brightness=round(brightness, 2),
        is_blurry=blur_score < BLUR_THRESHOLD,
        is_overexposed=brightness > BRIGHTNESS_HIGH,
        is_underexposed=brightness < BRIGHTNESS_LOW,
        is_empty=white_ratio > EMPTY_WHITE_THRESHOLD,
        is_corrupt=False,
    )


def save_jpg(img_np: np.ndarray, out_path: str) -> None:
    Image.fromarray(img_np).save(out_path, 'JPEG', quality=JPEG_QUALITY)


def load_image_safe(path: str) -> Optional[np.ndarray]:
    try:
        img = Image.open(path)
        img.verify()
        img = Image.open(path).convert('RGB')
        return np.array(img)
    except Exception:
        return None


def load_image_rgba_safe(path: str) -> Optional[np.ndarray]:
    try:
        img = Image.open(path)
        img.verify()
        img = Image.open(path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        else:
            img = img.convert('RGB')
        return np.array(img)
    except Exception:
        return None


def parse_xyc(xyc_path: str) -> List[Tuple[int, int]]:
    centroids = []
    if not os.path.exists(xyc_path):
        return centroids
    with open(xyc_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    centroids.append((int(float(parts[0])), int(float(parts[1]))))
                except ValueError:
                    continue
    return centroids


def _circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter < 1e-5:
        return 0.0
    return 4 * np.pi * area / (perimeter ** 2)


def _nms_centroids(centroids, min_dist):
    centroids = sorted(centroids, key=lambda c: c[2], reverse=True)
    kept = []
    for cx, cy, area in centroids:
        too_close = any(
            (cx - kx) ** 2 + (cy - ky) ** 2 < min_dist ** 2
            for kx, ky, _ in kept
        )
        if not too_close:
            kept.append((cx, cy, area))
    return kept


def detect_wbc_centroids(img_rgb: np.ndarray) -> List[Tuple[int, int]]:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, WBC_HSV_LOWER, WBC_HSV_UPPER)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (WBC_AREA_MIN < area < WBC_AREA_MAX):
            continue
        if _circularity(cnt) < WBC_CIRCULARITY_MIN:
            continue
        M = cv2.moments(cnt)
        if M['m00'] < 1:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        candidates.append((cx, cy, area))
    candidates = _nms_centroids(candidates, WBC_MIN_CELL_DIST)
    candidates = candidates[:WBC_MAX_CELLS_PER_IMG]
    return [(cx, cy) for cx, cy, _ in candidates]


def crop_around_centroid(img_np, cx, cy, size):
    h, w = img_np.shape[:2]
    half = size // 2
    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = x1 + size
    y2 = y1 + size
    if x2 > w:
        x2 = w
        x1 = max(w - size, 0)
    if y2 > h:
        y2 = h
        y1 = max(h - size, 0)
    crop = img_np[y1:y2, x1:x2]
    if crop.shape[0] != size or crop.shape[1] != size:
        crop = np.array(Image.fromarray(crop).resize((size, size), Image.LANCZOS))
    return crop


def detect_centroids_from_mask(mask_path: str, area_min: int = SNAM_NUCLEUS_AREA_MIN) -> List[Tuple[int, int]]:
    mask_img = load_image_safe(mask_path)
    if mask_img is None:
        return []
    gray = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_min:
            continue
        M = cv2.moments(cnt)
        if M['m00'] < 1:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        centroids.append((cx, cy))
    return centroids


class ALLIDBProcessor:

    def __init__(self, raw_dir: str, output_dir: str):
        self.idb1_im_dir = os.path.join(raw_dir, 'ALL_IDB', 'ALL_IDB1', 'im')
        self.idb1_xyc_dir = os.path.join(raw_dir, 'ALL_IDB', 'ALL_IDB1', 'xyc')
        self.idb2_img_dir = os.path.join(raw_dir, 'ALL_IDB', 'ALL_IDB2', 'img')
        self.output_dir = os.path.join(output_dir, 'ALL_IDB')
        self.stats = DatasetStats(name='ALL-IDB')
        self.entries: List[ManifestEntry] = []

    def _get_class_label(self, stem: str) -> Optional[str]:
        parts = stem.rsplit('_', 1)
        if len(parts) == 2:
            if parts[1] == '1':
                return ABNORMAL
            if parts[1] == '0':
                return NORMAL
        return None

    def _process_idb1(self) -> None:
        print('  Processing ALL-IDB1 (full-field segmentation)')
        img_files = sorted(
            f for f in os.listdir(self.idb1_im_dir)
            if Path(f).suffix.lower() in VALID_IMG_EXT
        )
        self.stats.raw_files += len(img_files)

        for fname in img_files:
            stem = Path(fname).stem
            class_name = self._get_class_label(stem)
            if class_name is None:
                continue

            img_np = load_image_safe(os.path.join(self.idb1_im_dir, fname))
            if img_np is None:
                self.stats.rejected += 1
                continue

            if class_name == ABNORMAL:
                xyc_path = os.path.join(self.idb1_xyc_dir, stem + '.xyc')
                centroids = parse_xyc(xyc_path)
            else:
                centroids = detect_wbc_centroids(img_np)

            if not centroids:
                self.stats.rejected += 1
                continue

            out_dir = os.path.join(self.output_dir, class_name)
            os.makedirs(out_dir, exist_ok=True)

            for i, (cx, cy) in enumerate(centroids):
                crop = crop_around_centroid(img_np, cx, cy, ALLIDB_CROP_SIZE)
                qr = quality_check(crop)

                out_name = f'ALLIDB1_{stem}_cell{i:03d}.jpg'
                out_path = os.path.join(out_dir, out_name)

                if not qr.is_ok:
                    self.stats.rejected += 1
                    continue

                if qr.flag == 'FLAGGED':
                    self.stats.flagged += 1

                save_jpg(crop, out_path)
                self.stats.output_files += 1
                if class_name == ABNORMAL:
                    self.stats.abnormal_count += 1
                else:
                    self.stats.normal_count += 1

                self.entries.append(ManifestEntry(
                    output_filename=out_name,
                    source_dataset='ALL-IDB',
                    source_path=f'ALL_IDB/ALL_IDB1/im/{fname}',
                    original_label=stem.rsplit('_', 1)[1],
                    mapped_label=class_name,
                    patient_id='',
                    split='train',
                    width=crop.shape[1],
                    height=crop.shape[0],
                    blur_score=qr.blur_score,
                    brightness=qr.brightness,
                    quality_flag=qr.flag,
                ))

    def _process_idb2(self) -> None:
        print('  Processing ALL-IDB2 (single-cell copy)')
        img_files = sorted(
            f for f in os.listdir(self.idb2_img_dir)
            if Path(f).suffix.lower() in VALID_IMG_EXT
        )
        self.stats.raw_files += len(img_files)

        for fname in img_files:
            stem = Path(fname).stem
            class_name = self._get_class_label(stem)
            if class_name is None:
                continue

            img_np = load_image_safe(os.path.join(self.idb2_img_dir, fname))
            if img_np is None:
                self.stats.rejected += 1
                continue

            if img_np.shape[0] != ALLIDB_CROP_SIZE or img_np.shape[1] != ALLIDB_CROP_SIZE:
                img_np = np.array(
                    Image.fromarray(img_np).resize(
                        (ALLIDB_CROP_SIZE, ALLIDB_CROP_SIZE), Image.LANCZOS
                    )
                )

            qr = quality_check(img_np)
            if not qr.is_ok:
                self.stats.rejected += 1
                continue

            if qr.flag == 'FLAGGED':
                self.stats.flagged += 1

            out_dir = os.path.join(self.output_dir, class_name)
            os.makedirs(out_dir, exist_ok=True)
            out_name = f'ALLIDB2_{stem}.jpg'
            save_jpg(img_np, os.path.join(out_dir, out_name))
            self.stats.output_files += 1
            if class_name == ABNORMAL:
                self.stats.abnormal_count += 1
            else:
                self.stats.normal_count += 1

            self.entries.append(ManifestEntry(
                output_filename=out_name,
                source_dataset='ALL-IDB',
                source_path=f'ALL_IDB/ALL_IDB2/img/{fname}',
                original_label=stem.rsplit('_', 1)[1],
                mapped_label=class_name,
                patient_id='',
                split='train',
                width=ALLIDB_CROP_SIZE,
                height=ALLIDB_CROP_SIZE,
                blur_score=qr.blur_score,
                brightness=qr.brightness,
                quality_flag=qr.flag,
            ))

    def process(self) -> Tuple[DatasetStats, List[ManifestEntry]]:
        print('[1/5] Processing ALL-IDB.')
        self._process_idb1()
        self._process_idb2()
        print(f'  Done: {self.stats.output_files} images '
              f'(Abn={self.stats.abnormal_count}, Nor={self.stats.normal_count}, '
              f'rejected={self.stats.rejected}, flagged={self.stats.flagged})')
        return self.stats, self.entries


class CNMCProcessor:

    def __init__(self, raw_dir: str, output_dir: str):
        self.all_dir = os.path.join(raw_dir, 'C-NMC', 'C-NMC_train_merged', 'all')
        self.hem_dir = os.path.join(raw_dir, 'C-NMC', 'C-NMC_train_merged', 'hem')
        self.output_dir = os.path.join(output_dir, 'C_NMC')
        self.stats = DatasetStats(name='C-NMC')
        self.entries: List[ManifestEntry] = []
        self.patient_mapping: Dict[str, List[str]] = {}

    @staticmethod
    def _extract_patient_id(filename: str) -> str:
        parts = Path(filename).stem.split('_')
        if len(parts) >= 2:
            return parts[1]
        return 'unknown'

    def _process_folder(self, folder: str, original_label: str, mapped_label: str) -> None:
        files = sorted(
            f for f in os.listdir(folder)
            if Path(f).suffix.lower() in VALID_IMG_EXT
        )
        self.stats.raw_files += len(files)

        for fname in files:
            patient_id = self._extract_patient_id(fname)
            if patient_id not in self.patient_mapping:
                self.patient_mapping[patient_id] = []
            self.patient_mapping[patient_id].append(fname)

            img_np = load_image_safe(os.path.join(folder, fname))
            if img_np is None:
                self.stats.rejected += 1
                continue

            qr = quality_check(img_np)
            if not qr.is_ok:
                self.stats.rejected += 1
                continue

            if qr.flag == 'FLAGGED':
                self.stats.flagged += 1

            out_dir = os.path.join(self.output_dir, mapped_label)
            os.makedirs(out_dir, exist_ok=True)
            stem = Path(fname).stem
            out_name = f'CNMC_{stem}.jpg'
            save_jpg(img_np, os.path.join(out_dir, out_name))
            self.stats.output_files += 1
            if mapped_label == ABNORMAL:
                self.stats.abnormal_count += 1
            else:
                self.stats.normal_count += 1

            self.entries.append(ManifestEntry(
                output_filename=out_name,
                source_dataset='C-NMC',
                source_path=f'C-NMC/C-NMC_train_merged/{original_label}/{fname}',
                original_label=original_label,
                mapped_label=mapped_label,
                patient_id=patient_id,
                split='',
                width=img_np.shape[1],
                height=img_np.shape[0],
                blur_score=qr.blur_score,
                brightness=qr.brightness,
                quality_flag=qr.flag,
            ))

    def process(self) -> Tuple[DatasetStats, List[ManifestEntry]]:
        print('[2/5] Processing C-NMC.')
        self._process_folder(self.all_dir, 'all', ABNORMAL)
        self._process_folder(self.hem_dir, 'hem', NORMAL)
        print(f'  Done: {self.stats.output_files} images '
              f'(Abn={self.stats.abnormal_count}, Nor={self.stats.normal_count}, '
              f'rejected={self.stats.rejected}, flagged={self.stats.flagged})')
        print(f'  Unique patients: {len(self.patient_mapping)}')
        return self.stats, self.entries


class TaleqaniProcessor:

    SUBTYPE_MAP = {
        'Benign': NORMAL,
        'Early': ABNORMAL,
        'Pre': ABNORMAL,
        'Pro': ABNORMAL,
    }

    def __init__(self, raw_dir: str, output_dir: str):
        self.source_dir = os.path.join(raw_dir, 'Taleqani', 'Original')
        self.output_dir = os.path.join(output_dir, 'Taleqani')
        self.stats = DatasetStats(name='Taleqani')
        self.entries: List[ManifestEntry] = []

    def process(self) -> Tuple[DatasetStats, List[ManifestEntry]]:
        print('[3/5] Processing Taleqani (Original).')

        for subtype, mapped_label in self.SUBTYPE_MAP.items():
            subtype_dir = os.path.join(self.source_dir, subtype)
            if not os.path.isdir(subtype_dir):
                print(f'  WARNING: {subtype_dir} not found, skipping')
                continue

            files = sorted(
                f for f in os.listdir(subtype_dir)
                if Path(f).suffix.lower() in VALID_IMG_EXT
            )
            self.stats.raw_files += len(files)

            for fname in files:
                img_np = load_image_safe(os.path.join(subtype_dir, fname))
                if img_np is None:
                    self.stats.rejected += 1
                    continue

                qr = quality_check(img_np)
                if not qr.is_ok:
                    self.stats.rejected += 1
                    continue

                if qr.flag == 'FLAGGED':
                    self.stats.flagged += 1

                out_dir = os.path.join(self.output_dir, mapped_label)
                os.makedirs(out_dir, exist_ok=True)
                stem = Path(fname).stem
                out_name = f'TALEQANI_{subtype}_{stem}.jpg'
                save_jpg(img_np, os.path.join(out_dir, out_name))
                self.stats.output_files += 1
                if mapped_label == ABNORMAL:
                    self.stats.abnormal_count += 1
                else:
                    self.stats.normal_count += 1

                self.entries.append(ManifestEntry(
                    output_filename=out_name,
                    source_dataset='Taleqani',
                    source_path=f'Taleqani/Original/{subtype}/{fname}',
                    original_label=subtype,
                    mapped_label=mapped_label,
                    patient_id='',
                    split='',
                    width=img_np.shape[1],
                    height=img_np.shape[0],
                    blur_score=qr.blur_score,
                    brightness=qr.brightness,
                    quality_flag=qr.flag,
                ))

        print(f'  Done: {self.stats.output_files} images '
              f'(Abn={self.stats.abnormal_count}, Nor={self.stats.normal_count}, '
              f'rejected={self.stats.rejected}, flagged={self.stats.flagged})')
        return self.stats, self.entries


class SNAMProcessor:

    def __init__(self, raw_dir: str, output_dir: str):
        self.all_dir = os.path.join(raw_dir, 'PKG - SN-AM', 'ALL')
        self.mm_dir = os.path.join(raw_dir, 'PKG - SN-AM', 'MM')
        self.output_dir = os.path.join(output_dir, 'SN_AM')
        self.stats = DatasetStats(name='SN-AM')
        self.entries: List[ManifestEntry] = []

    @staticmethod
    def _is_raw_image(fname: str) -> bool:
        lower = fname.lower()
        if 'mask' in lower:
            return False
        return Path(fname).suffix.lower() in VALID_IMG_EXT

    def _find_nucleus_mask(self, folder: str, raw_name: str) -> Optional[str]:
        stem = Path(raw_name).stem
        mask_name = f'{stem}_nucleus_mask.bmp'
        mask_path = os.path.join(folder, mask_name)
        if os.path.exists(mask_path):
            return mask_path
        return None

    def _process_folder(self, folder: str, label_prefix: str, mapped_label: str) -> None:
        raw_files = sorted(f for f in os.listdir(folder) if self._is_raw_image(f))
        self.stats.raw_files += len(raw_files)

        for fname in raw_files:
            img_np = load_image_safe(os.path.join(folder, fname))
            if img_np is None:
                self.stats.rejected += 1
                continue

            mask_path = self._find_nucleus_mask(folder, fname)
            if mask_path:
                centroids = detect_centroids_from_mask(mask_path)
            else:
                centroids = detect_wbc_centroids(img_np)

            if not centroids:
                self.stats.rejected += 1
                print(f'    WARNING: No cells found in {fname}')
                continue

            stem = Path(fname).stem
            out_dir = os.path.join(self.output_dir, mapped_label)
            os.makedirs(out_dir, exist_ok=True)

            for i, (cx, cy) in enumerate(centroids):
                crop = crop_around_centroid(img_np, cx, cy, SNAM_CROP_SIZE)
                crop_resized = np.array(
                    Image.fromarray(crop).resize(
                        (SNAM_OUTPUT_SIZE, SNAM_OUTPUT_SIZE), Image.LANCZOS
                    )
                )

                qr = quality_check(crop_resized)
                if not qr.is_ok:
                    self.stats.rejected += 1
                    continue

                if qr.flag == 'FLAGGED':
                    self.stats.flagged += 1

                out_name = f'SNAM_{label_prefix}_{stem}_cell{i:03d}.jpg'
                save_jpg(crop_resized, os.path.join(out_dir, out_name))
                self.stats.output_files += 1
                if mapped_label == ABNORMAL:
                    self.stats.abnormal_count += 1
                else:
                    self.stats.normal_count += 1

                self.entries.append(ManifestEntry(
                    output_filename=out_name,
                    source_dataset='SN-AM',
                    source_path=f'PKG - SN-AM/{label_prefix}/{fname}',
                    original_label=label_prefix,
                    mapped_label=mapped_label,
                    patient_id='',
                    split='train',
                    width=SNAM_OUTPUT_SIZE,
                    height=SNAM_OUTPUT_SIZE,
                    blur_score=qr.blur_score,
                    brightness=qr.brightness,
                    quality_flag=qr.flag,
                ))

    def process(self) -> Tuple[DatasetStats, List[ManifestEntry]]:
        print('[4/4] Processing SN-AM...')
        self._process_folder(self.all_dir, 'ALL', ABNORMAL)
        print(f'  Done: {self.stats.output_files} images '
              f'(Abn={self.stats.abnormal_count}, Nor={self.stats.normal_count}, '
              f'rejected={self.stats.rejected}, flagged={self.stats.flagged})')
        return self.stats, self.entries


class PoolingEngine:

    def __init__(self, data_dir: str, dataset_dir: str):
        self.data_dir = data_dir
        self.dataset_dir = dataset_dir
        self.metadata_dir = os.path.join(dataset_dir, 'metadata')

    def _taleqani_stratified_split(
        self,
        taleqani_entries: List[ManifestEntry],
    ) -> Tuple[List[str], List[str]]:
        random.seed(SEED)

        abn_files = [e.output_filename for e in taleqani_entries if e.mapped_label == ABNORMAL]
        nor_files = [e.output_filename for e in taleqani_entries if e.mapped_label == NORMAL]

        random.shuffle(abn_files)
        random.shuffle(nor_files)

        # 20% Train for Unlabeled Target Adaptation
        TALEQANI_TRAIN_RATIO = 0.20
        abn_split = int(len(abn_files) * TALEQANI_TRAIN_RATIO)
        nor_split = int(len(nor_files) * TALEQANI_TRAIN_RATIO)

        train_files = set(abn_files[:abn_split]) | set(nor_files[:nor_split])
        test_files = set(abn_files[abn_split:]) | set(nor_files[nor_split:])

        overlap = train_files & test_files
        assert len(overlap) == 0, f'File overlap detected: {overlap}'

        print(f'  Taleqani split: {len(train_files)} train (Unlabeled), '
              f'{len(test_files)} test (Evaluation)')

        return sorted(train_files), sorted(test_files)

    def pool(
        self,
        all_entries: List[ManifestEntry],
        taleqani_entries: List[ManifestEntry],
    ) -> List[ManifestEntry]:
        print('\n[Pooling] Merging datasets.')

        train_files, test_files = self._taleqani_stratified_split(taleqani_entries)
        train_file_set = set(train_files)
        test_file_set = set(test_files)

        for d in [
            os.path.join(self.dataset_dir, 'train', ABNORMAL),
            os.path.join(self.dataset_dir, 'train', NORMAL),
            os.path.join(self.dataset_dir, 'test', ABNORMAL),
            os.path.join(self.dataset_dir, 'test', NORMAL),
            self.metadata_dir,
        ]:
            os.makedirs(d, exist_ok=True)

        final_entries = []
        train_abn, train_nor, test_abn, test_nor = 0, 0, 0, 0

        for entry in all_entries:
            if entry.source_dataset == 'Taleqani':
                if entry.output_filename in train_file_set:
                    entry.split = 'train'
                elif entry.output_filename in test_file_set:
                    entry.split = 'test'
                else:
                    entry.split = 'train'
            else:
                entry.split = 'train'

            _dataset_dir_map = {
                'ALL-IDB': 'ALL_IDB',
                'C-NMC': 'C_NMC',
                'Taleqani': 'Taleqani',
                'SN-AM': 'SN_AM',
            }
            if entry.source_dataset not in _dataset_dir_map:
                print(f'  WARNING: Unknown dataset "{entry.source_dataset}", skipping.')
                continue
            src_dir = os.path.join(self.data_dir, _dataset_dir_map[entry.source_dataset], entry.mapped_label)

            src_path = os.path.join(src_dir, entry.output_filename)
            dst_dir = os.path.join(self.dataset_dir, entry.split, entry.mapped_label)
            dst_path = os.path.join(dst_dir, entry.output_filename)

            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
            else:
                continue

            if entry.split == 'train':
                if entry.mapped_label == ABNORMAL:
                    train_abn += 1
                else:
                    train_nor += 1
            else:
                if entry.mapped_label == ABNORMAL:
                    test_abn += 1
                else:
                    test_nor += 1

            final_entries.append(entry)

        print(f'  Train: {train_abn} Abnormal + {train_nor} Normal = {train_abn + train_nor}')
        print(f'  Test:  {test_abn} Abnormal + {test_nor} Normal = {test_abn + test_nor}')
        print(f'  Total: {len(final_entries)}')

        return final_entries

    def generate_manifest(self, entries: List[ManifestEntry]) -> None:
        manifest_path = os.path.join(self.metadata_dir, 'dataset_manifest.csv')
        with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'output_filename', 'source_dataset', 'source_path',
                'original_label', 'mapped_label', 'patient_id', 'split',
                'width', 'height', 'blur_score', 'brightness', 'quality_flag',
            ])
            for e in entries:
                writer.writerow([
                    e.output_filename, e.source_dataset, e.source_path,
                    e.original_label, e.mapped_label, e.patient_id, e.split,
                    e.width, e.height, e.blur_score, e.brightness, e.quality_flag,
                ])
        print(f'  Manifest saved: {manifest_path}')

    def generate_split_assignments(
        self,
        train_files: List[str],
        test_files: List[str],
    ) -> None:
        path = os.path.join(self.metadata_dir, 'split_assignments.csv')
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['output_filename', 'split'])
            for p in train_files:
                writer.writerow([p, 'train'])
            for p in test_files:
                writer.writerow([p, 'test'])
        print(f'  Split assignments saved: {path}')

    def generate_preprocessing_log(
        self,
        all_stats: List[DatasetStats],
        entries: List[ManifestEntry],
        train_files: List[str],
        test_files: List[str],
    ) -> None:
        train_count = sum(1 for e in entries if e.split == 'train')
        test_count = sum(1 for e in entries if e.split == 'test')

        log = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'pipeline_version': '2.1_UDA',
            'seed': SEED,
            'taleqani_train_ratio': 0.20,
            'datasets_processed': [s.name for s in all_stats],
            'per_dataset': {s.name: asdict(s) for s in all_stats},
            'total_train': train_count,
            'total_test': test_count,
            'total': len(entries),
            'taleqani_train_files': train_files,
            'taleqani_test_files': test_files,
        }

        path = os.path.join(self.metadata_dir, 'preprocessing_log.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
        print(f'  Preprocessing log saved: {path}')

    def generate_quality_report(self, entries: List[ManifestEntry]) -> None:
        report: Dict = {'per_dataset': {}, 'overall': {}}

        datasets = set(e.source_dataset for e in entries)
        for ds in sorted(datasets):
            ds_entries = [e for e in entries if e.source_dataset == ds]
            ok = sum(1 for e in ds_entries if e.quality_flag == 'OK')
            flagged = sum(1 for e in ds_entries if e.quality_flag == 'FLAGGED')
            total = len(ds_entries)
            report['per_dataset'][ds] = {
                'total': total,
                'ok': ok,
                'flagged': flagged,
                'flagged_pct': round(flagged / total * 100, 2) if total > 0 else 0,
            }

        total = len(entries)
        ok = sum(1 for e in entries if e.quality_flag == 'OK')
        flagged = sum(1 for e in entries if e.quality_flag == 'FLAGGED')
        report['overall'] = {
            'total': total,
            'ok': ok,
            'flagged': flagged,
            'flagged_pct': round(flagged / total * 100, 2) if total > 0 else 0,
        }

        path = os.path.join(self.metadata_dir, 'quality_report.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f'  Quality report saved: {path}')


def run_sanity_checks(entries: List[ManifestEntry], dataset_dir: str) -> bool:
    print('\n[Sanity Checks]')
    passed = True

    train_entries = [e for e in entries if e.split == 'train']
    test_entries = [e for e in entries if e.split == 'test']
    train_abn = sum(1 for e in train_entries if e.mapped_label == ABNORMAL)
    train_nor = sum(1 for e in train_entries if e.mapped_label == NORMAL)

    checks = [
        ('Total train > 10000', len(train_entries) > 10000),
        ('Total test > 1000', len(test_entries) > 1000),
    ]

    taleqani_train_files = set(
        e.output_filename for e in entries
        if e.source_dataset == 'Taleqani' and e.split == 'train'
    )
    taleqani_test_files = set(
        e.output_filename for e in entries
        if e.source_dataset == 'Taleqani' and e.split == 'test'
    )
    file_overlap = taleqani_train_files & taleqani_test_files
    checks.append(('No Taleqani file overlap', len(file_overlap) == 0))

    filenames = [e.output_filename for e in entries]
    unique_filenames = set(filenames)
    checks.append(('No filename duplicates', len(filenames) == len(unique_filenames)))

    missing = 0
    for e in entries:
        fpath = os.path.join(dataset_dir, e.split, e.mapped_label, e.output_filename)
        if not os.path.exists(fpath):
            missing += 1
    checks.append(('All manifest files exist on disk', missing == 0))

    for desc, result in checks:
        status = 'PASS' if result else 'FAIL'
        if not result:
            passed = False
        print(f'  [{status}] {desc}')

    if not passed:
        print('\n  WARNING: Some sanity checks failed. Review output carefully.')
    else:
        print('\n  All sanity checks passed.')

    return passed


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(project_root, 'raw_data')
    data_dir = os.path.join(project_root, 'data')
    dataset_dir = os.path.join(project_root, 'dataset')

    print('=' * 60)
    print('Multi-Dataset Preprocessing Pipeline v2.0')
    print('=' * 60)
    print(f'Raw data:  {raw_dir}')
    print(f'Data dir:  {data_dir}')
    print(f'Dataset:   {dataset_dir}')
    print(f'Seed:      {SEED}')
    print()

    random.seed(SEED)

    all_stats = []
    all_entries = []

    allidb = ALLIDBProcessor(raw_dir, data_dir)
    stats, entries = allidb.process()
    all_stats.append(stats)
    all_entries.extend(entries)
    print()

    cnmc = CNMCProcessor(raw_dir, data_dir)
    stats, entries = cnmc.process()
    all_stats.append(stats)
    cnmc_entries = entries
    all_entries.extend(entries)
    print()

    taleqani = TaleqaniProcessor(raw_dir, data_dir)
    stats, entries = taleqani.process()
    all_stats.append(stats)
    taleqani_entries = entries
    all_entries.extend(entries)
    print()

    snam = SNAMProcessor(raw_dir, data_dir)
    stats, entries = snam.process()
    all_stats.append(stats)
    all_entries.extend(entries)
    print()

    print('=' * 60)
    print('Per-Dataset Summary:')
    print('-' * 60)
    for s in all_stats:
        print(f'  {s.name:20s} | raw={s.raw_files:6d} | out={s.output_files:6d} | '
              f'abn={s.abnormal_count:6d} | nor={s.normal_count:6d} | '
              f'rej={s.rejected:4d} | flag={s.flagged:4d}')
    print()

    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    pooler = PoolingEngine(data_dir, dataset_dir)
    train_files, test_files = pooler._taleqani_stratified_split(taleqani_entries)
    final_entries = pooler.pool(all_entries, taleqani_entries)

    print()
    pooler.generate_manifest(final_entries)
    pooler.generate_split_assignments(train_files, test_files)
    pooler.generate_preprocessing_log(all_stats, final_entries, train_files, test_files)
    pooler.generate_quality_report(final_entries)

    print()
    run_sanity_checks(final_entries, dataset_dir)

    print()
    print('=' * 60)
    print('Preprocessing complete.')
    print('=' * 60)


if __name__ == '__main__':
    main()