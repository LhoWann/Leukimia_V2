from __future__ import annotations

import csv
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

DATASET_DIR = PROJECT_ROOT / 'dataset'
MANIFEST_CSV = DATASET_DIR / 'metadata' / 'dataset_manifest.csv'
TRAIN_DIR = DATASET_DIR / 'train'
VAL_DIR = DATASET_DIR / 'val'

SOURCE_VAL_RATIO = 0.15
CNMC_VAL_PATIENTS_RATIO = 0.15
SEED = 42

ABNORMAL = 'Abnormal'
NORMAL = 'Normal'


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_cnmc_patient(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.split('_')
    if parts[0] == 'CNMC' and len(parts) >= 3:
        return parts[1] + '_' + parts[2]
    if len(parts) >= 2:
        return parts[1]
    return 'unknown'


def select_val_files_cnmc(
    train_rows: List[Dict[str, str]],
) -> Set[str]:
    patient_to_files: Dict[str, List[str]] = defaultdict(list)
    for row in train_rows:
        pid = extract_cnmc_patient(row['output_filename'])
        patient_to_files[pid].append(row['output_filename'])

    all_patients = sorted(patient_to_files.keys())
    random.seed(SEED)
    random.shuffle(all_patients)

    n_val = max(1, int(len(all_patients) * CNMC_VAL_PATIENTS_RATIO))
    val_patients = set(all_patients[:n_val])

    val_files: Set[str] = set()
    for pid in val_patients:
        val_files.update(patient_to_files[pid])

    print(f'  C-NMC: {len(all_patients)} patients, {n_val} val patients '
          f'-> {len(val_files)} val images')
    return val_files


def select_val_files_stratified(
    train_rows: List[Dict[str, str]],
    dataset_name: str,
) -> Set[str]:
    abn_files = [r['output_filename'] for r in train_rows if r['mapped_label'] == ABNORMAL]
    nor_files = [r['output_filename'] for r in train_rows if r['mapped_label'] == NORMAL]

    random.seed(SEED)
    random.shuffle(abn_files)
    random.shuffle(nor_files)

    n_abn_val = max(1, int(len(abn_files) * SOURCE_VAL_RATIO)) if abn_files else 0
    n_nor_val = max(1, int(len(nor_files) * SOURCE_VAL_RATIO)) if nor_files else 0

    val_files = set(abn_files[:n_abn_val]) | set(nor_files[:n_nor_val])
    print(f'  {dataset_name}: {len(abn_files)} Abn + {len(nor_files)} Nor in train '
          f'-> {n_abn_val} + {n_nor_val} = {len(val_files)} val images')
    return val_files


def main() -> None:
    if not MANIFEST_CSV.exists():
        print(f'ERROR: Manifest not found: {MANIFEST_CSV}')
        sys.exit(1)

    if VAL_DIR.exists():
        print(f'WARNING: {VAL_DIR} already exists. Aborting to prevent duplication.')
        print('Remove it manually first: Remove-Item -Recurse -Force dataset/val')
        sys.exit(1)

    (VAL_DIR / ABNORMAL).mkdir(parents=True, exist_ok=True)
    (VAL_DIR / NORMAL).mkdir(parents=True, exist_ok=True)

    rows = read_manifest(MANIFEST_CSV)

    source_train_by_ds: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row['source_dataset'] != 'Taleqani' and row['split'] == 'train':
            source_train_by_ds[row['source_dataset']].append(row)

    print('Source train images by dataset:')
    for ds, ds_rows in source_train_by_ds.items():
        print(f'  {ds}: {len(ds_rows)}')
    print()

    val_file_set: Set[str] = set()

    cnmc_val = select_val_files_cnmc(source_train_by_ds.get('C-NMC', []))
    val_file_set.update(cnmc_val)

    allidb_val = select_val_files_stratified(source_train_by_ds.get('ALL-IDB', []), 'ALL-IDB')
    val_file_set.update(allidb_val)

    snam_val = select_val_files_stratified(source_train_by_ds.get('SN-AM', []), 'SN-AM')
    val_file_set.update(snam_val)

    print(f'\nTotal val files selected: {len(val_file_set)}')
    print()

    moved = 0
    not_found = 0
    for row in rows:
        fname = row['output_filename']
        if fname not in val_file_set:
            continue
        label = row['mapped_label']
        src = TRAIN_DIR / label / fname
        dst = VAL_DIR / label / fname
        if src.exists():
            shutil.move(str(src), str(dst))
            row['split'] = 'val'
            moved += 1
        else:
            print(f'  WARNING: Not found in train: {src}')
            not_found += 1

    print(f'Moved to val: {moved} files')
    if not_found:
        print(f'Not found: {not_found} files')

    write_manifest(MANIFEST_CSV, rows)
    print(f'Manifest updated: {MANIFEST_CSV}')

    val_abn = list((VAL_DIR / ABNORMAL).iterdir())
    val_nor = list((VAL_DIR / NORMAL).iterdir())
    train_abn = list((TRAIN_DIR / ABNORMAL).iterdir())
    train_nor = list((TRAIN_DIR / NORMAL).iterdir())

    print()
    print('=' * 50)
    print('Final split summary:')
    print(f'  Train: {len(train_abn)} Abnormal + {len(train_nor)} Normal = {len(train_abn) + len(train_nor)}')
    print(f'  Val:   {len(val_abn)} Abnormal + {len(val_nor)} Normal = {len(val_abn) + len(val_nor)}')

    train_fnames = set(f.name for f in (TRAIN_DIR / ABNORMAL).iterdir()) | \
                   set(f.name for f in (TRAIN_DIR / NORMAL).iterdir())
    val_fnames = set(f.name for f in (VAL_DIR / ABNORMAL).iterdir()) | \
                 set(f.name for f in (VAL_DIR / NORMAL).iterdir())
    overlap = train_fnames & val_fnames
    if overlap:
        print(f'  ERROR: {len(overlap)} files exist in both train and val!')
    else:
        print('  No train/val overlap. OK.')
    print('=' * 50)


if __name__ == '__main__':
    main()
