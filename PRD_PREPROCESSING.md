# PRD: Multi-Dataset Preprocessing Pipeline

Dokumen spesifikasi preprocessing untuk 4 dataset leukemia yang akan digabung menjadi
satu pooled training set. Setiap keputusan preprocessing berdampak langsung pada kualitas
model -- tidak ada ruang untuk error di tahap ini.

---

## 1. Ringkasan Kondisi Raw Data

### 1.1 Inventaris Dataset

| Dataset | Lokasi Raw | Jumlah File | Format | Dimensi | Mode | Level |
|---------|-----------|-------------|--------|---------|------|-------|
| ALL-IDB1 | `raw_data/ALL_IDB/ALL_IDB1/im/` | 108 gambar | JPEG | 1712x1368 | RGB | Full-field (perlu segmentasi) |
| ALL-IDB2 | `raw_data/ALL_IDB/ALL_IDB2/img/` | 260 gambar | TIFF | 257x257 | RGB | Single-cell (sudah crop) |
| C-NMC | `raw_data/C-NMC/C-NMC_train_merged/` | 7.272 (all) + 3.389 (hem) = 10.661 | BMP | 450x450 | RGB | Single-cell |
| Taleqani (Original) | `raw_data/Taleqani/Original/` | 504 (Benign) + 985 (Early) + 963 (Pre) + 804 (Pro) = 3.256 | JPEG | 224x224 | RGB | Single-cell |
| Taleqani (Segmented) | `raw_data/Taleqani/Segmented/` | 504 + 985 + 963 + 804 = 3.256 | JPEG | 224x224 | RGB | Single-cell (background removed) |
| SN-AM | `raw_data/PKG - SN-AM/` | 31 (ALL) + 30 (MM) = 61 raw images | BMP | 2560x1920 | RGB | Full-field (perlu segmentasi) |

### 1.2 Detail SN-AM

- Format: BMP 2560x1920, magnifikasi 1000x (10x lebih tinggi dari dataset lain)
- File mask tersedia: `*_background_mask.bmp`, `*_nucleus_mask.bmp`, `*_cyto_mask.bmp` (sebagian)
- ALL: 31 gambar full-field (29 numbered + 1 extra + 1 ref)
- MM: 30 gambar full-field

### 1.3 Catatan Kritis Per Dataset

**ALL-IDB1:**
- Gambar full-field 1712x1368, memerlukan segmentasi sel.
- File `.xyc` di `raw_data/ALL_IDB/ALL_IDB1/xyc/` berisi koordinat centroid sel blast.
- Sel normal dideteksi via HSV thresholding (pipeline V1 sudah ada di `src/segment_dataset.py`).
- Label di-encode dalam nama file: `Im0XX_Y.jpg` dimana Y=0 (Normal), Y=1 (Abnormal).
- Crop size V1: 257x257.

**ALL-IDB2:**
- Sudah single-cell 257x257.
- Label di-encode dalam nama file: `Im0XX_Y.tif` (Y=0/1).

**C-NMC:**
- Sudah single-cell 450x450.
- Struktur folder: `all/` (leukemia) dan `hem/` (healthy).
- Nama file mengandung Patient ID: `UID_XX_YY_Z_class.bmp` -- krusial untuk patient-aware split.

**Taleqani:**
- Tersedia 2 versi: Original dan Segmented (background sudah di-remove).
- Sudah single-cell 224x224.
- Subtype: Benign, Early (Pre-B), Pre (Pre-B), Pro (Pro-B).
- Ground truth dari flow cytometry (lebih reliable dari visual labeling).

---

## 2. Pipeline Preprocessing

### 2.1 Arsitektur Pipeline

```
raw_data/                          data/                          dataset/
(sumber asli)                      (intermediate, per-dataset)    (final, pooled)
                                                                  
ALL_IDB/ ------> [Step 1-5] ------> ALL_IDB/Abnormal/             train/
                                   ALL_IDB/Normal/                   Abnormal/
C-NMC/   ------> [Step 1-5] ------> C_NMC/Abnormal/                  Normal/
                                   C_NMC/Normal/                  test/
Taleqani/ -----> [Step 1-5] ------> Taleqani/Abnormal/                Abnormal/
                                   Taleqani/Normal/                  Normal/
SN-AM/   ------> [Step 1-5] ------> SN_AM/Abnormal/               
                                                                  metadata/
                                                                     dataset_manifest.csv
                                                                     split_assignments.csv
                                                                     preprocessing_log.json
                                                                     quality_report.json
```

### 2.2 Step-by-Step Per Dataset

---

#### DATASET 1: ALL-IDB

**Sumber:** `raw_data/ALL_IDB/ALL_IDB1/` dan `raw_data/ALL_IDB/ALL_IDB2/`

**Step 1 -- Parsing Label**
- IDB1 dan IDB2: label dari nama file. `ImXXX_0` = Normal, `ImXXX_1` = Abnormal.
- Validasi: pastikan setiap file memiliki suffix `_0` atau `_1`. Log file yang tidak match.

**Step 2 -- Segmentasi Sel (IDB1 saja)**
- IDB1 adalah full-field image 1712x1368. Perlu di-crop menjadi single-cell.
- Sel Abnormal: gunakan koordinat centroid dari file `.xyc` di folder `xyc/`.
- Sel Normal: deteksi otomatis via HSV thresholding (pipeline V1 `detect_wbc_centroids()`).
- Crop 257x257 per centroid menggunakan `crop_around_centroid()`.
- **Validasi**: log jumlah centroid per gambar, skip gambar dengan 0 centroid.

**Step 3 -- IDB2 Direct Copy**
- IDB2 sudah single-cell 257x257. Direct copy dengan konversi TIFF ke JPEG (quality=95).
- Validasi dimensi: reject jika bukan 257x257.

**Step 4 -- Quality Check**
- Blur detection: Laplacian variance < 50 -> flag sebagai blurry.
- Brightness check: mean grayscale < 30 (underexposed) atau > 230 (overexposed).
- Minimum cell content: jika > 90% pixel adalah background (putih/sangat terang), reject.

**Step 5 -- Output**
- Simpan ke `data/ALL_IDB/Abnormal/` dan `data/ALL_IDB/Normal/`.
- Format output: JPEG quality 95.
- Naming: `ALLIDB1_{original_stem}_cell{NNN}.jpg` atau `ALLIDB2_{original_stem}.jpg`.
- Estimasi output: ~400-550 single-cell images.

---

#### DATASET 2: C-NMC 2019

**Sumber:** `raw_data/C-NMC/C-NMC_train_merged/all/` dan `hem/`

**Step 1 -- Label Mapping**
- Folder `all/` -> Abnormal (ALL leukemia cells).
- Folder `hem/` -> Normal (healthy hematogones).
- Total: 7.272 Abnormal + 3.389 Normal = 10.661.

**Step 2 -- Patient ID Extraction**
- Nama file: `UID_XX_YY_ZZ_class.bmp` dimana `XX` adalah patient ID.
- Regex parsing: extract field kedua setelah `UID_` sebagai patient_id.
- KRITIS: patient ID diperlukan untuk patient-aware split (mencegah data leakage).
- Buat mapping `{patient_id: [list of file paths]}`.
- Validasi: log jumlah pasien unik dan distribusi sel per pasien.

**Step 3 -- Format Conversion**
- Konversi BMP 450x450 ke JPEG quality 95 (hemat storage ~85%).
- Tidak perlu resize: 450x450 akan di-resize saat training (transforms.Resize ke 224x224).

**Step 4 -- Quality Check**
- Blur detection (Laplacian variance).
- Brightness validation.
- Duplicate detection: hash-based check untuk gambar identik.

**Step 5 -- Output**
- Simpan ke `data/C_NMC/Abnormal/` dan `data/C_NMC/Normal/`.
- Naming: `CNMC_UID{patient_id}_{original_idx}_{class}.jpg`.
- Simpan patient mapping ke `data/C_NMC/patient_mapping.json`.
- Estimasi output: ~10.661 images.

---

#### DATASET 3: Taleqani Hospital

**Sumber:** `raw_data/Taleqani/Original/` atau `Segmented/`

**Step 1 -- Pemilihan Versi**
- Tersedia 2 versi: Original (dengan background) dan Segmented (background removed).
- **Keputusan: gunakan Original** -- konsisten dengan dataset lain yang memiliki background.
- Alasan: model harus belajar dari gambar dengan konteks background alami. Segmented version
  menghilangkan informasi yang ada di dataset lain, menyebabkan distribution mismatch.
- **Alternatif jika Original tidak optimal**: lakukan ablation study membandingkan kedua versi.

**Step 2 -- Label Mapping**
- `Benign/` -> Normal (504 images).
- `Early/` -> Abnormal (985 images, Early Pre-B ALL).
- `Pre/` -> Abnormal (963 images, Pre-B ALL).
- `Pro/` -> Abnormal (804 images, Pro-B ALL).
- Total: 504 Normal + 2.752 Abnormal = 3.256.
- **Catatan**: rasio sangat imbalanced (1:5.5 Normal:Abnormal). Akan ditangani di weighted sampling.

**Step 3 -- Validasi Integritas**
- Dimensi harus 224x224 untuk semua file. Log anomali.
- Format JPEG -- direct copy, tidak perlu konversi.
- Cross-check: jumlah file di Original vs Segmented harus identik per subfolder.

**Step 4 -- Quality Check**
- Brightness dan blur check standar.
- Taleqani images sudah 224x224 (ukuran terkecil), jadi detail lebih rendah.
- Flag gambar dengan sel yang terpotong di edge (> 30% area sel di border).

**Step 5 -- Output**
- Simpan ke `data/Taleqani/Abnormal/` dan `data/Taleqani/Normal/`.
- Naming: `TALEQANI_{subtype}_{original_name}.jpg`.
- Simpan subtype info di metadata (berguna untuk analisis error per subtype nanti).
- Estimasi output: ~3.256 images.

---

#### DATASET 4: SN-AM

**Sumber:** `raw_data/PKG - SN-AM/ALL/` dan `MM/`

**Step 1 -- Filtering Mask Files**
- Folder berisi raw images DAN mask files (`*_background_mask.bmp`, `*_nucleus_mask.bmp`,
  `*_cyto_mask.bmp`).
- Filter: ambil hanya file tanpa suffix `_mask`, `_background_mask`, `_nucleus_mask`, `_cyto_mask`.
- ALL: 31 raw images (ALL_1.bmp sampai ALL_29.bmp + ALL_ref.bmp + ALL_ALL_extra_images.bmp).
- MM: 30 raw images (MM_1.bmp sampai MM_30.bmp, termasuk MM_5_also_ref_image.bmp).

**Step 2 -- Label Mapping dan Keputusan MM**
- ALL folder -> Abnormal (B-cell ALL).
- MM folder (Multiple Myeloma): **EXCLUDE**.
  - Sesuai keputusan terbaru, hanya kelas ALL dan Normal yang digunakan. Karena MM bukan ALL atau Normal, seluruh folder MM dibuang.

**Step 3 -- Single-Cell Segmentation**
- Gambar full-field 2560x1920 pada magnifikasi 1000x.
- **KRITIS**: magnifikasi 10x lebih tinggi dari dataset lain (100x). Sel terlihat JAUH lebih besar.
- Mask yang tersedia: nucleus_mask dan background_mask (sebagian punya cyto_mask).
- **Strategi segmentasi**:
  1. Gunakan `nucleus_mask.bmp` untuk mendeteksi lokasi sel (connected components).
  2. Hitung centroid setiap connected component dari nucleus mask.
  3. Crop region sekitar centroid. Crop size perlu disesuaikan karena magnifikasi 1000x.
  4. Estimasi crop size: sel WBC di 1000x ~200-400px diameter. Crop 512x512 lalu resize ke 257x257.
- **Fallback** jika mask tidak tersedia: HSV thresholding (parameter perlu tuning ulang dari V1).
- **PENTING**: setelah crop dan resize, sel harus terlihat proporsional dengan dataset lain.
  Validasi visual WAJIB.

**Step 4 -- Resolusi Mismatch Handling**
- Post-crop resize ke 257x257 (konsisten dengan ALL-IDB crop size).
- Gunakan LANCZOS interpolation untuk downscale (preserve detail).
- **Risiko**: downscale dari 1000x ke skala visual ~100x mengubah texture granularity.
- **Mitigasi**: tambahkan mild Gaussian blur sebelum resize untuk simulasi lower magnification.

**Step 5 -- Quality Check**
- Validasi visual wajib untuk SETIAP crop (dataset kecil, feasible).
- Blur detection (threshold mungkin perlu adjustment karena magnifikasi berbeda).
- Reject crop yang mostly background.

**Step 6 -- Output**
- Simpan ke `data/SN_AM/Abnormal/` (hanya Abnormal karena MM di-exclude).
- Format: JPEG quality 95.
- Naming: `SNAM_ALL_{source_num}_cell{NNN}.jpg`.
- Output aktual: **1.278** single-cell images (dari ALL saja, hasil re-segmentasi ketat).
- **Dataset terkecil setelah processing.** Weighted sampling di training harus mengkompensasi.

---

## 3. Pooling dan Split Strategy

### 3.1 Dataset yang Masuk Training Pool (100%)

| Dataset | Abnormal | Normal | Total | Alasan 100% Pool |
|---------|----------|--------|-------|-------------------|
| ALL-IDB | ~640 | ~390 | 1.030 | Masuk Source Domain |
| C-NMC | 7.272 | 3.389 | 10.661 | Masuk Source Domain |
| SN-AM | 1.278 | 0 | 1.278 | Masuk Source Domain |

### 3.2 Dataset dengan Split (Taleqani - Target Domain)

| Split | Proporsi | Jumlah Estimasi | Catatan |
|-------|----------|-----------------|---------|
| Train (Unlabeled)| 20% | ~650 | Masuk training pool (untuk adaptasi pseudo-labeling) |
| Test (Evaluation)| 80% | ~2.606 | Evaluasi lintas-domain (label dirahasiakan saat training) |

**Split Rules:**
1. Split secara **stratified**: rasio Abnormal/Normal harus proporsional di kedua split (20/80).
2. Split dilakukan SEKALI dengan seed=42, kemudian di-freeze.
3. Simpan split assignment di `dataset/metadata/split_assignments.csv`.

**Implementasi Split Taleqani:**
```
1. Parse semua nama file Taleqani
2. Stratified split image-level ke 20% train / 80% test (seed=42)
3. Verifikasi: rasio kelas di train dan test mendekati rasio global
```

### 3.3 Struktur Output Final

```
dataset/
  train/
    Abnormal/
      ALLIDB1_Im001_1_cell000.jpg
      CNMC_UID11_10_1_all.jpg        (100% of C-NMC)
      TALEQANI_Early_WBC-Malignant-Early-001.jpg (20% of Taleqani)
      SNAM_ALL_1_cell000.jpg
      ...
    Normal/
      ALLIDB1_Im002_0_cell000.jpg
      CNMC_UIDH10_100_1_hem.jpg      (100% of C-NMC)
      TALEQANI_Benign_WBC-Benign-001.jpg (20% of Taleqani)
      
      ...
  test/
    Abnormal/                         (80% Taleqani only)
    Normal/                           (80% Taleqani only)
  metadata/
    dataset_manifest.csv
    split_assignments.csv
    preprocessing_log.json
    quality_report.json
```

### 3.4 Estimasi Distribusi Final

| Sumber | Train Abnormal | Train Normal | Test Abnormal | Test Normal |
|--------|---------------|-------------|---------------|-------------|
| ALL-IDB | 640 | 390 | - | - |
| C-NMC | 7.272 | 3.389 | - | - |
| SN-AM | 1.278 | 0 | - | - |
| Taleqani (20%) | ~550 | ~100 | - | - |
| Taleqani (80%) | - | - | ~2.202 | ~404 |
| **Total** | **~9.740** | **~3.879** | **~2.202** | **~404** |

Total train: ~13.619, Total test: ~2.606.
Rasio train Abnormal:Normal ~2.5:1 (akan ditangani weighted sampling di dataloader).

---

## 4. Standar Quality Control

### 4.1 Quality Checks Wajib

| Check | Metode | Threshold | Aksi |
|-------|--------|-----------|------|
| Blur detection | Laplacian variance | < 50 | Flag, review manual |
| Overexposure | Mean grayscale | > 230 | Flag, review manual |
| Underexposure | Mean grayscale | < 30 | Flag, review manual |
| Empty crop | % white pixels (> 240) | > 85% area | Reject otomatis |
| Corrupt file | PIL Image.verify() | Exception | Reject otomatis |
| Wrong dimensions | Size check | != expected | Reject otomatis |
| Duplicate | Perceptual hash (pHash) | distance < 5 | Flag, keep satu |

### 4.2 Validasi Visual Wajib

| Dataset | Jumlah Sampel Visual | Alasan |
|---------|---------------------|--------|
| ALL-IDB1 crops | SEMUA (dataset kecil) | Segmentasi otomatis bisa gagal |
| SN-AM crops | SEMUA (dataset kecil) | Magnifikasi berbeda, risiko crop salah |
| C-NMC | 50 per kelas (random) | Dataset besar, spot-check cukup |
| Taleqani | 30 per kelas (random) | Verifikasi label mapping |

### 4.3 Cross-Dataset Consistency Check

| Aspek | Metode | Target |
|-------|--------|--------|
| Ukuran sel relatif | Visual comparison grid | Sel harus terlihat proporsional antar dataset |
| Color distribution | LAB histogram per dataset | Dokumentasikan perbedaan, bukan normalisasi |
| Background pattern | Visual inspection | Dokumentasikan variasi background |
| Cell-to-background ratio | Foreground segmentation ratio | Dokumentasikan range |

---

## 5. Metadata dan Traceability

### 5.1 dataset_manifest.csv

Setiap gambar output harus tercatat dengan kolom:

| Kolom | Deskripsi | Contoh |
|-------|-----------|--------|
| output_filename | Nama file di dataset/ | CNMC_UID11_10_1_all.jpg |
| source_dataset | Dataset asal | C-NMC |
| source_path | Path relatif ke raw_data/ | C-NMC/C-NMC_train_merged/all/UID_11_10_1_all.bmp |
| original_label | Label asli sebelum mapping | all |
| mapped_label | Label binary | Abnormal |
| patient_id | ID pasien (jika tersedia) | 11 |
| split | train/test | train |
| width | Lebar output | 450 |
| height | Tinggi output | 450 |
| blur_score | Laplacian variance | 234.5 |
| brightness | Mean grayscale | 156.3 |
| quality_flag | OK/FLAGGED/REJECTED | OK |

### 5.2 preprocessing_log.json

```
{
  "timestamp": "2026-06-24T...",
  "pipeline_version": "2.0",
  "seed": 42,
  "datasets_processed": [...],
  "per_dataset": {
    "ALL_IDB": {
      "raw_files": 368,
      "output_files": 550,
      "rejected": 5,
      "flagged": 12,
      "abnormal_count": 250,
      "normal_count": 300
    },
    ...
  },
  "total_train": 30114,
  "total_test": 3199,
  "cnmc_split_seed": 42,
  "cnmc_train_patients": [...],
  "cnmc_test_patients": [...]
}
```

---

## 6. Potensi Risiko dan Mitigasi

### 6.1 Risiko Tinggi (Bisa Invalidate Seluruh Eksperimen)

| Risiko | Dampak | Mitigasi | Validasi |
|--------|--------|----------|----------|
| Data leakage Taleqani (overlap train/test) | Metrik test tidak valid | Deterministic split | Assert: intersection file train vs test = empty set |
| Label mapping error (kelas salah) | Model belajar sinyal salah | Manual review sampel per mapping | Visual spot-check 150+ gambar |
| SN-AM magnifikasi mismatch | Sel terlihat beda ukuran, model bingung | Crop size calibration + resize + visual comparison | Side-by-side grid comparison |
| Duplicate gambar antar dataset | Inflasi metrik | pHash cross-dataset duplicate check | Report duplicates |

### 6.2 Risiko Sedang

| Risiko | Dampak | Mitigasi |
|--------|--------|----------|
| Taleqani extreme imbalance (504 Normal vs 2752 Abnormal) | Bias model | Weighted sampling per dataset di training |
| BMP/TIFF to JPEG conversion loss | Minimal, tapi ada | Quality 95 cukup, verify PSNR > 45 dB |

### 6.3 Risiko Rendah

| Risiko | Dampak | Mitigasi |
|--------|--------|----------|
| File corrupt | Satu gambar hilang | try/except + log |
| Nama file collision | Overwrite file | Prefix dataset di setiap filename |
| Disk space | Pipeline gagal | Estimasi: ~5 GB output (JPEG), pastikan cukup space |

---

## 7. Implementasi Script

### 7.1 File: `src/prepare_datasets.py`

**Struktur:**
```
prepare_datasets.py
  |-- Config dataclass (paths, thresholds, seeds)
  |-- BaseDatasetProcessor (abstract)
  |   |-- process() -> manifest entries
  |   |-- quality_check(img) -> QualityResult
  |
  |-- ALLIDBProcessor(BaseDatasetProcessor)
  |   |-- process_idb1() -- segmentation pipeline
  |   |-- process_idb2() -- direct copy
  |
  |-- CNMCProcessor(BaseDatasetProcessor)
  |   |-- extract_patient_id(filename) -> str
  |   |-- patient_stratified_split()
  |
  |-- TaleqaniProcessor(BaseDatasetProcessor)
  |   |-- map_subtype_to_binary()
  |
  |-- SNAMProcessor(BaseDatasetProcessor)
  |   |-- segment_from_mask() -- menggunakan nucleus_mask
  |   |-- segment_from_hsv() -- fallback
  |   |-- calibrate_crop_size()
  |
  |
  |-- PoolingEngine
  |   |-- merge_to_final_dataset()
  |   |-- generate_manifest()
  |   |-- generate_quality_report()
  |
  |-- main()
```

### 7.2 Urutan Eksekusi

```
1. Buat output directories (data/, dataset/)
2. Process ALL-IDB   -> data/ALL_IDB/{Abnormal,Normal}/
3. Process C-NMC     -> data/C_NMC/{Abnormal,Normal}/ + patient_mapping.json
4. Process Taleqani  -> data/Taleqani/{Abnormal,Normal}/
5. Process SN-AM     -> data/SN_AM/{Abnormal,Normal}/  [NEEDS VISUAL VALIDATION]
7. Print per-dataset statistics
8. [PAUSE] User validates SN-AM crops and reviews statistics
9. Split Taleqani -> train (20%) / test (80%) assignment
10. Pool all train data (Source + 20% Taleqani) -> dataset/train/{Abnormal,Normal}/
11. Pool Taleqani test (80%) -> dataset/test/{Abnormal,Normal}/
12. Generate manifest, log, quality report
13. Print final statistics + sanity checks
```

### 7.3 Sanity Checks (Automated, Run at End)

| Check | Expected | Action if Fail |
|-------|----------|----------------|
| Total train images > 13.000 | True | WARNING |
| Total test images > 2.500 | True | WARNING |
| Train Abnormal > 7.000 | True | WARNING |
| Train Normal > 3.000 | True | WARNING |
| No overlap Taleqani train/test | True | ERROR: abort |
| All output files loadable by PIL | True | ERROR: list corrupt files |
| No filename duplicates in dataset/ | True | ERROR: list duplicates |
| Every file in manifest exists on disk | True | ERROR: list missing files |
| Every file on disk is in manifest | True | ERROR: list orphan files |

---

## 8. Keputusan Desain yang Memerlukan Konfirmasi

### 8.1 Taleqani: Original vs Segmented

**Rekomendasi**: Original (dengan background).
**Alasan**: konsistensi dengan dataset lain. Model harus robust terhadap background variance.
**Alternatif**: gunakan keduanya (augmentasi data 2x). Risiko: duplikat konseptual.

### 8.2 SN-AM: Include atau Exclude MM (Multiple Myeloma)

**Rekomendasi**: Exclude MM.
**Alasan**: Keputusan spesifik untuk hanya mengambil ALL dan normal.


### 8.3 Crop Size untuk SN-AM (1000x magnification)

**Rekomendasi**: Crop 512x512 lalu resize ke 257x257.
**Alasan**: pada 1000x, sel WBC diameter ~200-400px. Crop 512x512 memberikan konteks
sekitar sel yang proporsional dengan crop 257x257 di dataset 100x.
**Perlu**: kalibrasi empiris dengan inspeksi visual perbandingan.

### 8.4 Stain Normalization di Preprocessing vs Training-time

**Rekomendasi**: TIDAK melakukan stain normalization di preprocessing.
**Alasan**:
- ReinhardJitter stain augmentation sudah aktif di training pipeline.
- Stain normalization dan stain augmentation saling menggantikan (temuan V1).
- Pooled training dari multi-stain sudah memaksa model belajar stain invariance.
- Normalisasi di preprocessing menghilangkan variasi yang justru berguna untuk robustness.
- **Exception**: stain normalization AKAN diterapkan di Fase 2 (fine-tuning data RS)
  dimana referensi normalizer di-fit ke distribusi RS.

---

## 9. Dependensi dan Environment

### 9.1 Package yang Diperlukan

| Package | Versi Min | Kegunaan |
|---------|----------|----------|
| Pillow | 9.0+ | Image I/O, resize |
| opencv-python | 4.5+ | HSV thresholding, morphology, blur detection |
| numpy | 1.21+ | Array operations |
| scikit-learn | 1.0+ | StratifiedGroupKFold untuk patient split |
| pandas | 1.3+ | Manifest CSV generation |
| imagehash | 4.2+ | Perceptual hashing untuk duplicate detection |
| tqdm | 4.0+ | Progress bars |

### 9.2 Hardware Requirements

| Resource | Estimasi |
|----------|----------|
| Disk space input (raw_data) | ~25 GB |
| Disk space output (data/ + dataset/) | ~5 GB (JPEG) |
| RAM | 4 GB minimum (processing image-by-image) |
| Waktu estimasi | ~30-60 menit total (tergantung SN-AM segmentation) |

---

## 10. Acceptance Criteria

Pipeline preprocessing dianggap COMPLETE jika dan hanya jika:

1. Semua 4 dataset berhasil diproses tanpa ERROR.
2. `dataset/train/` berisi > 13.000 gambar di 2 kelas.
3. `dataset/test/` berisi > 2.500 gambar di 2 kelas (Taleqani 80% hold-out).
4. Tidak ada image overlap pada Taleqani train/test split (assert).
5. Semua file dalam manifest valid (loadable, correct dimensions).
6. Quality report menunjukkan < 5% flagged images per dataset.
7. Visual validation SN-AM crops selesai dan disetujui.
8. Visual validation label mapping (150+ sampel) selesai dan disetujui.
9. `preprocessing_log.json` lengkap dengan semua statistik.
10. Pipeline reproducible: running ulang dengan seed=42 menghasilkan output identik.

---

Last updated: 2026-06-24
