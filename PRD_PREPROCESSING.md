# PRD: Multi-Dataset Preprocessing Pipeline

## 1. Raw Data Condition Summary

### 1.1 Dataset Inventory

| Dataset | Raw Location | File Count | Format | Dimensions | Mode | Level |
|---------|-----------|-------------|--------|---------|------|-------|
| ALL-IDB1 | \
aw_data/ALL_IDB/ALL_IDB1/im/\ | 108 images | JPEG | 1712x1368 | RGB | Full-field (segmentation required) |
| ALL-IDB2 | \
aw_data/ALL_IDB/ALL_IDB2/img/\ | 260 images | TIFF | 257x257 | RGB | Single-cell (pre-cropped) |
| C-NMC | \
aw_data/C-NMC/C-NMC_train_merged/\ | 7,272 (all) + 3,389 (hem) = 10,661 | BMP | 450x450 | RGB | Single-cell |
| Taleqani (Original) | \
aw_data/Taleqani/Original/\ | 504 (Benign) + 985 (Early) + 963 (Pre) + 804 (Pro) = 3,256 | JPEG | 224x224 | RGB | Single-cell |
| Taleqani (Segmented) | \
aw_data/Taleqani/Segmented/\ | 504 + 985 + 963 + 804 = 3,256 | JPEG | 224x224 | RGB | Single-cell (background removed) |
| SN-AM | \
aw_data/PKG - SN-AM/\ | 31 (ALL) + 30 (MM) = 61 raw images | BMP | 2560x1920 | RGB | Full-field (segmentation required) |

### 1.2 SN-AM Details

- Format: BMP 2560x1920, 1000x magnification (10x higher than other datasets)
- Mask files available: \*_background_mask.bmp\, \*_nucleus_mask.bmp\, \*_cyto_mask.bmp\
- ALL: 31 full-field images (29 numbered + 1 extra + 1 ref)
- MM: 30 full-field images

### 1.3 Critical Notes per Dataset

#### ALL-IDB
- **IDB1**: Contains 108 full-field images. We must use \im/\ (images) and \xyc/\ (coordinates). XYC files contain centroid coordinates of ALL blast cells. Normal cells must be cropped using saliency segmentation to avoid false positives.
- **IDB2**: Contains 260 cropped single cells. Can be used directly.

#### C-NMC
- Contains 10,661 cropped single cells.
- **Critical Issue**: Images are very dark and heavily stained with hematoxylin (blue/purple). Global brightness thresholding will flag 100% of these images, but this is expected and should not result in filtering.

#### Taleqani
- Contains 3,256 cropped single cells.
- Subtypes: Benign, Early Pre-B, Pre-B, Pro-B.
- We will map Benign -> Normal (0), and (Early Pre-B + Pre-B + Pro-B) -> Abnormal (1).

#### SN-AM
- **Critical Issue**: The MM class (Multiple Myeloma) is NOT Acute Lymphoblastic Leukemia (ALL). It must be discarded.
- Only the 31 ALL images will be processed. We will use the provided \
ucleus_mask.bmp\ to crop the cells directly.

---

## 2. Preprocessing Goals

1. **Format Standardization**: Convert all outputs to \257x257\ RGB JPEG files.
2. **Class Mapping**: Unify labels to binary: \Abnormal\ (Leukemia) and \Normal\ (Benign/Hem).
3. **Quality Control (QC)**: Implement automatic blur and exposure checks. Outliers will be flagged in a JSON report but NOT discarded from the pipeline.
4. **Data Splitting**: Create an Unsupervised Domain Adaptation (UDA) split where Taleqani forms the unlabeled target domain.

---

## 3. Pipeline Implementation

The preprocessing pipeline is fully implemented in \src/prepare_datasets.py\. It performs:
- Saliency-based cell segmentation for ALL-IDB1.
- Mask-based cell segmentation for SN-AM.
- Direct copy/resize for ALL-IDB2, C-NMC, and Taleqani.
- Unification into \dataset/train/\, \dataset/test/\.
- Generation of \dataset_manifest.csv\, \preprocessing_log.json\, and \quality_report.json\.

All preprocessing runs automatically with a single command and takes approximately 2-5 minutes depending on CPU threads.
