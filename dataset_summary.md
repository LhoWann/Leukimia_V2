# Leukocyte Dataset Summary (ALL & Normal)

The following is the final summary of the dataset preprocessing results for the Unsupervised Domain Adaptation (UDA) architecture.

### 1. Composition per Dataset

| Source Dataset | Abnormal Class (ALL) | Normal Class | Total Images | Output Resolution | Preprocessing Notes |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **ALL-IDB** | 640 | 390 | **1,030** | ¯ x 257\ | Automatically cropped (Saliency/XYC) per cell from high-resolution microscope images |
| **C-NMC** | 7,272 | 3,389 | **10,661** | Ĩ x 450\ | Original size (single cell). No additional cropping applied |
| **SN-AM** | 1,278 | 0 | **1,278** | ¯ x 257\ | Automatically cropped (512x512) from full-field image, then resized to 257x257. (MM class excluded) |
| **Taleqani** | 2,752 | 504 | **3,256** | Varies | Original size. Kept intact for full-scale experiments |
| **TOTAL** | **11,942** | **4,283** | **16,225** | - | *Combined total of all single white blood cell images* |

<br>

### 2. Distribution Split (UDA Split)

Based on the Pooling Engine algorithm, the dataset above is specifically distributed into several partitions:

| Split Purpose | Included Datasets | Total Images | Role in Training |
| :--- | :--- | :---: | :--- |
| **Train (*Source Domain*)** | 100% ALL-IDB, 100% C-NMC, 100% SN-AM | 12,969 | Trains the base morphological knowledge of the model |
| **Train (*Target Unlabeled*)** | 20% Taleqani | 650 | Source of target adaptation via *Iterative Pseudo-Labeling* (>90% confidence) |
| **Test (*Target Evaluation*)** | 80% Taleqani | 2,606 | Tests the robustness of the model's transferability to a new environment (without *data leakage*) |
| **TOTAL** | | **16,225** | |
