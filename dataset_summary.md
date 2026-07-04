# Ringkasan Dataset Leukosit (ALL & Normal)

Berikut adalah ringkasan hasil akhir dari *preprocessing* data untuk arsitektur *Unsupervised Domain Adaptation* (UDA).

### 1. Komposisi per Dataset

| Dataset Sumber | Kelas Abnormal (ALL) | Kelas Normal | Total Data | Ukuran Resolusi Output | Keterangan Preprocessing |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **ALL-IDB** | 640 | 390 | **1.030** | `257 x 257` | Di-*crop* otomatis (Saliency/XYC) per sel dari gambar mikroskop resolusi tinggi |
| **C-NMC** | 7.272 | 3.389 | **10.661** | `450 x 450` | Original size (sel tunggal). Tidak ada pemotongan tambahan |
| **SN-AM** | 1.278 | 0 | **1.278** | `257 x 257` | Di-*crop* otomatis (512x512) dari *full-field image* lalu di-*resize* ke 257x257. (Kelas MM dibuang) |
| **Taleqani** | 2.752 | 504 | **3.256** | Bervariasi | Original size. Disimpan utuh untuk eksperimen skala penuh |
| **TOTAL** | **11.942** | **4.283** | **16.225** | - | *Total gabungan seluruh gambar tunggal sel darah putih* |

<br>

### 2. Pembagian Distribusi (UDA Split)

Berdasarkan algoritma *Pooling Engine*, dataset di atas didistribusikan secara spesifik ke dalam beberapa ruang:

| Tujuan Split | Dataset yang Masuk | Total Gambar | Peran dalam Pelatihan |
| :--- | :--- | :---: | :--- |
| **Train (*Source Domain*)** | 100% ALL-IDB, 100% C-NMC, 100% SN-AM | 12.969 | Melatih pengetahuan dasar morfologi model |
| **Train (*Target Unlabeled*)** | 20% Taleqani | 650 | Sumber adaptasi target melalui *Iterative Pseudo-Labeling* (>90% confidence) |
| **Test (*Target Evaluation*)** | 80% Taleqani | 2.606 | Menguji ketahanan transfer model pada lingkungan baru (tanpa *data leakage*) |
| **TOTAL** | | **16.225** | |
