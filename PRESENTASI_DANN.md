# Penjelasan Konsep DANN (Domain-Adversarial Neural Network) 

Dokumen ini disusun sebagai panduan poin-poin presentasi kepada dosen terkait implementasi **DANN** dalam memecahkan masalah *Domain Shift* pada klasifikasi sel Leukemia lintas-dataset.

---

## 1. Latar Belakang Masalah: *Domain Shift*
Ketika kita menggabungkan dataset dari berbagai institusi (C-NMC, ALL-IDB, SN-AM sebagai *Source*, dan Taleqani sebagai *Target*), kita menghadapi masalah fundamental yang disebut **Domain Shift**. 

Gambar dari rumah sakit Taleqani memiliki karakteristik visual yang sangat berbeda:
- Perbedaan warna pewarnaan kimia (*staining*).
- Tingkat pencahayaan latar belakang yang berbeda.
- Jenis mikroskop dan sensor kamera yang berbeda.

**Masalahnya:** Jika model hanya dilatih di *Source*, akurasinya akan anjlok saat diuji di Taleqani karena model 'menghafal' warna dan tekstur spesifik dari *Source*, bukan mempelajari struktur sejati dari sel leukemianya.

---

## 2. Solusi: Arsitektur DANN
Untuk mengatasi masalah ini, kita menerapkan arsitektur **Domain-Adversarial Neural Network (DANN)**. 
Tujuan utamanya adalah memaksa *Feature Extractor* (ConvNeXtV2) untuk **buta terhadap asal domain dataset** dan hanya fokus mengekstrak fitur anatomis sel yang krusial untuk membedakan sel normal dan abnormal.

### Cara Kerja (3 Komponen Utama)
Saat proses *training*, gambar dari *Source* (berlabel) dan *Target* (tanpa label) dimasukkan ke dalam model secara bersamaan. Model kemudian memprosesnya melalui 3 cabang:

1. **Label Classifier (Tugas Utama):** Berusaha mengklasifikasikan apakah sel tersebut Leukemia atau Normal (hanya belajar dari data *Source*).
2. **Domain Discriminator (Tugas Adversarial):** Berusaha menebak apakah gambar ini berasal dari dataset *Source* atau dari dataset *Target* (Taleqani).
3. **Gradient Reversal Layer / GRL (Kunci Utama):** Layer ajaib ini diletakkan di antara *Feature Extractor* dan *Domain Discriminator*. Saat backpropagation, GRL akan **membalikkan gradien** (mengalikan dengan angka negatif).

### Proses "Tipu-Menipu" (Analogi Adversarial)
Karena adanya GRL, terjadi kompetisi di dalam model:
*Domain Discriminator* berusaha sekuat tenaga mengenali ciri khas warna Taleqani. Namun, karena arah pembelajarannya dibalik oleh GRL, *Feature Extractor* justru belajar untuk **menghapus semua ciri khas warna Taleqani tersebut** dari fitur yang diekstrak. Ia berusaha menipu *Domain Discriminator* agar kebingungan dan tidak bisa lagi membedakan asal usul gambar.

**Hasil Akhir:** Model menghasilkan **Domain-Invariant Features**. Di mata klasifikator utama, sel dari Taleqani secara matematis kini terlihat identik dengan sel dari C-NMC. Akurasi pada Taleqani meningkat drastis meskipun model tidak pernah melihat labelnya sama sekali selama fase *training*.

---

## 3. Komparasi: Mengapa Tidak Menggunakan Pseudo-Labeling Saja?

*(Penting untuk menjelaskan bedanya pendekatan saat ini dengan pendekatan sebelumnya)*

### Kondisi SEBELUM (Hanya Pseudo-Labeling)
Metode awal kita adalah membiarkan model belajar dari *Source*, lalu menyuruhnya menebak (memberi pseudo-label) dataset Taleqani, dan memasukkan tebakan tersebut kembali ke dalam pelatihan. 

**Kelemahan Fatal:**
- Karena adanya *domain shift*, tebakan awal model pada dataset Taleqani sangat buruk dan penuh bias (*Target Domain Specificity* hancur).
- Terjadi fenomena **"Garbage In, Garbage Out"**. Karena tebakan awalnya salah, pseudo-label yang dihasilkan justru menyesatkan. Model melatih dirinya sendiri menggunakan label yang salah, sehingga ia semakin terjerumus mengulangi kesalahannya sendiri (**Confirmation Bias**).

### Kondisi SESUDAH (Hybrid DANN + Pseudo-Labeling)
Dengan masuknya DANN ke dalam arsitektur, prosesnya berubah secara fundamental:
1. **Pondasi yang Lurus:** Sebelum model mencoba menebak pseudo-label, DANN (melalui *Gradient Reversal Layer*) sudah meratakan dan menyamakan fitur visual antara Taleqani dan *Source*.
2. **Kualitas Tebakan Akurat:** Karena secara internal model sudah tidak terpengaruh oleh "warna" Taleqani, tebakan probabilitas pseudo-label menjadi jauh lebih akurat sejak awal.
3. **Sinergi Sempurna:** DANN bertugas mengatasi **"perbedaan gaya visual"** secara global, sedangkan Pseudo-Labeling bertugas untuk **"menajamkan batasan kelas"** (memastikan sel yang meragukan ditegaskan menjadi Abnormal atau Normal).

### Kesimpulan / Analogi
*"Tanpa DANN, Pseudo-Labeling itu seperti menyuruh orang yang memakai kacamata merah untuk menilai lukisan berwarna biru; tebakannya pasti salah dan menyesatkan. DANN berfungsi untuk 'melepas kacamata merah' tersebut terlebih dahulu, sehingga saat Pseudo-Labeling dilakukan, model sudah melihat dataset Taleqani dengan warna yang objektif dan benar, menghindari bias konfirmasi."*