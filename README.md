# Klasifikasi Tingkat Tekanan Darah Berbasis Sinyal Photoplethysmogram (PPG)

Repositori ini berisi implementasi model pembelajaran mendalam (Deep Learning) dengan arsitektur gabungan Convolutional Neural Network (CNN) dan Long Short-Term Memory (LSTM) untuk melakukan klasifikasi tingkat tekanan darah (hipertensi vs. normal/prehipertensi) secara langsung dari sinyal mentah Photoplethysmogram (PPG). Proyek ini dirancang secara modular agar mudah dikembangkan, diintegrasikan ke perangkat berbasis komputasi tepi (edge computing), serta dijalankan secara fleksibel di lingkungan seperti Google Colab.

## Deskripsi Proyek

Pemantauan tekanan darah secara berkelanjutan dan non-invasif menjadi fokus krusial dalam rekayasa biomedis modern. Berbeda dengan pendekatan regresi konvensional yang memprediksi nilai numerik sistolik (SBP) dan diastolik (DBP) secara absolut, proyek ini menerapkan pendekatan klasifikasi biner untuk mendeteksi indikasi klinis hipertensi berdasarkan pola gelombang PPG.

Sinyal masukan diklasifikasikan menjadi dua kategori utama:

1. Kelas 0 (Normal / Prehypertension): Kondisi di mana nilai SBP < 140 mmHg DAN DBP < 90 mmHg.
2. Kelas 1 (Hypertension): Kondisi di mana nilai SBP >= 140 mmHg ATAU DBP >= 90 mmHg (mencakup Hipertensi Stadium I dan Stage II).

## Struktur Repositori

Proyek ini disusun menggunakan arsitektur pipeline modular untuk memisahkan setiap fungsionalitas logika pemrograman, sehingga mempermudah proses debugging, pengujian komponen secara terisolasi, dan kolaborasi tim:

```

```text
File README.md berhasil dibuat.

```text
blood_pressure_classification/
│
├── data/
│   └── raw/                   <-- Direktori penyimpanan berkas .npy dari MIMIC-BP
│
├── src/
│   ├── __init__.py            <-- Berkas inisialisasi modul python
│   ├── preprocessing.py       <-- Modul pembersihan sinyal dan filtrasi noise
│   ├── dataset.py             <-- Modul pelabelan klinis dan generator DataLoader
│   ├── model.py               <-- Arsitektur model neural network CNN-LSTM
│   ├── engine.py              <-- Algoritma pengulangan pelatihan dan Early Stopping
│   └── evaluate.py            <-- Kalkulasi metrik evaluasi dan visualisasi grafik
│
└── main_train.py              <-- Orkestrator skrip utama untuk eksekusi program

```

## Dataset

Proyek ini menggunakan dataset publik MIMIC-BP yang tersedia di Harvard Dataverse (https://doi.org/10.7910/DVN/DBM1NF). Dataset ini menyimpan data fisiologis pasien dalam format matriks NumPy (`.npy`) yang terpisah menjadi dua komponen utama:

* Berkas `[ID_Pasien]_ppg.npy`: Menyimpan matriks sinyal gelombang PPG mentah yang terbagi menjadi beberapa segmen waktu dengan frekuensi sampling (sampling rate) sebesar 125 Hz.
* Berkas `[ID_Pasien]_labels.npy`: Menyimpan label nilai referensi medis numerik [SBP, DBP] yang berpasangan tepat dengan setiap segmen gelombang PPG.

## Pipeline Pemrosesan Sinyal dan Klasifikasi

Alur kerja pemrograman di dalam repositori ini mencakup tahapan sekuensial berikut:

1. **Pre-processing Sinyal (src/preprocessing.py)**

* **Bandpass Filtering**: Menerapkan filter Butterworth order 3 dengan rentang frekuensi cut-off 0.5 Hz hingga 8 Hz untuk mereduksi interferensi frekuensi tinggi serta artefak pergerakan otot.
* **Baseline Correction**: Menggunakan metode detrend linear untuk menghilangkan fluktuasi garis dasar (baseline drift) akibat pernapasan atau pergeseran posisi sensor pada kulit.
* **Z-score Normalization**: Menormalisasi amplitudo sinyal agar memiliki rata-rata nol dan standar deviasi satu untuk mempercepat konvergensi pelatihan model.

2. **Penyusunan Dataset (src/dataset.py)**

* Membaca seluruh file pasien, menjodohkan file sinyal dengan labelnya, dan mengonversi nilai kontinu SBP/DBP menjadi label kelas integer (0 atau 1) secara otomatis.
* Membagi data secara proporsional menjadi 70% Data Pelatihan (Train), 15% Data Validasi (Validation), dan 15% Data Pengujian (Test).

3. **Arsitektur Model AI (src/model.py)**

* **Lapis Konvolusi 1D (Conv1D)**: Mengekstraksi fitur spasial lokal dan morfologi gelombang pulsa PPG menggunakan kernel berukuran 7.
* **Lapis LSTM**: Menangkap ketergantungan urutan waktu jangka panjang (sekuensial temporal) dari fitur gelombang yang dihasilkan oleh CNN.
* **Lapis Fully Connected (FC)**: Memetakan representasi fitur temporal terakhir dari sirkuit LSTM menjadi dua probabilitas kelas keluaran menggunakan fungsi CrossEntropy.

4. **Mesin Pelatihan dan Pengujian (src/engine.py & src/evaluate.py)**

* Mengoptimasi parameter bobot menggunakan Adam Optimizer dan meminimalkan kerugian lewat Cross Entropy Loss Function.
* Mengintegrasikan fungsi Early Stopping untuk memantau nilai kerugian validasi (validation loss) guna menghindari kondisi overfitting dan menghentikan pelatihan secara adaptif saat model terbaik tercapai.
* Menghasilkan visualisasi Confusion Matrix serta laporan performa klasifikasi secara komprehensif.

## Kebutuhan Sistem dan Instalasi

Pastikan lingkungan eksekusi Anda telah terinstal pustaka dependensi berikut:

* Python >= 3.8
* PyTorch >= 1.10
* NumPy
* SciPy
* Scikit-Learn
* Seaborn
* Matplotlib

Untuk menginstal seluruh pustaka pendukung secara cepat, Anda dapat menjalankan perintah berikut di terminal Anda:

```bash
pip install numpy scipy torch scikit-learn seaborn matplotlib

```

## Cara Penggunaan di Google Colab

Repositori ini dioptimasi penuh agar dapat dijalankan langsung di Google Colab dengan memanfaatkan akselerator T4 GPU. Berikut adalah urutan langkah eksekusinya:

1. **Kloning Repositori GitHub**
   Unduh struktur folder kode ini ke dalam ruang kerja Colab Anda:

```python
!git clone <URL_REPOSITORY_GITHUB_ANDA>
%cd blood_pressure_classification

```

2. **Ekstrak Dataset dari Google Drive**
   Untuk mencegah perlambatan browser akibat mengunggah ribuan file kecil `.npy` secara manual, unggah file `ppg.zip` dan `labels.zip` yang didapat dari Harvard Dataverse ke Google Drive Anda terlebih dahulu. Sambungkan Drive ke Colab, lalu jalankan perintah ekstraksi otomatis ini untuk menyatukan berkas ke folder target:

```python
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p data/raw
!unzip -q -j "/content/drive/MyDrive/ppg.zip" -d data/raw/
!unzip -q -j "/content/drive/MyDrive/labels.zip" -d data/raw/
print("Proses penyatuan berkas selesai. Seluruh file npy berada di data/raw/")

```

3. **Jalankan Proses Pelatihan dan Evaluasi**
   Eksekusi skrip utama orkestrator untuk memulai siklus pembelajaran kecerdasan buatan:

```bash
!python main_train.py

```

## Metrik Evaluasi

Model akhir dievaluasi secara ketat menggunakan parameter penilaian matriks kebingungan (Confusion Matrix) untuk menghitung metrik-metrik klinis utama:

* **Precision**: Mengukur tingkat kepastian prediksi positif model (menghindari kesalahan positif/false positive).
* **Recall / Sensitivity**: Mengukur kemampuan model dalam menjaring seluruh pasien hipertensi secara tepat (sangat krusial untuk menghindari false negative yang berbahaya dalam dunia medis).
* **F1-Score**: Rata-rata harmonis antara presisi dan recall, yang menjadi indikator performa paling andal jika distribusi jumlah sampel kelas normal dan hipertensi tidak seimbang.
* **Specificity**: Mengukur tingkat keberhasilan model dalam mengidentifikasi subjek yang benar-benar sehat/normal.
* **Accuracy**: Rasio ketepatan prediksi total terhadap keseluruhan data uji yang tersedia.
  """
