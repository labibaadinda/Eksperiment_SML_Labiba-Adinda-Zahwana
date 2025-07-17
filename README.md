# Sleep Health Life Style Analysis

## Deskripsi

Proyek ini bertujuan untuk melakukan analisis pada dataset **sleep-health\_life-style**, yang berisi data terkait gaya hidup sehat dan gangguan tidur. Proses ini mencakup **preprocessing data**, eksplorasi data, serta eksperimen model machine learning untuk memprediksi gangguan tidur berdasarkan berbagai fitur yang relevan.

Proyek ini juga mengimplementasikan **Continuous Integration (CI)** untuk otomatisasi preprocessing menggunakan GitHub Actions, dan eksperimen model dilakukan menggunakan Jupyter Notebook.

---

## Struktur Proyek

```
/project-root
│
├── preprocessing/                  # Folder untuk preprocessing data
│   ├── sleep-health_life-style_raw.csv   # Dataset mentah
│   ├── sleep-health_life-style_preprocessing.csv  # Dataset yang sudah diproses
│   └── preprocess.py               # Script preprocessing data
│
├── experiments/                    # Folder untuk eksperimen dan analisis
│   ├── experiment_notebook.ipynb    # Jupyter notebook untuk eksperimen model
├── requirements.txt
├── sleep-health_life-style_raw.csv
└── README.md                       # Dokumentasi proyek ini
```

---

## Teknologi yang Digunakan

* **Pandas**: untuk manipulasi dan analisis data.
* **Numpy**: untuk operasi numerik pada data.
* **Scikit-learn**: untuk preprocessing, pemodelan, dan evaluasi model.
* **Logistic Regression, XGBoost, RandomForest, GradientBoosting**: model machine learning yang digunakan untuk klasifikasi.
* **Matplotlib, Seaborn, Plotly**: untuk visualisasi data.
* **GitHub Actions**: untuk mengatur Continuous Integration (CI) dan otomatisasi preprocessing.

---

## Langkah-langkah Penggunaan

### 1. Persiapan

Pastikan Anda sudah menginstal semua dependensi yang diperlukan. Anda dapat menginstalnya dengan menjalankan perintah berikut:

```bash
pip install -r requirements.txt
```

### 2. Preprocessing Data

Script `preprocess.py` yang ada di dalam folder `preprocessing/` akan melakukan beberapa tahapan preprocessing pada dataset **sleep-health\_life-style\_raw\.csv**, termasuk:

* **Pemeriksaan kualitas data** (Descriptive Statistics, Data Types, Missing Values, Duplicates).
* **Penanganan nilai yang hilang** pada kolom 'Sleep Disorder'.
* **Pemecahan kolom 'Blood Pressure'** menjadi 'Systolic BP' dan 'Diastolic BP'.
* **Pembersihan data dari outlier** menggunakan metode IQR (Interquartile Range).
* **Penyandian fitur kategorikal** dengan menggunakan `LabelEncoder`.

Setelah preprocessing selesai, dataset akan disimpan sebagai **sleep-health\_life-style\_preprocessing.csv**.

### 3. Eksperimen Model

Proyek ini juga melibatkan eksperimen untuk memprediksi gangguan tidur. Berikut adalah tahapan eksperimen yang dilakukan:

* **Eksplorasi Data**: Menggunakan visualisasi untuk memahami distribusi dan hubungan antar fitur.

* **Eksperimen Model**: Melibatkan beberapa algoritma klasifikasi, termasuk:

  * **Logistic Regression**
  * **Random Forest Classifier**
  * **Gradient Boosting**
  * **XGBoost**

  Model-model ini dievaluasi menggunakan metrik **accuracy**, **precision**, **recall**, dan **F1-score**.

* **Evaluasi Model**: Hasil evaluasi menggunakan confusion matrix dan laporan klasifikasi untuk melihat kinerja masing-masing model.

---

## Continuous Integration (CI)

Proyek ini juga menerapkan **Continuous Integration (CI)** menggunakan **GitHub Actions** untuk menjalankan preprocessing setiap kali perubahan dilakukan pada repositori. Ini memastikan bahwa data yang digunakan dalam eksperimen selalu terjaga kualitasnya.

---

## Eksperimen dan Hasil

### 1. Klasifikasi Gangguan Tidur

Model yang diterapkan pada dataset ini menunjukkan bahwa **XGBoost** memberikan hasil akurasi tertinggi, dengan performa yang stabil pada data pelatihan dan pengujian.

* **Random Forest** dan **Gradient Boosting** juga memberikan akurasi tinggi, namun lebih cenderung mengalami **overfitting** pada data pelatihan.
* **Logistic Regression** memberikan hasil yang lebih sederhana dan interpretatif, cocok untuk kasus-kasus dengan data lebih sederhana.

### 2. Evaluasi Model

Evaluasi dilakukan menggunakan **confusion matrix** dan **classification report** yang menunjukkan kinerja model dalam mengklasifikasikan gangguan tidur dengan sangat baik.

---

## Instruksi Penggunaan

1. **Clone Repositori**

   ```bash
   git clone https://github.com/username/sleep-health-analysis.git
   cd sleep-health-analysis
   ```

2. **Preprocessing Data**
   Jalankan script `preprocess.py` untuk memproses dataset mentah:

   ```bash
   python preprocessing/preprocess.py
   ```

3. **Eksperimen Model**
   Untuk melakukan eksperimen dan evaluasi model, buka notebook `experiment_notebook.ipynb` dan ikuti langkah-langkah di dalamnya untuk mempersiapkan data dan menjalankan eksperimen.
