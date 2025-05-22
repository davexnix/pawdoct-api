# 🐾 Pawdoct ML-API

API Flask untuk mendeteksi penyakit hewan berdasarkan gejala menggunakan model Machine Learning (Random Forest). Proyek ini melibatkan:

* Dataset gejala hewan (`Yes/No`)
* Training model menggunakan RandomForestClassifier
* API untuk menerima input gejala dan mengembalikan prediksi penyakit

## 📁 Struktur Proyek

```bash
├── app.py                 # Flask API untuk prediksi penyakit
├── data/
│   └── dataset.csv        # Dataset gejala dan label penyakit
├── model/
│   └── pawdoct.joblib     # Model hasil training
├── train_model.py         # Script training ML
├── requirements.txt       # Daftar dependensi Python
├── README.md              # Dokumentasi proyek
└── venv/                  # Virtual environment (opsional)
```

## 📦 Instalasi

### 1. Clone Repository

```bash
cd pawdoct-ml-api
```

### 2. Buat Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

## 🧠 Training Model

Jika belum ada model, jalankan:

```bash
python train_model.py
```

Script ini akan:

* Membaca `data/dataset.csv`
* Melatih model Random Forest
* Menyimpan model ke `model/pawdoct.joblib`
* Menampilkan akurasi, klasifikasi, dan feature importance

## 🚀 Menjalankan API

```bash
python app.py
```

API akan aktif di: `http://127.0.0.1:5000`

## 📨 Endpoint API

### `/predict` - POST

**Deskripsi:** Prediksi penyakit berdasarkan input gejala

**Request Body:**

Semua fitur harus disediakan (`Yes`/`No` atau `1`/`0`):

```json
{
  "anorexia": "Yes",
  "muntah": "Yes",
  "lemah": "Yes",
  "kurang respon": "No",
  "dehidrasi": "No",
  "demam": "No",
  "diare": "No",
  "hipersevalis": "No",
  "radang telinga": "No",
  "batuk": "No",
  "hidung meler": "No",
  "gatal": "No",
  "telinga keropeng": "No",
  "pilek": "No",
  "bersin2": "No",
  "mata berair": "No"
}
```

**Response:**

```json
{
  "prediction": "panleukopenia"
}
```

## 🧪 Contoh Request (cURL)

```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{
  "anorexia": "Yes",
  "muntah": "Yes",
  "lemah": "Yes",
  "kurang respon": "No",
  "dehidrasi": "No",
  "demam": "No",
  "diare": "No",
  "hipersevalis": "No",
  "radang telinga": "No",
  "batuk": "No",
  "hidung meler": "No",
  "gatal": "No",
  "telinga keropeng": "No",
  "pilek": "No",
  "bersin2": "No",
  "mata berair": "No"
}'
```

## 🧰 Teknologi yang Digunakan

* Python 3.10+
* Flask
* scikit-learn
* pandas
* joblib
* numpy

## 📌 Catatan

* Model hanya seakurat datanya. Dapat ditambahkan data gejala lebih banyak untuk meningkatkan akurasi.

## 🐶 Contoh Penyakit yang Dideteksi

* Panleukopenia
* Scabies
* FCV (Feline Calicivirus)
* Enteritis
