# Energeek Box Monitor 🎯

**Sistem Monitoring Deteksi Objek Kotak Biru (In/Out Container Putih)**
Menggunakan **YOLOv8 Nano** + **OpenCV** untuk computer vision.

---

## 📋 Deskripsi Proyek

Proyek ini mendeteksi dan menghitung objek **kotak biru** (_blue_box_) yang berada di **dalam** vs **di luar** sebuah **kontainer putih** (_container_box_) menggunakan:

- **YOLOv8 Nano** — Model deteksi objek ringan dan cepat
- **OpenCV** — Point-in-Polygon test untuk logika spasial
- **Python** — Bahasa pemrograman utama

---

## 📁 Struktur Proyek

```text
detection-box/
├── dataset/
│   ├── data.yaml              # Konfigurasi dataset YOLO
│   ├── train/
│   │   ├── images/            # Gambar training (jpg/png)
│   │   └── labels/            # Label training (txt format YOLO)
│   ├── val/
│   │   ├── images/            # Gambar validasi
│   │   └── labels/            # Label validasi
│   └── test/                  # Opsional
│       ├── images/
│       └── labels/
├── runs/
│   └── detect/
│       └── energeek_box_monitor/  # Hasil training (akan digenerate)
│           └── weights/
│               ├── best.pt        # Bobot model terbaik
│               └── last.pt        # Bobot model terakhir
├── train.py                   # Skrip training
├── augment_dataset.py         # 📸 Augmentasi data (perbanyak dataset otomatis)
├── inference_test.py          # Skrip inference & logika spasial
├── api_service.py             # 🌐 AI Service API (FastAPI)
├── test_api.py                # 🧪 Test client untuk API
├── requirements.txt           # Dependencies
├── README.md                  # Dokumentasi ini
└── project.md                 # Panduan pengembangan lengkap
```

---

## 🚀 Instalasi

### 1. Clone / Siapkan Proyek

```bash
cd l:\Portofolio\detection box
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Atau install manual:

```bash
pip install ultralytics opencv-python numpy matplotlib
```

---

## 📸 Augmentasi Dataset (Perbanyak Data)

Jika data Anda masih sedikit (< 50 gambar), **wajib** melakukan augmentasi agar model tidak overfit.

### Cara pakai:

```bash
python augment_dataset.py
```

Lalu pilih:

1. **Preview** — Lihat contoh hasil augmentasi 1 gambar
2. **Augmentasi penuh** — Perbanyak semua data training & validasi
3. **Hapus augmentasi** — Reset dataset ke kondisi asli

### Contoh: Perbanyak 5x lipat

```bash
python augment_dataset.py
# Pilih opsi 2, masukkan multiplier: 5
# Hasil: 10 gambar asli → 50 gambar (asli + augmentasi)
```

### Augmentasi yang Didukung:

| Augmentasi          | Label Berubah? | Keterangan        |
| ------------------- | :------------: | ----------------- |
| Flip Horizontal     |       ✅       | Mirror kiri↔kanan |
| Flip Vertikal       |       ✅       | Mirror atas↔bawah |
| Rotasi 90°          |       ✅       | Diputar 90° CW    |
| Rotasi 180°         |       ✅       | Diputar 180°      |
| Rotasi 270°         |       ✅       | Diputar 90° CCW   |
| Brightness/Contrast |       ❌       | Cerah/redup       |
| HSV Shift           |       ❌       | Warna berubah     |
| Gaussian Blur       |       ❌       | Efek blur ringan  |
| Random Noise        |       ❌       | Efek noise        |

> ✅ = Koordinat bounding box ikut berubah secara presisi  
> ❌ = Hanya tampilan gambar, label tetap sama

---

## 🏋️‍♂️ Training Model

### 1. Siapkan Dataset

Masukkan gambar dan label Anda ke folder:

- `dataset/train/images/` & `dataset/train/labels/`
- `dataset/val/images/` & `dataset/val/labels/`

### 2. (Opsional) Perbanyak Data dengan Augmentasi

```bash
python augment_dataset.py
# Pilih opsi 2, masukkan multiplier (contoh: 5)
```

### 3. Konfigurasi `dataset/data.yaml`

Pastikan path dan class sudah sesuai:

```yaml
path: ./dataset
train: train/images
val: val/images
nc: 2
names: ["container_box", "blue_box"]
```

### 4. Jalankan Training

```bash
python train.py
```

Training akan menghasilkan model di:
`runs/detect/energeek_box_monitor/weights/best.pt`

---

## 🔍 Inference & Pengujian

### Jalankan Inference

```bash
python inference_test.py
```

Kemudian pilih mode:

1. **Gambar statis** — Deteksi pada satu file gambar
2. **Video file** — Deteksi pada file video
3. **Webcam** — Deteksi real-time dari kamera

### Logika Spasial

Sistem menggunakan `cv2.pointPolygonTest()` untuk menentukan apakah centroid _blue_box_ berada di dalam polygon _container_box_:

- ✅ **INSIDE** (Hijau) — Centroid ada di dalam kontainer
- ❌ **OUTSIDE** (Merah) — Centroid ada di luar kontainer

---

## 📊 Output

- **Gambar**: Hasil deteksi disimpan sebagai `result_output.jpg`
- **Video**: Hasil deteksi disimpan sebagai `result_video.mp4`
- **Info**: Menampilkan jumlah _Inside_ dan _Outside_ di frame

---

## 🌐 AI Service API (FastAPI)

Service REST API yang menerima gambar dan mengembalikan JSON.

### Format Response

```json
{
  "inside_box": 7,
  "outside_box": 2,
  "total": 9
}
```

### Cara Jalankan

```bash
# Install dependencies tambahan
pip install fastapi uvicorn python-multipart

# Jalankan server
python api_service.py
```

Server akan berjalan di `http://localhost:8000`.  
Dokumentasi API otomatis: `http://localhost:8000/docs`

### Endpoints

| Method | Path              | Deskripsi                        |
| ------ | ----------------- | -------------------------------- |
| `GET`  | `/`               | Info service                     |
| `GET`  | `/health`         | Health check                     |
| `POST` | `/predict`        | Upload file gambar → JSON hasil  |
| `POST` | `/predict/base64` | Kirim gambar base64 → JSON hasil |

### Contoh Penggunaan

**Via curl:**

```bash
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/predict
```

**Via Python:**

```python
import requests

with open("test_image.jpg", "rb") as f:
    resp = requests.post("http://localhost:8000/predict", files={"file": f})
    print(resp.json())
# Output: {"inside_box": 7, "outside_box": 2, "total": 9}
```

**Via test client:**

```bash
python test_api.py test_image.jpg
```

**Via base64:**

```bash
# Encode dulu
$base64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("test_image.jpg"))
Invoke-RestMethod -Uri http://localhost:8000/predict/base64 `
  -Method Post -Body (@{image=$base64} | ConvertTo-Json) `
  -ContentType "application/json"
```

---

## ⚡ Optimasi Deployment

### Ekspor ke ONNX

```python
from ultralytics import YOLO
model = YOLO("runs/detect/energeek_box_monitor/weights/best.pt")
model.export(format="onnx")
```

### Ekspor ke TensorRT

```python
model.export(format="engine", device=0)  # GPU required
```

---

## 🛠️ Troubleshooting

| Masalah                  | Solusi                                         |
| ------------------------ | ---------------------------------------------- |
| Model tidak ditemukan    | Jalankan `python train.py` terlebih dahulu     |
| Akurasi rendah           | Tambah dataset, variasikan sudut & pencahayaan |
| Container miring ekstrem | Gunakan model segmentasi (`yolov8n-seg.pt`)    |
| Performa lambat          | Ekspor ke ONNX atau TensorRT                   |

---
