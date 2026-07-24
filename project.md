# Panduan Pengembangan Computer Vision: YOLOv8 Nano + OpenCV
## Sistem Monitoring Deteksi Objek Kotak Biru (In/Out Container Putih) — Energeek

---

## 1. Persiapan Dataset & Labeling

### 1.1 Struktur Folder Dataset Format YOLO
Susun direktori proyek Anda dengan struktur standar Ultralytics YOLOv8 berikut:

```text
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/ (Opsional)
    ├── images/
    └── labels/
```

---

### 1.2 Ketentuan Class Label
Tentukan 2 *class* utama saat melabeli data:
* **`0` : `container_box`** (Kotak Putih / Bounding area utama)
* **`1` : `blue_box`** (Kotak Biru / Objek yang akan dihitung)

---

### 1.3 Alur Labeling Data (Rekomendasi: Roboflow / Label Studio)
1. **Pengambilan Sampel Gambar (100–200 Gambar Awal):**
   * Ambil foto dari berbagai sudut (tegak lurus, miring/rotasi seperti Gambar 1, jarak dekat, jarak jauh).
   * Sertakan variasi kondisi pencahayaan (terang, redup, bayangan).
2. **Aturan Labeling:**
   * Untuk **`blue_box`**: Gunakan **Rectangle/Bounding Box** rapat di sekeliling kotak biru.
   * Untuk **`container_box`**: Gunakan **Polygon/Segmentation Label** jika menggunakan model segmentasi (`yolov8n-seg`), atau **Oriented Bounding Box (OBB)** jika kontainer sering miring secara ekstrem.
3. **Ekspor Dataset:**
   * Pilih format ekspor **YOLOv8 PyTorch** (menghasilkan file `.yaml` beserta folder `train/` dan `val/`).

---

## 2. Alur Pelatihan Model (Training Pipeline)

### 2.1 Instalasi Dependency
Jalankan di terminal/environment Python Anda:

```bash
pip install ultralytics opencv-python numpy matplotlib
```

---

### 2.2 Konfigurasi `data.yaml`
Pastikan isi file `data.yaml` sesuai dengan path dataset Anda:

```yaml
path: ./dataset  # root dir dataset
train: train/images
val: val/images

nc: 2
names: ['container_box', 'blue_box']
```

---

### 2.3 Skrip Pelatihan Model (`train.py`)

```python
from ultralytics import YOLO

def train_model():
    # 1. Load Pretrained YOLOv8 Nano Model
    model = YOLO("yolov8n.pt")  # Atau 'yolov8n-seg.pt' jika menggunakan segmentasi

    # 2. Mulai Training
    results = model.train(
        data="dataset/data.yaml",
        epochs=100,               # Sesuaikan dengan tingkat konvergensi
        imgsz=640,                # Resolusi input standar
        batch=16,                 # Sesuaikan dengan VRAM GPU/RAM CPU
        name="energeek_box_monitor",
        project="runs/detect",
        save=True,
        device=0                  # '0' untuk GPU, 'cpu' jika tanpa GPU
    )

    print("Training Selesai! Model terbaik disimpan di: runs/detect/energeek_box_monitor/weights/best.pt")

if __name__ == "__main__":
    train_model()
```

---

### 2.4 Evaluasi Model
Setel *training* selesai, periksa folder `runs/detect/energeek_box_monitor/`:
* **`confusion_matrix.png`**: Pastikan tidak ada bentrok deteksi antara `container_box` dan `blue_box`.
* **`results.png`**: Perhatikan grafik **mAP50** dan **mAP50-95**. Model siap digunakan jika mAP50 telah mencapai **> 0.90 (90%)**.

---

## 3. Alur Pengujian & Logika Spasial (Inference & OpenCV Logic)

Setelah mendapatkan file bobot model `best.pt`, gunakan skrip di bawah ini untuk melakukan deteksi spasial (*inside/outside counting*).

### 3.1 Skrip Pengujian Lengkap (`inference_test.py`)

```python
import cv2
import numpy as np
from ultralytics import YOLO

class BoxMonitorSystem:
    def __init__(self, model_path="runs/detect/energeek_box_monitor/weights/best.pt", conf_thresh=0.5):
        # Load Model YOLOv8 Hasil Training
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def process_frame(self, frame):
        # 1. Inference dengan YOLOv8
        results = self.model.predict(source=frame, conf=self.conf_thresh, verbose=False)[0]
        
        container_polygon = None
        blue_boxes = []

        # 2. Ekstraksi Bounding Box / Polygon
        for box in results.boxes:
            cls_id = int(box.cls[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int) # [xmin, ymin, xmax, ymax]
            conf = float(box.conf[0])

            if cls_id == 0:  # container_box
                # Untuk kasus miring, buat Polygon dari Bounding Box / MinAreaRect
                x1, y1, x2, y2 = xyxy
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
                container_polygon = pts
            
            elif cls_id == 1:  # blue_box
                x1, y1, x2, y2 = xyxy
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                blue_boxes.append({'bbox': (x1, y1, x2, y2), 'centroid': (cx, cy)})

        # Kopian frame untuk visualisasi
        output_frame = frame.copy()
        count_inside = 0
        count_outside = 0

        # Jika Kontainer Terdeteksi
        if container_polygon is not None:
            # Gambarkan Garis Batas Kontainer Putih
            cv2.polylines(output_frame, [container_polygon], isClosed=True, color=(255, 255, 255), thickness=3)

            # 3. Logika Spasial: Point-in-Polygon Test
            for item in blue_boxes:
                cx, cy = item['centroid']
                x1, y1, x2, y2 = item['bbox']

                # Uji apakah centroid berada di dalam Polygon Kontainer
                # >0 : Di Dalam, 0 : Di Garis, <0 : Di Luar
                status = cv2.pointPolygonTest(container_polygon, (float(cx), float(cy)), measureDist=False)

                if status >= 0:
                    count_inside += 1
                    color = (0, 255, 0) # Hijau untuk INSIDE
                    label = "INSIDE"
                else:
                    count_outside += 1
                    color = (0, 0, 255) # Merah untuk OUTSIDE
                    label = "OUTSIDE"

                # Drawing UI
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(output_frame, (cx, cy), 4, (255, 255, 0), -1)
                cv2.putText(output_frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 4. Tampilkan Ringkasan di Frame
        info_text = f"Inside: {count_inside} | Outside: {count_outside}"
        cv2.putText(output_frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return output_frame, count_inside, count_outside

# ==================== UJI COBA ====================
if __name__ == "__main__":
    # Inisialisasi Sistem
    # Ganti path model dengan model Anda (atau 'yolov8n.pt' untuk demo awal)
    monitor = BoxMonitorSystem(model_path="runs/detect/energeek_box_monitor/weights/best.pt")

    # Buka File Gambar / Video
    image_path = "test_image.jpg"
    frame = cv2.imread(image_path)

    if frame is not None:
        result_frame, inside, outside = monitor.process_frame(frame)
        
        print(f"Hasil Perhitungan -> Di Dalam: {inside} | Di Luar: {outside}")
        
        # Simpan & Tampilkan Gambar
        cv2.imwrite("result_output.jpg", result_frame)
        cv2.imshow("Energeek Monitoring Result", result_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Gagal memuat gambar uji coba.")
```

---

## 4. Rencana Pemeliharaan & Troubleshooting

### 4.1 Kasus Rotasi Ekstrem Kotak Putih (Gambar 1)
Jika kotak putih miring secara signifikan dan Bounding Box biasa ($xyxy$) kurang presisi:
1. Ganti deteksi `container_box` menggunakan **YOLOv8-Oriented Bounding Box (OBB)** atau **YOLOv8 Segmentation (`yolov8n-seg.pt`)**.
2. Ekstrak koordinat polygon kontainer secara presisi dari *contour mask* hasil segmentasi:
   ```python
   # Dapatkan polygon langsung dari mask YOLO Segmentation
   mask = results.masks[0].xy[0] # Mengembalikan array [(x1,y1), (x2,y2), ...]
   container_polygon = np.int32(mask)
   ```

### 4.2 Optimasi Kecepatan untuk Deployment Produk
* **Konversi ke TensorRT / ONNX:**
  Gunakan skrip ekspor berikut agar performa di CPU/Edge Device melesat naik hingga 2x-3x lipat:
  ```python
  model = YOLO("runs/detect/energeek_box_monitor/weights/best.pt")
  model.export(format="onnx")  # atau format="engine" untuk TensorRT
  ```

---
*Dokumen arahan teknis ini siap diimplementasikan untuk proyek Energeek Computer Vision.*
