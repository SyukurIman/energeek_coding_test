"""
Training Script for YOLOv8 Nano - Energeek Box Monitor
Melatih model untuk mendeteksi container_box (kotak putih) dan blue_box (kotak biru).
"""

from ultralytics import YOLO


def train_model():
    """
    Melatih model YOLOv8 Nano dengan dataset kustom.
    Menghasilkan bobot model terbaik di runs/detect/energeek_box_monitor/weights/best.pt
    """
    # 1. Load Pretrained YOLOv8 Nano Model
    # Gunakan 'yolov8n.pt' untuk deteksi standar
    # Gunakan 'yolov8n-seg.pt' jika ingin menggunakan segmentasi
    # Gunakan 'yolov8n-obb.pt' jika ingin menggunakan Oriented Bounding Box
    model = YOLO("yolo26n.pt")

    # 2. Mulai Training
    # Catatan: YOLO otomatis menyimpan ke runs/detect/{name}/weights/best.pt
    results = model.train(
        data="dataset/data.yaml",
        epochs=100,                     # Sesuaikan dengan tingkat konvergensi
        imgsz=640,                      # Resolusi input standar YOLOv8
        batch=8,                       # Sesuaikan dengan VRAM GPU / RAM CPU
        name="energeek_box_monitor",
        save=True,
        save_period=50,                 # Simpan checkpoint setiap 10 epoch
        device="0",                       # '0' untuk GPU pertama, 'cpu' jika tanpa GPU
        patience=20,                    # Early stopping jika tidak ada improvement
        lr0=0.01,                       # Learning rate awal
        augment=True,                   # Gunakan data augmentation
        cos_lr=True,                    # Cosine learning rate scheduler
        warmup_epochs=10,                # Warmup epochs
        verbose=True,

        mosaic=1.0,                      # Menggabungkan 4 gambar jadi 1 (wajib untuk data sedikit)
        scale=0.5,                       # Mengubah skala gambar secara acak
        
        optimizer="AdamW",
    )

    print("\n" + "=" * 60)
    print("Training Selesai!")
    print("Model terbaik disimpan di: runs/detect/energeek_box_monitor/weights/best.pt")
    print("Model terakhir disimpan di: runs/detect/energeek_box_monitor/weights/last.pt")
    print("=" * 60)


def export_model_onnx():
    """
    Opsional: Ekspor model ke format ONNX untuk optimasi deployment.
    """
    model = YOLO("runs/detect/energeek_box_monitor/weights/best.pt")
    model.export(format="onnx", imgsz=640)
    print("Model berhasil diekspor ke format ONNX!")


def export_model_tensorrt():
    """
    Opsional: Ekspor model ke format TensorRT untuk akselerasi GPU maksimal.
    """
    model = YOLO("runs/detect/energeek_box_monitor/weights/best.pt")
    model.export(format="engine", imgsz=640, device="0")
    print("Model berhasil diekspor ke format TensorRT Engine!")


if __name__ == "__main__":
    train_model()

    # Uncomment salah satu atau keduanya jika ingin ekspor setelah training
    # export_model_onnx()
    # export_model_tensorrt()
