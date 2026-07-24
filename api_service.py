"""
AI Service API - Energeek Box Monitor
Menerima gambar sebagai input, mengembalikan hasil analisis dalam format JSON.

Endpoint:
  POST /predict
    - Input: file gambar (multipart/form-data)
    - Output: JSON { inside_box, outside_box, total }

Cara jalankan:
  uvicorn api_service:app --reload --host 0.0.0.0 --port 8000

Atau langsung:
  python api_service.py
"""

import os
import json
import io
import uuid
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.responses import JSONResponse
except ImportError:
    print("FastAPI tidak terinstall. Install dengan: pip install fastapi uvicorn python-multipart")
    raise

# ==================== KONFIGURASI ====================

MODEL_PATH = "runs/detect/energeek_box_monitor/weights/best.pt"
FALLBACK_MODEL = "yolov8n.pt"
CONF_THRESH = 0.5
USE_SEGMENTATION = False

# ==================== INISIALISASI MODEL ====================

# Cek model, fallback ke YOLO pretrained jika belum training
if not os.path.exists(MODEL_PATH):
    print(f"[WARNING] Model tidak ditemukan: {os.path.abspath(MODEL_PATH)}")
    print(f"[WARNING] Menggunakan model pretrained: {FALLBACK_MODEL}")
    print(f"[WARNING] Model pretrained hanya mendeteksi class COCO (80 class), bukan container_box/blue_box.")
    print(f"[WARNING] Jalankan train.py terlebih dahulu untuk hasil optimal.")
    MODEL_PATH = FALLBACK_MODEL

model = YOLO(MODEL_PATH)
print(f"[INFO] Model loaded: {MODEL_PATH}")


def analyze_image(image_bytes: bytes) -> dict:
    """
    Analisis satu gambar dan kembalikan hasil hitungan.
    
    Args:
        image_bytes: Raw bytes dari file gambar
        
    Returns:
        dict dengan keys: inside_box, outside_box, total
    """
    # Decode gambar dari bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="Gambar tidak valid atau format tidak didukung")
    
    # Inference
    results = model.predict(source=frame, conf=CONF_THRESH, verbose=False)[0]
    
    container_polygon = None
    blue_boxes = []
    
    # Ekstraksi hasil deteksi
    if USE_SEGMENTATION and results.masks is not None:
        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            if cls_id == 0:  # container_box
                mask_xy = results.masks[i].xy[0]
                container_polygon = np.int32(mask_xy)
                break
    else:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            
            if cls_id == 0:  # container_box
                x1, y1, x2, y2 = xyxy
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
                container_polygon = pts
            
            elif cls_id == 1:  # blue_box
                x1, y1, x2, y2 = xyxy
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                blue_boxes.append({'centroid': (cx, cy)})
    
    # Logika Spasial: Point-in-Polygon
    count_inside = 0
    count_outside = 0
    
    if container_polygon is not None:
        for item in blue_boxes:
            cx, cy = item['centroid']
            status = cv2.pointPolygonTest(
                container_polygon,
                (float(cx), float(cy)),
                measureDist=False
            )
            if status >= 0:
                count_inside += 1
            else:
                count_outside += 1
    else:
        # Jika container tidak terdeteksi, semua blue_box dianggap outside
        count_outside = len(blue_boxes)
    
    return {
        "inside_box": count_inside,
        "outside_box": count_outside,
        "total": count_inside + count_outside
    }


# ==================== FASTAPI APP ====================

app = FastAPI(
    title="Energeek Box Monitor - AI Service",
    description="API untuk mendeteksi blue_box di dalam vs di luar container_box",
    version="1.0.0"
)


@app.get("/")
def root():
    """Root endpoint - informasi service."""
    return {
        "service": "Energeek Box Monitor AI Service",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "Informasi service",
            "GET /health": "Health check",
            "POST /predict": "Upload gambar untuk analisis"
        },
        "model": str(MODEL_PATH),
        "conf_threshold": CONF_THRESH
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload gambar dan dapatkan hasil analisis.
    
    Args:
        file: File gambar (jpg, jpeg, png)
        
    Returns:
        JSON: { inside_box, outside_box, total }
    """
    # Validasi tipe file
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        # Fallback: cek dari ekstensi file
        ext = Path(file.filename).suffix.lower() if file.filename else ""
        if ext not in [".jpg", ".jpeg", ".png"]:
            raise HTTPException(
                status_code=400,
                detail=f"Format tidak didukung: {file.content_type}. Gunakan jpg, jpeg, atau png."
            )
    
    # Baca file
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal membaca file: {str(e)}")
    
    if not image_bytes:
        raise HTTPException(status_code=400, detail="File kosong")
    
    # Analisis
    result = analyze_image(image_bytes)
    
    return JSONResponse(content=result)


@app.post("/predict/base64")
async def predict_base64(data: dict):
    """
    Upload gambar dalam format base64.
    
    Request body:
    {
        "image": "base64_encoded_string",
        "filename": "optional_name.jpg"
    }
    
    Returns:
        JSON: { inside_box, outside_box, total }
    """
    import base64
    
    if "image" not in data:
        raise HTTPException(status_code=400, detail="Field 'image' diperlukan (base64 encoded)")
    
    try:
        image_bytes = base64.b64decode(data["image"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Base64 tidak valid: {str(e)}")
    
    result = analyze_image(image_bytes)
    return JSONResponse(content=result)


# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("  Energeek Box Monitor - AI Service")
    print("=" * 50)
    print()
    print(f"  Model: {MODEL_PATH}")
    print(f"  Confidence threshold: {CONF_THRESH}")
    print()
    print("  Running at: http://localhost:8000")
    print("  API Docs:    http://localhost:8000/docs")
    print()
    print("  Endpoints:")
    print("    POST /predict        - Upload image file")
    print("    POST /predict/base64 - Base64 encoded image")
    print("    GET  /health         - Health check")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
