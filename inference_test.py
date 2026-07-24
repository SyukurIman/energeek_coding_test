"""
Inference & Spatial Logic Script - Energeek Box Monitor
Mendeteksi container_box (kotak putih) dan blue_box (kotak biru),
lalu menghitung jumlah blue_box yang berada di dalam vs di luar container_box
menggunakan Point-in-Polygon Test (cv2.pointPolygonTest).
"""

import cv2
import numpy as np
from ultralytics import YOLO


class BoxMonitorSystem:
    """
    Sistem Monitoring Deteksi Objek Kotak Biru (In/Out Container Putih).
    
    Args:
        model_path: Path ke model YOLOv8 (.pt)
        conf_thresh: Confidence threshold untuk deteksi (default: 0.5)
        use_segmentation: Jika True, gunakan mask segmentasi untuk container_box (yolov8n-seg)
    """

    def __init__(self, model_path: str = "runs/detect/energeek_box_monitor/weights/best.pt",
                 conf_thresh: float = 0.5,
                 use_segmentation: bool = False):
        # Load Model YOLOv8 Hasil Training

        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.use_segmentation = use_segmentation

    def process_frame(self, frame: np.ndarray):
        """
        Proses satu frame gambar untuk deteksi dan logika spasial.
        
        Args:
            frame: Input image numpy array (BGR format)
            
        Returns:
            output_frame: Frame dengan visualisasi hasil deteksi
            count_inside: Jumlah blue_box di dalam container
            count_outside: Jumlah blue_box di luar container
        """
        # 1. Inference dengan YOLOv8
        results = self.model.predict(source=frame, conf=self.conf_thresh, verbose=False)[0]

        container_polygon = None
        blue_boxes = []

        # ========== EKSTRAKSI HASIL DETEKSI ==========

        # Cek apakah model menggunakan segmentasi
        if self.use_segmentation and results.masks is not None:
            # Ekstraksi polygon dari mask segmentasi (lebih presisi untuk bentuk miring)
            for i, box in enumerate(results.boxes):
                cls_id = int(box.cls[0])
                if cls_id == 0:  # container_box
                    # Ambil polygon dari mask segmentasi
                    mask_xy = results.masks[i].xy[0]  # Array of (x, y) points
                    container_polygon = np.int32(mask_xy)
                    break  # Ambil container pertama (asumsi hanya ada 1)
        else:
            # Ekstraksi dari Bounding Box biasa (xyxy)
            for box in results.boxes:
                cls_id = int(box.cls[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)  # [xmin, ymin, xmax, ymax]
                conf = float(box.conf[0])

                if cls_id == 0:  # container_box
                    x1, y1, x2, y2 = xyxy
                    # Buat polygon dari 4 sudut bounding box
                    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
                    container_polygon = pts

                elif cls_id == 1:  # blue_box
                    x1, y1, x2, y2 = xyxy
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    blue_boxes.append({
                        'bbox': (x1, y1, x2, y2),
                        'centroid': (cx, cy),
                        'confidence': conf
                    })

        # Salin frame untuk visualisasi
        output_frame = frame.copy()
        count_inside = 0
        count_outside = 0

        # ========== LOGIKA SPASIAL: POINT-IN-POLYGON TEST ==========
        if container_polygon is not None:
            # Gambarkan Garis Batas Kontainer Putih
            cv2.polylines(output_frame, [container_polygon], isClosed=True,
                          color=(255, 255, 255), thickness=3)

            # Hitung luas container untuk scaling font
            container_area = cv2.contourArea(container_polygon)

            for item in blue_boxes:
                cx, cy = item['centroid']
                x1, y1, x2, y2 = item['bbox']

                # Point-in-Polygon Test:
                #   > 0  : Di Dalam polygon
                #   == 0 : Tepat di garis tepi
                #   < 0  : Di Luar polygon
                status = cv2.pointPolygonTest(
                    container_polygon,
                    (float(cx), float(cy)),
                    measureDist=False
                )

                if status >= 0:
                    count_inside += 1
                    color = (0, 255, 0)      # Hijau untuk INSIDE
                    label = "INSIDE"
                else:
                    count_outside += 1
                    color = (0, 0, 255)      # Merah untuk OUTSIDE
                    label = "OUTSIDE"

                # Drawing Bounding Box untuk blue_box
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                # Gambar centroid
                cv2.circle(output_frame, (cx, cy), 4, (255, 255, 0), -1)
                # Label INSIDE/OUTSIDE
                cv2.putText(output_frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Gambarkan container_box label jika terdeteksi
        if container_polygon is not None:
            # Tulis label "CONTAINER" di atas polygon
            M = cv2.moments(container_polygon)
            if M['m00'] > 0:
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                cv2.putText(output_frame, "CONTAINER", (cX - 40, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ========== TAMPILKAN RINGKASAN DI FRAME ==========
        info_text = f"Inside: {count_inside} | Outside: {count_outside}"
        cv2.putText(output_frame, info_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Tambah info tambahan
        cv2.putText(output_frame, f"Total Blue Boxes: {count_inside + count_outside}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return output_frame, count_inside, count_outside

    def process_video(self, video_path: str, output_path: str = None, display: bool = True):
        """
        Proses video file atau stream kamera.
        
        Args:
            video_path: Path ke file video (atau 0 untuk webcam)
            output_path: Path untuk menyimpan video output (None = tidak disimpan)
            display: Jika True, tampilkan video secara real-time
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Tidak dapat membuka video {video_path}")
            return

        # Dapatkan properti video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Inisialisasi VideoWriter jika menyimpan output
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Memproses video: {video_path}")
        print(f"Resolusi: {width}x{height}, FPS: {fps}, Total Frame: {total_frames}")
        print("-" * 40)

        frame_count = 0
        total_inside = 0
        total_outside = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Proses frame
            output_frame, inside, outside = self.process_frame(frame)

            # Akumulasi untuk rata-rata
            total_inside += inside
            total_outside += outside
            frame_count += 1

            # Simpan frame jika diperlukan
            if writer:
                writer.write(output_frame)

            # Tampilkan frame (jika environment mendukung GUI)
            if display:
                try:
                    cv2.imshow("Energeek Box Monitor - Video", output_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' atau ESC untuk keluar
                        break
                except (cv2.error, AttributeError):
                    # Fallback: tidak bisa menampilkan GUI, tetap lanjut proses
                    pass

            # Progress indicator
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}/{total_frames} - Inside: {inside}, Outside: {outside}")

        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if display:
            try:
                cv2.destroyAllWindows()
            except (cv2.error, AttributeError):
                pass

        # Print summary
        avg_inside = total_inside / max(frame_count, 1)
        avg_outside = total_outside / max(frame_count, 1)
        print("-" * 40)
        print(f"Video selesai diproses: {frame_count} frame")
        print(f"Rata-rata Inside: {avg_inside:.1f}")
        print(f"Rata-rata Outside: {avg_outside:.1f}")


# ==================== MAIN / UJI COBA ====================
if __name__ == "__main__":
    import os

    # Tentukan path model - gunakan model default YOLOv8 jika belum training
    model_path = "runs/detect/energeek_box_monitor/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Model tidak ditemukan di {os.path.abspath(model_path)}")
        print("Menggunakan model pretrained YOLOv8n untuk demo...")
        model_path = "yolov8n.pt"
        print("Catatan: Model pretrained hanya akan mendeteksi class COCO (bukan container_box/blue_box).")
        print("         Jalankan train.py terlebih dahulu untuk melatih model kustom.")
        print()

    # Inisialisasi Sistem Monitoring
    monitor = BoxMonitorSystem(
        model_path=model_path,
        conf_thresh=0.5,
        use_segmentation=False  # Set True jika menggunakan model segmentasi
    )

    # Pilih mode: Gambar, Video, atau Webcam
    print("Pilih mode pengujian:")
    print("1. Gambar statis (test_image.jpg)")
    print("2. Video file (test_video.mp4)")
    print("3. Webcam (kamera real-time)")

    choice = input("Masukkan pilihan (1/2/3): ").strip()

    if choice == "1":
        # ===== UJI COBA GAMBAR STATIS =====
        image_path = input("Masukkan path gambar (default: test_image.jpg): ").strip()
        if not image_path:
            image_path = "test_image.jpg"

        frame = cv2.imread(image_path)
        if frame is not None:
            result_frame, inside, outside = monitor.process_frame(frame)
            print(f"\nHasil Perhitungan -> Di Dalam: {inside} | Di Luar: {outside}")

            # Simpan hasil
            output_path = "result_output.jpg"
            cv2.imwrite(output_path, result_frame)
            print(f"Hasil disimpan ke: {output_path}")

            # Tampilkan (jika environment mendukung GUI)
            try:
                cv2.imshow("Energeek Monitoring Result", result_frame)
                print("Tekan sembarang tombol untuk menutup jendela...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except cv2.error:
                print("(Preview tidak tersedia di environment ini)")
            except AttributeError:
                print("(Preview tidak tersedia di environment ini)")
        else:
            print(f"Error: Gagal memuat gambar {image_path}")

    elif choice == "2":
        # ===== UJI COBA VIDEO =====
        video_path = input("Masukkan path video (default: test_video.mp4): ").strip()
        if not video_path:
            video_path = "test_video.mp4"

        output_video = input("Simpan video output? (y/n, default: n): ").strip().lower()
        save_path = "result_video.mp4" if output_video == 'y' else None

        if os.path.exists(video_path):
            monitor.process_video(video_path, output_path=save_path, display=True)
        else:
            print(f"Error: File video {video_path} tidak ditemukan")

    elif choice == "3":
        # ===== UJI COBA WEBCAM =====
        print("Mengakses webcam (tekan 'q' atau ESC untuk keluar)...")
        monitor.process_video(0, display=True)

    else:
        print("Pilihan tidak valid.")
