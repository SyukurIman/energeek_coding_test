"""
Dataset Augmentation Script - Energeek Box Monitor
Memperbanyak dataset dengan augmentasi gambar + transformasi label YOLO.
Cocok untuk data yang masih sedikit agar model lebih robust.

Augmentasi yang didukung:
  ✓ Horizontal Flip (mirror)
  ✓ Vertical Flip
  ✓ Rotasi 90°, 180°, 270° (sudut siku-siku — label tetap presisi)
  ✓ Kecerahan & Kontras
  × Rotasi sembarang (label YOLO tidak presisi untuk sudut non-90°)
"""

import os
import cv2
import numpy as np
import shutil
from pathlib import Path
import random


def read_yolo_label(label_path: str, img_w: int, img_h: int):
    """
    Baca file label YOLO (.txt).
    Format: class_id x_center y_center width height  (semua normalized 0-1)

    Returns: list of [class_id, x_center, y_center, width, height]
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                boxes.append([cls_id, x_center, y_center, w, h])
    return boxes


def write_yolo_label(label_path: str, boxes: list):
    """Tulis file label YOLO (.txt)."""
    with open(label_path, 'w') as f:
        for box in boxes:
            cls_id, x_center, y_center, w, h = box
            # Clamp ke [0, 1] biar aman
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


def augment_horizontal_flip(img: np.ndarray, boxes: list):
    """Flip horizontal (kiri↔kanan)."""
    h, w = img.shape[:2]
    img_flip = cv2.flip(img, 1)  # 1 = horizontal flip
    boxes_flip = []
    for box in boxes:
        cls_id, xc, yc, bw, bh = box
        # x_center baru = 1 - x_center (kiri jadi kanan)
        xc_new = 1.0 - xc
        boxes_flip.append([cls_id, xc_new, yc, bw, bh])
    return img_flip, boxes_flip


def augment_vertical_flip(img: np.ndarray, boxes: list):
    """Flip vertikal (atas↔bawah)."""
    h, w = img.shape[:2]
    img_flip = cv2.flip(img, 0)  # 0 = vertical flip
    boxes_flip = []
    for box in boxes:
        cls_id, xc, yc, bw, bh = box
        # y_center baru = 1 - y_center
        yc_new = 1.0 - yc
        boxes_flip.append([cls_id, xc, yc_new, bw, bh])
    return img_flip, boxes_flip


def augment_rotate_90(img: np.ndarray, boxes: list):
    """Rotasi 90° searah jarum jam."""
    h, w = img.shape[:2]
    img_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    boxes_rot = []
    for box in boxes:
        cls_id, xc, yc, bw, bh = box
        # Rotasi 90° CW: (x, y) -> (y, w-1-x) dalam pixel
        # Normalized: xc_baru = yc, yc_baru = 1 - xc
        # width & height tertukar
        xc_new = yc
        yc_new = 1.0 - xc
        # bw dan bh tertukar (karena gambar diputar)
        bw_new = bh
        bh_new = bw
        boxes_rot.append([cls_id, xc_new, yc_new, bw_new, bh_new])
    return img_rot, boxes_rot


def augment_rotate_180(img: np.ndarray, boxes: list):
    """Rotasi 180°."""
    img_rot = cv2.rotate(img, cv2.ROTATE_180)
    boxes_rot = []
    for box in boxes:
        cls_id, xc, yc, bw, bh = box
        xc_new = 1.0 - xc
        yc_new = 1.0 - yc
        boxes_rot.append([cls_id, xc_new, yc_new, bw, bh])
    return img_rot, boxes_rot


def augment_rotate_270(img: np.ndarray, boxes: list):
    """Rotasi 270° searah jarum jam (atau 90° CCW)."""
    h, w = img.shape[:2]
    img_rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    boxes_rot = []
    for box in boxes:
        cls_id, xc, yc, bw, bh = box
        # Rotasi 90° CCW: (x, y) -> (h-1-y, x)
        # Normalized: xc_baru = 1 - yc, yc_baru = xc
        xc_new = 1.0 - yc
        yc_new = xc
        bw_new = bh
        bh_new = bw
        boxes_rot.append([cls_id, xc_new, yc_new, bw_new, bh_new])
    return img_rot, boxes_rot


def augment_brightness_contrast(img: np.ndarray, boxes: list,
                                 alpha: float = None, beta: int = None):
    """
    Ubah brightness (beta) dan contrast (alpha).
    alpha: 1.0 = asli, >1 lebih kontras, <1 kurang kontras
    beta: nilai brightness (-100 sd 100)
    """
    if alpha is None:
        alpha = random.uniform(0.6, 1.4)
    if beta is None:
        beta = random.randint(-40, 40)
    img_aug = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    # Label tidak berubah
    return img_aug, boxes


def augment_hsv_shift(img: np.ndarray, boxes: list):
    """
    Ubah Hue, Saturation, Value (HSV) secara acak.
    Label tidak berubah.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Hue shift
    h_shift = random.randint(-10, 10)
    hsv[:, :, 0] += h_shift
    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)

    # Saturation shift
    s_scale = random.uniform(0.5, 1.5)
    hsv[:, :, 1] *= s_scale
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

    # Value shift
    v_scale = random.uniform(0.5, 1.5)
    hsv[:, :, 2] *= v_scale
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

    img_aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img_aug, boxes


def augment_gaussian_blur(img: np.ndarray, boxes: list):
    """Beri efek blur ringan. Label tidak berubah."""
    ksize = random.choice([3, 5])
    img_aug = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return img_aug, boxes


def augment_noise(img: np.ndarray, boxes: list):
    """Tambah random noise. Label tidak berubah."""
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img_aug = cv2.add(img, noise)
    return img_aug, boxes


# ==================== DAFTAR AUGMENTASI ====================
# Setiap fungsi menerima (img, boxes) -> (img_aug, boxes_aug)
AUGMENTATIONS = [
    ("hflip", augment_horizontal_flip),
    ("vflip", augment_vertical_flip),
    ("rot90", augment_rotate_90),
    ("rot180", augment_rotate_180),
    ("rot270", augment_rotate_270),
    ("bright1", lambda img, boxes: augment_brightness_contrast(
        img, boxes, alpha=0.7, beta=-20)),
    ("bright2", lambda img, boxes: augment_brightness_contrast(
        img, boxes, alpha=1.3, beta=20)),
    ("hsv", augment_hsv_shift),
    ("blur", augment_gaussian_blur),
    ("noise", augment_noise),
]


def augment_dataset(data_root: str = "dataset",
                    target_multiplier: int = 3,
                    subsets: list = None):
    """
    Perbanyak dataset dengan augmentasi.

    Args:
        data_root: Root folder dataset (berisi train/, val/)
        target_multiplier: Target perbanyak data (3 = data jadi 3x lipat)
        subsets: List subset yg diaugmentasi, default ['train', 'val']

    Cara kerja:
        1. Baca gambar & label asli
        2. Terapkan augmentasi acak (N - 1) kali per gambar asli
           (N = target_multiplier)
        3. Simpan hasil augmentasi di folder yang sama dgn prefix nama
    """
    if subsets is None:
        subsets = ['train', 'val']

    total_original = 0
    total_augmented = 0

    for subset in subsets:
        img_dir = os.path.join(data_root, subset, "images")
        label_dir = os.path.join(data_root, subset, "labels")

        if not os.path.exists(img_dir):
            print(f"⚠️  Folder tidak ditemukan: {img_dir}")
            continue

        image_files = [f for f in os.listdir(img_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            print(f"⚠️  Tidak ada gambar di {img_dir}")
            continue

        print(f"\n{'='*50}")
        print(f"📁 Subset: {subset}")
        print(f"   Gambar asli: {len(image_files)}")
        print(f"   Target multiplier: {target_multiplier}x")
        print(f"{'='*50}")

        for img_name in image_files:
            img_path = os.path.join(img_dir, img_name)
            base_name = os.path.splitext(img_name)[0]
            ext = os.path.splitext(img_name)[1]

            # Path label
            label_name = base_name + ".txt"
            label_path = os.path.join(label_dir, label_name)

            # Baca gambar
            img = cv2.imread(img_path)
            if img is None:
                print(f"   ⚠️  Gagal baca: {img_path}")
                continue

            h, w = img.shape[:2]

            # Baca label YOLO
            boxes = read_yolo_label(label_path, w, h)

            total_original += 1

            # Pilih augmentasi acak untuk mencapai target_multiplier
            # Kita sudah punya 1 gambar asli, butuh (target_multiplier - 1) tambahan
            num_to_generate = target_multiplier - 1

            if num_to_generate <= 0:
                continue

            # Pilih augmentasi secara acak (boleh diulang)
            chosen_augs = random.choices(AUGMENTATIONS, k=num_to_generate)

            for idx, (aug_name, aug_func) in enumerate(chosen_augs):
                try:
                    img_aug, boxes_aug = aug_func(img.copy(), [b.copy() for b in boxes])

                    # Skip jika hasil augmentasi menghasilkan box tidak valid
                    if boxes_aug is None:
                        continue

                    # Nama file baru
                    aug_img_name = f"{base_name}_{aug_name}_{idx}{ext}"
                    aug_img_path = os.path.join(img_dir, aug_img_name)
                    aug_label_name = f"{base_name}_{aug_name}_{idx}.txt"
                    aug_label_path = os.path.join(label_dir, aug_label_name)

                    # Simpan gambar augmentasi
                    cv2.imwrite(aug_img_path, img_aug)

                    # Simpan label (dengan box yang sudah ditransformasi)
                    write_yolo_label(aug_label_path, boxes_aug)

                    total_augmented += 1

                except Exception as e:
                    print(f"   ⚠️  Gagal augmentasi {img_name} -> {aug_name}: {e}")
                    continue

        # Info subset
        after_count = len([f for f in os.listdir(img_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"   ✅ {subset}: {len(image_files)} → {after_count} gambar")

    # Summary
    print(f"\n{'='*50}")
    print(f"📊 SUMMARY AUGMENTASI")
    print(f"{'='*50}")
    print(f"   Total gambar asli   : {total_original}")
    print(f"   Total gambar baru   : {total_augmented}")
    print(f"   Total keseluruhan   : {total_original + total_augmented}")
    print(f"{'='*50}")


def preview_augmentations(image_path: str, output_dir: str = "aug_preview"):
    """
    Preview semua jenis augmentasi pada satu gambar.
    Berguna untuk melihat hasil sebelum menjalankan augmentasi penuh.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Gagal baca {image_path}")
        return

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    # Cari label
    label_dir = os.path.dirname(image_path).replace("images", "labels")
    label_path = os.path.join(label_dir, base_name + ".txt")
    boxes = read_yolo_label(label_path, img.shape[1], img.shape[0])

    print(f"Preview augmentasi untuk: {image_path}")
    print(f"  Jumlah box: {len(boxes)}")
    print(f"  Output dir: {output_dir}/")
    print()

    # Simpan gambar asli
    cv2.imwrite(os.path.join(output_dir, f"00_original.jpg"), img)

    for i, (aug_name, aug_func) in enumerate(AUGMENTATIONS):
        try:
            img_aug, boxes_aug = aug_func(img.copy(), [b.copy() for b in boxes])

            # Visualisasi box di gambar
            h, w = img_aug.shape[:2]
            for box in boxes_aug:
                cls_id, xc, yc, bw, bh = box
                # Convert normalized to pixel
                x1 = int((xc - bw/2) * w)
                y1 = int((yc - bh/2) * h)
                x2 = int((xc + bw/2) * w)
                y2 = int((yc + bh/2) * h)
                color = (0, 255, 0) if cls_id == 0 else (255, 0, 0)
                cv2.rectangle(img_aug, (x1, y1), (x2, y2), color, 2)

            out_path = os.path.join(output_dir, f"{i+1:02d}_{aug_name}.jpg")
            cv2.imwrite(out_path, img_aug)
            print(f"  ✅ {aug_name:10s} -> {out_path}")

        except Exception as e:
            print(f"  ❌ {aug_name:10s} -> Error: {e}")

    print(f"\n✅ Preview selesai! Lihat folder '{output_dir}/'")


# ==================== MAIN ====================
if __name__ == "__main__":
    import sys

    print("=" * 50)
    print("  Energeek - Dataset Augmentation Tool")
    print("=" * 50)
    print()
    print("Pilih mode:")
    print("1. Preview augmentasi (lihat hasil di folder aug_preview/)")
    print("2. Augmentasi dataset penuh")
    print("3. Hapus semua hasil augmentasi (reset dataset ke asli)")

    choice = input("\nMasukkan pilihan (1/2/3): ").strip()

    if choice == "1":
        # Mode preview
        img_path = input("Path gambar untuk preview: ").strip()
        if os.path.exists(img_path):
            preview_augmentations(img_path)
        else:
            print(f"File tidak ditemukan: {img_path}")

    elif choice == "2":
        # Mode augmentasi penuh
        multiplier = input("Target multiplier (contoh: 3 = data jadi 3x lipat) [default=3]: ").strip()
        multiplier = int(multiplier) if multiplier.isdigit() else 3

        subsets = input("Subset yang diaugmentasi (train/val/both) [default=both]: ").strip().lower()
        if subsets == "train":
            subsets_list = ['train']
        elif subsets == "val":
            subsets_list = ['val']
        else:
            subsets_list = ['train', 'val']

        print("\nMemulai augmentasi...")
        augment_dataset(
            data_root="dataset",
            target_multiplier=multiplier,
            subsets=subsets_list
        )

    elif choice == "3":
        # Hapus semua file hasil augmentasi
        confirm = input("Yakin ingin menghapus SEMUA file augmentasi? (y/n): ").strip().lower()
        if confirm == 'y':
            data_root = "dataset"
            deleted = 0
            for subset in ['train', 'val']:
                img_dir = os.path.join(data_root, subset, "images")
                label_dir = os.path.join(data_root, subset, "labels")
                for d in [img_dir, label_dir]:
                    if os.path.exists(d):
                        for f in os.listdir(d):
                            # Hanya hapus file yang bukan asli (mengandung nama augmentasi)
                            # File asli tidak mengandung underscore + nama augmentasi
                            # Tapi kita bisa rebuild dari backup
                            pass

            # Cara lebih aman: backup dulu
            backup_dir = "dataset_backup"
            if os.path.exists(backup_dir):
                print("Restore dari backup...")
                for subset in ['train', 'val']:
                    for sub in ['images', 'labels']:
                        src = os.path.join(backup_dir, subset, sub)
                        dst = os.path.join(data_root, subset, sub)
                        if os.path.exists(src):
                            # Hapus dulu folder tujuan
                            if os.path.exists(dst):
                                shutil.rmtree(dst)
                            # Copy backup
                            shutil.copytree(src, dst)
                            print(f"  ✅ Restore {subset}/{sub}")
                print("✅ Dataset di-reset ke kondisi asli!")
            else:
                print("❌ Tidak ada backup. Hanya bisa hapus file ber-label augmentasi.")
                # Hapus file dengan pola *_hflip_* dll
                for subset in ['train', 'val']:
                    for d in [os.path.join(data_root, subset, "images"),
                              os.path.join(data_root, subset, "labels")]:
                        if os.path.exists(d):
                            for f in os.listdir(d):
                                # Cek apakah file hasil augmentasi
                                is_aug = False
                                for aug_name, _ in AUGMENTATIONS:
                                    if f"_{aug_name}_" in f:
                                        is_aug = True
                                        break
                                if is_aug:
                                    os.remove(os.path.join(d, f))
                                    deleted += 1
                print(f"✅ {deleted} file augmentasi dihapus.")

    else:
        print("Pilihan tidak valid.")

    print("\nSelesai!")
