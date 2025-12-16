import os
import random
from pathlib import Path

import cv2
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


# Import đúng các hàm bạn đã dùng để build dataset
from dataset_lisa import (
    find_box_csv_files,
    load_annotations,
    infer_image_base_dir,
    resolve_image_path,
    read_rgb,
    crop_patch,
    map_label_to_4,
)

# =======================
# CONFIG (bạn sửa 3 dòng)
# =======================
DATASET_ROOT = r"C:\Users\PC\.cache\kagglehub\datasets\mbornoe\lisa-traffic-light-dataset\versions\2"
OUT_DIR = r"demo_assets/patches"
PATCH_SIZE = 64

# Mỗi lớp (green/red/yellow) lấy bao nhiêu patch đèn để demo
N_PER_CLASS = 30

# Số patch "other/background" để demo reject
N_OTHER = 60

RANDOM_SEED = 42
MAX_TRIALS_PER_OTHER = 300  # tăng nếu dataset khó lấy negative
IOU_MAX = 0.05              # patch other không được overlap bbox đèn quá mức này
# =======================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CLASS_ID2NAME = {0: "other", 1: "green", 2: "red", 3: "yellow"}


def ensure_dirs(base: str):
    for name in ["green", "red", "yellow", "other"]:
        Path(base, name).mkdir(parents=True, exist_ok=True)


def clamp_xyxy(x1, y1, x2, y2, W, H):
    x1 = max(0, min(int(x1), W - 1))
    y1 = max(0, min(int(y1), H - 1))
    x2 = max(1, min(int(x2), W))
    y2 = max(1, min(int(y2), H))
    return x1, y1, x2, y2


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def pick_cols(df):
    """Copy y hệt logic pick_col trong dataset_lisa.py nhưng trả về tên cột gốc."""
    cols = [c.lower() for c in df.columns]

    def pick_col(cands):
        for c in cands:
            if c in cols:
                return df.columns[cols.index(c)]
        return None

    col_fname = pick_col(["filename", "file", "image", "frame"])
    col_xmin  = pick_col(["xmin", "x1", "x_min", "upper left corner x"])
    col_ymin  = pick_col(["ymin", "y1", "y_min", "upper left corner y"])
    col_xmax  = pick_col(["xmax", "x2", "x_max", "lower right corner x"])
    col_ymax  = pick_col(["ymax", "y2", "y_max", "lower right corner y"])
    col_label = pick_col(["annotation", "label", "state", "type", "trafficlight", "annotation tag"])

    return col_fname, col_xmin, col_ymin, col_xmax, col_ymax, col_label


def save_rgb_patch_as_jpg(patch_rgb: np.ndarray, out_path: str):
    # patch_rgb -> BGR để cv2.imwrite
    patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, patch_bgr)


def main():
    ensure_dirs(OUT_DIR)

    csv_files = find_box_csv_files(DATASET_ROOT)
    if not csv_files:
        raise RuntimeError("Không tìm thấy frameAnnotationsBOX.csv. Kiểm tra DATASET_ROOT.")

    # 1) Thu thập patches đèn theo bbox để đủ green/red/yellow
    counts = {"green": 0, "red": 0, "yellow": 0}
    picked_records = []  # lưu (img_path, bbox_xyxy) để lát lấy OTHER từ chính các ảnh này

    for csv_path in csv_files:
        df = load_annotations(csv_path)
        col_fname, col_xmin, col_ymin, col_xmax, col_ymax, col_label = pick_cols(df)
        if not all([col_fname, col_xmin, col_ymin, col_xmax, col_ymax, col_label]):
            continue

        image_base_dir = infer_image_base_dir(csv_path)

        for _, row in df.iterrows():
            y = map_label_to_4(row[col_label])
            if y not in (1, 2, 3):  # chỉ lấy 3 màu
                continue

            lab = CLASS_ID2NAME[y]
            if counts[lab] >= N_PER_CLASS:
                continue

            img_path = resolve_image_path(image_base_dir, row[col_fname])
            if img_path is None or (not os.path.exists(img_path)):
                continue

            try:
                img_rgb = read_rgb(img_path)
            except Exception:
                continue

            H, W, _ = img_rgb.shape
            x1, y1, x2, y2 = clamp_xyxy(row[col_xmin], row[col_ymin], row[col_xmax], row[col_ymax], W, H)

            patch = crop_patch(img_rgb, x1, y1, x2, y2, out_size=(PATCH_SIZE, PATCH_SIZE))
            if patch is None:
                continue

            out_path = os.path.join(OUT_DIR, lab, f"{lab}_{counts[lab]:04d}.jpg")
            save_rgb_patch_as_jpg(patch, out_path)

            counts[lab] += 1
            picked_records.append((img_path, (x1, y1, x2, y2)))

            if all(counts[c] >= N_PER_CLASS for c in counts):
                break
        if all(counts[c] >= N_PER_CLASS for c in counts):
            break

    print("Saved traffic-light patches:", counts)

    # 2) Tạo OTHER patches: random crop 64x64 không overlap bbox đèn (từ các ảnh đã pick)
    other_saved = 0
    if not picked_records:
        print("WARNING: Không pick được ảnh/bbox nào để tạo OTHER.")
        return

    trials = 0
    while other_saved < N_OTHER and trials < N_OTHER * MAX_TRIALS_PER_OTHER:
        trials += 1
        img_path, bbox = random.choice(picked_records)

        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        if W < PATCH_SIZE or H < PATCH_SIZE:
            continue

        rx1 = random.randint(0, W - PATCH_SIZE)
        ry1 = random.randint(0, H - PATCH_SIZE)
        rx2 = rx1 + PATCH_SIZE
        ry2 = ry1 + PATCH_SIZE

        if iou_xyxy((rx1, ry1, rx2, ry2), bbox) > IOU_MAX:
            continue

        patch_bgr = bgr[ry1:ry2, rx1:rx2]
        out_path = os.path.join(OUT_DIR, "other", f"other_{other_saved:04d}.jpg")
        cv2.imwrite(out_path, patch_bgr)
        other_saved += 1

    print("Saved OTHER patches:", other_saved)
    if other_saved == 0:
        print(
            "WARNING: OTHER=0 thường do bbox quá lớn hoặc ảnh quá nhỏ. "
            "Thử tăng MAX_TRIALS_PER_OTHER hoặc nới IOU_MAX lên 0.1."
        )


if __name__ == "__main__":
    main()
