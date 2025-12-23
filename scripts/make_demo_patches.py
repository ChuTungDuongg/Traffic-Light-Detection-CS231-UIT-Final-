import os
import random
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset_lisa import (
    find_box_csv_files,
    load_annotations,
    infer_image_base_dir,
    resolve_image_path,
    read_rgb,
    crop_patch,
    map_label_to_4,
)

DATASET_ROOT = r"C:\Users\PC\.cache\kagglehub\datasets\mbornoe\lisa-traffic-light-dataset\versions\2"
OUT_DIR = r"demo_assets/patches_test_only"
PATCH_SIZE = 64

RANDOM_SEED = 42
TEST_SIZE = 0.2

N_PER_CLASS = 200
N_OTHER = 100

IOU_MAX = 0.05
MAX_TRIALS_PER_OTHER = 3000

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
    cols = [c.lower() for c in df.columns]

    def pick_col(cands):
        for c in cands:
            if c in cols:
                return df.columns[cols.index(c)]
        return None

    col_fname = pick_col(["filename", "file", "image", "frame"])
    col_xmin = pick_col(["xmin", "x1", "x_min", "upper left corner x"])
    col_ymin = pick_col(["ymin", "y1", "y_min", "upper left corner y"])
    col_xmax = pick_col(["xmax", "x2", "x_max", "lower right corner x"])
    col_ymax = pick_col(["ymax", "y2", "y_max", "lower right corner y"])
    col_label = pick_col(["annotation", "label", "state", "type", "trafficlight", "annotation tag"])
    return col_fname, col_xmin, col_ymin, col_xmax, col_ymax, col_label


def save_rgb_patch_as_jpg(patch_rgb: np.ndarray, out_path: str):
    patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, patch_bgr)


def build_test_image_set():
    csv_files = find_box_csv_files(DATASET_ROOT)
    if not csv_files:
        raise RuntimeError("Không tìm thấy frameAnnotationsBOX.csv. Kiểm tra DATASET_ROOT.")

    img_set = set()

    for csv_path in csv_files:
        df = load_annotations(csv_path)
        col_fname, col_xmin, col_ymin, col_xmax, col_ymax, col_label = pick_cols(df)
        if not all([col_fname, col_xmin, col_ymin, col_xmax, col_ymax, col_label]):
            continue

        image_base_dir = infer_image_base_dir(csv_path)
        for _, row in df.iterrows():
            y = map_label_to_4(row[col_label])
            if y not in (1, 2, 3):
                continue
            img_path = resolve_image_path(image_base_dir, row[col_fname])
            if img_path is None or (not os.path.exists(img_path)):
                continue
            img_set.add(img_path)

    img_list = sorted(list(img_set))
    train_imgs, test_imgs = train_test_split(
        img_list,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    return set(test_imgs)


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ensure_dirs(OUT_DIR)
    test_imgs = build_test_image_set()

    csv_files = find_box_csv_files(DATASET_ROOT)
    counts = {"green": 0, "red": 0, "yellow": 0}

    bbox_by_img = {}
    picked_test_imgs = []

    for csv_path in csv_files:
        df = load_annotations(csv_path)
        col_fname, col_xmin, col_ymin, col_xmax, col_ymax, col_label = pick_cols(df)
        if not all([col_fname, col_xmin, col_ymin, col_xmax, col_ymax, col_label]):
            continue

        image_base_dir = infer_image_base_dir(csv_path)

        for _, row in df.iterrows():
            img_path = resolve_image_path(image_base_dir, row[col_fname])
            if img_path is None or (not os.path.exists(img_path)):
                continue
            if img_path not in test_imgs:
                continue

            try:
                img_rgb = read_rgb(img_path)
            except Exception:
                continue

            H, W, _ = img_rgb.shape
            x1, y1, x2, y2 = clamp_xyxy(row[col_xmin], row[col_ymin], row[col_xmax], row[col_ymax], W, H)

            if img_path not in bbox_by_img:
                bbox_by_img[img_path] = []
            bbox_by_img[img_path].append((x1, y1, x2, y2))

            y = map_label_to_4(row[col_label])
            if y not in (1, 2, 3):
                continue

            lab = CLASS_ID2NAME[y]
            if counts[lab] >= N_PER_CLASS:
                continue

            patch = crop_patch(img_rgb, x1, y1, x2, y2, out_size=(PATCH_SIZE, PATCH_SIZE))
            if patch is None:
                continue

            out_path = os.path.join(OUT_DIR, lab, f"{lab}_{counts[lab]:04d}.jpg")
            save_rgb_patch_as_jpg(patch, out_path)

            counts[lab] += 1
            picked_test_imgs.append(img_path)

            if all(counts[c] >= N_PER_CLASS for c in counts):
                break

        if all(counts[c] >= N_PER_CLASS for c in counts):
            break

    print("Saved TEST traffic-light patches:", counts)

    picked_test_imgs = list(dict.fromkeys(picked_test_imgs))
    other_saved = 0
    trials = 0
    while other_saved < N_OTHER and trials < N_OTHER * MAX_TRIALS_PER_OTHER:
        trials += 1
        img_path = random.choice(picked_test_imgs)
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

        ok = True
        for bb in bbox_by_img.get(img_path, []):
            if iou_xyxy((rx1, ry1, rx2, ry2), bb) > IOU_MAX:
                ok = False
                break
        if not ok:
            continue

        patch_bgr = bgr[ry1:ry2, rx1:rx2]
        out_path = os.path.join(OUT_DIR, "other", f"other_{other_saved:04d}.jpg")
        cv2.imwrite(out_path, patch_bgr)
        other_saved += 1

    print("Saved TEST OTHER patches:", other_saved)


if __name__ == "__main__":
    main()
