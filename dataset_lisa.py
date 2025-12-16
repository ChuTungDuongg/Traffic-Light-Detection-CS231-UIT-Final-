import os
import glob
import pandas as pd
import cv2
import numpy as np

# Map label string -> {0,1,2,3} = {other, green, red, yellow}
# LISA label có thể khác nhau (go/stop/warning/none/arrow...), ta gom về 4 lớp.
def map_label_to_4(label: str) -> int:
    s = str(label).lower()
    if "green" in s or "go" in s:
        return 1
    if "red" in s or "stop" in s:
        return 2
    if "yellow" in s or "warning" in s:
        return 3
    return 0

def read_rgb(img_path: str) -> np.ndarray:
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise ValueError(f"Cannot read image: {img_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def crop_patch(image_rgb: np.ndarray, x1, y1, x2, y2, out_size=(64,64)) -> np.ndarray:
    H, W, _ = image_rgb.shape
    x1 = max(0, min(int(x1), W-1))
    x2 = max(0, min(int(x2), W))
    y1 = max(0, min(int(y1), H-1))
    y2 = max(0, min(int(y2), H))
    patch = image_rgb[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    patch = cv2.resize(patch, out_size, interpolation=cv2.INTER_AREA)
    return patch

def find_box_csv_files(dataset_root: str):
    pattern = os.path.join(dataset_root, "Annotations", "Annotations", "**", "frameAnnotationsBOX.csv")
    return sorted(glob.glob(pattern, recursive=True))

def load_annotations(csv_path: str):
    # LISA thường dùng ';' nhưng có file dùng ','
    for sep in [";", ","]:
        try:
            df = pd.read_csv(csv_path, sep=sep)
            if df.shape[1] >= 5:
                return df
        except Exception:
            pass
    raise RuntimeError(f"Cannot parse CSV: {csv_path}")

def infer_image_base_dir(csv_path: str) -> str:
    """
    csv_path: .../Annotations/Annotations/dayTrain/dayClip1/frameAnnotationsBOX.csv
    images:   .../dayTrain/dayClip1/frames/...
    thực tế LISA thường có folder ảnh nằm ngoài Annotations, cùng level dataset_root.
    Ta suy ra: dataset_root = phần trước "/Annotations/Annotations/..."
    """
    marker = os.path.join("Annotations", "Annotations")
    idx = csv_path.find(marker)
    if idx == -1:
        # fallback: assume dataset_root is 2 levels up
        return os.path.dirname(os.path.dirname(csv_path))
    dataset_root = csv_path[:idx]
    # phần còn lại sau marker: dayTrain/dayClip1/...
    tail = csv_path[idx+len(marker)+1:]
    # tail folder: dayTrain/dayClip1/frameAnnotationsBOX.csv
    rel_dir = os.path.dirname(tail)  # dayTrain/dayClip1
    # ảnh thường ở: dataset_root/dayTrain/dayClip1/frames/*.jpg
    return os.path.join(dataset_root, rel_dir)

def resolve_image_path(image_base_dir: str, filename: str) -> str:
    # filename trong CSV thường dạng: "dayTrain/dayClip1/frames/..." hoặc chỉ "frame_..."
    f = str(filename)
    if os.path.isabs(f) and os.path.exists(f):
        return f

    # thử join trực tiếp
    p1 = os.path.join(image_base_dir, f)
    if os.path.exists(p1):
        return p1

    # thử nếu filename chỉ là basename -> tìm trong frames
    p2 = os.path.join(image_base_dir, "frames", os.path.basename(f))
    if os.path.exists(p2):
        return p2

    # fallback: glob search
    candidates = glob.glob(os.path.join(image_base_dir, "**", os.path.basename(f)), recursive=True)
    if candidates:
        return candidates[0]

    return None

def build_patch_dataset_from_box_csvs(dataset_root: str,
                                      max_samples: int = 5000,
                                      add_other: bool = True,
                                      n_other_per_image: int = 1,
                                      other_max_iou: float = 0.02,
                                      seed: int = 42):
    """
    add_other=True: thêm negative patches (label=0) bằng random crops nền.
    n_other_per_image: mỗi ảnh sinh bao nhiêu patch other.
    other_max_iou: ngưỡng IOU tối đa giữa patch other và bbox đèn.
    """
    np.random.seed(seed)

    csv_files = find_box_csv_files(dataset_root)
    if not csv_files:
        raise RuntimeError("No frameAnnotationsBOX.csv found. Check dataset_root.")

    X_patches = []
    y_labels = []

    for csv_path in csv_files:
        df = load_annotations(csv_path)
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

        if not all([col_fname, col_xmin, col_ymin, col_xmax, col_ymax, col_label]):
            continue

        image_base_dir = infer_image_base_dir(csv_path)

        # ---------
        # 1) Gom rows theo từng ảnh
        # ---------
        grouped = {}
        for _, row in df.iterrows():
            img_path = resolve_image_path(image_base_dir, row[col_fname])
            if img_path is None:
                continue
            grouped.setdefault(img_path, []).append(row)

        # ---------
        # 2) Với mỗi ảnh: lấy positives + negatives
        # ---------
        for img_path, rows in grouped.items():
            try:
                img = read_rgb(img_path)
            except Exception:
                continue

            H, W, _ = img.shape

            bboxes = []
            # 2a) positives
            for row in rows:
                patch = crop_patch(
                    img,
                    row[col_xmin], row[col_ymin],
                    row[col_xmax], row[col_ymax],
                    out_size=(64,64)
                )
                if patch is None:
                    continue

                y = map_label_to_4(row[col_label])
                # giữ y=0 nếu annotation thật sự là other (hiếm), còn chủ yếu y=1/2/3
                X_patches.append(patch)
                y_labels.append(y)

                # lưu bbox để tránh khi sample other
                # clamp bbox về toạ độ ảnh
                x1 = max(0, min(int(row[col_xmin]), W-1))
                y1 = max(0, min(int(row[col_ymin]), H-1))
                x2 = max(1, min(int(row[col_xmax]), W))
                y2 = max(1, min(int(row[col_ymax]), H))
                bboxes.append((x1, y1, x2, y2))

                if len(X_patches) >= max_samples:
                    return np.array(X_patches), np.array(y_labels)

            # 2b) negatives (other)
            if add_other and bboxes and len(X_patches) < max_samples:
                others = sample_other_patches(
                    img_rgb=img,
                    bboxes_xyxy=bboxes,
                    n_samples=n_other_per_image,
                    patch_size=64,
                    max_iou=other_max_iou,
                    max_tries=300
                )
                for op in others:
                    X_patches.append(op)
                    y_labels.append(0)
                    if len(X_patches) >= max_samples:
                        return np.array(X_patches), np.array(y_labels)

    return np.array(X_patches), np.array(y_labels)


def iou_xyxy(a, b):
    # a,b: (x1,y1,x2,y2)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def sample_other_patches(img_rgb, bboxes_xyxy, n_samples=1, patch_size=64,
                         max_iou=0.02, max_tries=200):
    """
    Cắt patch nền (other) sao cho không overlap bbox đèn.
    bboxes_xyxy: list[(x1,y1,x2,y2)] đã clamp trong ảnh
    """
    H, W, _ = img_rgb.shape
    out = []
    if W < patch_size or H < patch_size:
        return out

    tries = 0
    while len(out) < n_samples and tries < max_tries:
        tries += 1
        x1 = np.random.randint(0, W - patch_size + 1)
        y1 = np.random.randint(0, H - patch_size + 1)
        x2 = x1 + patch_size
        y2 = y1 + patch_size
        cand = (x1, y1, x2, y2)

        ok = True
        for bb in bboxes_xyxy:
            if iou_xyxy(cand, bb) > max_iou:
                ok = False
                break
        if not ok:
            continue

        patch = img_rgb[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        out.append(patch.copy())

    return out
