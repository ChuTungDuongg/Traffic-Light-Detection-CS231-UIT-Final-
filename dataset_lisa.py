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

def build_patch_dataset_from_box_csvs(dataset_root: str, max_samples: int = 5000):
    csv_files = find_box_csv_files(dataset_root)
    if not csv_files:
        raise RuntimeError("No frameAnnotationsBOX.csv found. Check dataset_root.")

    X_patches = []
    y_labels = []

    for csv_path in csv_files:
        df = load_annotations(csv_path)
        cols = [c.lower() for c in df.columns]

        # cố gắng tìm tên cột phổ biến
        def pick_col(cands):
            for c in cands:
                if c in cols:
                    return df.columns[cols.index(c)]
            return None

        col_fname = pick_col(["filename", "file", "image", "frame"])

        # LISA BOX columns
        col_xmin  = pick_col(["xmin", "x1", "x_min", "upper left corner x"])
        col_ymin  = pick_col(["ymin", "y1", "y_min", "upper left corner y"])
        col_xmax  = pick_col(["xmax", "x2", "x_max", "lower right corner x"])
        col_ymax  = pick_col(["ymax", "y2", "y_max", "lower right corner y"])

        col_label = pick_col(["annotation", "label", "state", "type", "trafficlight", "annotation tag"])



        if not all([col_fname, col_xmin, col_ymin, col_xmax, col_ymax, col_label]):
            # nếu không match được cột, bỏ qua file đó
            continue

        image_base_dir = infer_image_base_dir(csv_path)

        for _, row in df.iterrows():
            img_path = resolve_image_path(image_base_dir, row[col_fname])
            if img_path is None:
                continue

            try:
                img = read_rgb(img_path)
            except Exception:
                continue

            patch = crop_patch(
                img,
                row[col_xmin], row[col_ymin],
                row[col_xmax], row[col_ymax],
                out_size=(64,64)
            )
            if patch is None:
                continue

            y = map_label_to_4(row[col_label])

            X_patches.append(patch)
            y_labels.append(y)

            if len(X_patches) >= max_samples:
                return np.array(X_patches), np.array(y_labels)

    return np.array(X_patches), np.array(y_labels)
