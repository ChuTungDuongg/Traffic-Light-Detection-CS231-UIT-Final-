import json
import joblib
import numpy as np
import cv2
import gradio as gr
from skimage.feature import hog

# ====== EDIT THESE ======
MODEL_PATH = "outputs/svm_hog.joblib"
HOG_CONFIG_PATH = "outputs/hog_config.json"

# Kích thước patch sau khi chuẩn hóa (letterbox + resize về vuông)
PATCH_SIZE = 64

# Ngưỡng reject mặc định: nếu model không tự tin -> trả OTHER
DEFAULT_REJECT_THRESHOLD = 0.55
# =======================


def load_hog_cfg():
    default = {
        "orientations": 9,
        "pixels_per_cell": [10, 10],
        "cells_per_block": [2, 2],
        "block_norm": "L2-Hys"
    }
    try:
        with open(HOG_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


hog_cfg = load_hog_cfg()
clf = joblib.load(MODEL_PATH)

# Map id -> tên chuẩn của bạn
ID2NAME = {0: "other", 1: "green", 2: "red", 3: "yellow"}


def get_class_names_from_model(model):
    """
    Lấy danh sách class đúng theo model đã train.
    - Nếu model classes_ là số (0/1/2/3) -> map sang other/green/red/yellow
    - Nếu là string -> giữ nguyên string đó (ví dụ go/stop/warning)
    """
    if not hasattr(model, "classes_"):
        # fallback: giả định 3 lớp green/red/yellow
        return ["green", "red", "yellow"], [1, 2, 3]

    classes = list(model.classes_)  # có thể là np array
    names = []
    for c in classes:
        try:
            ci = int(c)
            names.append(ID2NAME.get(ci, str(ci)))
        except Exception:
            # class là string
            names.append(str(c))
    return names, classes


CLASS_NAMES, MODEL_CLASSES = get_class_names_from_model(clf)


def _letterbox_resize(img_bgr, dst_size):
    """Resize giữ tỉ lệ, padding letterbox để ra ảnh vuông dst_size×dst_size."""
    h, w = img_bgr.shape[:2]
    scale = dst_size / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_top = (dst_size - new_h) // 2
    pad_bottom = dst_size - new_h - pad_top
    pad_left = (dst_size - new_w) // 2
    pad_right = dst_size - new_w - pad_left

    return cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )


def preprocess_patch(img_bgr, apply_clahe=False):
    # Resize giữ tỉ lệ và padding để không méo hình
    img = _letterbox_resize(img_bgr, PATCH_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    return img, gray


def extract_hog(gray):
    feat = hog(
        gray,
        orientations=int(hog_cfg["orientations"]),
        pixels_per_cell=tuple(hog_cfg["pixels_per_cell"]),
        cells_per_block=tuple(hog_cfg["cells_per_block"]),
        block_norm=hog_cfg.get("block_norm", "L2-Hys"),
        feature_vector=True
    )
    return feat.astype(np.float32)


def _softmax(scores_1d: np.ndarray) -> np.ndarray:
    s = scores_1d - np.max(scores_1d)
    exps = np.exp(s)
    return exps / (np.sum(exps) + 1e-9)


def predict(img, reject_threshold=DEFAULT_REJECT_THRESHOLD, apply_clahe=False):
    if img is None:
        return "No image", {}

    # gradio trả RGB -> BGR
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, gray = preprocess_patch(img_bgr, apply_clahe=apply_clahe)
    x = extract_hog(gray).reshape(1, -1)

    # 1) Lấy proba nếu có
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(x)[0]  # shape = (n_classes,)
    else:
        # 2) Fallback: decision_function -> chuyển về pseudo-proba bằng softmax
        scores = clf.decision_function(x)
        # scores có thể là (n_classes,) hoặc (1,n_classes)
        scores = np.array(scores).reshape(-1)
        proba = _softmax(scores)

    # Số lớp theo model
    n_cls = len(CLASS_NAMES)
    proba = proba[:n_cls]  # phòng trường hợp lạ

    pred_idx = int(np.argmax(proba))
    conf = float(proba[pred_idx])
    pred_label = CLASS_NAMES[pred_idx]

    # Reject: nếu không đủ tự tin thì trả other (dù model không train other)
    if conf < reject_threshold:
        pred_label = "other"

    # Build dict probabilities đúng số lớp
    probs = {CLASS_NAMES[i]: float(proba[i]) for i in range(len(CLASS_NAMES))}

    title = f"Prediction: {pred_label.upper()} | confidence={conf:.3f}"
    # Thêm info classes để bạn debug nhanh
    title += f" | model_classes={CLASS_NAMES}"
    return title, probs


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(
            type="numpy",
            label="Upload a patch (any aspect ratio; sẽ letterbox về 64×64)",
        ),
        gr.Slider(
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=DEFAULT_REJECT_THRESHOLD,
            label="Reject threshold (confidence < threshold → other)",
        ),
        gr.Checkbox(
            value=False,
            label="Apply CLAHE (normalize brightness for low-light/glare)",
        ),
    ],
    outputs=[
        gr.Textbox(label="Result"),
        gr.Label(label="Probabilities")
    ],
    title="Traffic Light Patch Classifier (HOG + Classifier)",
    description=(
        "Demo classification trên patch. Ảnh sẽ được letterbox để giữ tỉ lệ, "
        "có tùy chọn chuẩn hóa sáng (CLAHE) và chỉnh ngưỡng reject để thử nghiệm"
    ),
    examples=[
        ["demo_assets/patches/red/red_0000.jpg", DEFAULT_REJECT_THRESHOLD, False],
        ["demo_assets/patches/green/green_0000.jpg", DEFAULT_REJECT_THRESHOLD, False],
        ["demo_assets/patches/yellow/yellow_0000.jpg", DEFAULT_REJECT_THRESHOLD, False],
        ["demo_assets/patches/other/other_0000.jpg", DEFAULT_REJECT_THRESHOLD, False],
        ["demo/Modern_British_LED_Traffic_Light.jpg", DEFAULT_REJECT_THRESHOLD, True],
        ["demo/z7324240384988_c909efa49a8f0b3eefdd79dec70a68b6.jpg", 0.35, True],
    ]
)

if __name__ == "__main__":
    demo.launch()
