import json
import joblib
import numpy as np
import cv2
import gradio as gr
from skimage.feature import hog

# ====== EDIT THESE ======
MODEL_PATH = "outputs/svm_hog.joblib"
HOG_CONFIG_PATH = "outputs/hog_config.json"

PATCH_SIZE = 64
DEFAULT_REJECT_THRESHOLD = 0.55
# =======================


def load_hog_cfg():
    default = {
        "orientations": 9,
        "pixels_per_cell": [10, 10],
        "cells_per_block": [2, 2],
        "block_norm": "L2-Hys",
    }
    try:
        with open(HOG_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


hog_cfg = load_hog_cfg()
clf = joblib.load(MODEL_PATH)

ID2NAME = {0: "other", 1: "green", 2: "red", 3: "yellow"}


def get_class_names_from_model(model):
    """Return display names in the same order as model.classes_."""
    if not hasattr(model, "classes_"):
        return ["other", "green", "red", "yellow"], [0, 1, 2, 3]

    classes = list(model.classes_)
    names = []
    for c in classes:
        try:
            ci = int(c)
            names.append(ID2NAME.get(ci, str(ci)))
        except Exception:
            names.append(str(c))
    return names, classes


CLASS_NAMES, MODEL_CLASSES = get_class_names_from_model(clf)


def preprocess_patch(img_bgr, apply_clahe=False):
    """
    Quy tắc:
    - Nếu đúng 64x64 => giữ nguyên (direct)
    - Nếu không => resize nén (squash) về 64x64 (KHÔNG letterbox)
    """
    h, w = img_bgr.shape[:2]
    is_patch_64 = (h == PATCH_SIZE and w == PATCH_SIZE)

    if is_patch_64:
        img = img_bgr.copy()
    else:
        img = cv2.resize(img_bgr, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    return img, gray, is_patch_64


def extract_hog(gray):
    feat = hog(
        gray,
        orientations=int(hog_cfg["orientations"]),
        pixels_per_cell=tuple(hog_cfg["pixels_per_cell"]),
        cells_per_block=tuple(hog_cfg["cells_per_block"]),
        block_norm=hog_cfg.get("block_norm", "L2-Hys"),
        feature_vector=True,
    )
    return feat.astype(np.float32)


def _softmax(scores_1d: np.ndarray) -> np.ndarray:
    s = scores_1d - np.max(scores_1d)
    exps = np.exp(s)
    return exps / (np.sum(exps) + 1e-9)


def model_predict_probs(patch_bgr, apply_clahe=False):
    """Predict on a patch/image. DOES NOT apply reject threshold."""
    patch_bgr_64, gray, _ = preprocess_patch(patch_bgr, apply_clahe=apply_clahe)
    x = extract_hog(gray).reshape(1, -1)

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(x)[0]
    else:
        scores = clf.decision_function(x)
        scores = np.array(scores).reshape(-1)
        proba = _softmax(scores)

    n_cls = len(CLASS_NAMES)
    proba = proba[:n_cls]

    pred_idx = int(np.argmax(proba))
    conf = float(proba[pred_idx])
    pred_label = CLASS_NAMES[pred_idx]
    probs = {CLASS_NAMES[i]: float(proba[i]) for i in range(n_cls)}
    return probs, pred_label, conf, patch_bgr_64


def pick_non_other(probs: dict):
    """Chọn label tốt nhất trong {green, red, yellow} (loại other)."""
    probs2 = {k: float(v) for k, v in probs.items() if k != "other"}
    if len(probs2) == 0:
        # nếu model không có các lớp kia (cực hiếm), fallback
        return probs, max(probs, key=probs.get), float(max(probs.values()))
    best = max(probs2, key=probs2.get)
    return probs2, best, float(probs2[best])


def probs_force_other_full():
    return {"other": 1.0, "green": 0.0, "red": 0.0, "yellow": 0.0}


def predict(image_rgb, reject_threshold=DEFAULT_REJECT_THRESHOLD, apply_clahe=False):
    if image_rgb is None:
        return None, None, "No image", {}

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    is_patch_64 = (h == PATCH_SIZE and w == PATCH_SIZE)

    meta_lines = [f"[INPUT] {h}x{w}"]

    # Luôn tạo patch_show (đã về 64x64 theo rule)
    probs, pred_label, conf, patch_bgr_64 = model_predict_probs(bgr, apply_clahe=apply_clahe)
    patch_rgb_64 = cv2.cvtColor(patch_bgr_64, cv2.COLOR_BGR2RGB)

    # Case A: đúng 64x64 -> cho phép OTHER bằng reject threshold
    if is_patch_64:
        meta_lines.append("[RULE] 64x64 => direct + reject threshold (can be OTHER)")
        final_label = pred_label
        final_conf = conf

        if final_conf < float(reject_threshold):
            final_label = "other"

        title = (
            f"Prediction: {final_label.upper()} | conf={final_conf:.3f}\n"
            + "\n".join(meta_lines)
            + f"\nmodel_classes={CLASS_NAMES}"
        )
        return image_rgb, patch_rgb_64, title, probs

    # Case B: không phải 64x64 -> DISABLE OTHER (chỉ chọn green/red/yellow)
    meta_lines.append("[RULE] not 64x64 => resize to 64x64 + DISABLE OTHER")
    probs_no_other, final_label, final_conf = pick_non_other(probs)

    title = (
        f"Prediction: {final_label.upper()} | conf={final_conf:.3f}\n"
        + "\n".join(meta_lines)
        + f"\nmodel_classes={CLASS_NAMES}"
    )
    return image_rgb, patch_rgb_64, title, probs_no_other


with gr.Blocks() as demo:
    gr.Markdown(
        "# Traffic Light Demo (NO DETECTOR) – HOG+SVM\n"
        "- Nếu input **64×64**: đưa thẳng vào model (**reject threshold** có thể ra **OTHER**)\n"
        "- Nếu input **khác 64×64**: **resize nén** về 64×64 và **DISABLE OTHER** (chỉ chọn green/red/yellow)\n"
    )

    with gr.Row():
        inp = gr.Image(type="numpy", label="Upload image (patch 64×64 hoặc ảnh bất kỳ)")
        with gr.Column():
            thr = gr.Slider(
                0.0, 1.0, step=0.01,
                value=DEFAULT_REJECT_THRESHOLD,
                label="Reject threshold (chỉ áp dụng cho input 64×64)"
            )
            clahe = gr.Checkbox(value=False, label="Apply CLAHE")
            btn = gr.Button("Submit")

    with gr.Row():
        out_full = gr.Image(type="numpy", label="Full image (as-is)")
        out_patch = gr.Image(type="numpy", label="Patch dùng để phân loại (đã resize về 64×64)")

    out_txt = gr.Textbox(label="Result", lines=7)
    out_prob = gr.Label(label="Probabilities")

    btn.click(
        predict,
        inputs=[inp, thr, clahe],
        outputs=[out_full, out_patch, out_txt, out_prob],
    )


if __name__ == "__main__":
    demo.launch()
