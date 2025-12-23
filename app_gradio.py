import json
import joblib
import numpy as np
import cv2
import gradio as gr
from skimage.feature import hog

# Import detector (separate file). If TF is missing, app still runs (fallback).
try:
    from detector_tf1 import TLDetectorTF1
except Exception as e:
    print(f"[app_gradio] WARNING: cannot import detector_tf1: {e}")
    TLDetectorTF1 = None

# ====== EDIT THESE ======
MODEL_PATH = "outputs/svm_hog.joblib"
HOG_CONFIG_PATH = "outputs/hog_config.json"

PATCH_SIZE = 64
DEFAULT_REJECT_THRESHOLD = 0.55
DEFAULT_DETECT_SCORE_THRESHOLD = 0.10
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
        # fallback
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


def _letterbox_resize(img_bgr, dst_size):
    """Resize giữ tỉ lệ, padding letterbox để ra dst_size×dst_size."""
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
    # Nếu đã đúng 64x64 -> giữ nguyên đúng nghĩa "truyền thẳng".
    if img_bgr.shape[0] == PATCH_SIZE and img_bgr.shape[1] == PATCH_SIZE:
        img = img_bgr.copy()
    else:
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
        feature_vector=True,
    )
    return feat.astype(np.float32)


def _softmax(scores_1d: np.ndarray) -> np.ndarray:
    s = scores_1d - np.max(scores_1d)
    exps = np.exp(s)
    return exps / (np.sum(exps) + 1e-9)


def model_predict_probs(patch_bgr, apply_clahe=False):
    """Predict on a patch/image. This function DOES NOT apply reject threshold."""
    _, gray = preprocess_patch(patch_bgr, apply_clahe=apply_clahe)
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
    return probs, pred_label, conf


def pick_non_other(probs: dict):
    """Chọn label tốt nhất trong {green, red, yellow} (loại other)."""
    probs2 = {k: float(v) for k, v in probs.items() if k != "other"}
    if len(probs2) == 0:
        # fallback cực đoan
        return {"other": 1.0, "green": 0.0, "red": 0.0, "yellow": 0.0}, "other", 1.0
    best = max(probs2, key=probs2.get)
    return probs2, best, float(probs2[best])


def probs_force_other_full():
    """Luôn trả đủ 4 class để UI hiện đủ thanh (other=1, còn lại=0)."""
    return {"other": 1.0, "green": 0.0, "red": 0.0, "yellow": 0.0}


# Create one detector instance (avoid re-loading per request)
try:
    _tl_detector = TLDetectorTF1() if TLDetectorTF1 is not None else None
except Exception as e:
    print(f"[app_gradio] WARNING: detector init failed: {e}")
    _tl_detector = None


def _draw_box_norm(rgb, box_norm, color=(0, 255, 255), thickness=2):
    """Draw normalized box (ymin,xmin,ymax,xmax) on an RGB image."""
    if box_norm is None:
        return rgb
    h, w = rgb.shape[:2]
    y1, x1, y2, x2 = box_norm
    x1i, y1i = int(x1 * w), int(y1 * h)
    x2i, y2i = int(x2 * w), int(y2 * h)
    out = rgb.copy()
    cv2.rectangle(out, (x1i, y1i), (x2i, y2i), color, thickness)
    return out


def predict(image_rgb, reject_threshold=DEFAULT_REJECT_THRESHOLD, det_score=DEFAULT_DETECT_SCORE_THRESHOLD, apply_clahe=False):
    if image_rgb is None:
        return None, None, "No image", {}

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    is_patch_64 = (h == PATCH_SIZE and w == PATCH_SIZE)

    meta_lines = []

    # -----------------------------
    # Case 1: input is exactly 64x64 patch => feed directly
    # -----------------------------
    if is_patch_64:
        meta_lines.append("[INPUT] 64x64 patch => direct")

        probs, pred_label, conf = model_predict_probs(bgr, apply_clahe=apply_clahe)

        final_label = pred_label
        if conf < float(reject_threshold):
            final_label = "other"

        patch_show_bgr, _ = preprocess_patch(bgr, apply_clahe=apply_clahe)
        patch_show_rgb = cv2.cvtColor(patch_show_bgr, cv2.COLOR_BGR2RGB)

        title = (
            f"Prediction: {final_label.upper()} | conf={conf:.3f}\n"
            + "\n".join(meta_lines)
            + f"\nmodel_classes={CLASS_NAMES}"
        )
        return image_rgb, patch_show_rgb, title, probs

    # -----------------------------
    # Case 2: not 64x64 => MUST use detector
    # Rule:
    #   - If no TL box => FORCE OTHER + probs (other=1, rest=0)
    #   - If TL box found => DISABLE OTHER (predict but exclude 'other')
    # -----------------------------
    if _tl_detector is None:
        meta_lines.append("[DETECTOR] unavailable => FORCE OTHER")
        meta_lines.append("[RULE] not 64x64 AND no TL box => OTHER")

        patch_show_bgr, _ = preprocess_patch(bgr, apply_clahe=False)
        patch_show_rgb = cv2.cvtColor(patch_show_bgr, cv2.COLOR_BGR2RGB)

        title = (
            "Prediction: OTHER | conf=1.000\n"
            + "\n".join(meta_lines)
            + f"\nmodel_classes={CLASS_NAMES}"
        )
        return image_rgb, patch_show_rgb, title, probs_force_other_full()

    crop_bgr, meta = _tl_detector.crop_best(
        bgr,
        score_threshold=float(det_score),
        choose="best_score",
    )

    if (not meta.ok) or (crop_bgr is None) or (crop_bgr.size == 0):
        meta_lines.append(f"[DETECTOR] no box (reason={meta.reason}) => FORCE OTHER")
        meta_lines.append("[RULE] not 64x64 AND no TL box => OTHER")

        patch_show_bgr, _ = preprocess_patch(bgr, apply_clahe=False)
        patch_show_rgb = cv2.cvtColor(patch_show_bgr, cv2.COLOR_BGR2RGB)

        title = (
            "Prediction: OTHER | conf=1.000\n"
            + "\n".join(meta_lines)
            + f"\nmodel_classes={CLASS_NAMES}"
        )
        return image_rgb, patch_show_rgb, title, probs_force_other_full()

    # detector ok => crop ROI and DISABLE OTHER
    vis_full_rgb = _draw_box_norm(image_rgb, meta.box_norm)
    meta_lines.append(f"[DETECTOR] ok score={meta.score:.3f} => crop ROI")
    meta_lines.append("[RULE] not 64x64 AND TL box found => DISABLE OTHER")

    probs, pred_label, conf = model_predict_probs(crop_bgr, apply_clahe=apply_clahe)

    # disable other: pick among green/red/yellow
    probs_no_other, final_label, final_conf = pick_non_other(probs)

    patch_show_bgr, _ = preprocess_patch(crop_bgr, apply_clahe=apply_clahe)
    patch_show_rgb = cv2.cvtColor(patch_show_bgr, cv2.COLOR_BGR2RGB)

    title = (
        f"Prediction: {final_label.upper()} | conf={final_conf:.3f}\n"
        + "\n".join(meta_lines)
        + f"\nmodel_classes={CLASS_NAMES}"
    )

    return vis_full_rgb, patch_show_rgb, title, probs_no_other


with gr.Blocks() as demo:
    gr.Markdown(
        "# Traffic Light Demo (TF1 detector -> crop ROI -> HOG+SVM)\n"
        "- Nếu input **64×64**: đưa thẳng vào model (reject threshold có thể ra **OTHER**)\n"
        "- Nếu input **khác 64×64**:\n"
        "  - **Không detect ra đèn** => ép **OTHER** + probabilities: other=100%, còn lại=0%\n"
        "  - **Detect ra đèn** => **disable OTHER** (chỉ chọn green/red/yellow)\n"
    )

    with gr.Row():
        inp = gr.Image(type="numpy", label="Upload image (patch 64×64 hoặc ảnh gốc)")
        with gr.Column():
            thr = gr.Slider(
                0.0, 1.0, step=0.01,
                value=DEFAULT_REJECT_THRESHOLD,
                label="Reject threshold (chỉ áp dụng cho patch 64×64)"
            )
            det_thr = gr.Slider(
                0.0, 0.9, step=0.01,
                value=DEFAULT_DETECT_SCORE_THRESHOLD,
                label="Detector score threshold"
            )
            clahe = gr.Checkbox(value=False, label="Apply CLAHE")
            btn = gr.Button("Submit")

    with gr.Row():
        out_full = gr.Image(type="numpy", label="Full image (bbox nếu detect được)")
        out_patch = gr.Image(type="numpy", label="Patch dùng để phân loại (đã chuẩn hoá về 64×64)")

    out_txt = gr.Textbox(label="Result", lines=7)
    out_prob = gr.Label(label="Probabilities")

    btn.click(
        predict,
        inputs=[inp, thr, det_thr, clahe],
        outputs=[out_full, out_patch, out_txt, out_prob],
    )


if __name__ == "__main__":
    demo.launch()
