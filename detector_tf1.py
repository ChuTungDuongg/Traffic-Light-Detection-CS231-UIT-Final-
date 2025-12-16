"""detector_tf1.py
TF1 Object Detection API detector wrapper (traffic light ROI crop).

- Loads a frozen_inference_graph.pb (COCO pre-trained) and returns traffic-light boxes.
- Default COCO traffic light class id = 10.

Usage:
    from detector_tf1 import TLDetectorTF1
    det = TLDetectorTF1()
    crop_bgr, meta = det.crop_best(bgr, score_threshold=0.1)

Notes:
- Uses tensorflow.compat.v1 so it works under TF2 installs.
- If the checkpoint .pb is missing, it can auto-download a default model.
  If you are offline, download the model yourself and set PATH_TO_CKPT accordingly.
"""

from __future__ import annotations

import os
import tarfile
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np

try:
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
except Exception as e:
    raise ImportError(
        "TensorFlow is required for detector_tf1.py. Install tensorflow (or tensorflow-cpu). "
        f"Original error: {e}"
    )


# -----------------------------
# Config (edit if needed)
# -----------------------------
# Default TF1 detection model (COCO). You can change this to your local model.
MODEL_NAME = os.environ.get("TL_TF1_MODEL_NAME", "ssd_mobilenet_v1_coco_2017_11_17")
DOWNLOAD_BASE = os.environ.get("TL_TF1_DOWNLOAD_BASE", "http://download.tensorflow.org/models/object_detection")
MODEL_DIR = os.environ.get("TL_TF1_MODEL_DIR", os.path.join("outputs", "tf1_detector"))

# Path to frozen graph
PATH_TO_CKPT = os.environ.get(
    "TL_TF1_PATH_TO_CKPT",
    os.path.join(MODEL_DIR, MODEL_NAME, "frozen_inference_graph.pb"),
)

# COCO traffic light class id
COCO_TL_CLASS_ID = int(os.environ.get("TL_TF1_TARGET_CLASS", "10"))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_model(model_name: str = MODEL_NAME, model_dir: str = MODEL_DIR) -> str:
    """Download and extract a TF1 OD API model if checkpoint is missing.

    Returns the expected PATH_TO_CKPT.
    """
    _ensure_dir(model_dir)
    tar_name = model_name + ".tar.gz"
    url = f"{DOWNLOAD_BASE}/{tar_name}"
    tar_path = os.path.join(model_dir, tar_name)

    if not os.path.exists(tar_path):
        print(f"[detector_tf1] Downloading: {url}")
        urllib.request.urlretrieve(url, tar_path)

    print(f"[detector_tf1] Extracting: {tar_path}")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=model_dir)

    ckpt = os.path.join(model_dir, model_name, "frozen_inference_graph.pb")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            "Downloaded model but frozen_inference_graph.pb not found at: " + ckpt
        )
    return ckpt


def load_graph(path_to_ckpt: str = PATH_TO_CKPT) -> tf.Graph:
    if not os.path.exists(path_to_ckpt):
        # Try auto download (if internet). If fails, raise with clear msg.
        try:
            path_to_ckpt = download_model()
        except Exception as e:
            raise FileNotFoundError(
                f"PATH_TO_CKPT not found: {path_to_ckpt}. "
                "Either place frozen_inference_graph.pb there or set environment variable TL_TF1_PATH_TO_CKPT. "
                f"Auto-download failed: {e}"
            )

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_ckpt, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")

    return detection_graph


def select_boxes(
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
    score_threshold: float = 0.0,
    target_class: int = COCO_TL_CLASS_ID,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter boxes by target class and score threshold.

    Returns:
        (sel_boxes, sel_scores)
    """
    sq_scores = np.squeeze(scores)
    sq_classes = np.squeeze(classes)
    sq_boxes = np.squeeze(boxes)

    sel_id = np.logical_and(sq_classes == target_class, sq_scores > score_threshold)
    return sq_boxes[sel_id], sq_scores[sel_id]


def crop_roi_image(image_np: np.ndarray, sel_box: np.ndarray) -> np.ndarray:
    """Crop normalized box [ymin, xmin, ymax, xmax] from image."""
    im_height, im_width, _ = image_np.shape
    left = sel_box[1] * im_width
    right = sel_box[3] * im_width
    top = sel_box[0] * im_height
    bottom = sel_box[2] * im_height

    left_i = max(0, min(im_width - 1, int(left)))
    right_i = max(0, min(im_width, int(right)))
    top_i = max(0, min(im_height - 1, int(top)))
    bottom_i = max(0, min(im_height, int(bottom)))

    if right_i <= left_i or bottom_i <= top_i:
        return image_np[0:0, 0:0]

    return image_np[top_i:bottom_i, left_i:right_i, :]


@dataclass
class CropMeta:
    ok: bool
    reason: str = ""
    score: float = 0.0
    class_id: int = COCO_TL_CLASS_ID
    box_norm: Optional[Tuple[float, float, float, float]] = None  # (ymin,xmin,ymax,xmax)


class TLDetectorTF1:
    """TF1 detector wrapper for COCO traffic light."""

    def __init__(self, path_to_ckpt: str = PATH_TO_CKPT):
        self.path_to_ckpt = path_to_ckpt
        self.detection_graph = load_graph(self.path_to_ckpt)
        self._extract_graph_components()
        self.sess = tf.Session(graph=self.detection_graph)

        # warm-up
        dummy = np.zeros((128, 128, 3), dtype=np.uint8)
        _ = self.detect_multi_object(dummy, score_threshold=0.1)

    def _extract_graph_components(self):
        g = self.detection_graph
        self.image_tensor = g.get_tensor_by_name("image_tensor:0")
        self.detection_boxes = g.get_tensor_by_name("detection_boxes:0")
        self.detection_scores = g.get_tensor_by_name("detection_scores:0")
        self.detection_classes = g.get_tensor_by_name("detection_classes:0")
        self.num_detections = g.get_tensor_by_name("num_detections:0")

    def detect_multi_object(self, image_np: np.ndarray, score_threshold: float = 0.1):
        """Return selected traffic-light boxes (normalized) and scores."""
        image_np_expanded = np.expand_dims(image_np, axis=0)

        boxes, scores, classes, num = self.sess.run(
            [
                self.detection_boxes,
                self.detection_scores,
                self.detection_classes,
                self.num_detections,
            ],
            feed_dict={self.image_tensor: image_np_expanded},
        )

        sel_boxes, sel_scores = select_boxes(
            boxes=boxes,
            classes=classes,
            scores=scores,
            score_threshold=score_threshold,
            target_class=COCO_TL_CLASS_ID,
        )
        return sel_boxes, sel_scores

    def crop_best(
        self,
        bgr: np.ndarray,
        score_threshold: float = 0.1,
        choose: str = "best_score",
    ) -> Tuple[Optional[np.ndarray], CropMeta]:
        """Crop the best detected traffic light ROI from an image.

        Args:
            bgr: input image in BGR.
            score_threshold: detector threshold.
            choose: 'best_score' or 'largest_area'.

        Returns:
            (crop_bgr or None, meta)
        """
        if bgr is None or bgr.size == 0:
            return None, CropMeta(ok=False, reason="empty_image")

        sel_boxes, sel_scores = self.detect_multi_object(bgr, score_threshold=score_threshold)
        if sel_boxes is None or len(sel_boxes) == 0:
            return None, CropMeta(ok=False, reason="no_box")

        idx = 0
        if choose == "largest_area":
            areas = (sel_boxes[:, 2] - sel_boxes[:, 0]) * (sel_boxes[:, 3] - sel_boxes[:, 1])
            idx = int(np.argmax(areas))
        else:
            idx = int(np.argmax(sel_scores))

        box = sel_boxes[idx]
        score = float(sel_scores[idx])
        crop = crop_roi_image(bgr, box)
        if crop is None or crop.size == 0:
            return None, CropMeta(ok=False, reason="bad_crop", score=score, box_norm=tuple(map(float, box)))

        return crop, CropMeta(ok=True, score=score, box_norm=tuple(map(float, box)))

    def close(self):
        try:
            self.sess.close()
        except Exception:
            pass
