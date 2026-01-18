# 🚦 Traffic Light Detection (CS231 – UIT)

Final project for **CS231 – Introduction to Computer Vision (University of Information Technology, VNU-HCM)**. The repository implements a **classical computer vision pipeline** for traffic light detection and color classification using HSV heuristics for localization and HOG/VGG16 features for classification.

## 🗺️ What’s inside
- **Complete pipelines in notebooks**: full training/evaluation (`lisa_full_pipeline_clean.ipynb`) and hyperparameter search notes (`hypertuning_params.ipynb`).
- **Reusable dataset utilities**: `dataset_lisa.py` for loading LISA annotations and building patch datasets.
- **Feature extraction**: `features_hog.py` contains HOG helpers; VGG16 (frozen) extraction lives in the notebook.
- **Interactive demo**: `app_gradio_B.py` serves a Gradio UI powered by the trained SVM + HOG model stored in `outputs/`.
- **Demo assets**: `demo/` (sample images) and `demo_assets/` (patches/backgrounds for presentations).
- **Helpers**: `scripts/make_demo_patches.py` to export balanced patches from the raw dataset.

## 📂 Repository layout
```
Traffic-Light-Detection-CS231-UIT-Final-
├── lisa_full_pipeline_clean.ipynb   # Main notebook: dataset -> features -> training -> evaluation
├── hypertuning_params.ipynb         # Hyperparameter notes and experiments
├── dataset_lisa.py                  # Parse LISA annotations, crop patches, map labels
├── features_hog.py                  # HOG feature helpers used by notebooks and demo
├── detector_tf1.py                  # Optional TF1 detector wrapper (best-effort import in the app)
├── app_gradio_B.py                  # Gradio demo consuming trained model/config from outputs/
├── scripts/
│   └── make_demo_patches.py         # Export small, balanced patch set for demos
├── outputs/                         # Place for trained models/configs (svm_hog.joblib, ablation_light_best_config.json, ...)
├── demo/                            # Sample images for quick testing
└── demo_assets/                     # Additional demo-ready assets (patches, backgrounds)
```

## 🛠️ Environment
- Python ≥ 3.9 recommended.
- Core dependencies: `numpy`, `pandas`, `opencv-python`, `matplotlib`, `scikit-learn`, `tensorflow` (for VGG16 feature extraction), `scikit-image`, `joblib`, `gradio` (for the UI).

Quick start in a fresh virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install numpy pandas opencv-python matplotlib scikit-learn tensorflow scikit-image joblib gradio
```

> Tip: If you only need the HOG + SVM demo, you can omit `tensorflow` to speed up installation.

Activate your environment in future shells with `source .venv/bin/activate` (or the Windows equivalent).

### Quick commands
- **Launch the demo UI:** `python app_gradio_B.py`.
- **Open the main notebook:** `jupyter notebook lisa_full_pipeline_clean.ipynb`.
- **Export demo patches:** `python scripts/make_demo_patches.py` (after setting `DATASET_ROOT` and `OUT_DIR`).

### Quickstart workflow
1. **Use the included artifacts:** pre-trained HOG + SVM models and configs live in `outputs/` (e.g., `svm_hog.joblib`, `svm_best_hog.joblib`, `ablation_light_best_model.joblib`, and matching JSON configs). You can immediately load these files in the Gradio app without retraining.
2. **Train your own model (optional):** follow the steps in **Training & evaluation** to regenerate artifacts tailored to your dataset slice or hyperparameters.
3. **Launch the demo:** run `python app_gradio_B.py` and upload an image/patch to view predictions. The app auto-loads `outputs/svm_best_hog.joblib` and tries `outputs/hog_config.json` by default (falling back to a built-in HOG config if the JSON is missing); adjust the constants at the top of the script to point to alternative artifacts such as `outputs/ablation_light_best_config.json`.
4. **Swap detectors:** if you have TensorFlow 1.x installed and a compatible frozen detector in `outputs/tf1_detector/`, the optional detector wrapper (`detector_tf1.py`) will be used automatically; otherwise it is safely skipped.

## 📊 Dataset: LISA Traffic Light
Download the LISA Traffic Light Dataset from Kaggle:
- https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset

After extracting, note the dataset root (folder containing `Annotations/` and video folders). The utilities assume the original folder structure.

### Expected outputs
When you run the training notebook or custom pipeline, the following artifacts are written to `outputs/`:
- `svm_hog.joblib` / `svm_best_hog.joblib`: trained SVM models for HOG features.
- `ablation_light_best_config.json`: a saved HOG preprocessing configuration (patch size, cell size, etc.).
- `ablation_light_best_model.joblib`: best-performing model from the ablation notebook.

### Build a patch dataset
Use `dataset_lisa.py` to crop 64×64 patches from bounding boxes (and optional background patches):
```python
from dataset_lisa import build_patch_dataset_from_box_csvs

X, y = build_patch_dataset_from_box_csvs(
    dataset_root="/path/to/lisa-traffic-light-dataset",  # folder that contains Annotations/
    max_samples=5000,
    add_other=True,            # add background negatives
    n_other_per_image=1,       # negatives per annotated frame
    other_max_iou=0.02,        # avoid overlap with ground-truth boxes
)
```
The function returns NumPy arrays `X` (patches) and `y` (labels mapped to {0: other, 1: green, 2: red, 3: yellow}).

You can persist the patches for reuse with `numpy.savez_compressed`:
```python
import numpy as np
np.savez_compressed("lisa_patches.npz", X=X, y=y)
```

## 🧠 Training & evaluation
1. Launch Jupyter and open the main notebook:
   ```bash
   jupyter notebook lisa_full_pipeline_clean.ipynb
   ```
2. Update the dataset path variables in the first cells.
3. Run all cells to generate train/validation splits, extract features (HOG or VGG16), train SVM/SGD models, and log metrics/plots.
4. The notebook saves trained artifacts to `outputs/` (e.g., `svm_hog.joblib` and `ablation_light_best_config.json`).

For a purely scriptable path (no notebook), you can reuse the feature and training helpers inside `features_hog.py` and `dataset_lisa.py` to assemble your own pipeline.

## 🎛️ Gradio demo
Run the interactive patch/classification demo once a trained model exists in `outputs/`:
```bash
python app_gradio_B.py
```
- The app loads `outputs/svm_best_hog.joblib` and tries `outputs/hog_config.json` by default (edit the constants at the top of `app_gradio_B.py` to point to custom files, such as `outputs/ablation_light_best_config.json`).
- If TensorFlow 1.x support is unavailable, the app still runs; the TF1 detector is optional and guarded by a safe import.
- Upload an image or patch through the UI to view predicted color and class probabilities.

## 🧰 Utilities
- **Patch export for slides/demos**: edit `DATASET_ROOT` and `OUT_DIR` in `scripts/make_demo_patches.py`, then run:
  ```bash
  python scripts/make_demo_patches.py
  ```
  The script creates balanced folders (`green/`, `red/`, `yellow/`, `other/`) inside `OUT_DIR`.
- **Sample images**: use files in `demo/` to quickly test the demo app.

## 🚦 Notes & tips
- HOG is tuned for small objects; if you change `PATCH_SIZE` or `HOG_CONFIG_PATH`, retrain and save both the model and config.
- Negative sampling (`add_other`) improves robustness to non-traffic-light regions.
- Keep the original LISA directory layout; the loaders infer image paths relative to `Annotations/Annotations/*/frameAnnotationsBOX.csv`.

---
**Happy training & detecting!**
