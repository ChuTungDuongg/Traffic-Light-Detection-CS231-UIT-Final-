# 🚦 Traffic Light Detection (CS231 – UIT)

Final project for **CS231 – Introduction to Computer Vision (University of Information Technology, VNU-HCM)**. The repository implements a **classical computer vision pipeline** for traffic light detection and color classification using HSV heuristics for localization and HOG/VGG16 features for classification.

## 🗺️ What’s inside
- **Complete pipelines in notebooks**: full training/evaluation (`lisa_full_pipeline_clean.ipynb`) and ablation experiments (`Ablation_study.ipynb`).
- **Reusable dataset utilities**: `dataset_lisa.py` for loading LISA annotations and building patch datasets.
- **Feature extraction**: `features_hog.py` contains HOG helpers; VGG16 (frozen) extraction lives in the notebook.
- **Interactive demo**: `app_gradio.py` serves a Gradio UI powered by the trained SVM + HOG model stored in `outputs/`.
- **Demo assets**: `demo/` (sample images) and `demo_assets/` (patches/backgrounds for presentations).
- **Helpers**: `scripts/make_demo_patches.py` to export balanced patches from the raw dataset.

## 📂 Repository layout
```
Traffic-Light-Detection-CS231-UIT-Final-
├── lisa_full_pipeline_clean.ipynb   # Main notebook: dataset -> features -> training -> evaluation
├── Ablation_study.ipynb             # Experiments comparing HOG vs VGG16 (frozen) + SVM/SGD
├── dataset_lisa.py                  # Parse LISA annotations, crop patches, map labels
├── features_hog.py                  # HOG feature helpers used by notebooks and demo
├── detector_tf1.py                  # Optional TF1 detector wrapper (best-effort import in the app)
├── app_gradio.py                    # Gradio demo consuming trained model/config from outputs/
├── scripts/
│   └── make_demo_patches.py         # Export small, balanced patch set for demos
├── outputs/                         # Place for trained models (svm_hog.joblib, hog_config.json, ...)
├── demo/                            # Sample images for quick testing
└── demo_assets/                     # Additional demo-ready assets (patches, backgrounds)
```

## 🛠️ Environment
- Python ≥ 3.9 recommended.
- Core dependencies: `numpy`, `pandas`, `opencv-python`, `matplotlib`, `scikit-learn`, `tensorflow` (for VGG16 feature extraction), `scikit-image`, `joblib`, `gradio` (for the UI).

Install everything in a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install numpy pandas opencv-python matplotlib scikit-learn tensorflow scikit-image joblib gradio
```

## 📊 Dataset: LISA Traffic Light
Download the LISA Traffic Light Dataset from Kaggle:
- https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset

After extracting, note the dataset root (folder containing `Annotations/` and video folders). The utilities assume the original folder structure.

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

## 🧠 Training & evaluation
1. Launch Jupyter and open the main notebook:
   ```bash
   jupyter notebook lisa_full_pipeline_clean.ipynb
   ```
2. Update the dataset path variables in the first cells.
3. Run all cells to generate train/validation splits, extract features (HOG or VGG16), train SVM/SGD models, and log metrics/plots.
4. The notebook saves trained artifacts to `outputs/` (e.g., `svm_hog.joblib` and `hog_config.json`).

For additional comparisons, open `Ablation_study.ipynb` and execute selected cells to reproduce the feature/classifier ablations.

## 🎛️ Gradio demo
Run the interactive patch/classification demo once a trained model exists in `outputs/`:
```bash
python app_gradio.py
```
- The app loads `outputs/svm_hog.joblib` and `outputs/hog_config.json` by default (edit the constants at the top of `app_gradio.py` to point to custom files).
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
