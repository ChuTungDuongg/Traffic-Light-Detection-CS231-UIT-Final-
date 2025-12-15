# 🚦 Traffic Light Detection

**CS231 – Introduction to Computer Vision (UIT)**

> 🎓 **Final Project – CS231**
> 👨‍💻 Classical Computer Vision + Machine Learning
> 📍 University of Information Technology (UIT)

---

## 📌 Overview

This project focuses on **traffic light detection and classification** using a **hybrid classical computer vision pipeline**.
The system combines:

* **Heuristic color-based localization** (HSV, S×V peak)
* **Feature extraction** (HOG & VGG16 pretrained – frozen)
* **Classical classifiers** (SVM, SGDClassifier)

The goal is to build a **robust, interpretable, and practical pipeline** that works well even with **small objects and limited training data**, as commonly encountered in real-world traffic scenes.

---

## 🧠 Key Contributions

✔ Stable **Conditioned Sliding Window** based on HSV (S×V peak)
✔ Comparison between **Handcrafted features (HOG)** and **Deep features (VGG16 freeze)**
✔ End-to-end **training – evaluation – demo pipeline**
✔ Clear analysis using **confusion matrix, learning curves, epoch curves**
✔ Fully reproducible experimental setup

---

## 📂 Project Structure

```
Traffic-Light-Detection-CS231-UIT-Final/
│
├── LISA_Trafficlight_clean_pipeline.ipynb   # Main training & evaluation pipeline
├── dataset_lisa.py                          # Dataset loader from LISA annotations
├── features_hog.py                          # HOG feature extraction (batch)
│
├── outputs/
│   ├── svm_hog.joblib                       # Trained HOG + SVM model
│   └── svm_vgg16.joblib                     # Trained VGG16(freeze) + SVM model
│
├── demo/
│   ├── *.jpg / *.png / *.webp               # Images for demo testing
│
└── README.md
```

---

## 📊 Dataset

This project uses the **LISA Traffic Light Dataset**, available on Kaggle:

🔗 [https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset)

* Images with bounding-box annotations
* Traffic light colors: **Red / Yellow / Green**
* Outdoor scenes with varying lighting conditions

---

## ⚙️ Methodology

### 1️⃣ Patch-based Dataset Construction

* Crop image patches from bounding-box annotations
* Normalize patch size to **64×64**
* Convert detection problem → classification problem

### 2️⃣ Feature Extraction

* **HOG (Histogram of Oriented Gradients)**

  * Cell: 4×4, Block: 2×2
  * Optimized for small objects (traffic lights)
* **VGG16 (Frozen, ImageNet pretrained)**

  * Deep feature extractor
  * No fine-tuning to avoid overfitting

### 3️⃣ Classification

* **Support Vector Machine (RBF kernel)**
* **SGDClassifier (log loss)** for epoch-based analysis

### 4️⃣ Evaluation

* Accuracy
* Precision / Recall / F1-score
* Confusion Matrix
* Learning Curves (Train vs Validation)
* Epoch-based Accuracy/Loss Curves (SGD)

---

## 🎥 Final Demo – Stable Detection Pipeline

A practical demo is implemented using:

**Conditioned Sliding Window via HSV S×V Peak**

* No hard thresholds
* No contour or circularity dependency
* Robust against illumination changes
* Always proposes a candidate region if strong color evidence exists

This makes the demo **stable and reliable for real-world testing**.

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
pip install numpy opencv-python matplotlib scikit-learn tensorflow scikit-image joblib
```

### 2️⃣ Download Dataset

Download and extract the dataset from Kaggle:

```
https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset
```

Update the dataset path in the notebook:

```python
DATASET_ROOT = "path/to/lisa_dataset"
```

### 3️⃣ Train & Evaluate

Open and run:

```
LISA_Trafficlight_clean_pipeline.ipynb
```

### 4️⃣ Run Demo

Use images in the `demo/` folder or add your own traffic images.

---

## 🧪 Experimental Highlights

* **HOG + SVM** performs strongly on small, well-defined objects
* **VGG16(freeze) + SVM** provides competitive performance with better generalization
* Learning curves reveal data-limited behavior
* Epoch curves illustrate convergence dynamics (via SGD)

---

## ⚠️ Limitations

* Heuristic detection may fail on:

  * Traffic lights heavily occluded
  * Bright LED billboards or vehicle lights
* No temporal modeling (single-frame only)
* No end-to-end deep detection (e.g., YOLO)

---

## 🔮 Future Work

* Integrate CNN-based detector (YOLO / SSD)
* Temporal smoothing for video input
* Fine-tune deep backbone on traffic-light-specific data
* Deploy lightweight version for embedded systems

---

## 👤 Author

**CS231 – Final Project**
University of Information Technology (UIT)
Vietnam National University – HCMC

---

## ⭐ Acknowledgements

* CS231 Teaching Team – UIT
* LISA Traffic Light Dataset authors
* Open-source Computer Vision community

---

🚦 *Happy training & happy detecting!*


Chỉ cần nói 👍

