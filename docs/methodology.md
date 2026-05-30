# Technical Methodology

## Overview

This system classifies Alzheimer's disease severity from MRI brain scans using classical computer vision feature extraction combined with scikit-learn ensemble classifiers. The pipeline is fully explainable and does not rely on deep learning.

---

## 1. Dataset

**Source:** [Alzheimer's Dataset — 4 class (Kaggle)](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images)

- ~6,400 brain MRI images (augmented from ~1,000 unique scans)
- 4 classes: NonDemented, VeryMildDemented, MildDemented, ModerateDemented
- Pre-split into train/test directories

**Class imbalance:** The dataset is imbalanced (NonDemented > VeryMild > Mild > Moderate). All classifiers use `class_weight="balanced"` to compensate.

---

## 2. Preprocessing

Every image is:
1. Loaded as **grayscale** (MRI scans carry diagnostic information in intensity, not colour)
2. Resized to **128 × 128 pixels** (balance between information retention and compute)
3. No normalisation at load time — StandardScaler handles this in the pipeline

---

## 3. Feature Engineering

Three complementary feature types are concatenated into a single vector per image:

### 3.1 HOG — Histogram of Oriented Gradients

HOG captures **local edge orientations and structural patterns**.

- Configuration: `orientations=9`, `pixels_per_cell=(16,16)`, `cells_per_block=(2,2)`, L2-Hys normalisation
- Sensitive to the geometric shape of brain structures (ventricular enlargement, cortical atrophy)
- Output: ~324 floats

### 3.2 LBP — Local Binary Patterns

LBP encodes **micro-texture** by comparing each pixel to its neighbours.

- Configuration: `n_points=24`, `radius=3`, uniform LBP
- Captures surface texture changes associated with cortical thinning
- Output: 26-bin normalised histogram

### 3.3 Intensity Statistics

Global and regional statistics summarise **overall brain tissue appearance**.

- **Global (6 values):** mean, std, skewness, kurtosis, Q1, Q3
- **Quadrant means/stds (12 values):** 2×2 spatial decomposition with contrast ratio
- **Projection profiles (6 values):** horizontal & vertical mean projections, their range
- Total: 24 floats

**Combined feature vector length:** ~354 floats per image.

---

## 4. Classifiers

All classifiers are wrapped in a `sklearn.pipeline.Pipeline` with a `StandardScaler` prefix.

| Model | Key Hyperparameters | Notes |
|-------|---------------------|-------|
| **Random Forest** | `n_estimators=300`, `class_weight=balanced` | Best overall; ensemble reduces variance |
| **SVM (RBF)** | `C=10`, `gamma=scale`, `class_weight=balanced` | Strong with high-dim features |
| **Gradient Boosting** | `n_estimators=200`, `lr=0.1`, `max_depth=5` | Slower but competitive |
| **k-NN** | `k=7`, `weights=distance` | Baseline; sensitive to feature scale |

---

## 5. Evaluation

- **Primary metric:** Balanced accuracy (handles class imbalance)
- **Secondary metric:** Per-class F1-score
- **Validation:** 5-fold stratified cross-validation on training set
- **Final evaluation:** Held-out test split

### Confusion Matrix Interpretation

The most clinically important errors are:
- Predicting **NonDemented** when actually **MildDemented** (under-detection)
- Predicting **ModerateDemented** when actually **MildDemented** (over-detection)

---

## 6. Limitations

1. **Feature handcrafting:** HOG/LBP may miss subtle volumetric changes that CNNs detect
2. **2D slices:** Real MRI diagnosis uses 3D volumetric analysis; we use 2D projections
3. **Data augmentation leakage:** The Kaggle dataset contains augmented copies — results may slightly overestimate real-world performance
4. **Not clinical-grade:** No regulatory approval; intended for research only

---

## 7. Future Work

- Replace feature extraction with a fine-tuned CNN (ResNet-18, EfficientNet)
- 3D volumetric analysis using nibabel + SimpleITK
- Incorporate MMSE clinical scores as supplementary features
- SHAP-based explainability for individual predictions
- DICOM file support
