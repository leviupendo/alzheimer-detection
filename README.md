 🧠 Alzheimer's Detection via MRI Brain Scans

A classical machine learning pipeline that classifies Alzheimer's Disease severity from MRI brain scan images using feature extraction + scikit-learn classifiers.

---

## 📋 Classification Classes

| Class | Description |
|-------|-------------|
| `NonDemented` | No signs of dementia |
| `VeryMildDemented` | Very mild cognitive decline |
| `MildDemented` | Mild cognitive decline |
| `ModerateDemented` | Moderate cognitive decline |

---

## 🗂️ Project Structure

```
alzheimer-detection/
├── data/
│   ├── raw/                  # Original MRI images (place dataset here)
│   └── processed/            # Extracted features (auto-generated)
├── models/                   # Saved trained models (.pkl)
├── src/
│   ├── preprocess.py         # Image loading & feature extraction
│   ├── train.py              # Model training & evaluation
│   ├── predict.py            # Single-image inference
│   └── utils.py              # Shared helpers
├── app/
│   └── app.py                # Streamlit web UI
├── notebooks/
│   └── exploration.ipynb     # EDA & experiment notebook
├── tests/
│   └── test_pipeline.py      # Unit tests
├── docs/
│   └── methodology.md        # Technical methodology
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/alzheimer-detection.git
cd alzheimer-detection
pip install -r requirements.txt
```

### 2. Get the Dataset

Download the **Alzheimer's Dataset (4 class)** from Kaggle:
> https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images

Place it under `data/raw/` so the structure looks like:
```
data/raw/
├── train/
│   ├── NonDemented/
│   ├── VeryMildDemented/
│   ├── MildDemented/
│   └── ModerateDemented/
└── test/
    ├── NonDemented/
    └── ...
```

### 3. Train the Model

```bash
python src/train.py
```

### 4. Run Inference on a Single MRI

```bash
python src/predict.py --image path/to/mri_scan.jpg
```

### 5. Launch the Web UI

```bash
streamlit run app/app.py
```

---

## 🧪 Run Tests

```bash
pytest tests/
```

---

## 🐳 Docker

```bash
docker build -t alzheimer-detection .
docker run -p 8501:8501 alzheimer-detection
```

---

## 🔬 Methodology

Feature extraction pipeline:
- **HOG** (Histogram of Oriented Gradients) — captures tissue structure
- **LBP** (Local Binary Patterns) — encodes texture
- **Pixel intensity statistics** — mean, std, skewness per region

Classifiers evaluated:
- Random Forest ✅ (best overall)
- SVM (RBF kernel)
- Gradient Boosting
- k-NN

See [`docs/methodology.md`](docs/methodology.md) for full details.

---

## 📊 Expected Performance

| Classifier | Accuracy |
|------------|----------|
| Random Forest | ~82–88% |
| SVM (RBF) | ~79–85% |
| Gradient Boosting | ~80–86% |

*Results vary with dataset split and hyperparameters.*

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**. It is **not** a medical device and should not be used for clinical diagnosis.

---

## 📄 License

MIT License © 2024
