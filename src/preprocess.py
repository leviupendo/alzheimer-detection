"""
preprocess.py — Load MRI images and extract classical vision features.

Feature vector per image:
  • HOG  (Histogram of Oriented Gradients) — captures edge/shape structure
  • LBP  (Local Binary Patterns)           — encodes micro-texture
  • Stat (pixel intensity statistics)      — global intensity summary
"""

import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
from scipy.stats import skew, kurtosis
import joblib

from utils import (
    DATA_RAW, DATA_PROC, CLASSES, CLASS_TO_IDX, IMG_SIZE, get_logger
)

log = get_logger(__name__)


# ── Low-level image I/O ────────────────────────────────────────────────────

def load_image(path: str | Path) -> np.ndarray:
    """Load an image as a grayscale numpy array resized to IMG_SIZE."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
    return img


# ── Feature extractors ─────────────────────────────────────────────────────

def extract_hog_features(img: np.ndarray) -> np.ndarray:
    """
    HOG captures gradients & structural patterns (sulci/gyri boundaries).
    Returns a 1-D float32 vector.
    """
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features.astype(np.float32)


def extract_lbp_features(img: np.ndarray, n_points: int = 24, radius: int = 3) -> np.ndarray:
    """
    LBP encodes local texture — sensitive to atrophy-related surface changes.
    Returns a normalised histogram vector of length n_points + 2.
    """
    lbp = local_binary_pattern(img, n_points, radius, method="uniform")
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def extract_stat_features(img: np.ndarray) -> np.ndarray:
    """
    Global and regional intensity statistics.
    Returns a 1-D float32 vector of length 24.
    """
    flat = img.ravel().astype(np.float64)
    global_stats = np.array([
        flat.mean(), flat.std(),
        skew(flat), kurtosis(flat),
        np.percentile(flat, 25), np.percentile(flat, 75),
    ], dtype=np.float32)

    # Divide image into 2×2 quadrants and compute mean/std per quadrant
    h, w = img.shape
    quads = [
        img[:h//2, :w//2], img[:h//2, w//2:],
        img[h//2:, :w//2], img[h//2:, w//2:],
    ]
    quad_stats = np.array(
        [[q.mean(), q.std(), q.mean() / (q.std() + 1e-6)] for q in quads],
        dtype=np.float32,
    ).ravel()  # 12 values

    # Horizontal & vertical projection profiles
    h_proj = img.mean(axis=1)
    v_proj = img.mean(axis=0)
    proj_stats = np.array([
        h_proj.mean(), h_proj.std(),
        v_proj.mean(), v_proj.std(),
        h_proj.max() - h_proj.min(),
        v_proj.max() - v_proj.min(),
    ], dtype=np.float32)

    return np.concatenate([global_stats, quad_stats, proj_stats])


def extract_features(img: np.ndarray) -> np.ndarray:
    """Concatenate all feature types into a single feature vector."""
    return np.concatenate([
        extract_hog_features(img),
        extract_lbp_features(img),
        extract_stat_features(img),
    ])


# ── Dataset builder ────────────────────────────────────────────────────────

def build_dataset(split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """
    Walk data/raw/<split>/<ClassName>/*.jpg and return (X, y).

    Args:
        split: 'train' or 'test'

    Returns:
        X: float32 array of shape (n_samples, n_features)
        y: int32 array of shape (n_samples,)
    """
    cache_X = DATA_PROC / f"{split}_X.npy"
    cache_y = DATA_PROC / f"{split}_y.npy"

    if cache_X.exists() and cache_y.exists():
        log.info(f"Loading cached {split} features from {DATA_PROC}")
        return np.load(cache_X), np.load(cache_y)

    split_dir = DATA_RAW / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {split_dir}.\n"
            "Download from https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images "
            "and place under data/raw/"
        )

    X_list, y_list = [], []
    for cls in CLASSES:
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            log.warning(f"Class folder missing: {cls_dir} — skipping.")
            continue
        images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        log.info(f"  {cls}: {len(images)} images")
        for img_path in tqdm(images, desc=cls, leave=False):
            try:
                img = load_image(img_path)
                feats = extract_features(img)
                X_list.append(feats)
                y_list.append(CLASS_TO_IDX[cls])
            except Exception as e:
                log.warning(f"Skipping {img_path}: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    np.save(cache_X, X)
    np.save(cache_y, y)
    log.info(f"Saved {split} features → {DATA_PROC}  shape={X.shape}")
    return X, y


if __name__ == "__main__":
    log.info("Extracting train features…")
    X_tr, y_tr = build_dataset("train")
    log.info(f"Train: {X_tr.shape}, labels: {np.bincount(y_tr)}")

    log.info("Extracting test features…")
    X_te, y_te = build_dataset("test")
    log.info(f"Test:  {X_te.shape}, labels: {np.bincount(y_te)}")