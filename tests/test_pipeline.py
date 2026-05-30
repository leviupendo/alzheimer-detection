"""
test_pipeline.py — Unit tests for the Alzheimer Detection pipeline.

Run:
    pytest tests/
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocess import (
    extract_hog_features,
    extract_lbp_features,
    extract_stat_features,
    extract_features,
)
from utils import CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, IMG_SIZE


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_image():
    """Random grayscale image of the expected input size."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, IMG_SIZE, dtype=np.uint8)


@pytest.fixture
def blank_image():
    return np.zeros(IMG_SIZE, dtype=np.uint8)


@pytest.fixture
def bright_image():
    return np.full(IMG_SIZE, 255, dtype=np.uint8)


# ── Feature extraction tests ───────────────────────────────────────────────

class TestHOG:
    def test_returns_1d_array(self, dummy_image):
        feat = extract_hog_features(dummy_image)
        assert feat.ndim == 1

    def test_dtype_float32(self, dummy_image):
        feat = extract_hog_features(dummy_image)
        assert feat.dtype == np.float32

    def test_consistent_length(self, dummy_image, blank_image):
        assert len(extract_hog_features(dummy_image)) == len(extract_hog_features(blank_image))

    def test_no_nan(self, dummy_image):
        feat = extract_hog_features(dummy_image)
        assert not np.any(np.isnan(feat))


class TestLBP:
    def test_returns_1d_array(self, dummy_image):
        feat = extract_lbp_features(dummy_image)
        assert feat.ndim == 1

    def test_probability_histogram(self, dummy_image):
        feat = extract_lbp_features(dummy_image)
        # Should be normalised (sums to ~1)
        assert pytest.approx(feat.sum(), abs=0.01) == 1.0

    def test_consistent_length(self, dummy_image, blank_image):
        assert len(extract_lbp_features(dummy_image)) == len(extract_lbp_features(blank_image))


class TestStatFeatures:
    def test_returns_1d_array(self, dummy_image):
        feat = extract_stat_features(dummy_image)
        assert feat.ndim == 1

    def test_expected_length(self, dummy_image):
        # 6 global + 12 quadrant + 6 projection = 24
        assert len(extract_stat_features(dummy_image)) == 24

    def test_blank_image_std_zero(self, blank_image):
        feat = extract_stat_features(blank_image)
        # std of blank image should be 0
        assert feat[1] == pytest.approx(0.0, abs=1e-5)

    def test_no_nan(self, dummy_image):
        feat = extract_stat_features(dummy_image)
        assert not np.any(np.isnan(feat))


class TestFullFeatureVector:
    def test_consistent_shape(self, dummy_image, blank_image):
        f1 = extract_features(dummy_image)
        f2 = extract_features(blank_image)
        assert f1.shape == f2.shape

    def test_dtype(self, dummy_image):
        assert extract_features(dummy_image).dtype == np.float32

    def test_no_nan_or_inf(self, dummy_image):
        feat = extract_features(dummy_image)
        assert not np.any(np.isnan(feat))
        assert not np.any(np.isinf(feat))

    def test_different_images_produce_different_features(self, dummy_image, blank_image):
        f1 = extract_features(dummy_image)
        f2 = extract_features(blank_image)
        assert not np.allclose(f1, f2)


# ── Utils tests ────────────────────────────────────────────────────────────

class TestUtils:
    def test_classes_count(self):
        assert len(CLASSES) == 4

    def test_class_idx_roundtrip(self):
        for cls in CLASSES:
            idx = CLASS_TO_IDX[cls]
            assert IDX_TO_CLASS[idx] == cls

    def test_img_size_tuple(self):
        assert len(IMG_SIZE) == 2
        assert all(s > 0 for s in IMG_SIZE)

    def test_expected_class_names(self):
        expected = {"NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"}
        assert set(CLASSES) == expected