"""
predict.py — Run inference on a single MRI image.

Usage:
    python src/predict.py --image path/to/mri.jpg [--model models/best_randomforest.pkl]
"""

import argparse
import sys
import numpy as np
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from preprocess import load_image, extract_features
from utils import IDX_TO_CLASS, CLASSES, MODELS_DIR, get_logger

log = get_logger(__name__)


def predict_single(image_path: str | Path, model_path: str | Path | None = None) -> dict:
    """
    Predict Alzheimer's class for a single MRI image.

    Returns:
        dict with keys: predicted_class, confidence, all_probabilities
    """
    # ── Load model ─────────────────────────────────────────────────────────
    if model_path is None:
        candidates = sorted(MODELS_DIR.glob("best_randomforest.pkl"))
        if not candidates:
            candidates = sorted(MODELS_DIR.glob("best_*.pkl"))
        if not candidates:
            raise FileNotFoundError(
                "No saved model found in models/. Run `python src/train.py` first."
            )
        model_path = candidates[-1]

    model = joblib.load(model_path)
    log.info(f"Loaded model: {model_path}")

    # ── Extract features ───────────────────────────────────────────────────
    img = load_image(image_path)
    feats = extract_features(img).reshape(1, -1)

    # ── Predict ────────────────────────────────────────────────────────────
    pred_idx = model.predict(feats)[0]
    predicted_class = IDX_TO_CLASS[pred_idx]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feats)[0]
        confidence = float(proba[pred_idx])
        all_probs = {CLASSES[i]: round(float(p), 4) for i, p in enumerate(proba)}
    else:
        confidence = 1.0
        all_probs = {predicted_class: 1.0}

    return {
        "image": str(image_path),
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "all_probabilities": all_probs,
    }


def pretty_print(result: dict):
    print("\n" + "="*50)
    print("  ALZHEIMER DETECTION — PREDICTION RESULT")
    print("="*50)
    print(f"  Image            : {Path(result['image']).name}")
    print(f"  Predicted Class  : {result['predicted_class']}")
    print(f"  Confidence       : {result['confidence']*100:.1f}%")
    print("\n  Class Probabilities:")
    for cls, prob in result["all_probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"    {cls:<22} {prob*100:5.1f}%  {bar}")
    print("="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Alzheimer class from an MRI scan.")
    parser.add_argument("--image", required=True, help="Path to MRI image file")
    parser.add_argument("--model", default=None, help="Path to .pkl model file (optional)")
    args = parser.parse_args()

    result = predict_single(args.image, args.model)
    pretty_print(result)
