"""
train.py — Train and evaluate multiple scikit-learn classifiers on MRI features.

Usage:
    python src/train.py [--model {rf,svm,gb,knn,all}] [--tune]
"""

import argparse
import json
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, balanced_accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

# Fix relative imports when run as script
import sys
sys.path.insert(0, str(Path(__file__).parent))

from preprocess import build_dataset
from utils import CLASSES, IDX_TO_CLASS, MODELS_DIR, get_logger

log = get_logger(__name__)


# ── Classifier catalogue ───────────────────────────────────────────────────

def get_pipelines() -> dict[str, Pipeline]:
    return {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=4,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=42,
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                random_state=42,
            )),
        ]),
        "kNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=7,
                weights="distance",
                metric="euclidean",
                n_jobs=-1,
            )),
        ]),
    }


# ── Evaluation helpers ─────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, pad=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"Confusion matrix saved → {save_path}")


def evaluate(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray,
             model_name: str, out_dir: Path) -> dict:
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test, y_pred,
        target_names=CLASSES,
        output_dict=True,
    )

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Balanced Accuracy : {bacc:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    plot_confusion_matrix(
        y_test, y_pred, model_name,
        out_dir / f"cm_{model_name.lower().replace(' ', '_')}.png",
    )

    return {
        "model": model_name,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bacc, 4),
        "report": report,
    }


# ── Training loop ──────────────────────────────────────────────────────────

def train(model_key: str = "all", cross_validate: bool = True):
    log.info("Loading features…")
    X_train, y_train = build_dataset("train")
    X_test,  y_test  = build_dataset("test")
    log.info(f"Train: {X_train.shape}  Test: {X_test.shape}")

    pipelines = get_pipelines()
    if model_key != "all":
        key_map = {
            "rf":  "RandomForest",
            "svm": "SVM_RBF",
            "gb":  "GradientBoosting",
            "knn": "kNN",
        }
        name = key_map.get(model_key, model_key)
        pipelines = {name: pipelines[name]}

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, pipeline in pipelines.items():
        log.info(f"\nTraining {name}…")

        if cross_validate:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=cv, scoring="balanced_accuracy", n_jobs=-1,
            )
            log.info(f"  5-fold CV balanced accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        pipeline.fit(X_train, y_train)

        metrics = evaluate(pipeline, X_test, y_test, name, out_dir)
        results.append(metrics)

        # Save model
        model_path = out_dir / f"{name.lower()}_{timestamp}.pkl"
        joblib.dump(pipeline, model_path)
        log.info(f"  Model saved → {model_path}")

        # Save best model pointer (always overwrite with latest trained)
        best_path = out_dir / f"best_{name.lower()}.pkl"
        joblib.dump(pipeline, best_path)

    # Save summary JSON
    summary_path = out_dir / f"results_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults summary → {summary_path}")

    # Print leaderboard
    print("\n" + "="*60)
    print("  LEADERBOARD")
    print("="*60)
    sorted_results = sorted(results, key=lambda r: r["balanced_accuracy"], reverse=True)
    for i, r in enumerate(sorted_results, 1):
        print(f"  {i}. {r['model']:<22} balanced_acc={r['balanced_accuracy']:.4f}  acc={r['accuracy']:.4f}")
    print("="*60)

    return sorted_results


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Alzheimer detection models.")
    parser.add_argument(
        "--model", default="all",
        choices=["all", "rf", "svm", "gb", "knn"],
        help="Which model(s) to train (default: all)",
    )
    parser.add_argument(
        "--no-cv", action="store_true",
        help="Skip cross-validation (faster)",
    )
    args = parser.parse_args()
    train(model_key=args.model, cross_validate=not args.no_cv)