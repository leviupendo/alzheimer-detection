"""
utils.py — Shared helpers for the Alzheimer Detection pipeline.
"""

import os
import logging
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent.parent
DATA_RAW      = ROOT_DIR / "data" / "raw"
DATA_PROC     = ROOT_DIR / "data" / "processed"
MODELS_DIR    = ROOT_DIR / "models"

for d in (DATA_PROC, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Class labels (must match folder names in dataset) ──────────────────────
CLASSES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# ── Image settings ─────────────────────────────────────────────────────────
IMG_SIZE = (128, 128)   # resize target (height, width)

# ── Logging ────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def label_from_path(path: str | Path) -> str:
    """Infer class label from the parent directory name of an image file."""
    return Path(path).parent.name