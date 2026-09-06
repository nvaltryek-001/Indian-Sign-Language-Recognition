"""Validation helpers for extracted Include-50 sequences."""

from pathlib import Path

import numpy as np

from app.core.feature_pipeline import FEATURES_PER_FRAME, SEQUENCE_LENGTH


def validate_sequence_file(path):
    """Validate one .npy sequence and return its shape."""

    array = np.load(path, allow_pickle=False)
    expected_shape = (SEQUENCE_LENGTH, FEATURES_PER_FRAME)
    if array.shape != expected_shape:
        raise ValueError(f"{path}: expected {expected_shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: contains NaN or Inf")
    return array.shape


def validate_sequence_directory(directory):
    """Return a summary for all extracted sequences below *directory*."""

    paths = sorted(Path(directory).rglob("*.npy"))
    for path in paths:
        validate_sequence_file(path)
    return {"files": len(paths), "shape": (SEQUENCE_LENGTH, FEATURES_PER_FRAME)}