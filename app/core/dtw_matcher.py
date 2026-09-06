"""DTW reference matching for normalized, hand-only INCLUDE-50 sequences.

The returned score is a reference-trajectory similarity, not an accuracy or a
classifier probability.
"""

import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


LOGGER = logging.getLogger(__name__)
SEQUENCE_LENGTH = 45
FEATURES_PER_FRAME = 126
ROOT = Path(__file__).resolve().parents[2]


def _configured_path(name, default):
    value = os.getenv(name)
    path = Path(value) if value else default
    return path if path.is_absolute() else ROOT / path


DEFAULT_REFERENCE_DIR = _configured_path(
    "ISL_HAND_REFERENCE_DIR", ROOT / "data" / "include50" / "reference_hand_sequences"
)


def validate_hand_sequence(sequence):
    """Validate and return a model-ready `(45, 126)` hand sequence."""

    array = np.asarray(sequence, dtype=np.float32)
    expected_shape = (SEQUENCE_LENGTH, FEATURES_PER_FRAME)
    if array.shape != expected_shape:
        raise ValueError(f"Expected hand sequence shape {expected_shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Hand sequence contains NaN or Inf")
    return array


def _dtw_distance(reference, query):
    """Mean frame cost along the optimal DTW path (speed-tolerant)."""

    costs = np.sqrt(np.mean((reference[:, None, :] - query[None, :, :]) ** 2, axis=2))
    rows, columns = costs.shape
    accumulated = np.full((rows + 1, columns + 1), np.inf, dtype=np.float32)
    path_lengths = np.zeros((rows + 1, columns + 1), dtype=np.int16)
    accumulated[0, 0] = 0.0

    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            candidates = (
                accumulated[row - 1, column],
                accumulated[row, column - 1],
                accumulated[row - 1, column - 1],
            )
            best = int(np.argmin(candidates))
            if best == 0:
                previous_row, previous_column = row - 1, column
            elif best == 1:
                previous_row, previous_column = row, column - 1
            else:
                previous_row, previous_column = row - 1, column - 1
            accumulated[row, column] = costs[row - 1, column - 1] + candidates[best]
            path_lengths[row, column] = path_lengths[previous_row, previous_column] + 1

    return float(accumulated[rows, columns] / path_lengths[rows, columns])


class DTWReferenceMatcher:
    """Cache reference sequences and match a captured hand sequence."""

    def __init__(self, reference_dir=DEFAULT_REFERENCE_DIR):
        self.reference_dir = Path(reference_dir)
        self.references = defaultdict(list)
        self.loaded = False

    def load(self):
        if self.loaded:
            return self
        if not self.reference_dir.is_dir():
            LOGGER.warning("Hand reference directory is unavailable: %s", self.reference_dir)
            self.loaded = True
            return self

        skipped = 0
        for path in sorted(self.reference_dir.rglob("*.npy")):
            try:
                sequence = validate_hand_sequence(np.load(path, allow_pickle=False))
            except (OSError, ValueError) as error:
                skipped += 1
                LOGGER.warning("Skipping invalid hand reference %s: %s", path, error)
                continue
            self.references[path.parent.name].append(sequence)

        self.loaded = True
        LOGGER.info(
            "Loaded hand DTW references classes=%d sequences=%d skipped=%d",
            len(self.references), sum(len(items) for items in self.references.values()), skipped,
        )
        return self

    def match(self, sequence, candidate_label=None):
        """Match the LSTM's label to references, or find the nearest label."""

        query = validate_hand_sequence(sequence)
        self.load()
        labels = [candidate_label] if candidate_label else sorted(self.references)
        candidates = [(label, self.references.get(label, [])) for label in labels]
        candidates = [(label, items) for label, items in candidates if items]
        reference_total = sum(len(items) for items in self.references.values())
        if not candidates:
            return {
                "available": False,
                "match_label": candidate_label,
                "reason": "No hand reference sequence is available for this class.",
                "reference_classes": len(self.references),
                "reference_sequences": reference_total,
            }

        best_label, best_distance, reference_count = None, np.inf, 0
        for label, references in candidates:
            distance = min(_dtw_distance(reference, query) for reference in references)
            if distance < best_distance:
                best_label, best_distance, reference_count = label, distance, len(references)

        # A bounded display similarity: exact reference matches score 100%.
        # It is intentionally not presented as a calibrated probability.
        similarity = float(np.exp(-3.0 * best_distance))
        return {
            "available": True,
            "match_label": best_label,
            "similarity": similarity,
            "match_percent": round(similarity * 100, 1),
            "dtw_distance": round(float(best_distance), 6),
            "reference_count": reference_count,
            "reference_classes": len(self.references),
            "reference_sequences": reference_total,
        }
