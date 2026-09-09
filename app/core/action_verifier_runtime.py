"""Runtime for the INCLUDE-50 same-action verifier."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "include50_action_verifier.keras"
META_PATH = ROOT / "models" / "include50_action_verifier.json"

SEQUENCE_LENGTH = 45
FEATURES_PER_FRAME = 126


class ActionVerifierRuntime:
    def __init__(self):
        self.model = None
        self.mean = None
        self.std = None

    def load(self):
        if self.model is not None:
            return self

        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Action verifier not found: {MODEL_PATH}"
            )

        self.model = tf.keras.models.load_model(
            MODEL_PATH,
            safe_mode=False,
        )

        if META_PATH.is_file():
            metadata = json.loads(
                META_PATH.read_text(encoding="utf-8")
            )

            if "normalization_mean" in metadata:
                self.mean = np.asarray(
                    metadata["normalization_mean"],
                    dtype=np.float32,
                )

            if "normalization_std" in metadata:
                self.std = np.asarray(
                    metadata["normalization_std"],
                    dtype=np.float32,
                )

        return self

    def predict_same_action(
        self,
        webcam_sequence,
        reference_sequences,
    ):
        self.load()

        webcam = np.asarray(
            webcam_sequence,
            dtype=np.float32,
        )

        if webcam.shape != (
            SEQUENCE_LENGTH,
            FEATURES_PER_FRAME,
        ):
            raise ValueError(
                f"Unexpected webcam shape: {webcam.shape}; "
                f"expected ({SEQUENCE_LENGTH}, {FEATURES_PER_FRAME})"
            )

        if not reference_sequences:
            return {
                "available": False,
                "probability": 0.0,
                "best_probability": 0.0,
                "reference_count": 0,
            }

        references = np.asarray(
            reference_sequences,
            dtype=np.float32,
        )

        if references.ndim != 3:
            raise ValueError(
                f"Unexpected reference shape: {references.shape}"
            )

        # Apply normalization only when metadata contains it.
        if (
            self.mean is not None
            and self.std is not None
            and self.mean.shape == (FEATURES_PER_FRAME,)
            and self.std.shape == (FEATURES_PER_FRAME,)
        ):
            std = np.maximum(
                self.std,
                1e-4,
            )

            webcam = (
                webcam - self.mean
            ) / std

            references = (
                references - self.mean
            ) / std

        webcam_batch = np.repeat(
            webcam[None, :, :],
            len(references),
            axis=0,
        )

        predictions = self.model.predict(
            [
                references,
                webcam_batch,
            ],
            verbose=0,
        )

        scores = np.asarray(
            predictions,
            dtype=np.float32,
        ).reshape(-1)

        scores = scores[
            np.isfinite(scores)
        ]

        if len(scores) == 0:
            return {
                "available": False,
                "probability": 0.0,
                "best_probability": 0.0,
                "reference_count": 0,
            }

        best = float(
            np.max(scores)
        )

        top_count = min(
            5,
            len(scores),
        )

        top_scores = np.sort(scores)[
            -top_count:
        ]

        robust = float(
            np.mean(top_scores)
        )

        probability = (
            0.60 * best
            + 0.40 * robust
        )

        probability = float(
            np.clip(
                probability,
                0.0,
                1.0,
            )
        )

        return {
            "available": True,
            "probability": probability,
            "best_probability": best,
            "top_reference_average": robust,
            "reference_count": int(len(scores)),
        }
