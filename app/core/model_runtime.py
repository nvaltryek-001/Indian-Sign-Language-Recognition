"""Model loading and prediction for the production V2 inference API."""

import json
import logging
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from app.core.feature_pipeline import validate_sequence


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


def _configured_path(name, default):
    value = os.getenv(name)
    path = Path(value) if value else default
    return path if path.is_absolute() else ROOT / path


DEFAULT_MODEL_PATH = _configured_path(
    "ISL_MODEL_PATH", ROOT / "models" / "include50_lstm_v2_best.keras"
)
DEFAULT_CLASS_MAP_PATH = _configured_path(
    "ISL_CLASS_MAP_PATH", ROOT / "models" / "include50_classes_v2.json"
)


class ModelRuntime:
    """Load the model once and expose validated predictions."""

    model_version = "include50_lstm_v2_best"

    def __init__(self, model_path=DEFAULT_MODEL_PATH, class_map_path=DEFAULT_CLASS_MAP_PATH):
        self.model_path = Path(model_path)
        self.class_map_path = Path(class_map_path)
        self.model = None
        self.class_names = []

    def load(self):
        if self.model is not None:
            return self

        self.model = tf.keras.models.load_model(self.model_path)
        if self.model.input_shape[1:] != (45, 258):
            raise ValueError(f"Unexpected model input shape: {self.model.input_shape}")
        if self.model.output_shape[-1] != 50:
            raise ValueError(f"Unexpected model output shape: {self.model.output_shape}")

        data = json.loads(self.class_map_path.read_text(encoding="utf-8"))
        self.class_names = [str(data[str(index)]) for index in range(50)]
        LOGGER.info("Loaded model=%s classes=%d", self.model_path.name, len(self.class_names))
        return self

    def predict(self, sequence):
        self.load()
        array = validate_sequence(sequence)
        probabilities = np.asarray(self.model.predict(array[None, :, :], verbose=0)[0])
        if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1.0, atol=1e-5):
            raise RuntimeError("Model returned an invalid probability distribution")
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = [
            {
                "index": int(index),
                "label": self.class_names[index],
                "class": self.class_names[index],
                "confidence": float(probabilities[index]),
            }
            for index in top_indices
        ]
        return {
            "prediction": top_predictions[0]["label"],
            "confidence": top_predictions[0]["confidence"],
            "top_predictions": top_predictions,
            "model_version": self.model_version,
        }
