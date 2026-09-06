"""Runtime for the hand-focused `(45, 126)` INCLUDE-50 LSTM."""

import json
import logging
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from app.core.dtw_matcher import FEATURES_PER_FRAME, SEQUENCE_LENGTH, validate_hand_sequence


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


def _configured_path(name, default):
    value = os.getenv(name)
    path = Path(value) if value else default
    return path if path.is_absolute() else ROOT / path


DEFAULT_MODEL_PATH = _configured_path(
    "ISL_HAND_MODEL_PATH", ROOT / "models" / "include50_hand_lstm_savedmodel"
)
DEFAULT_CLASS_MAP_PATH = _configured_path(
    "ISL_HAND_CLASS_MAP_PATH", ROOT / "models" / "include50_hand_classes.json"
)


class HandModelRuntime:
    """Load and predict with the hand-only model without touching the V2 model."""

    model_version = "include50_hand_lstm_best"

    def __init__(self, model_path=DEFAULT_MODEL_PATH, class_map_path=DEFAULT_CLASS_MAP_PATH):
        self.model_path = Path(model_path)
        self.class_map_path = Path(class_map_path)
        self.model = None
        self.class_names = []
        self._saved_model_signature = None

    def load(self):
        if self.model is not None:
            return self

        if self.model_path.is_dir():
            self.model = tf.saved_model.load(str(self.model_path))
            if "serve" in self.model.signatures:
                self._saved_model_signature = self.model.signatures["serve"]
            elif "serving_default" in self.model.signatures:
                self._saved_model_signature = self.model.signatures["serving_default"]
            else:
                raise ValueError("SavedModel does not contain a usable serving signature")
            
            input_specs = self._saved_model_signature.structured_input_signature[1]
            output_specs = self._saved_model_signature.structured_outputs

            input_spec = next(iter(input_specs.values()))
            output_spec = next(iter(output_specs.values()))

            if input_spec.shape[1:] != (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
                raise ValueError(f"Unexpected hand-model input shape: {input_spec.shape}")

            if output_spec.shape[-1] != 50:
                raise ValueError(f"Unexpected hand-model output shape: {output_spec.shape}")

        else:
            self.model = tf.keras.models.load_model(self.model_path)

            if self.model.input_shape[1:] != (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
                raise ValueError(f"Unexpected hand-model input shape: {self.model.input_shape}")

            if self.model.output_shape[-1] != 50:
                raise ValueError(f"Unexpected hand-model output shape: {self.model.output_shape}")

        data = json.loads(self.class_map_path.read_text(encoding="utf-8"))
        self.class_names = [str(data[str(index)]) for index in range(50)]

        LOGGER.info(
            "Loaded hand model=%s classes=%d",
            self.model_path,
            len(self.class_names),
        )
        return self

    def predict(self, sequence):
        self.load()

        array = validate_hand_sequence(sequence)
        batch = array[None, :, :].astype(np.float32)

        if self._saved_model_signature is not None:
            input_name = next(iter(self._saved_model_signature.structured_input_signature[1]))
            outputs = self._saved_model_signature(
                **{input_name: tf.convert_to_tensor(batch)}
            )
            probabilities = np.asarray(next(iter(outputs.values()))[0])
        else:
            probabilities = np.asarray(
                self.model.predict(batch, verbose=0)[0]
            )

        if (
            not np.isfinite(probabilities).all()
            or not np.isclose(probabilities.sum(), 1.0, atol=1e-5)
        ):
            raise RuntimeError("Hand model returned an invalid probability distribution")

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
