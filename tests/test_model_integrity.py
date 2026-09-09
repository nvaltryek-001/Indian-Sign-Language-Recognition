import json
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "include50_hand_lstm_savedmodel"
CLASS_PATH = ROOT / "models" / "include50_hand_classes.json"


def _load_signature():
    model = tf.saved_model.load(str(MODEL_PATH))

    if "serve" in model.signatures:
        return model, model.signatures["serve"]

    if "serving_default" in model.signatures:
        return model, model.signatures["serving_default"]

    raise AssertionError("SavedModel has no usable serving signature")


def test_hand_model_and_class_map_contract():
    assert MODEL_PATH.is_dir(), f"Missing SavedModel: {MODEL_PATH}"
    assert CLASS_PATH.is_file(), f"Missing class map: {CLASS_PATH}"

    model, signature = _load_signature()

    input_specs = signature.structured_input_signature[1]
    output_specs = signature.structured_outputs

    input_spec = next(iter(input_specs.values()))
    output_spec = next(iter(output_specs.values()))

    assert tuple(input_spec.shape[1:]) == (45, 126)
    assert output_spec.shape[-1] == 50

    class_map = json.loads(CLASS_PATH.read_text(encoding="utf-8"))

    assert len(class_map) == 50
    assert all(str(index) in class_map for index in range(50))


def test_hand_model_returns_probability_distribution():
    _, signature = _load_signature()

    input_name = next(
        iter(signature.structured_input_signature[1])
    )

    sequence = np.zeros(
        (1, 45, 126),
        dtype=np.float32,
    )

    outputs = signature(
        **{
            input_name: tf.convert_to_tensor(sequence)
        }
    )

    probabilities = np.asarray(
        next(iter(outputs.values()))
    )

    assert probabilities.shape == (1, 50)
    assert np.isfinite(probabilities).all()
    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-5,
    )
