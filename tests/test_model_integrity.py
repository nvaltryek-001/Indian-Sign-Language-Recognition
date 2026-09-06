import json
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "include50_lstm_v2_best.keras"
CLASS_PATH = ROOT / "models" / "include50_classes_v2.json"


def test_model_and_class_map_contract():
    model = tf.keras.models.load_model(MODEL_PATH)
    assert model.input_shape[1:] == (45, 258)
    assert model.output_shape[-1] == 50

    class_map = json.loads(CLASS_PATH.read_text(encoding="utf-8"))
    assert sorted(class_map, key=int) == [str(index) for index in range(50)]


def test_model_returns_finite_probability_distribution():
    model = tf.keras.models.load_model(MODEL_PATH)
    probabilities = model.predict(np.zeros((1, 45, 258), dtype=np.float32), verbose=0)
    assert probabilities.shape == (1, 50)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)