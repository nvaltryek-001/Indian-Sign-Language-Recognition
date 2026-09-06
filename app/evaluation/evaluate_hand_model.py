"""Evaluate the trained INCLUDE-50 hand-focused model."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from app.training.train_hand_lstm import BEST_MODEL, CLASS_MAP, load_sequences, BASE


OUTPUT_DIR = Path("models/evaluation/hand_model")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(BEST_MODEL)
    features, labels, frame = load_sequences(BASE / "test.csv")
    probabilities = model.predict(features, verbose=0)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    class_map = json.loads(CLASS_MAP.read_text(encoding="utf-8"))
    labels_text = [class_map[str(index)] for index in range(50)]

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(50), zero_division=0,
    )
    report = classification_report(
        labels, predictions, labels=np.arange(50), target_names=labels_text,
        output_dict=True, zero_division=0,
    )
    per_class = []
    for index, label in enumerate(labels_text):
        actual = labels == index
        per_class.append({
            "class_id": index,
            "label": label,
            "accuracy": float((predictions[actual] == index).mean()) if actual.any() else None,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        })

    prediction_frame = frame.copy()
    prediction_frame["predicted_class_id"] = predictions
    prediction_frame["predicted_label"] = [labels_text[index] for index in predictions]
    prediction_frame["model_confidence"] = confidence
    prediction_frame.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    np.save(OUTPUT_DIR / "confusion_matrix.npy", confusion_matrix(labels, predictions, labels=np.arange(50)))
    (OUTPUT_DIR / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = {
        "model": str(BEST_MODEL),
        "samples": int(len(labels)),
        "classes": 50,
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confidence_is_not_accuracy": True,
        "per_class": per_class,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("samples", "classes", "accuracy", "macro_precision", "macro_recall", "macro_f1")}, indent=2))


if __name__ == "__main__":
    main()