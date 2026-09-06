from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# INCLUDE-50 — MODEL EVALUATION
# ============================================================

BASE = Path("data/include50")
MODEL_PATH = Path("models/include50_lstm_v2_best.keras")
CLASS_MAP_PATH = Path("models/include50_classes_v2.json")

TEST_CSV = BASE / "test.csv"

print("=" * 70)
print("INCLUDE-50 MODEL EVALUATION")
print("=" * 70)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded:")
print(MODEL_PATH)

# ------------------------------------------------------------
# Load class mapping
# ------------------------------------------------------------

with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
    class_mapping = json.load(f)

class_names = [
    class_mapping[str(i)]
    for i in range(len(class_mapping))
]

print(f"\nClasses: {len(class_names)}")

# ------------------------------------------------------------
# Load test dataset
# ------------------------------------------------------------

test_df = pd.read_csv(TEST_CSV)

X_test = []
y_test = []

print("\nLoading test sequences...")

for i, row in enumerate(
    test_df.itertuples(index=False),
    start=1
):
    data = np.load(
        Path(row.sequence_path)
    ).astype(np.float32)

    X_test.append(data)
    y_test.append(int(row.class_id))

    if i % 50 == 0 or i == len(test_df):
        print(f"  Loaded {i}/{len(test_df)}")

X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.int32)

print("\nTest shape:")
print(X_test.shape)

# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

print("\nGenerating predictions...")

probabilities = model.predict(
    X_test,
    verbose=1
)

y_pred = np.argmax(
    probabilities,
    axis=1
)

# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)
top3_accuracy = float(np.mean([
    actual in np.argsort(probabilities[index])[::-1][:3]
    for index, actual in enumerate(y_test)
]))

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)
print(f"Top-3 Accuracy: {top3_accuracy * 100:.2f}%")

# ------------------------------------------------------------
# Classification report
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
)

print(report)

# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(len(class_names)))
)

OUTPUT_DIR = Path("models/evaluation")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

np.save(
    OUTPUT_DIR / "confusion_matrix.npy",
    cm
)

# ------------------------------------------------------------
# Save classification report
# ------------------------------------------------------------

report_df = pd.DataFrame(
    classification_report(
        y_test,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
).transpose()

report_df.to_csv(
    OUTPUT_DIR / "classification_report.csv"
)

# ------------------------------------------------------------
# Save predictions
# ------------------------------------------------------------

prediction_df = test_df.copy()

prediction_df["predicted_class_id"] = y_pred

prediction_df["predicted_label"] = [
    class_names[i]
    for i in y_pred
]

prediction_df["confidence"] = np.max(
    probabilities,
    axis=1
)

prediction_df.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
)

# ------------------------------------------------------------
# Most confused pairs
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MOST CONFUSED CLASS PAIRS")
print("=" * 70)

pairs = []

for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm[i, j] > 0:
            pairs.append(
                (
                    int(cm[i, j]),
                    class_names[i],
                    class_names[j]
                )
            )

pairs.sort(
    reverse=True
)

for count, actual, predicted in pairs[:20]:
    print(
        f"{count:3d} | "
        f"Actual: {actual} "
        f"-> Predicted: {predicted}"
    )

# ------------------------------------------------------------
# Save summary
# ------------------------------------------------------------

summary = {
    "test_samples": int(len(y_test)),
    "classes": int(len(class_names)),
    "accuracy": float(accuracy),
    "accuracy_percent": float(accuracy * 100),
    "top3_accuracy": top3_accuracy,
    "top3_accuracy_percent": float(top3_accuracy * 100),
}

with open(
    OUTPUT_DIR / "evaluation_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=2
    )

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    "Confusion matrix : "
    "models/evaluation/confusion_matrix.npy"
)

print(
    "Classification   : "
    "models/evaluation/classification_report.csv"
)

print(
    "Predictions      : "
    "models/evaluation/test_predictions.csv"
)

print(
    "Summary          : "
    "models/evaluation/evaluation_summary.json"
)

print("\nSTATUS: READY FOR VIDEO PREDICTION")