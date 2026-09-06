from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# CONFIG
# ============================================================

BASE = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE / "models" / "include50_lstm_v2_best.keras"

TRAIN_CSV = BASE / "data/include50/train_augmented.csv"
VAL_CSV = BASE / "data/include50/val.csv"
TEST_CSV = BASE / "data/include50/test.csv"

RESULTS = BASE / "models/evaluation/graphs"
RESULTS.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 45
FEATURES = 258
NUM_CLASSES = 50


# ============================================================
# START
# ============================================================

print("=" * 70)
print("INCLUDE-50 MODEL EVALUATION + GRAPH GENERATION")
print("=" * 70)

print()
print("Model:", MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

print()
print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# LOAD DATA
# ============================================================

def load_sequences(csv_path, name):

    df = pd.read_csv(csv_path)

    X = []
    y = []

    print()
    print(f"Loading {name}: {len(df)} records")

    for i, row in enumerate(
        df.itertuples(index=False),
        start=1
    ):

        path = BASE / Path(row.sequence_path)

        if not path.exists():
            print("MISSING:", path)
            continue

        data = np.load(path).astype(np.float32)

        if data.shape != (
            SEQUENCE_LENGTH,
            FEATURES
        ):
            print(
                "INVALID:",
                path,
                data.shape
            )
            continue

        if not np.isfinite(data).all():
            print(
                "INVALID VALUES:",
                path
            )
            continue

        X.append(data)
        y.append(int(row.class_id))

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int32),
        df
    )


X_val, y_val, val_df = load_sequences(
    VAL_CSV,
    "VALIDATION"
)

X_test, y_test, test_df = load_sequences(
    TEST_CSV,
    "TEST"
)


print()
print("=" * 70)
print("DATA SHAPES")
print("=" * 70)

print("Validation:", X_val.shape, y_val.shape)
print("Test      :", X_test.shape, y_test.shape)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = (
    train_df := pd.read_csv(TRAIN_CSV)
).sort_values("class_id").drop_duplicates(
    "class_id"
)["folder_label"].tolist()


if len(class_names) != NUM_CLASSES:
    raise RuntimeError(
        f"Expected 50 classes, found {len(class_names)}"
    )


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print()
print("Running validation prediction...")

val_prob = model.predict(
    X_val,
    verbose=1
)

val_pred = np.argmax(
    val_prob,
    axis=1
)

val_confidence = np.max(
    val_prob,
    axis=1
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

print()
print("Running test prediction...")

test_prob = model.predict(
    X_test,
    verbose=1
)

test_pred = np.argmax(
    test_prob,
    axis=1
)

test_confidence = np.max(
    test_prob,
    axis=1
)


# ============================================================
# OVERALL RESULTS
# ============================================================

val_accuracy = accuracy_score(
    y_val,
    val_pred
)

test_accuracy = accuracy_score(
    y_test,
    test_pred
)

val_mean_confidence = np.mean(
    val_confidence
)

test_mean_confidence = np.mean(
    test_confidence
)


print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    f"Validation Accuracy : {val_accuracy * 100:.2f}%"
)

print(
    f"Test Accuracy       : {test_accuracy * 100:.2f}%"
)

print(
    f"Validation Confidence: "
    f"{val_mean_confidence * 100:.2f}%"
)

print(
    f"Test Confidence      : "
    f"{test_mean_confidence * 100:.2f}%"
)


# ============================================================
# PER-CLASS RESULTS
# ============================================================

rows = []

for class_id in range(NUM_CLASSES):

    mask = y_test == class_id

    count = int(np.sum(mask))

    if count == 0:
        continue

    correct = int(
        np.sum(
            test_pred[mask] == class_id
        )
    )

    accuracy = correct / count

    confidence = float(
        np.mean(
            test_confidence[mask]
        )
    )

    rows.append({

        "class_id": class_id,

        "sign": class_names[class_id],

        "test_samples": count,

        "correct": correct,

        "incorrect": count - correct,

        "accuracy_percent":
            accuracy * 100,

        "mean_confidence_percent":
            confidence * 100
    })


results_df = pd.DataFrame(rows)

results_df.to_csv(
    RESULTS / "50_sign_per_class_results.csv",
    index=False
)


# ============================================================
# GRAPH 1 — PER SIGN ACCURACY
# ============================================================

plt.figure(figsize=(20, 9))

plt.bar(
    results_df["sign"],
    results_df["accuracy_percent"]
)

plt.axhline(
    95,
    linestyle="--",
    label="95% Target"
)

plt.axhline(
    98,
    linestyle="--",
    label="98% Target"
)

plt.title(
    "INCLUDE-50 — Test Accuracy for All 50 Signs"
)

plt.xlabel("Sign")

plt.ylabel("Accuracy (%)")

plt.xticks(
    rotation=90,
    fontsize=8
)

plt.ylim(0, 105)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS / "01_per_sign_accuracy.png",
    dpi=200
)

plt.close()


# ============================================================
# GRAPH 2 — PER SIGN CONFIDENCE
# ============================================================

plt.figure(figsize=(20, 9))

plt.bar(
    results_df["sign"],
    results_df["mean_confidence_percent"]
)

plt.axhline(
    95,
    linestyle="--",
    label="95% Confidence Target"
)

plt.axhline(
    98,
    linestyle="--",
    label="98% Confidence Target"
)

plt.title(
    "INCLUDE-50 — Mean Prediction Confidence for All 50 Signs"
)

plt.xlabel("Sign")

plt.ylabel("Confidence (%)")

plt.xticks(
    rotation=90,
    fontsize=8
)

plt.ylim(0, 105)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS / "02_per_sign_confidence.png",
    dpi=200
)

plt.close()


# ============================================================
# GRAPH 3 — CORRECT VS INCORRECT
# ============================================================

plt.figure(figsize=(20, 9))

x = np.arange(
    len(results_df)
)

width = 0.4

plt.bar(
    x - width / 2,
    results_df["correct"],
    width,
    label="Correct"
)

plt.bar(
    x + width / 2,
    results_df["incorrect"],
    width,
    label="Incorrect"
)

plt.title(
    "INCLUDE-50 — Correct vs Incorrect Predictions"
)

plt.xlabel("Sign")

plt.ylabel("Number of Test Samples")

plt.xticks(
    x,
    results_df["sign"],
    rotation=90,
    fontsize=8
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS / "03_correct_vs_incorrect.png",
    dpi=200
)

plt.close()


# ============================================================
# GRAPH 4 — OVERALL ACCURACY
# ============================================================

plt.figure(figsize=(8, 6))

labels = [
    "Validation",
    "Test"
]

values = [
    val_accuracy * 100,
    test_accuracy * 100
]

plt.bar(
    labels,
    values
)

plt.axhline(
    95,
    linestyle="--",
    label="95% Target"
)

plt.axhline(
    98,
    linestyle="--",
    label="98% Target"
)

plt.title(
    "INCLUDE-50 — Overall Model Accuracy"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.ylim(0, 105)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS / "04_overall_accuracy.png",
    dpi=200
)

plt.close()


# ============================================================
# GRAPH 5 — CONFIDENCE DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 6))

plt.hist(
    test_confidence * 100,
    bins=20
)

plt.axvline(
    95,
    linestyle="--",
    label="95%"
)

plt.title(
    "INCLUDE-50 — Test Prediction Confidence Distribution"
)

plt.xlabel(
    "Confidence (%)"
)

plt.ylabel(
    "Number of Predictions"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS / "05_confidence_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {

    "model":
        str(MODEL_PATH),

    "num_classes":
        NUM_CLASSES,

    "validation_samples":
        int(len(y_val)),

    "test_samples":
        int(len(y_test)),

    "validation_accuracy_percent":
        round(val_accuracy * 100, 2),

    "test_accuracy_percent":
        round(test_accuracy * 100, 2),

    "validation_mean_confidence_percent":
        round(val_mean_confidence * 100, 2),

    "test_mean_confidence_percent":
        round(test_mean_confidence * 100, 2)
}


with open(
    RESULTS / "evaluation_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# PRINT PER-SIGN TABLE
# ============================================================

print()
print("=" * 70)
print("PER-SIGN TEST RESULTS")
print("=" * 70)

print(
    results_df[
        [
            "class_id",
            "sign",
            "test_samples",
            "correct",
            "incorrect",
            "accuracy_percent",
            "mean_confidence_percent"
        ]
    ].to_string(index=False)
)


print()
print("=" * 70)
print("GRAPHS GENERATED")
print("=" * 70)

print(
    RESULTS / "01_per_sign_accuracy.png"
)

print(
    RESULTS / "02_per_sign_confidence.png"
)

print(
    RESULTS / "03_correct_vs_incorrect.png"
)

print(
    RESULTS / "04_overall_accuracy.png"
)

print(
    RESULTS / "05_confidence_distribution.png"
)

print()
print(
    "CSV:",
    RESULTS / "50_sign_per_class_results.csv"
)

print(
    "SUMMARY:",
    RESULTS / "evaluation_summary.json"
)

print()
print("DONE.")