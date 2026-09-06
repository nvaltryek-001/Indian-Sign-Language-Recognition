from pathlib import Path
import json

import numpy as np
import pandas as pd

import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE
    / "models"
    / "include50_lstm_v2_best.keras"
)

TRAIN_CSV = (
    BASE
    / "data"
    / "include50"
    / "train_augmented.csv"
)

VAL_CSV = (
    BASE
    / "data"
    / "include50"
    / "val.csv"
)

TEST_CSV = (
    BASE
    / "data"
    / "include50"
    / "test.csv"
)

OUTPUT_DIR = (
    BASE
    / "models"
    / "evaluation"
    / "graphs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEQUENCE_LENGTH = 45
FEATURES = 258
NUM_CLASSES = 50


# ============================================================
# PRINT HEADER
# ============================================================

print()
print("=" * 80)
print("ISL INCLUDE-50 FINAL MODEL EVALUATION")
print("=" * 80)

print()
print("PROJECT :", BASE)
print("MODEL   :", MODEL_PATH)
print("OUTPUT  :", OUTPUT_DIR)


# ============================================================
# CHECK FILES
# ============================================================

print()
print("=" * 80)
print("CHECKING FILES")
print("=" * 80)

required_files = [
    MODEL_PATH,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{file_path}"
        )

    print("OK:", file_path)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 80)
print("LOADING MODEL")
print("=" * 80)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")

print()
print("Model input shape :", model.input_shape)
print("Model output shape:", model.output_shape)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

print()
print("=" * 80)
print("LOADING 50 CLASS NAMES")
print("=" * 80)

train_df = pd.read_csv(
    TRAIN_CSV
)

class_table = (
    train_df
    .sort_values("class_id")
    .drop_duplicates("class_id")
)

class_names = (
    class_table["folder_label"]
    .astype(str)
    .tolist()
)

if len(class_names) != NUM_CLASSES:

    raise RuntimeError(
        f"Expected {NUM_CLASSES} classes "
        f"but found {len(class_names)}"
    )

for i, name in enumerate(class_names):

    print(
        f"{i:02d} -> {name}"
    )


# ============================================================
# LOAD SEQUENCES
# ============================================================

def load_sequences(csv_path, dataset_name):

    df = pd.read_csv(csv_path)

    X = []
    y = []

    print()
    print(
        f"Loading {dataset_name}: "
        f"{len(df)} records"
    )

    missing = 0
    invalid = 0

    for row in df.itertuples(index=False):

        sequence_path = BASE / Path(
            row.sequence_path
        )

        if not sequence_path.exists():

            missing += 1

            print(
                "MISSING:",
                sequence_path
            )

            continue

        try:

            data = np.load(
                sequence_path
            ).astype(
                np.float32
            )

        except Exception as error:

            invalid += 1

            print(
                "LOAD ERROR:",
                sequence_path,
                error
            )

            continue

        if data.shape != (
            SEQUENCE_LENGTH,
            FEATURES
        ):

            invalid += 1

            print(
                "INVALID SHAPE:",
                sequence_path,
                data.shape
            )

            continue

        if not np.isfinite(data).all():

            invalid += 1

            print(
                "INVALID VALUES:",
                sequence_path
            )

            continue

        X.append(data)

        y.append(
            int(row.class_id)
        )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int32
    )

    print()
    print(
        dataset_name,
        "loaded successfully."
    )

    print(
        "X shape:",
        X.shape
    )

    print(
        "Y shape:",
        y.shape
    )

    print(
        "Missing:",
        missing
    )

    print(
        "Invalid:",
        invalid
    )

    return X, y


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

X_val, y_val = load_sequences(
    VAL_CSV,
    "VALIDATION"
)


# ============================================================
# LOAD TEST DATA
# ============================================================

X_test, y_test = load_sequences(
    TEST_CSV,
    "TEST"
)


# ============================================================
# DATA CHECK
# ============================================================

print()
print("=" * 80)
print("DATASET SUMMARY")
print("=" * 80)

print(
    f"Validation samples : {len(y_val)}"
)

print(
    f"Test samples       : {len(y_test)}"
)

print(
    f"Validation classes : {len(np.unique(y_val))}"
)

print(
    f"Test classes       : {len(np.unique(y_test))}"
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_dataset(
    X,
    y,
    dataset_name
):

    print()
    print("=" * 80)
    print(
        f"RUNNING {dataset_name} PREDICTION"
    )
    print("=" * 80)

    probabilities = model.predict(
        X,
        verbose=1
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    confidence = np.max(
        probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        y,
        predictions
    )

    mean_confidence = np.mean(
        confidence
    )

    print()
    print(
        f"{dataset_name} ACCURACY    : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"{dataset_name} CONFIDENCE  : "
        f"{mean_confidence * 100:.2f}%"
    )

    return (
        probabilities,
        predictions,
        confidence,
        accuracy,
        mean_confidence
    )


# ============================================================
# VALIDATION PREDICTION
# ============================================================

(
    val_prob,
    val_pred,
    val_confidence,
    val_accuracy,
    val_mean_confidence
) = predict_dataset(
    X_val,
    y_val,
    "VALIDATION"
)


# ============================================================
# TEST PREDICTION
# ============================================================

(
    test_prob,
    test_pred,
    test_confidence,
    test_accuracy,
    test_mean_confidence
) = predict_dataset(
    X_test,
    y_test,
    "TEST"
)


# ============================================================
# PER-SIGN RESULTS
# ============================================================

print()
print("=" * 80)
print("CALCULATING PER-SIGN RESULTS")
print("=" * 80)

results = []

for class_id in range(NUM_CLASSES):

    sign = class_names[class_id]

    # ------------------------------
    # VALIDATION
    # ------------------------------

    val_mask = (
        y_val == class_id
    )

    val_samples = int(
        np.sum(val_mask)
    )

    if val_samples > 0:

        val_correct = int(
            np.sum(
                val_pred[val_mask]
                == class_id
            )
        )

        val_accuracy_class = (
            val_correct
            / val_samples
            * 100
        )

        val_confidence_class = (
            np.mean(
                val_confidence[val_mask]
            )
            * 100
        )

    else:

        val_correct = 0
        val_accuracy_class = np.nan
        val_confidence_class = np.nan

    # ------------------------------
    # TEST
    # ------------------------------

    test_mask = (
        y_test == class_id
    )

    test_samples = int(
        np.sum(test_mask)
    )

    if test_samples > 0:

        test_correct = int(
            np.sum(
                test_pred[test_mask]
                == class_id
            )
        )

        test_accuracy_class = (
            test_correct
            / test_samples
            * 100
        )

        test_confidence_class = (
            np.mean(
                test_confidence[test_mask]
            )
            * 100
        )

    else:

        test_correct = 0
        test_accuracy_class = np.nan
        test_confidence_class = np.nan

    results.append({

        "class_id":
            class_id,

        "sign":
            sign,

        "validation_samples":
            val_samples,

        "validation_correct":
            val_correct,

        "validation_incorrect":
            val_samples - val_correct,

        "validation_accuracy_percent":
            round(
                val_accuracy_class,
                2
            )
            if not np.isnan(
                val_accuracy_class
            )
            else None,

        "validation_confidence_percent":
            round(
                val_confidence_class,
                2
            )
            if not np.isnan(
                val_confidence_class
            )
            else None,

        "test_samples":
            test_samples,

        "test_correct":
            test_correct,

        "test_incorrect":
            test_samples - test_correct,

        "test_accuracy_percent":
            round(
                test_accuracy_class,
                2
            )
            if not np.isnan(
                test_accuracy_class
            )
            else None,

        "test_confidence_percent":
            round(
                test_confidence_class,
                2
            )
            if not np.isnan(
                test_confidence_class
            )
            else None
    })


results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE CSV
# ============================================================

csv_output = (
    OUTPUT_DIR
    / "50_sign_validation_test_confidence.csv"
)

results_df.to_csv(
    csv_output,
    index=False
)

print()
print(
    "CSV SAVED:"
)

print(
    csv_output
)


# ============================================================
# GRAPH 1
# VALIDATION + TEST ACCURACY
# ============================================================

print()
print(
    "Creating Graph 1..."
)

plt.figure(
    figsize=(24, 10)
)

x = np.arange(
    NUM_CLASSES
)

width = 0.38

plt.bar(
    x - width / 2,
    results_df[
        "validation_accuracy_percent"
    ],
    width,
    label="Validation Accuracy"
)

plt.bar(
    x + width / 2,
    results_df[
        "test_accuracy_percent"
    ],
    width,
    label="Test Accuracy"
)

plt.axhline(
    95,
    linestyle="--",
    label="95% Target"
)

plt.axhline(
    98,
    linestyle=":",
    label="98% Target"
)

plt.title(
    "ISL INCLUDE-50 — Validation vs Test Accuracy for All 50 Signs",
    fontsize=18
)

plt.xlabel(
    "ISL Sign",
    fontsize=13
)

plt.ylabel(
    "Accuracy (%)",
    fontsize=13
)

plt.xticks(
    x,
    results_df["sign"],
    rotation=90,
    fontsize=8
)

plt.ylim(
    0,
    105
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.legend()

plt.tight_layout()

graph1 = (
    OUTPUT_DIR
    / "01_50sign_validation_vs_test_accuracy.png"
)

plt.savefig(
    graph1,
    dpi=250,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 2
# VALIDATION CONFIDENCE
# ============================================================

print(
    "Creating Graph 2..."
)

plt.figure(
    figsize=(24, 10)
)

plt.bar(
    x,
    results_df[
        "validation_confidence_percent"
    ]
)

plt.axhline(
    95,
    linestyle="--",
    label="95% Target"
)

plt.axhline(
    98,
    linestyle=":",
    label="98% Target"
)

plt.title(
    "ISL INCLUDE-50 — Validation Confidence for All 50 Signs",
    fontsize=18
)

plt.xlabel(
    "ISL Sign"
)

plt.ylabel(
    "Mean Confidence (%)"
)

plt.xticks(
    x,
    results_df["sign"],
    rotation=90,
    fontsize=8
)

plt.ylim(
    0,
    105
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.legend()

plt.tight_layout()

graph2 = (
    OUTPUT_DIR
    / "02_50sign_validation_confidence.png"
)

plt.savefig(
    graph2,
    dpi=250,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 3
# TEST CONFIDENCE
# ============================================================

print(
    "Creating Graph 3..."
)

plt.figure(
    figsize=(24, 10)
)

plt.bar(
    x,
    results_df[
        "test_confidence_percent"
    ]
)

plt.axhline(
    95,
    linestyle="--",
    label="95% Target"
)

plt.axhline(
    98,
    linestyle=":",
    label="98% Target"
)

plt.title(
    "ISL INCLUDE-50 — Test Confidence for All 50 Signs",
    fontsize=18
)

plt.xlabel(
    "ISL Sign"
)

plt.ylabel(
    "Mean Confidence (%)"
)

plt.xticks(
    x,
    results_df["sign"],
    rotation=90,
    fontsize=8
)

plt.ylim(
    0,
    105
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.legend()

plt.tight_layout()

graph3 = (
    OUTPUT_DIR
    / "03_50sign_test_confidence.png"
)

plt.savefig(
    graph3,
    dpi=250,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 4
# COMBINED ACCURACY + CONFIDENCE
# ============================================================

print(
    "Creating Graph 4..."
)

plt.figure(
    figsize=(26, 11)
)

plt.plot(
    x,
    results_df[
        "validation_accuracy_percent"
    ],
    marker="o",
    linewidth=2,
    label="Validation Accuracy"
)

plt.plot(
    x,
    results_df[
        "test_accuracy_percent"
    ],
    marker="s",
    linewidth=2,
    label="Test Accuracy"
)

plt.plot(
    x,
    results_df[
        "validation_confidence_percent"
    ],
    marker="^",
    linewidth=2,
    label="Validation Confidence"
)

plt.plot(
    x,
    results_df[
        "test_confidence_percent"
    ],
    marker="D",
    linewidth=2,
    label="Test Confidence"
)

plt.axhline(
    95,
    linestyle="--",
    label="95% Target"
)

plt.axhline(
    98,
    linestyle=":",
    label="98% Target"
)

plt.title(
    "ISL INCLUDE-50 — Accuracy and Confidence Across All 50 Signs",
    fontsize=19
)

plt.xlabel(
    "ISL Sign",
    fontsize=13
)

plt.ylabel(
    "Percentage (%)",
    fontsize=13
)

plt.xticks(
    x,
    results_df["sign"],
    rotation=90,
    fontsize=8
)

plt.ylim(
    0,
    105
)

plt.grid(
    alpha=0.25
)

plt.legend(
    loc="lower left"
)

plt.tight_layout()

graph4 = (
    OUTPUT_DIR
    / "04_50sign_accuracy_confidence_combined.png"
)

plt.savefig(
    graph4,
    dpi=250,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 5
# CORRECT VS INCORRECT — VALIDATION
# ============================================================

print(
    "Creating Graph 5..."
)

plt.figure(
    figsize=(24, 10)
)

plt.bar(
    x - width / 2,
    results_df[
        "validation_correct"
    ],
    width,
    label="Correct"
)

plt.bar(
    x + width / 2,
    results_df[
        "validation_incorrect"
    ],
    width,
    label="Incorrect"
)

plt.title(
    "ISL INCLUDE-50 — Validation Correct vs Incorrect Predictions",
    fontsize=18
)

plt.xlabel(
    "ISL Sign"
)

plt.ylabel(
    "Number of Samples"
)

plt.xticks(
    x,
    results_df["sign"],
    rotation=90,
    fontsize=8
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.legend()

plt.tight_layout()

graph5 = (
    OUTPUT_DIR
    / "05_50sign_validation_correct_incorrect.png"
)

plt.savefig(
    graph5,
    dpi=250,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 6
# CORRECT VS INCORRECT — TEST
# ============================================================

print(
    "Creating Graph 6..."
)

plt.figure(
    figsize=(24, 10)
)

plt.bar(
    x - width / 2,
    results_df[
        "test_correct"
    ],
    width,
    label="Correct"
)

plt.bar(
    x + width / 2,
    results_df[
        "test_incorrect"
    ],
    width,
    label="Incorrect"
)

plt.title(
    "ISL INCLUDE-50 — Test Correct vs Incorrect Predictions",
    fontsize=18
)

plt.xlabel(
    "ISL Sign"
)

plt.ylabel(
    "Number of Samples"
)

plt.xticks(
    x,
    results_df["sign"],
    rotation=90,
    fontsize=8
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.legend()

plt.tight_layout()

graph6 = (
    OUTPUT_DIR
    / "06_50sign_test_correct_incorrect.png"
)

plt.savefig(
    graph6,
    dpi=250,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# OVERALL GRAPH
# ============================================================

print(
    "Creating Overall Graph..."
)

plt.figure(
    figsize=(10, 7)
)

labels = [
    "Validation\nAccuracy",
    "Test\nAccuracy",
    "Validation\nConfidence",
    "Test\nConfidence"
]

values = [
    val_accuracy * 100,
    test_accuracy * 100,
    val_mean_confidence * 100,
    test_mean_confidence * 100
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
    linestyle=":",
    label="98% Target"
)

plt.title(
    "ISL INCLUDE-50 — Overall Model Performance",
    fontsize=18
)

plt.ylabel(
    "Percentage (%)"
)

plt.ylim(
    0,
    105
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.legend()

plt.tight_layout()

overall_graph = (
    OUTPUT_DIR
    / "07_overall_model_performance.png"
)

plt.savefig(
    overall_graph,
    dpi=250,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {

    "project":
        "ISL INCLUDE-50 Sign Recognition",

    "model":
        str(MODEL_PATH),

    "num_classes":
        NUM_CLASSES,

    "sequence_length":
        SEQUENCE_LENGTH,

    "features":
        FEATURES,

    "validation_samples":
        int(len(y_val)),

    "test_samples":
        int(len(y_test)),

    "validation_accuracy_percent":
        round(
            val_accuracy * 100,
            2
        ),

    "test_accuracy_percent":
        round(
            test_accuracy * 100,
            2
        ),

    "validation_mean_confidence_percent":
        round(
            val_mean_confidence * 100,
            2
        ),

    "test_mean_confidence_percent":
        round(
            test_mean_confidence * 100,
            2
        )
}

summary_path = (
    OUTPUT_DIR
    / "final_50sign_evaluation_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# PRINT FINAL TABLE
# ============================================================

print()
print("=" * 80)
print("FINAL 50-SIGN RESULTS")
print("=" * 80)

display_columns = [

    "class_id",

    "sign",

    "validation_samples",

    "validation_correct",

    "validation_accuracy_percent",

    "validation_confidence_percent",

    "test_samples",

    "test_correct",

    "test_accuracy_percent",

    "test_confidence_percent"
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 80)
print("OVERALL RESULTS")
print("=" * 80)

print(
    f"Validation Accuracy   : "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"Test Accuracy         : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Validation Confidence : "
    f"{val_mean_confidence * 100:.2f}%"
)

print(
    f"Test Confidence       : "
    f"{test_mean_confidence * 100:.2f}%"
)


# ============================================================
# OUTPUT FILES
# ============================================================

print()
print("=" * 80)
print("FILES GENERATED")
print("=" * 80)

for file_path in [

    graph1,
    graph2,
    graph3,
    graph4,
    graph5,
    graph6,
    overall_graph,
    csv_output,
    summary_path

]:

    print(
        file_path
    )


print()
print("=" * 80)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 80)
print()