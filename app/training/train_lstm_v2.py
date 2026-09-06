from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam


# ============================================================
# INCLUDE-50 — LSTM V2 TRAINING
# ============================================================

BASE = Path("data/include50")

TRAIN_CSV = BASE / "train_augmented.csv"
VAL_CSV = BASE / "val.csv"
TEST_CSV = BASE / "test.csv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL = (
    MODEL_DIR /
    "include50_lstm_v2_best.keras"
)

FINAL_MODEL = (
    MODEL_DIR /
    "include50_lstm_v2_final.keras"
)

CLASS_MAP = (
    MODEL_DIR /
    "include50_classes_v2.json"
)

SEQUENCE_LENGTH = 45
FEATURES = 258
NUM_CLASSES = 50

SEED = 42

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0005

np.random.seed(SEED)
tf.random.set_seed(SEED)


print("=" * 70)
print("INCLUDE-50 LSTM V2 TRAINING")
print("=" * 70)


# ------------------------------------------------------------
# Load CSVs
# ------------------------------------------------------------

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

print(
    f"Train samples      : {len(train_df)}"
)

print(
    f"Validation samples : {len(val_df)}"
)

print(
    f"Test samples       : {len(test_df)}"
)


# ------------------------------------------------------------
# Load sequences
# ------------------------------------------------------------

def load_sequences(df, name):

    X = []
    y = []

    print()
    print(f"Loading {name}...")

    for i, row in enumerate(
        df.itertuples(index=False),
        start=1
    ):

        path = Path(row.sequence_path)

        data = np.load(path).astype(
            np.float32
        )

        if data.shape != (
            SEQUENCE_LENGTH,
            FEATURES
        ):
            raise ValueError(
                f"Invalid shape {data.shape}: {path}"
            )

        if not np.isfinite(data).all():
            raise ValueError(
                f"Invalid values: {path}"
            )

        X.append(data)
        y.append(int(row.class_id))

        if (
            i % 250 == 0
            or i == len(df)
        ):
            print(
                f"  Loaded {i}/{len(df)}"
            )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int32)
    )


X_train, y_train = load_sequences(
    train_df,
    "TRAIN"
)

X_val, y_val = load_sequences(
    val_df,
    "VALIDATION"
)

X_test, y_test = load_sequences(
    test_df,
    "TEST"
)


# ------------------------------------------------------------
# Dataset verification
# ------------------------------------------------------------

print()
print("=" * 70)
print("DATASET SHAPES")
print("=" * 70)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val  :", X_val.shape)
print("y_val  :", y_val.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ------------------------------------------------------------
# Verify classes
# ------------------------------------------------------------

unique_classes = np.unique(y_train)

print()
print(
    "Training classes:",
    len(unique_classes)
)

if len(unique_classes) != NUM_CLASSES:
    raise RuntimeError(
        f"Expected {NUM_CLASSES} classes, "
        f"found {len(unique_classes)}"
    )


# ------------------------------------------------------------
# Class mapping
# ------------------------------------------------------------

class_mapping = {}

for class_id in sorted(
    train_df["class_id"].unique()
):

    matches = train_df[
        train_df["class_id"] == class_id
    ]

    class_mapping[
        str(int(class_id))
    ] = matches[
        "folder_label"
    ].iloc[0]


with open(
    CLASS_MAP,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        class_mapping,
        f,
        indent=2,
        ensure_ascii=False
    )


print()
print(
    "Class mapping saved:",
    CLASS_MAP
)


# ------------------------------------------------------------
# Build V2 model
# ------------------------------------------------------------

print()
print("=" * 70)
print("BUILDING LSTM V2")
print("=" * 70)


model = Sequential([

    Input(
        shape=(
            SEQUENCE_LENGTH,
            FEATURES
        )
    ),

    LSTM(
        128,
        return_sequences=True
    ),

    BatchNormalization(),

    Dropout(0.25),

    LSTM(
        96,
        return_sequences=True
    ),

    BatchNormalization(),

    Dropout(0.25),

    LSTM(
        64,
        return_sequences=False
    ),

    BatchNormalization(),

    Dropout(0.30),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.30),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.20),

    Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


model.compile(

    optimizer=Adam(
        learning_rate=LEARNING_RATE
    ),

    loss=(
        "sparse_categorical_crossentropy"
    ),

    metrics=[
        "accuracy"
    ]
)


model.summary()


# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------

checkpoint = ModelCheckpoint(

    str(BEST_MODEL),

    monitor="val_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1
)


early_stopping = EarlyStopping(

    monitor="val_accuracy",

    mode="max",

    patience=18,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=6,

    min_lr=1e-6,

    verbose=1
)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

print()
print("=" * 70)
print("STARTING V2 TRAINING")
print("=" * 70)


history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    shuffle=True,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ------------------------------------------------------------
# Save final model
# ------------------------------------------------------------

model.save(
    str(FINAL_MODEL)
)


print()
print(
    "Final V2 model saved:",
    FINAL_MODEL
)


# ------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------

print()
print("=" * 70)
print("V2 TEST EVALUATION")
print("=" * 70)


test_loss, test_accuracy = (
    model.evaluate(
        X_test,
        y_test,
        verbose=1
    )
)


best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_train_accuracy = max(
    history.history["accuracy"]
)


print()
print("=" * 70)
print("V2 TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best Train Accuracy : "
    f"{best_train_accuracy * 100:.2f}%"
)

print(
    f"Best Val Accuracy   : "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Test Accuracy       : "
    f"{test_accuracy * 100:.2f}%"
)

print()
print("Models:")
print(
    f"Best  : {BEST_MODEL}"
)

print(
    f"Final : {FINAL_MODEL}"
)

print()
print("STATUS: LSTM V2 TRAINING COMPLETE")