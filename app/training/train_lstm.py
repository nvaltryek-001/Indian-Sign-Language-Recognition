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
# INCLUDE-50 — 50 CLASS LSTM TRAINING
# ============================================================

BASE = Path("data/include50")

TRAIN_CSV = BASE / "train.csv"
VAL_CSV = BASE / "val.csv"
TEST_CSV = BASE / "test.csv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL = MODEL_DIR / "include50_lstm_best.keras"
FINAL_MODEL = MODEL_DIR / "include50_lstm_final.keras"
CLASS_MAP = MODEL_DIR / "include50_classes.json"

SEED = 42
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001

np.random.seed(SEED)
tf.random.set_seed(SEED)


print("=" * 70)
print("INCLUDE-50 LSTM TRAINING")
print("=" * 70)


# ============================================================
# LOAD DATASET CSV FILES
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

print(f"Train samples      : {len(train_df)}")
print(f"Validation samples  : {len(val_df)}")
print(f"Test samples        : {len(test_df)}")


# ============================================================
# LOAD SEQUENCES
# ============================================================

def load_sequences(df, name):
    X = []
    y = []

    print(f"\nLoading {name}...")

    for i, row in enumerate(df.itertuples(index=False), start=1):

        path = Path(row.sequence_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Sequence not found: {path}"
            )

        data = np.load(path).astype(np.float32)

        if data.shape != (45, 258):
            raise ValueError(
                f"Invalid shape {data.shape}: {path}"
            )

        if np.isnan(data).any():
            raise ValueError(
                f"NaN detected: {path}"
            )

        if np.isinf(data).any():
            raise ValueError(
                f"Inf detected: {path}"
            )

        X.append(data)
        y.append(int(row.class_id))

        if i % 100 == 0 or i == len(df):
            print(f"  Loaded {i}/{len(df)}")

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


# ============================================================
# DATASET VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET SHAPES")
print("=" * 70)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val  :", X_val.shape)
print("y_val  :", y_val.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


if X_train.shape[1:] != (45, 258):
    raise RuntimeError(
        "Training input shape is incorrect."
    )

if X_val.shape[1:] != (45, 258):
    raise RuntimeError(
        "Validation input shape is incorrect."
    )

if X_test.shape[1:] != (45, 258):
    raise RuntimeError(
        "Test input shape is incorrect."
    )


# ============================================================
# NUMBER OF CLASSES
# ============================================================

NUM_CLASSES = 50

train_classes = np.unique(y_train)
val_classes = np.unique(y_val)
test_classes = np.unique(y_test)

print("\nTraining classes   :", len(train_classes))
print("Validation classes :", len(val_classes))
print("Test classes       :", len(test_classes))

if len(train_classes) != NUM_CLASSES:
    raise RuntimeError(
        f"Expected 50 training classes, "
        f"found {len(train_classes)}"
    )

if len(val_classes) != NUM_CLASSES:
    raise RuntimeError(
        f"Expected 50 validation classes, "
        f"found {len(val_classes)}"
    )

if len(test_classes) != NUM_CLASSES:
    raise RuntimeError(
        f"Expected 50 test classes, "
        f"found {len(test_classes)}"
    )


# ============================================================
# SAVE CLASS MAPPING
# ============================================================

class_mapping = {}

for class_id in range(NUM_CLASSES):

    rows = train_df[
        train_df["class_id"] == class_id
    ]

    if len(rows) == 0:
        raise RuntimeError(
            f"Class {class_id} missing from training data."
        )

    label = rows["folder_label"].iloc[0]

    class_mapping[str(class_id)] = label


with open(
    str(CLASS_MAP),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        class_mapping,
        f,
        indent=2,
        ensure_ascii=False
    )


print("\nClass mapping saved:")
print(CLASS_MAP)


# ============================================================
# BUILD LSTM MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING LSTM MODEL")
print("=" * 70)

model = Sequential([

    Input(
        shape=(45, 258)
    ),

    LSTM(
        128,
        return_sequences=True
    ),

    BatchNormalization(),

    Dropout(0.30),

    LSTM(
        64,
        return_sequences=False
    ),

    BatchNormalization(),

    Dropout(0.30),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.30),

    Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


model.compile(
    optimizer=Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(

    filepath=str(BEST_MODEL),

    monitor="val_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1
)


early_stopping = EarlyStopping(

    monitor="val_accuracy",

    mode="max",

    patience=15,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=5,

    min_lr=1e-6,

    verbose=1
)


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print(f"Epochs     : {EPOCHS}")
print(f"Batch size : {BATCH_SIZE}")
print(f"Learning rate : {LEARNING_RATE}")
print()


history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    str(FINAL_MODEL)
)

print("\nFinal model saved:")
print(FINAL_MODEL)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best validation model...")

best_model = tf.keras.models.load_model(
    str(BEST_MODEL)
)

print("Best model loaded successfully.")


# ============================================================
# TEST SET EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("TEST SET EVALUATION")
print("=" * 70)

test_loss, test_accuracy = best_model.evaluate(

    X_test,

    y_test,

    verbose=1
)


# ============================================================
# TRAINING STATISTICS
# ============================================================

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_train_accuracy = max(
    history.history["accuracy"]
)


best_epoch = (
    int(
        np.argmax(
            history.history["val_accuracy"]
        )
    )
    + 1
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best Epoch          : {best_epoch}"
)

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

print(
    f"Test Loss           : "
    f"{test_loss:.4f}"
)

print("\nModels:")
print(
    f"Best  : {BEST_MODEL}"
)

print(
    f"Final : {FINAL_MODEL}"
)

print("\nClass mapping:")
print(CLASS_MAP)

print("\nSTATUS: LSTM TRAINING COMPLETE")