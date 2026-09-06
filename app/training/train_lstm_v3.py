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
# INCLUDE-50 — V3 TARGETED AUGMENTATION LSTM TRAINING
# ============================================================

BASE = Path("data/include50")

TRAIN_CSV = BASE / "train.csv"
VAL_CSV = BASE / "val.csv"
TEST_CSV = BASE / "test.csv"

AUGMENTED_DIR = BASE / "v3_augmented"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL = MODEL_DIR / "include50_lstm_v3_best.keras"
FINAL_MODEL = MODEL_DIR / "include50_lstm_v3_final.keras"

CLASS_MAP = MODEL_DIR / "include50_classes_v3.json"

SEED = 42

EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.0005

SEQUENCE_LENGTH = 45
FEATURES = 258
NUM_CLASSES = 50

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# TARGET CLASSES
# ============================================================

TARGET_CLASSES = {
    "34. Pen",
    "78. Girl",
    "84. small little",
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("INCLUDE-50 V3 TARGETED AUGMENTATION LSTM TRAINING")
print("=" * 70)


# ============================================================
# LOAD CSV
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

print("\nOriginal dataset:")
print(f"Train samples      : {len(train_df)}")
print(f"Validation samples  : {len(val_df)}")
print(f"Test samples        : {len(test_df)}")


# ============================================================
# LOAD ORIGINAL TRAINING SEQUENCES
# ============================================================

def load_original_sequences(df, name):

    X = []
    y = []

    print(f"\nLoading {name}...")

    for i, row in enumerate(
        df.itertuples(index=False),
        start=1
    ):

        path = Path(row.sequence_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Sequence not found: {path}"
            )

        data = np.load(
            path
        ).astype(np.float32)

        if data.shape != (
            SEQUENCE_LENGTH,
            FEATURES
        ):
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

        if (
            i % 100 == 0
            or i == len(df)
        ):
            print(
                f"  Loaded {i}/{len(df)}"
            )

    return (
        np.asarray(
            X,
            dtype=np.float32
        ),
        np.asarray(
            y,
            dtype=np.int32
        )
    )


X_train_original, y_train_original = (
    load_original_sequences(
        train_df,
        "ORIGINAL TRAINING"
    )
)


# ============================================================
# LOAD AUGMENTED TRAINING SEQUENCES
# ============================================================

print("\n" + "=" * 70)
print("LOADING V3 AUGMENTED SEQUENCES")
print("=" * 70)

augmented_files = sorted(
    AUGMENTED_DIR.glob("*.npy")
)

print(
    f"Augmented files found: "
    f"{len(augmented_files)}"
)

if len(augmented_files) == 0:
    raise RuntimeError(
        "No augmented sequences found."
    )


X_augmented = []
y_augmented = []


# Create label lookup from original training CSV.
# The filename contains the class name, so map based
# on the safe class prefix.

class_to_id = {}

for row in train_df.itertuples(index=False):

    class_to_id[row.folder_label] = int(
        row.class_id
    )


def detect_class_id(filename):

    name = filename.lower()

    for class_name in TARGET_CLASSES:

        safe_name = (
            class_name
            .replace(" ", "_")
            .replace(".", "")
            .lower()
        )

        if name.startswith(
            safe_name + "__"
        ):
            return class_to_id[class_name]

    raise RuntimeError(
        f"Could not determine class: "
        f"{filename}"
    )


for i, path in enumerate(
    augmented_files,
    start=1
):

    data = np.load(
        path
    ).astype(np.float32)

    if data.shape != (
        SEQUENCE_LENGTH,
        FEATURES
    ):
        raise ValueError(
            f"Invalid augmented shape "
            f"{data.shape}: {path}"
        )

    if np.isnan(data).any():
        raise ValueError(
            f"NaN detected: {path}"
        )

    if np.isinf(data).any():
        raise ValueError(
            f"Inf detected: {path}"
        )

    class_id = detect_class_id(
        path.name
    )

    X_augmented.append(data)
    y_augmented.append(class_id)

    if (
        i % 50 == 0
        or i == len(augmented_files)
    ):
        print(
            f"  Loaded "
            f"{i}/{len(augmented_files)}"
        )


X_augmented = np.asarray(
    X_augmented,
    dtype=np.float32
)

y_augmented = np.asarray(
    y_augmented,
    dtype=np.int32
)


# ============================================================
# COMBINE ORIGINAL + AUGMENTED TRAINING
# ============================================================

X_train = np.concatenate(
    [
        X_train_original,
        X_augmented,
    ],
    axis=0
)

y_train = np.concatenate(
    [
        y_train_original,
        y_augmented,
    ],
    axis=0
)


# ============================================================
# SHUFFLE TRAINING DATA
# ============================================================

rng = np.random.default_rng(
    SEED
)

indices = rng.permutation(
    len(X_train)
)

X_train = X_train[
    indices
]

y_train = y_train[
    indices
]


# ============================================================
# LOAD VALIDATION / TEST
# ============================================================

X_val, y_val = load_original_sequences(
    val_df,
    "VALIDATION"
)

X_test, y_test = load_original_sequences(
    test_df,
    "TEST"
)


# ============================================================
# DATASET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("V3 DATASET SHAPES")
print("=" * 70)

print(
    "Original training :",
    X_train_original.shape
)

print(
    "Augmented training :",
    X_augmented.shape
)

print(
    "Combined training  :",
    X_train.shape
)

print(
    "Validation          :",
    X_val.shape
)

print(
    "Test                :",
    X_test.shape
)


# ============================================================
# VERIFY SHAPES
# ============================================================

expected_shape = (
    SEQUENCE_LENGTH,
    FEATURES
)

if X_train.shape[1:] != expected_shape:
    raise RuntimeError(
        f"Training shape incorrect: "
        f"{X_train.shape}"
    )

if X_val.shape[1:] != expected_shape:
    raise RuntimeError(
        f"Validation shape incorrect: "
        f"{X_val.shape}"
    )

if X_test.shape[1:] != expected_shape:
    raise RuntimeError(
        f"Test shape incorrect: "
        f"{X_test.shape}"
    )


# ============================================================
# VERIFY 50 CLASSES
# ============================================================

unique_classes = np.unique(
    y_train
)

print(
    "\nUnique training classes:",
    len(unique_classes)
)

if len(unique_classes) != NUM_CLASSES:
    raise RuntimeError(
        f"Expected {NUM_CLASSES} classes, "
        f"found {len(unique_classes)}"
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nV3 training class distribution:")

counts = np.bincount(
    y_train,
    minlength=NUM_CLASSES
)

for class_id in range(NUM_CLASSES):

    label = train_df[
        train_df["class_id"] == class_id
    ]["folder_label"].iloc[0]

    print(
        f"{class_id:2d} | "
        f"{label:25s} | "
        f"{counts[class_id]}"
    )


# ============================================================
# SAVE CLASS MAPPING
# ============================================================

class_mapping = {}

for class_id in sorted(
    train_df["class_id"].unique()
):

    label = train_df[
        train_df["class_id"] == class_id
    ]["folder_label"].iloc[0]

    class_mapping[
        str(int(class_id))
    ] = label


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


print(
    "\nClass mapping saved:",
    CLASS_MAP
)


# ============================================================
# BUILD V3 LSTM
# ============================================================

print("\n" + "=" * 70)
print("BUILDING V3 LSTM MODEL")
print("=" * 70)


model = Sequential(
    [

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
        ),

    ]
)


model.compile(
    optimizer=Adam(
        learning_rate=LEARNING_RATE
    ),

    loss=(
        "sparse_categorical_crossentropy"
    ),

    metrics=[
        "accuracy"
    ],
)


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(

    filepath=str(
        BEST_MODEL
    ),

    monitor="val_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1,
)


early_stopping = EarlyStopping(

    monitor="val_accuracy",

    mode="max",

    patience=15,

    restore_best_weights=True,

    verbose=1,
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=5,

    min_lr=1e-6,

    verbose=1,
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING V3 TRAINING")
print("=" * 70)

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Validation samples:",
    len(X_val)
)

print(
    "Test samples:",
    len(X_test)
)


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
        reduce_lr,
    ],

    verbose=1,
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    str(FINAL_MODEL)
)

print(
    "\nFinal V3 model saved:",
    FINAL_MODEL
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print(
    "\nLoading best V3 model..."
)

best_model = tf.keras.models.load_model(
    str(BEST_MODEL)
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("V3 TEST EVALUATION")
print("=" * 70)

test_loss, test_accuracy = (
    best_model.evaluate(
        X_test,
        y_test,
        verbose=1
    )
)


# ============================================================
# BEST TRAIN / VALIDATION
# ============================================================

best_train_accuracy = max(
    history.history["accuracy"]
)

best_val_accuracy = max(
    history.history["val_accuracy"]
)


# ============================================================
# V2 BASELINE
# ============================================================

V2_ACCURACY = 0.9735


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("V3 TRAINING COMPLETE")
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
    f"V3 Test Accuracy    : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"V2 Baseline         : "
    f"{V2_ACCURACY * 100:.2f}%"
)

difference = (
    test_accuracy - V2_ACCURACY
) * 100

print(
    f"V3 - V2 Difference  : "
    f"{difference:+.2f}%"
)


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\nModels:")

print(
    f"Best  : {BEST_MODEL}"
)

print(
    f"Final : {FINAL_MODEL}"
)

print(
    f"Class : {CLASS_MAP}"
)


# ============================================================
# FINAL STATUS
# ============================================================

if test_accuracy > V2_ACCURACY:

    print(
        "\nRESULT: V3 IMPROVED OVER V2"
    )

elif test_accuracy == V2_ACCURACY:

    print(
        "\nRESULT: V3 MATCHED V2"
    )

else:

    print(
        "\nRESULT: V2 REMAINS BETTER"
    )

print(
    "\nSTATUS: V3 TRAINING COMPLETE"
)