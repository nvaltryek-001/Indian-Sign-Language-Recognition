from pathlib import Path
import glob
import json
import shutil

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam


# ============================================================
# INCLUDE-50 WEBCAM DOMAIN ADAPTATION
# ============================================================

MODEL = Path("models/include50_lstm_v2_best.keras")
BACKUP = Path("models/include50_lstm_v2_best_backup.keras")

WEBCAM_DIR = Path(
    "data/include50/webcam_sequences/1. loud"
)

OUTPUT_MODEL = Path(
    "models/include50_lstm_v2_webcam_adapted.keras"
)

SEQUENCE_LENGTH = 45
FEATURES = 258

# class_id for "1. loud"
LOUD_CLASS_ID = 0

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


print("=" * 70)
print("INCLUDE-50 WEBCAM DOMAIN ADAPTATION")
print("=" * 70)


# ------------------------------------------------------------
# Safety checks
# ------------------------------------------------------------

if not MODEL.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL}"
    )

if not BACKUP.exists():
    raise FileNotFoundError(
        f"Backup not found: {BACKUP}"
    )


# ------------------------------------------------------------
# Load webcam samples
# ------------------------------------------------------------

files = sorted(
    glob.glob(
        str(WEBCAM_DIR / "*.npy")
    )
)

print()
print("Webcam samples:", len(files))

if len(files) < 10:
    raise RuntimeError(
        "Too few webcam samples."
    )


X = []
y = []


for f in files:

    data = np.load(f).astype(
        np.float32
    )

    if data.shape != (
        SEQUENCE_LENGTH,
        FEATURES
    ):
        raise ValueError(
            f"Invalid shape {data.shape}: {f}"
        )

    if not np.isfinite(data).all():
        raise ValueError(
            f"Invalid values: {f}"
        )

    X.append(data)
    y.append(LOUD_CLASS_ID)


X = np.asarray(
    X,
    dtype=np.float32
)

y = np.asarray(
    y,
    dtype=np.int32
)


print("X shape:", X.shape)
print("y shape:", y.shape)


# ------------------------------------------------------------
# Load original model
# ------------------------------------------------------------

print()
print("Loading original V2 model...")

model = load_model(
    MODEL
)

print("Model loaded successfully.")


# ------------------------------------------------------------
# Freeze most of the network
#
# Only the final Dense layers are adapted initially.
# This reduces the risk of destroying the existing
# 50-class knowledge.
# ------------------------------------------------------------

for layer in model.layers:
    layer.trainable = False


trainable_names = []

for layer in model.layers[-4:]:
    layer.trainable = True
    trainable_names.append(layer.name)


print()
print("=" * 70)
print("TRAINABLE LAYERS")
print("=" * 70)

for name in trainable_names:
    print(name)


# ------------------------------------------------------------
# Compile with very small learning rate
# ------------------------------------------------------------

model.compile(
    optimizer=Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------

checkpoint = ModelCheckpoint(
    str(OUTPUT_MODEL),
    monitor="loss",
    mode="min",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="loss",
    mode="min",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="loss",
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)


# ------------------------------------------------------------
# Adaptation
# ------------------------------------------------------------

print()
print("=" * 70)
print("STARTING WEBCAM ADAPTATION")
print("=" * 70)

history = model.fit(
    X,
    y,
    epochs=30,
    batch_size=4,
    shuffle=True,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],
    verbose=1
)


# ------------------------------------------------------------
# Save adapted model
# ------------------------------------------------------------

model.save(
    str(OUTPUT_MODEL)
)


print()
print("=" * 70)
print("ADAPTATION COMPLETE")
print("=" * 70)

print(
    "Original model :",
    MODEL
)

print(
    "Backup model   :",
    BACKUP
)

print(
    "Adapted model  :",
    OUTPUT_MODEL
)

print()
print(
    "STATUS: WEBCAM ADAPTED MODEL CREATED"
)
print("=" * 70)
