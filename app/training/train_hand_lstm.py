"""Train the INCLUDE-50 hand-only 50-class LSTM model."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input, LSTM
from tensorflow.keras.optimizers import Adam


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/include50"
HAND_SEQUENCE_DIR = BASE / "hand_sequences"
MODEL_DIR = ROOT / "models"
BEST_MODEL = MODEL_DIR / "include50_hand_lstm_best.keras"
FINAL_MODEL = MODEL_DIR / "include50_hand_lstm_final.keras"
CLASS_MAP = MODEL_DIR / "include50_hand_classes.json"
SEQUENCE_LENGTH = 45
FEATURES = 126
NUM_CLASSES = 50
SEED = 42


def hand_sequence_path(video_path):
    return HAND_SEQUENCE_DIR / Path(video_path).with_suffix(".npy")


def load_sequences(csv_path):
    frame = pd.read_csv(csv_path)
    sequences = []
    labels = []
    missing = []
    for row in frame.itertuples(index=False):
        path = hand_sequence_path(row.video_path)
        if not path.exists():
            missing.append(str(path))
            continue
        sequence = np.load(path)
        if sequence.shape != (SEQUENCE_LENGTH, FEATURES) or sequence.dtype != np.float32:
            raise ValueError(f"Invalid hand sequence: {path} {sequence.shape} {sequence.dtype}")
        if not np.isfinite(sequence).all():
            raise ValueError(f"Non-finite hand sequence: {path}")
        sequences.append(sequence)
        labels.append(int(row.class_id))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} hand sequences, first: {missing[0]}")
    return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.int32), frame


def build_model():
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, FEATURES)),
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.25),
        LSTM(96),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_x, train_y, train_df = load_sequences(BASE / "train.csv")
    val_x, val_y, val_df = load_sequences(BASE / "val.csv")
    test_x, test_y, _ = load_sequences(BASE / "test.csv")
    if len(np.unique(train_y)) != NUM_CLASSES:
        raise ValueError(f"Expected 50 training classes, found {len(np.unique(train_y))}")

    class_map = {
        str(class_id): train_df.loc[train_df["class_id"] == class_id, "folder_label"].iloc[0]
        for class_id in sorted(train_df["class_id"].unique())
    }
    CLASS_MAP.write_text(json.dumps(class_map, indent=2, ensure_ascii=False), encoding="utf-8")

    model = build_model()
    callbacks = [
        ModelCheckpoint(str(BEST_MODEL), monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", mode="max", patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1),
    ]
    print(f"Train: {train_x.shape}; validation: {val_x.shape}; test: {test_x.shape}")
    print(f"Classes: {len(class_map)}")
    model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    model.save(FINAL_MODEL)
    best_model = tf.keras.models.load_model(BEST_MODEL)
    print(f"Best model: {BEST_MODEL}")
    print(f"Final model: {FINAL_MODEL}")
    print(f"Test loss/accuracy: {best_model.evaluate(test_x, test_y, verbose=0)}")


if __name__ == "__main__":
    main()