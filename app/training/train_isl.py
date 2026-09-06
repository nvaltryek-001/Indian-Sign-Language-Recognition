import os
import json
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = "data/sequences"
MODEL_DIR = "app/models"

ACTIONS = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])

print("=" * 60)
print("ISL LSTM TRAINING")
print("=" * 60)
print("Classes:", ACTIONS)

X = []
y = []

for label, action in enumerate(ACTIONS):

    folder = os.path.join(DATA_DIR, action)

    files = sorted([
        f for f in os.listdir(folder)
        if f.endswith(".npy")
    ])

    print(action, ":", len(files), "samples")

    for file in files:

        data = np.load(
            os.path.join(folder, file)
        )

        if data.shape != (45, 258):
            print("Skipping invalid:", file, data.shape)
            continue

        X.append(data)
        y.append(label)

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int32)

print()
print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

# ------------------------------------------------------------
# Train / test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training:", X_train.shape)
print("Testing :", X_test.shape)

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(45, 258)
    ),

    tf.keras.layers.LSTM(
        64,
        return_sequences=True
    ),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.LSTM(
        128,
        return_sequences=True
    ),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.LSTM(
        64
    ),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(
        len(ACTIONS),
        activation="softmax"
    )
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=15,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(
            MODEL_DIR,
            "isl_14class.keras"
        ),
        monitor="val_accuracy",
        save_best_only=True
    )
]

# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=80,
    batch_size=8,
    callbacks=callbacks,
    verbose=1
)

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

model = tf.keras.models.load_model(
    os.path.join(
        MODEL_DIR,
        "isl_14class.keras"
    )
)

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print()
print("=" * 60)
print("FINAL TEST RESULT")
print("=" * 60)
print("Test accuracy:", round(accuracy * 100, 2), "%")
print("Test loss:", round(loss, 4))

predictions = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    predictions,
    axis=1
)

print()
print("CLASSIFICATION REPORT")
print()

print(
    classification_report(
        y_test,
        y_pred,
        target_names=ACTIONS,
        zero_division=0
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

np.save(
    "results/confusion_matrix.npy",
    cm
)

with open(
    os.path.join(
        MODEL_DIR,
        "labels.json"
    ),
    "w"
) as f:

    json.dump(
        ACTIONS,
        f,
        indent=2
    )

with open(
    "results/test_accuracy.txt",
    "w"
) as f:

    f.write(
        "Test Accuracy: "
        + str(round(accuracy * 100, 2))
        + "%\n"
    )

print()
print("MODEL SAVED:")
print("app/models/isl_14class.keras")
print("app/models/labels.json")
print()
print("TRAINING COMPLETE")
