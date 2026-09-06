import cv2
import numpy as np
import tensorflow as tf
import json
from collections import deque

MODEL_PATH = "models/include50_lstm_best.keras"
LABELS_PATH = "models/include50_classes_v2.json"

SEQUENCE_LENGTH = 45
CONFIDENCE_THRESHOLD = 0.70

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)

# JSON keys are class IDs
labels = {int(k): v for k, v in labels.items()}

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    raise SystemExit

print("WEBCAM OPENED")
print("Show the sign/action to the camera.")
print("Press Q to quit.")

sequence = deque(maxlen=SEQUENCE_LENGTH)

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    frame = cv2.flip(frame, 1)

    # ---------------------------------------------------------
    # TEMPORARY TEST:
    # We need MediaPipe landmark extraction here.
    # ---------------------------------------------------------
    #
    # For now this creates a zero feature vector.
    # This is ONLY to verify webcam access.
    #
    features = np.zeros(258, dtype=np.float32)

    sequence.append(features)

    prediction_text = "Collecting frames..."

    if len(sequence) == SEQUENCE_LENGTH:

        X = np.expand_dims(
            np.asarray(sequence, dtype=np.float32),
            axis=0
        )

        prediction = model.predict(X, verbose=0)[0]

        class_id = int(np.argmax(prediction))
        confidence = float(prediction[class_id])

        label = labels.get(
            class_id,
            f"Class {class_id}"
        )

        if confidence >= CONFIDENCE_THRESHOLD:
            prediction_text = (
                f"{label} - "
                f"{confidence * 100:.1f}%"
            )
        else:
            prediction_text = (
                f"No confident sign "
                f"({confidence * 100:.1f}%)"
            )

    cv2.putText(
        frame,
        prediction_text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "DOG WEBCAM TEST | Q = Quit",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "ISL Dog Webcam Test",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()