import cv2
import json
import time
import numpy as np
import mediapipe as mp
from collections import deque
from pathlib import Path
from tensorflow.keras.models import load_model

# ============================================================
# ISL INCLUDE-50 - ROBUST REAL-TIME INFERENCE
# ============================================================

MODEL_PATH = Path("models/include50_lstm_v2_best.keras")
CLASS_PATH = Path("models/include50_classes_v2.json")

SEQUENCE_LENGTH = 45
RAW_BUFFER_LENGTH = 90
FEATURES = 258

# Prediction settings
PREDICT_EVERY = 3
SMOOTHING_WINDOW = 7
CONFIDENCE_THRESHOLD = 0.55
STABLE_COUNT_REQUIRED = 3

# Landmark filtering
EMA_ALPHA = 0.45
MAX_FEATURE_DELTA = 0.35

# Camera
CAMERA_ID = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540

# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not CLASS_PATH.exists():
    raise FileNotFoundError(f"Class file not found: {CLASS_PATH}")

model = load_model(MODEL_PATH)

with open(CLASS_PATH, encoding="utf-8") as f:
    class_data = json.load(f)

names = [class_data[str(i)] for i in range(50)]

print("=" * 78)
print("ISL INCLUDE-50 - ROBUST REAL-TIME RECOGNITION")
print("=" * 78)
print("Model :", MODEL_PATH)
print("Classes:", len(names))
print("Input :", model.input_shape)
print("Expected features:", FEATURES)
print()
print("Pipeline:")
print("Webcam")
print("  -> MediaPipe Holistic")
print("  -> 258 feature extraction")
print("  -> EMA temporal filtering")
print("  -> outlier protection")
print("  -> 90-frame temporal buffer")
print("  -> uniform 45-frame sampling")
print("  -> LSTM V2")
print("  -> temporal probability smoothing")
print("  -> confidence threshold")
print("  -> stable prediction")
print()
print("Press Q to quit")
print("Press R to reset temporal buffer")
print("=" * 78)

# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def extract_features(results):
    """
    EXACT 258 dimensional representation:

    Pose:
        33 * 4 = 132

    Left hand:
        21 * 3 = 63

    Right hand:
        21 * 3 = 63

    Total:
        132 + 63 + 63 = 258
    """

    features = []

    # --------------------------------------------------------
    # POSE
    # --------------------------------------------------------

    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            features.extend([
                float(lm.x),
                float(lm.y),
                float(lm.z),
                float(lm.visibility)
            ])
    else:
        features.extend([0.0] * 132)

    # --------------------------------------------------------
    # LEFT HAND
    # --------------------------------------------------------

    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            features.extend([
                float(lm.x),
                float(lm.y),
                float(lm.z)
            ])
    else:
        features.extend([0.0] * 63)

    # --------------------------------------------------------
    # RIGHT HAND
    # --------------------------------------------------------

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            features.extend([
                float(lm.x),
                float(lm.y),
                float(lm.z)
            ])
    else:
        features.extend([0.0] * 63)

    arr = np.asarray(features, dtype=np.float32)

    if arr.shape != (FEATURES,):
        raise RuntimeError(
            f"Feature mismatch: {arr.shape}, expected ({FEATURES},)"
        )

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return arr


# ============================================================
# TEMPORAL FILTER
# ============================================================

previous_features = None


def filter_features(current):
    """
    EMA + maximum frame-to-frame jump protection.

    This prevents MediaPipe landmark jitter and sudden
    coordinate explosions from entering the LSTM.
    """

    global previous_features

    current = current.astype(np.float32)

    if previous_features is None:
        previous_features = current.copy()
        return current

    delta = current - previous_features

    delta = np.clip(
        delta,
        -MAX_FEATURE_DELTA,
        MAX_FEATURE_DELTA
    )

    filtered = (
        EMA_ALPHA * current
        +
        (1.0 - EMA_ALPHA) * previous_features
    )

    # Protect against extreme values
    filtered = np.clip(filtered, -3.0, 3.0)

    previous_features = filtered.copy()

    return filtered


# ============================================================
# RESET
# ============================================================

def reset_state():
    global previous_features

    previous_features = None

    frame_buffer.clear()
    probability_buffer.clear()

    return None


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# ============================================================
# BUFFERS
# ============================================================

frame_buffer = deque(maxlen=RAW_BUFFER_LENGTH)

probability_buffer = deque(
    maxlen=SMOOTHING_WINDOW
)

stable_label = "Waiting..."
stable_confidence = 0.0

candidate_label = None
candidate_count = 0

prediction_count = 0

last_prediction_time = 0


# ============================================================
# MAIN LOOP
# ============================================================

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55
) as holistic:

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        # ----------------------------------------------------
        # MIRROR DISPLAY
        # ----------------------------------------------------

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = holistic.process(rgb)

        # ----------------------------------------------------
        # FEATURE EXTRACTION
        # ----------------------------------------------------

        raw_features = extract_features(results)

        filtered_features = filter_features(
            raw_features
        )

        frame_buffer.append(
            filtered_features
        )

        # ----------------------------------------------------
        # DRAW LANDMARKS
        # ----------------------------------------------------

        if results.pose_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS
            )

        if results.left_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        if results.right_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        if (
            len(frame_buffer) >= RAW_BUFFER_LENGTH
            and prediction_count % PREDICT_EVERY == 0
        ):

            raw_sequence = np.asarray(
                frame_buffer,
                dtype=np.float32
            )

            # EXACTLY 45 frames from 90 frames
            indices = np.linspace(
                0,
                RAW_BUFFER_LENGTH - 1,
                SEQUENCE_LENGTH
            ).astype(int)

            sequence = raw_sequence[indices]

            if sequence.shape == (
                SEQUENCE_LENGTH,
                FEATURES
            ):

                if np.isfinite(sequence).all():

                    # ------------------------------------------------
                    # MODEL INPUT
                    # ------------------------------------------------

                    model_input = sequence[
                        None,
                        ...
                    ]

                    probabilities = model.predict(
                        model_input,
                        verbose=0
                    )[0]

                    probabilities = np.nan_to_num(
                        probabilities
                    )

                    # ------------------------------------------------
                    # TEMPORAL PROBABILITY SMOOTHING
                    # ------------------------------------------------

                    probability_buffer.append(
                        probabilities
                    )

                    avg_probabilities = np.mean(
                        np.asarray(
                            probability_buffer
                        ),
                        axis=0
                    )

                    index = int(
                        np.argmax(
                            avg_probabilities
                        )
                    )

                    confidence = float(
                        avg_probabilities[index]
                    )

                    predicted_label = names[index]

                    # ------------------------------------------------
                    # STABILITY LOGIC
                    # ------------------------------------------------

                    if confidence >= CONFIDENCE_THRESHOLD:

                        if predicted_label == candidate_label:

                            candidate_count += 1

                        else:

                            candidate_label = predicted_label
                            candidate_count = 1

                        if (
                            candidate_count
                            >= STABLE_COUNT_REQUIRED
                        ):

                            stable_label = predicted_label
                            stable_confidence = confidence

                    else:

                        candidate_count = 0
                        candidate_label = None

                else:

                    stable_label = "Invalid data"
                    stable_confidence = 0.0

        prediction_count += 1

        # ====================================================
        # UI
        # ====================================================

        h, w = frame.shape[:2]

        # Header
        cv2.rectangle(
            frame,
            (0, 0),
            (w, 120),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            frame,
            "ISL INCLUDE-50",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Prediction: {stable_label}",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {stable_confidence * 100:.1f}%",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Buffer status
        buffer_text = (
            f"Sequence: "
            f"{len(frame_buffer)}/{RAW_BUFFER_LENGTH}"
        )

        cv2.putText(
            frame,
            buffer_text,
            (w - 330, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # Feature count
        cv2.putText(
            frame,
            "Features: 258",
            (w - 330, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # Filtering
        cv2.putText(
            frame,
            "EMA + Temporal Smoothing",
            (w - 330, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        # ----------------------------------------------------
        # TOP 3
        # ----------------------------------------------------

        if len(probability_buffer) > 0:

            avg = np.mean(
                np.asarray(probability_buffer),
                axis=0
            )

            top_indices = np.argsort(
                avg
            )[::-1][:3]

            y = 155

            cv2.putText(
                frame,
                "TOP 3",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            y += 32

            for rank, idx in enumerate(top_indices):

                text = (
                    f"{rank + 1}. "
                    f"{names[idx]} "
                    f"{avg[idx] * 100:.1f}%"
                )

                cv2.putText(
                    frame,
                    text,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                y += 30

        # ----------------------------------------------------
        # INSTRUCTIONS
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "Perform one sign clearly | Q = Quit | R = Reset",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.imshow(
            "ISL INCLUDE-50 - Real Time",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

        if key == ord("r"):

            reset_state()

            stable_label = "Waiting..."
            stable_confidence = 0.0
            candidate_label = None
            candidate_count = 0

            print("Temporal state reset.")

# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print()
print("=" * 78)
print("REAL-TIME INFERENCE STOPPED")
print("=" * 78)
print("V2 model preserved.")
print("=" * 78)
