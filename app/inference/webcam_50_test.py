import json
import csv
import time
from pathlib import Path
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf


# ============================================================
# ISL INCLUDE-50 WEBCAM TEST + DATA COLLECTION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "include50_lstm_v2_best.keras"
CLASS_PATH = BASE_DIR / "models" / "include50_classes_v2.json"

WEBCAM_DATA_DIR = BASE_DIR / "data" / "webcam50"
RESULT_DIR = BASE_DIR / "results" / "webcam50"

WEBCAM_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

SEQUENCE_LENGTH = 45
FEATURES = 258

CAMERA_INDEX = 0

# Prediction smoothing
SMOOTHING_WINDOW = 9

# Minimum confidence for PASS
CONFIDENCE_THRESHOLD = 0.70

# Number of samples to collect per sign
SAMPLES_PER_SIGN = 10

# Seconds before recording starts
COUNTDOWN_SECONDS = 2


# ============================================================
# COLORS
# ============================================================

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
BLUE = (255, 150, 0)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# LOAD CLASSES
# ============================================================

def load_classes():

    if not CLASS_PATH.exists():
        raise FileNotFoundError(
            f"Classes file not found:\n{CLASS_PATH}"
        )

    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        classes = data

    elif isinstance(data, dict):

        # Dictionary: {"0": "1. Dog", ...}
        try:
            classes = [
                value
                for key, value in sorted(
                    data.items(),
                    key=lambda x: int(x[0])
                )
            ]

        except Exception:

            for key in [
                "classes",
                "class_names",
                "labels"
            ]:
                if key in data:
                    classes = data[key]
                    break
            else:
                classes = list(data.values())

    else:
        raise ValueError(
            "Invalid class JSON format."
        )

    if len(classes) != 50:
        print(
            f"WARNING: Expected 50 classes, "
            f"found {len(classes)}"
        )

    return classes


# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def extract_keypoints(results):

    """
    EXACTLY 258 FEATURES

    Pose:
        33 landmarks x 4 = 132

    Left hand:
        21 landmarks x 3 = 63

    Right hand:
        21 landmarks x 3 = 63

    Total:
        132 + 63 + 63 = 258
    """

    # -------------------------
    # Pose
    # -------------------------

    if results.pose_landmarks:

        pose = np.array(
            [
                [
                    lm.x,
                    lm.y,
                    lm.z,
                    lm.visibility
                ]
                for lm in results.pose_landmarks.landmark
            ],
            dtype=np.float32
        ).flatten()

    else:

        pose = np.zeros(
            132,
            dtype=np.float32
        )

    # -------------------------
    # Left hand
    # -------------------------

    if results.left_hand_landmarks:

        left_hand = np.array(
            [
                [
                    lm.x,
                    lm.y,
                    lm.z
                ]
                for lm in results.left_hand_landmarks.landmark
            ],
            dtype=np.float32
        ).flatten()

    else:

        left_hand = np.zeros(
            63,
            dtype=np.float32
        )

    # -------------------------
    # Right hand
    # -------------------------

    if results.right_hand_landmarks:

        right_hand = np.array(
            [
                [
                    lm.x,
                    lm.y,
                    lm.z
                ]
                for lm in results.right_hand_landmarks.landmark
            ],
            dtype=np.float32
        ).flatten()

    else:

        right_hand = np.zeros(
            63,
            dtype=np.float32
        )

    # -------------------------
    # Combine
    # -------------------------

    keypoints = np.concatenate(
        [
            pose,
            left_hand,
            right_hand
        ]
    )

    if keypoints.shape[0] != FEATURES:

        raise ValueError(
            f"Feature mismatch: "
            f"{keypoints.shape}"
        )

    return keypoints.astype(
        np.float32
    )


# ============================================================
# DRAW TEXT
# ============================================================

def put_text(
    frame,
    text,
    position,
    color=WHITE,
    size=0.7,
    thickness=2
):

    cv2.putText(
        frame,
        str(text),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# DRAW LANDMARKS
# ============================================================

def draw_landmarks(
    frame,
    results
):

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


# ============================================================
# COUNT HANDS
# ============================================================

def hand_count(results):

    count = 0

    if results.left_hand_landmarks:
        count += 1

    if results.right_hand_landmarks:
        count += 1

    return count


# ============================================================
# DRAW HEADER
# ============================================================

def draw_header(
    frame,
    expected,
    predicted,
    confidence,
    recording=False,
    frame_count=0
):

    h, w = frame.shape[:2]

    # Header
    cv2.rectangle(
        frame,
        (0, 0),
        (w, 170),
        BLACK,
        -1
    )

    put_text(
        frame,
        f"EXPECTED : {expected}",
        (20, 35),
        CYAN,
        0.75,
        2
    )

    if predicted:

        color = (
            GREEN
            if confidence >= CONFIDENCE_THRESHOLD
            else RED
        )

        put_text(
            frame,
            f"PREDICTED: {predicted}",
            (20, 70),
            color,
            0.75,
            2
        )

        put_text(
            frame,
            f"CONFIDENCE: {confidence * 100:.1f}%",
            (20, 105),
            YELLOW,
            0.75,
            2
        )

    if recording:

        put_text(
            frame,
            f"RECORDING: {frame_count}/{SEQUENCE_LENGTH}",
            (20, 140),
            RED,
            0.65,
            2
        )

    else:

        put_text(
            frame,
            "SPACE = record/test   R = repeat   ESC = quit",
            (20, 140),
            WHITE,
            0.55,
            1
        )


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("ISL INCLUDE-50 WEBCAM TESTER")
print("=" * 70)

print()
print("Loading model...")

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH
)

classes = load_classes()

print(
    f"Model loaded successfully."
)

print(
    f"Classes: {len(classes)}"
)

print()

for i, name in enumerate(classes):

    print(
        f"{i:2d} -> {name}"
    )


# ============================================================
# CAMERA
# ============================================================

print()
print("Opening webcam...")

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)

cap.set(
    cv2.CAP_PROP_FPS,
    30
)


# ============================================================
# SESSION DATA
# ============================================================

results_table = []

selected_index = 0

expected_sign = classes[selected_index]

last_prediction = None
last_confidence = 0.0

prediction_history = deque(
    maxlen=SMOOTHING_WINDOW
)


# ============================================================
# MEDIAPIPE
# ============================================================

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# SAVE SAMPLE
# ============================================================

def save_sample(
    sign_index,
    sign_name,
    sequence,
    prediction,
    confidence
):

    safe_name = (
        sign_name
        .replace(".", "")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    sign_dir = (
        WEBCAM_DATA_DIR
        / f"{sign_index:02d}_{safe_name}"
    )

    sign_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    existing = list(
        sign_dir.glob("*.npy")
    )

    sample_number = (
        len(existing) + 1
    )

    filename = (
        sign_dir
        / f"sample_{sample_number:03d}.npy"
    )

    np.save(
        filename,
        np.asarray(
            sequence,
            dtype=np.float32
        )
    )

    return filename


# ============================================================
# PREDICT SEQUENCE
# ============================================================

def predict_sequence(
    sequence
):

    X = np.asarray(
        sequence,
        dtype=np.float32
    )

    X = np.expand_dims(
        X,
        axis=0
    )

    prediction = model.predict(
        X,
        verbose=0
    )[0]

    prediction_history.append(
        int(np.argmax(prediction))
    )

    # Majority vote
    vote = Counter(
        prediction_history
    ).most_common(1)[0][0]

    # Average probability for selected class
    confidence = float(
        np.mean(
            [
                prediction_history[i] == vote
                for i in range(
                    len(prediction_history)
                )
            ]
        )
    )

    # Use current probability for display
    current_confidence = float(
        prediction[vote]
    )

    return (
        vote,
        current_confidence,
        prediction
    )


# ============================================================
# MAIN LOOP
# ============================================================

print()
print("=" * 70)
print("WEBCAM READY")
print("=" * 70)

print()
print("How to use:")
print()
print("UP/DOWN : select sign")
print("SPACE   : record 45-frame sample")
print("R       : clear current prediction")
print("ESC     : exit")
print()
print(
    f"Current sign: {expected_sign}"
)
print()


while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "ERROR: Could not read webcam frame."
        )

        break

    # Mirror webcam
    frame = cv2.flip(
        frame,
        1
    )

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = holistic.process(
        rgb
    )

    draw_landmarks(
        frame,
        results
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    draw_header(
        frame,
        expected_sign,
        last_prediction,
        last_confidence
    )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    hands = hand_count(
        results
    )

    put_text(
        frame,
        f"Hands detected: {hands}",
        (
            20,
            frame.shape[0] - 55
        ),
        GREEN if hands > 0 else RED,
        0.65,
        2
    )

    put_text(
        frame,
        f"Sign {selected_index + 1}/50",
        (
            frame.shape[1] - 180,
            frame.shape[0] - 25
        ),
        YELLOW,
        0.6,
        2
    )

    cv2.imshow(
        "ISL Include-50 Webcam Tester",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # --------------------------------------------------------
    # ESC
    # --------------------------------------------------------

    if key == 27:

        break

    # --------------------------------------------------------
    # UP
    # --------------------------------------------------------

    elif key == 82:

        selected_index = (
            selected_index - 1
        ) % len(classes)

        expected_sign = classes[
            selected_index
        ]

        last_prediction = None
        last_confidence = 0.0
        prediction_history.clear()

        print(
            f"\nSelected: "
            f"{selected_index + 1}. "
            f"{expected_sign}"
        )

    # --------------------------------------------------------
    # DOWN
    # --------------------------------------------------------

    elif key == 84:

        selected_index = (
            selected_index + 1
        ) % len(classes)

        expected_sign = classes[
            selected_index
        ]

        last_prediction = None
        last_confidence = 0.0
        prediction_history.clear()

        print(
            f"\nSelected: "
            f"{selected_index + 1}. "
            f"{expected_sign}"
        )

    # --------------------------------------------------------
    # R = RESET
    # --------------------------------------------------------

    elif key in [
        ord("r"),
        ord("R")
    ]:

        last_prediction = None
        last_confidence = 0.0

        prediction_history.clear()

        print(
            "\nPrediction reset."
        )

    # --------------------------------------------------------
    # SPACE = RECORD
    # --------------------------------------------------------

    elif key == 32:

        print()
        print(
            "=" * 60
        )

        print(
            f"Recording: "
            f"{selected_index + 1}. "
            f"{expected_sign}"
        )

        print(
            "Prepare your sign..."
        )

        # Countdown
        start_time = time.time()

        while (
            time.time() - start_time
            < COUNTDOWN_SECONDS
        ):

            ret2, frame2 = cap.read()

            if not ret2:
                break

            frame2 = cv2.flip(
                frame2,
                1
            )

            remaining = (
                COUNTDOWN_SECONDS
                - (
                    time.time()
                    - start_time
                )
            )

            cv2.rectangle(
                frame2,
                (0, 0),
                (
                    frame2.shape[1],
                    130
                ),
                BLACK,
                -1
            )

            put_text(
                frame2,
                f"GET READY: "
                f"{remaining:.1f}",
                (30, 70),
                YELLOW,
                1.2,
                3
            )

            cv2.imshow(
                "ISL Include-50 Webcam Tester",
                frame2
            )

            cv2.waitKey(1)

        # ----------------------------------------------------
        # Record 45 frames
        # ----------------------------------------------------

        sequence = []

        print(
            "Recording 45 frames..."
        )

        while len(sequence) < SEQUENCE_LENGTH:

            ret3, frame3 = cap.read()

            if not ret3:
                break

            frame3 = cv2.flip(
                frame3,
                1
            )

            rgb3 = cv2.cvtColor(
                frame3,
                cv2.COLOR_BGR2RGB
            )

            result3 = holistic.process(
                rgb3
            )

            draw_landmarks(
                frame3,
                result3
            )

            keypoints = extract_keypoints(
                result3
            )

            sequence.append(
                keypoints
            )

            # Recording overlay
            cv2.rectangle(
                frame3,
                (0, 0),
                (
                    frame3.shape[1],
                    120
                ),
                BLACK,
                -1
            )

            put_text(
                frame3,
                f"RECORDING "
                f"{len(sequence)}/45",
                (25, 55),
                RED,
                0.9,
                2
            )

            put_text(
                frame3,
                expected_sign,
                (25, 95),
                YELLOW,
                0.75,
                2
            )

            cv2.imshow(
                "ISL Include-50 Webcam Tester",
                frame3
            )

            key3 = cv2.waitKey(1) & 0xFF

            if key3 == 27:

                sequence = []
                break

        if len(sequence) != SEQUENCE_LENGTH:

            print(
                "Recording cancelled."
            )

            continue

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        pred_id, confidence, probabilities = (
            predict_sequence(
                sequence
            )
        )

        predicted_sign = classes[
            pred_id
        ]

        last_prediction = predicted_sign
        last_confidence = confidence

        correct = (
            pred_id == selected_index
        )

        # ----------------------------------------------------
        # Save sample
        # ----------------------------------------------------

        saved_file = save_sample(
            selected_index,
            expected_sign,
            sequence,
            predicted_sign,
            confidence
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        result = (
            "PASS"
            if correct
            else "FAIL"
        )

        result_color = (
            GREEN
            if correct
            else RED
        )

        print()
        print(
            f"EXPECTED   : {expected_sign}"
        )

        print(
            f"PREDICTED  : {predicted_sign}"
        )

        print(
            f"CONFIDENCE : "
            f"{confidence * 100:.2f}%"
        )

        print(
            f"RESULT     : {result}"
        )

        print(
            f"SAVED      : {saved_file}"
        )

        print(
            "=" * 60
        )

        # ----------------------------------------------------
        # Show result for 2 seconds
        # ----------------------------------------------------

        result_start = time.time()

        while (
            time.time() - result_start
            < 2.0
        ):

            ret4, frame4 = cap.read()

            if not ret4:
                break

            frame4 = cv2.flip(
                frame4,
                1
            )

            cv2.rectangle(
                frame4,
                (0, 0),
                (
                    frame4.shape[1],
                    180
                ),
                BLACK,
                -1
            )

            put_text(
                frame4,
                f"EXPECTED: "
                f"{expected_sign}",
                (25, 40),
                WHITE,
                0.7,
                2
            )

            put_text(
                frame4,
                f"PREDICTED: "
                f"{predicted_sign}",
                (25, 80),
                result_color,
                0.7,
                2
            )

            put_text(
                frame4,
                f"CONFIDENCE: "
                f"{confidence * 100:.1f}%",
                (25, 120),
                YELLOW,
                0.7,
                2
            )

            put_text(
                frame4,
                f"RESULT: {result}",
                (25, 160),
                result_color,
                0.8,
                2
            )

            cv2.imshow(
                "ISL Include-50 Webcam Tester",
                frame4
            )

            cv2.waitKey(1)

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results_table.append(
            {
                "expected": expected_sign,
                "predicted": predicted_sign,
                "confidence": round(
                    confidence * 100,
                    2
                ),
                "correct": int(correct),
                "saved_file": str(
                    saved_file
                )
            }
        )


# ============================================================
# CLEANUP
# ============================================================

holistic.close()
cap.release()
cv2.destroyAllWindows()


# ============================================================
# SAVE SESSION RESULTS
# ============================================================

if results_table:

    result_csv = (
        RESULT_DIR
        / "webcam_test_results.csv"
    )

    with open(
        result_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "expected",
                "predicted",
                "confidence",
                "correct",
                "saved_file"
            ]
        )

        writer.writeheader()

        writer.writerows(
            results_table
        )

    accuracy = (
        sum(
            r["correct"]
            for r in results_table
        )
        / len(results_table)
        * 100
    )

    average_confidence = (
        sum(
            r["confidence"]
            for r in results_table
        )
        / len(results_table)
    )

    print()
    print("=" * 70)
    print("WEBCAM SESSION COMPLETE")
    print("=" * 70)

    print(
        f"Tests              : "
        f"{len(results_table)}"
    )

    print(
        f"Correct            : "
        f"{sum(r['correct'] for r in results_table)}"
    )

    print(
        f"Accuracy           : "
        f"{accuracy:.2f}%"
    )

    print(
        f"Average confidence: "
        f"{average_confidence:.2f}%"
    )

    print()
    print(
        f"Results saved to:"
    )

    print(
        result_csv
    )

else:

    print()
    print(
        "No webcam tests recorded."
    )

print()
print("=" * 70)
print("PROGRAM EXITED")
print("=" * 70)