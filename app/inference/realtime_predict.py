"""
INCLUDE-50 REAL-TIME ISL RECOGNITION
====================================

IMPORTANT:
This realtime extractor MUST MATCH:

app/training/extract_landmarks.py

Training representation:

Pose       = 33 x 4 = 132
             x, y, z, visibility

Left hand  = 21 x 3 = 63
Right hand = 21 x 3 = 63

TOTAL      = 258

Sequence   = 45 frames

NO normalization
NO standardization
NO centering
NO scaling
NO coordinate transformation

The webcam landmarks are passed to the model in exactly
the same representation used during training.
"""

import json
import os
import sys
import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.feature_pipeline import (
    CaptureQualityGate,
    assess_landmark_quality,
    extract_landmarks as shared_extract_landmarks,
    validate_sequence,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = ROOT_DIR

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "include50_lstm_v2_best.keras"
)

CLASSES_PATH = (
    BASE_DIR
    / "models"
    / "include50_classes_v2.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 45
FEATURES = 258
NUM_CLASSES = 50

CAMERA_INDEX = int(os.getenv("ISL_CAMERA_INDEX", "0"))
CAMERA_WIDTH = int(os.getenv("ISL_CAMERA_WIDTH", "960"))
CAMERA_HEIGHT = int(os.getenv("ISL_CAMERA_HEIGHT", "720"))
MIN_DETECTION_CONFIDENCE = float(os.getenv("ISL_MIN_DETECTION_CONFIDENCE", "0.5"))
MIN_TRACKING_CONFIDENCE = float(os.getenv("ISL_MIN_TRACKING_CONFIDENCE", "0.5"))

# Prediction settings
PREDICT_EVERY = 3

CONFIDENCE_THRESHOLD = 0.35

# Number of predictions used for temporal voting
PREDICTION_HISTORY_SIZE = 5

# Minimum number of frames before prediction
MIN_SEQUENCE_FRAMES = 45

# Do not start collecting model input until MediaPipe has settled.
WARMUP_FRAMES = int(os.getenv("ISL_WARMUP_FRAMES", "15"))


# ============================================================
# DISPLAY
# ============================================================

WINDOW_NAME = "INCLUDE-50 ISL Recognition"

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

    if not CLASSES_PATH.exists():

        raise FileNotFoundError(
            f"Classes file not found:\n{CLASSES_PATH}"
        )

    with open(
        CLASSES_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    # --------------------------------------------------------
    # Dictionary:
    # {"0": "1. Dog", "1": "1. loud", ...}
    # --------------------------------------------------------

    if isinstance(data, dict):

        if all(
            str(i) in data
            for i in range(len(data))
        ):

            classes = [
                str(data[str(i)])
                for i in range(len(data))
            ]

            return classes

        if "classes" in data:

            return [
                str(x)
                for x in data["classes"]
            ]

        if "class_names" in data:

            return [
                str(x)
                for x in data["class_names"]
            ]

        if "labels" in data:

            return [
                str(x)
                for x in data["labels"]
            ]

    # --------------------------------------------------------
    # List
    # --------------------------------------------------------

    if isinstance(data, list):

        return [
            str(x)
            for x in data
        ]

    raise ValueError(
        "Unsupported classes JSON format."
    )


# ============================================================
# EXACT TRAINING LANDMARK EXTRACTION
# ============================================================

def extract_landmarks(results):

    """
    THIS FUNCTION MATCHES:

    app/training/extract_landmarks.py

    EXACTLY.

    There is intentionally NO normalization.
    """

    # ========================================================
    # POSE
    # ========================================================

    pose = []

    if results.pose_landmarks:

        for lm in results.pose_landmarks.landmark:

            pose.extend(
                [
                    lm.x,
                    lm.y,
                    lm.z,
                    lm.visibility
                ]
            )

    else:

        pose = [0.0] * 132


    return shared_extract_landmarks(results)
# ============================================================


# ============================================================
# DRAW LANDMARKS
# ============================================================

def draw_landmarks(
    frame,
    results
):

    # --------------------------------------------------------
    # Pose
    # --------------------------------------------------------

    if results.pose_landmarks:

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )


    # --------------------------------------------------------
    # Left hand
    # --------------------------------------------------------

    if results.left_hand_landmarks:

        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )


    # --------------------------------------------------------
    # Right hand
    # --------------------------------------------------------

    if results.right_hand_landmarks:

        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )


# ============================================================
# DRAW TEXT
# ============================================================

def draw_text(
    frame,
    text,
    position,
    color=WHITE,
    scale=0.6,
    thickness=2
):

    cv2.putText(
        frame,
        str(text),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# GET TOP PREDICTIONS
# ============================================================

def get_top_predictions(
    probabilities,
    class_names,
    count=5
):

    indices = np.argsort(
        probabilities
    )[::-1][:count]

    output = []

    for index in indices:

        output.append(
            (
                int(index),
                class_names[index],
                float(probabilities[index])
            )
        )

    return output


# ============================================================
# TEMPORAL VOTING
# ============================================================

def get_stable_prediction(
    prediction_history,
    class_names
):

    if not prediction_history:

        return (
            None,
            0.0
        )


    votes = {}

    confidence_sum = {}


    for index, confidence in prediction_history:

        votes[index] = (
            votes.get(index, 0)
            + 1
        )

        confidence_sum[index] = (
            confidence_sum.get(index, 0.0)
            + confidence
        )


    # --------------------------------------------------------
    # Most voted class
    # --------------------------------------------------------

    best_index = max(
        votes,
        key=lambda x: (
            votes[x],
            confidence_sum[x]
        )
    )


    average_confidence = (
        confidence_sum[best_index]
        /
        votes[best_index]
    )


    return (
        class_names[best_index],
        average_confidence
    )


# ============================================================
# RESET
# ============================================================

def reset_state(
    sequence,
    prediction_history
):

    sequence.clear()

    prediction_history.clear()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "INCLUDE-50 REAL-TIME ISL RECOGNITION"
    )

    print("=" * 70)

    print()

    print(
        "Training-compatible preprocessing:"
    )

    print(
        "Pose       : 33 x 4 = 132"
    )

    print(
        "Left hand  : 21 x 3 = 63"
    )

    print(
        "Right hand : 21 x 3 = 63"
    )

    print(
        "Total      : 258"
    )

    print(
        "Sequence   : 45"
    )

    print(
        "Normalization: NONE"
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print(
        "Loading model..."
    )


    if not MODEL_PATH.exists():

        print(
            "ERROR: Model not found:"
        )

        print(
            MODEL_PATH
        )

        return


    try:

        model = load_model(
            MODEL_PATH
        )

    except Exception as e:

        print(
            "ERROR loading model:"
        )

        print(e)

        return


    print(
        "Model loaded successfully."
    )

    print(
        "Input:",
        model.input_shape
    )

    print(
        "Output:",
        model.output_shape
    )


    # ========================================================
    # VALIDATE MODEL
    # ========================================================

    if (
        model.input_shape[-2]
        != SEQUENCE_LENGTH
    ):

        print(
            "ERROR: Sequence length mismatch."
        )

        return


    if (
        model.input_shape[-1]
        != FEATURES
    ):

        print(
            "ERROR: Feature count mismatch."
        )

        return


    if (
        model.output_shape[-1]
        != NUM_CLASSES
    ):

        print(
            "ERROR: Number of classes mismatch."
        )

        return


    # ========================================================
    # LOAD CLASSES
    # ========================================================

    print()
    print(
        "Loading classes..."
    )


    try:

        class_names = load_classes()

    except Exception as e:

        print(
            "ERROR loading classes:"
        )

        print(e)

        return


    if len(class_names) != NUM_CLASSES:

        print(
            "ERROR:"
        )

        print(
            f"Expected {NUM_CLASSES} classes."
        )

        print(
            f"Found {len(class_names)} classes."
        )

        return


    print(
        f"{len(class_names)} classes loaded."
    )


    # ========================================================
    # CLASS MAP
    # ========================================================

    print()
    print(
        "MODEL INDEX -> CLASS"
    )

    for i, name in enumerate(
        class_names
    ):

        print(
            f"{i:02d} -> {name}"
        )


    # ========================================================
    # CAMERA
    # ========================================================

    print()
    print(
        "Opening webcam..."
    )


    cap = cv2.VideoCapture(CAMERA_INDEX)


    if not cap.isOpened():

        print(
            "ERROR: Cannot open webcam."
        )

        return


    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if actual_width <= 0 or actual_height <= 0:
        print("ERROR: Camera returned an invalid resolution.")
        cap.release()
        return


    print(
        f"Webcam started at {actual_width}x{actual_height}."
    )


    print()
    print(
        "Controls:"
    )

    print(
        "Q = Quit"
    )

    print(
        "R = Reset sequence"
    )

    print(
        "SPACE = Show current top prediction"
    )

    print(
        "T = Show top 5 predictions"
    )


    # ========================================================
    # STATE
    # ========================================================

    sequence = deque(
        maxlen=SEQUENCE_LENGTH
    )


    prediction_history = deque(
        maxlen=PREDICTION_HISTORY_SIZE
    )


    frame_counter = 0

    fps = 0.0
    fps_frame_count = 0
    fps_started_at = time.perf_counter()

    prediction_counter = 0

    last_probabilities = np.zeros(
        NUM_CLASSES,
        dtype=np.float32
    )


    predicted_sign = "Waiting..."

    predicted_confidence = 0.0

    stable_sign = "Waiting..."

    stable_confidence = 0.0

    show_top5 = False

    quality_gate = CaptureQualityGate(WARMUP_FRAMES)
    capture_ready = False
    quality_status = "Warming up MediaPipe"


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    with mp_holistic.Holistic(

        static_image_mode=False,

        model_complexity=1,

        smooth_landmarks=True,

        enable_segmentation=False,

        refine_face_landmarks=False,

        min_detection_confidence=MIN_DETECTION_CONFIDENCE,

        min_tracking_confidence=MIN_TRACKING_CONFIDENCE

    ) as holistic:


        try:

            while cap.isOpened():

                # =================================================
                # READ
                # =================================================

                ret, frame = cap.read()


                if not ret:

                    print(
                        "Failed to read webcam."
                    )

                    break


                frame_counter += 1
                fps_frame_count += 1
                if fps_frame_count >= 10:
                    elapsed = time.perf_counter() - fps_started_at
                    fps = fps_frame_count / elapsed if elapsed > 0 else 0.0
                    fps_frame_count = 0
                    fps_started_at = time.perf_counter()


                # =================================================
                # MIRROR
                # =================================================

                frame = cv2.flip(
                    frame,
                    1
                )


                # =================================================
                # RGB
                # =================================================

                image = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )


                image.flags.writeable = False


                # =================================================
                # MEDIAPIPE
                # =================================================

                results = holistic.process(
                    image
                )


                image.flags.writeable = True


                # =================================================
                # DRAW
                # =================================================

                draw_landmarks(
                    frame,
                    results
                )


                # =================================================
                # EXACT TRAINING FEATURES
                # =================================================

                try:

                    features = extract_landmarks(
                        results
                    )

                except Exception as e:

                    draw_text(
                        frame,
                        "Feature extraction error",
                        (20, 40),
                        RED,
                        0.65,
                        2
                    )

                    print(
                        "Feature extraction error:",
                        e
                    )

                    continue


                # =================================================
                # DETECTION STATUS
                # =================================================

                quality = assess_landmark_quality(results)
                pose_detected = quality.pose_present
                left_detected = quality.left_hand_present
                right_detected = quality.right_hand_present

                capture_ready, quality_status, quality_lost = quality_gate.update(quality)
                if not capture_ready and not quality.all_required_present:
                    quality_status = "Improve hand visibility"
                if quality_lost:
                    print("Landmark quality lost; recovering and resetting sequence.")
                    sequence.clear()
                    prediction_history.clear()
                    predicted_sign = "Waiting..."
                    predicted_confidence = 0.0
                    stable_sign = "Waiting..."
                    stable_confidence = 0.0
                    quality_status = "Improve hand visibility"

                if capture_ready and quality.all_required_present:
                    sequence.append(features)


                hand_count = (
                    int(left_detected)
                    +
                    int(right_detected)
                )


                # =================================================
                # PREDICTION
                # =================================================

                if (
                    capture_ready
                    and
                    len(sequence)
                    >= MIN_SEQUENCE_FRAMES
                    and
                    frame_counter
                    % PREDICT_EVERY
                    == 0
                ):

                    input_sequence = np.asarray(
                        sequence,
                        dtype=np.float32
                    )


                    # ------------------------------------------------
                    # FINAL SAFETY CHECK
                    # ------------------------------------------------

                    try:
                        input_sequence = validate_sequence(input_sequence)
                    except ValueError as error:
                        print("Invalid sequence skipped:", error)
                        continue


                    # ------------------------------------------------
                    # IMPORTANT
                    # ------------------------------------------------
                    #
                    # NO NORMALIZATION HERE.
                    #
                    # The model receives:
                    #
                    # (1, 45, 258)
                    #
                    # exactly like training.
                    #
                    # ------------------------------------------------

                    model_input = (
                        input_sequence
                        [None, :, :]
                    )


                    try:

                        probabilities = (
                            model.predict(
                                model_input,
                                verbose=0
                            )[0]
                        )

                    except Exception as e:

                        print(
                            "Prediction error:",
                            e
                        )

                        continue


                    last_probabilities = (
                        probabilities
                    )


                    # ------------------------------------------------
                    # Top prediction
                    # ------------------------------------------------

                    best_index = int(
                        np.argmax(
                            probabilities
                        )
                    )


                    best_confidence = float(
                        probabilities[
                            best_index
                        ]
                    )


                    predicted_sign = (
                        class_names[
                            best_index
                        ]
                    )


                    predicted_confidence = (
                        best_confidence
                    )


                    # ------------------------------------------------
                    # Prediction history
                    # ------------------------------------------------

                    prediction_history.append(
                        (
                            best_index,
                            best_confidence
                        )
                    )


                    (
                        stable_sign,
                        stable_confidence
                    ) = get_stable_prediction(
                        prediction_history,
                        class_names
                    )


                    prediction_counter += 1


                # =================================================
                # HEADER
                # =================================================

                h, w = frame.shape[:2]


                cv2.rectangle(
                    frame,
                    (0, 0),
                    (w, 150),
                    BLACK,
                    -1
                )


                draw_text(
                    frame,
                    "ISL RECOGNITION",
                    (20, 35),
                    CYAN,
                    0.8,
                    2
                )

                draw_text(
                    frame,
                    f"Status: {quality_status}",
                    (20, 55),
                    GREEN if capture_ready else YELLOW,
                    0.50,
                    1
                )


                # =================================================
                # PREDICTION DISPLAY
                # =================================================

                if len(sequence) < 45:

                    draw_text(
                        frame,
                        (
                            f"Collecting: "
                            f"{len(sequence)}/45"
                        ),
                        (20, 90),
                        YELLOW,
                        0.65,
                        2
                    )

                else:

                    display_sign = (
                        stable_sign
                        if stable_sign != "Waiting..."
                        else predicted_sign
                    )


                    display_confidence = (
                        stable_confidence
                        if stable_sign != "Waiting..."
                        else predicted_confidence
                    )


                    if (
                        display_confidence
                        >= CONFIDENCE_THRESHOLD
                    ):

                        prediction_color = GREEN

                    else:

                        prediction_color = YELLOW


                    draw_text(
                        frame,
                        f"Sign: {display_sign}",
                        (20, 90),
                        prediction_color,
                        0.72,
                        2
                    )


                    draw_text(
                        frame,
                        (
                            f"Confidence: "
                            f"{display_confidence * 100:.1f}%"
                        ),
                        (20, 120),
                        WHITE,
                        0.50,
                        1
                    )


                # =================================================
                # DETECTION STATUS
                # =================================================

                if hand_count == 2:

                    hand_text = "Both Hands"

                elif hand_count == 1:

                    hand_text = "One Hand"

                else:

                    hand_text = "No Hands"


                draw_text(
                    frame,
                    f"Hands: {hand_text}",
                    (w - 250, 35),
                    WHITE,
                    0.50,
                    1
                )


                draw_text(
                    frame,
                    (
                        f"Pose: "
                        f"{'YES' if pose_detected else 'NO'}"
                    ),
                    (w - 250, 60),
                    WHITE,
                    0.50,
                    1
                )


                draw_text(
                    frame,
                    f"Frames: {len(sequence)}/45",
                    (w - 250, 85),
                    WHITE,
                    0.50,
                    1
                )


                draw_text(
                    frame,
                    f"Predictions: {prediction_counter}",
                    (w - 250, 110),
                    WHITE,
                    0.45,
                    1
                )

                draw_text(
                    frame,
                    f"FPS: {fps:.1f}",
                    (w - 250, 135),
                    WHITE,
                    0.45,
                    1
                )


                # =================================================
                # TOP 5
                # =================================================

                if (
                    show_top5
                    and
                    len(sequence) >= 45
                ):

                    top5 = get_top_predictions(
                        last_probabilities,
                        class_names,
                        5
                    )


                    x1 = 20
                    y1 = 150

                    box_height = 30 + (
                        len(top5) * 28
                    )


                    cv2.rectangle(
                        frame,
                        (
                            x1,
                            y1
                        ),
                        (
                            430,
                            y1 + box_height
                        ),
                        BLACK,
                        -1
                    )


                    draw_text(
                        frame,
                        "TOP 5",
                        (
                            x1 + 10,
                            y1 + 25
                        ),
                        CYAN,
                        0.55,
                        2
                    )


                    yy = y1 + 52


                    for rank, (
                        idx,
                        name,
                        probability
                    ) in enumerate(
                        top5,
                        start=1
                    ):

                        draw_text(
                            frame,
                            (
                                f"{rank}. "
                                f"{name}"
                            ),
                            (
                                x1 + 10,
                                yy
                            ),
                            WHITE,
                            0.45,
                            1
                        )


                        draw_text(
                            frame,
                            (
                                f"{probability * 100:.1f}%"
                            ),
                            (
                                x1 + 330,
                                yy
                            ),
                            GREEN,
                            0.45,
                            1
                        )


                        yy += 28


                # =================================================
                # BOTTOM STATUS
                # =================================================

                status_y = h - 65


                if len(sequence) < 45:

                    status = (
                        "Move into camera and perform sign..."
                    )

                    status_color = YELLOW

                elif predicted_confidence >= 0.50:

                    status = (
                        "SIGN DETECTED"
                    )

                    status_color = GREEN

                else:

                    status = (
                        "Low confidence - perform sign clearly"
                    )

                    status_color = YELLOW


                draw_text(
                    frame,
                    status,
                    (20, status_y),
                    status_color,
                    0.55,
                    2
                )


                draw_text(
                    frame,
                    "Q: Quit   R: Reset   SPACE: Select   T: Top 5",
                    (20, h - 25),
                    WHITE,
                    0.42,
                    1
                )


                # =================================================
                # SHOW
                # =================================================

                cv2.imshow(
                    WINDOW_NAME,
                    frame
                )


                # =================================================
                # KEYBOARD
                # =================================================

                key = cv2.waitKey(1) & 0xFF


                # ------------------------------------------------
                # Q
                # ------------------------------------------------

                if key == ord("q"):

                    break


                # ------------------------------------------------
                # R
                # ------------------------------------------------

                elif key == ord("r"):

                    reset_state(
                        sequence,
                        prediction_history
                    )

                    predicted_sign = (
                        "Waiting..."
                    )

                    predicted_confidence = 0.0

                    stable_sign = (
                        "Waiting..."
                    )

                    stable_confidence = 0.0

                    last_probabilities = (
                        np.zeros(
                            NUM_CLASSES,
                            dtype=np.float32
                        )
                    )

                    print(
                        "Sequence reset."
                    )


                # ------------------------------------------------
                # SPACE
                # ------------------------------------------------

                elif key == 32:

                    if (
                        predicted_sign
                        != "Waiting..."
                    ):

                        print()
                        print(
                            "CURRENT PREDICTION:"
                        )

                        print(
                            predicted_sign
                        )

                        print(
                            f"Confidence: "
                            f"{predicted_confidence:.4f}"
                        )


                # ------------------------------------------------
                # T
                # ------------------------------------------------

                elif key == ord("t"):

                    show_top5 = (
                        not show_top5
                    )


                    print(
                        "Top 5:",
                        (
                            "ON"
                            if show_top5
                            else
                            "OFF"
                        )
                    )


        finally:

            cap.release()

            cv2.destroyAllWindows()


    print()
    print(
        "Webcam stopped."
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()


