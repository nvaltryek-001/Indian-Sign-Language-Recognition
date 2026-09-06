import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from pathlib import Path
from collections import deque
import json
import time


# ============================================================
# INCLUDE-50 V2 — IMPROVED REAL-TIME ISL RECOGNITION
# ============================================================

MODEL_PATH = Path("models/include50_lstm_v2_best.keras")
CLASS_MAP_PATH = Path("models/include50_classes_v2.json")

SEQUENCE_LENGTH = 45
FEATURES = 258
NUM_CLASSES = 50

# ------------------------------------------------------------
# REAL-TIME SETTINGS
# ------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.60

# Number of recent predictions used for smoothing
SMOOTHING_WINDOW = 7

# Number of consecutive predictions required before
# changing the displayed sign
STABLE_REQUIRED = 4

# Predict every N frames
PREDICT_EVERY = 3

CAMERA_INDEX = 0

# Camera resolution
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("INCLUDE-50 V2 — IMPROVED REAL-TIME ISL RECOGNITION")
print("=" * 75)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading V2 model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"V2 model not found:\n{MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    str(MODEL_PATH)
)

print("V2 model loaded successfully.")

print(
    f"Model input : {model.input_shape}"
)

print(
    f"Model output: {model.output_shape}"
)


# ============================================================
# LOAD CLASS MAP
# ============================================================

if not CLASS_MAP_PATH.exists():
    raise FileNotFoundError(
        f"Class map not found:\n{CLASS_MAP_PATH}"
    )

with open(
    CLASS_MAP_PATH,
    "r",
    encoding="utf-8"
) as f:

    class_mapping = json.load(f)


class_names = [
    class_mapping[str(i)]
    for i in range(len(class_mapping))
]

print(
    f"Classes loaded: {len(class_names)}"
)

if len(class_names) != NUM_CLASSES:

    raise RuntimeError(
        f"Expected {NUM_CLASSES} classes, "
        f"found {len(class_names)}"
    )


# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# FEATURE EXTRACTION
#
# EXACT V2 FORMAT:
#
# Pose      = 33 × 4 = 132
# Left Hand = 21 × 3 = 63
# Right Hand= 21 × 3 = 63
#
# TOTAL = 258
# ============================================================

def extract_landmarks(results):

    # --------------------------------------------------------
    # POSE
    # --------------------------------------------------------

    pose = []

    if results.pose_landmarks:

        for lm in results.pose_landmarks.landmark:

            pose.extend([
                lm.x,
                lm.y,
                lm.z,
                lm.visibility
            ])

    else:

        pose = [0.0] * 132


    # --------------------------------------------------------
    # LEFT HAND
    # --------------------------------------------------------

    left_hand = []

    if results.left_hand_landmarks:

        for lm in results.left_hand_landmarks.landmark:

            left_hand.extend([
                lm.x,
                lm.y,
                lm.z
            ])

    else:

        left_hand = [0.0] * 63


    # --------------------------------------------------------
    # RIGHT HAND
    # --------------------------------------------------------

    right_hand = []

    if results.right_hand_landmarks:

        for lm in results.right_hand_landmarks.landmark:

            right_hand.extend([
                lm.x,
                lm.y,
                lm.z
            ])

    else:

        right_hand = [0.0] * 63


    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    features = (
        pose +
        left_hand +
        right_hand
    )

    features = np.asarray(
        features,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if features.shape != (FEATURES,):

        raise ValueError(
            f"Expected {FEATURES} features, "
            f"got {features.shape}"
        )


    if np.isnan(features).any():

        raise ValueError(
            "NaN detected in landmarks"
        )


    if np.isinf(features).any():

        raise ValueError(
            "Inf detected in landmarks"
        )


    return features


# ============================================================
# DRAW MEDIAPIPE LANDMARKS
# ============================================================

def draw_landmarks(frame, results):

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
# SMOOTHING
# ============================================================

prediction_history = deque(
    maxlen=SMOOTHING_WINDOW
)


# ============================================================
# STABLE PREDICTION STATE
# ============================================================

stable_index = None

candidate_index = None

candidate_count = 0


# ============================================================
# UPDATE STABLE PREDICTION
# ============================================================

def update_stable_prediction(index):

    global stable_index
    global candidate_index
    global candidate_count


    # First prediction
    if stable_index is None:

        stable_index = index

        candidate_index = None

        candidate_count = 0

        return stable_index


    # Same as currently displayed prediction
    if index == stable_index:

        candidate_index = None

        candidate_count = 0

        return stable_index


    # New candidate
    if candidate_index != index:

        candidate_index = index

        candidate_count = 1

    else:

        candidate_count += 1


    # Change displayed prediction only
    # after repeated confirmation
    if candidate_count >= STABLE_REQUIRED:

        stable_index = candidate_index

        candidate_index = None

        candidate_count = 0


    return stable_index


# ============================================================
# DRAW TEXT HELPER
# ============================================================

def put_text(
    frame,
    text,
    position,
    scale=0.7,
    thickness=2
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# DRAW TOP PREDICTIONS
# ============================================================

def draw_predictions(
    frame,
    probabilities,
    stable_index
):

    top_indices = np.argsort(
        probabilities
    )[::-1][:3]


    # --------------------------------------------------------
    # PANEL
    # --------------------------------------------------------

    panel_x = 20
    panel_y = 20

    panel_w = 440
    panel_h = 190


    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (
            panel_x,
            panel_y
        ),
        (
            panel_x + panel_w,
            panel_y + panel_h
        ),
        (0, 0, 0),
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0
    )


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    put_text(
        frame,
        "V2 ISL RECOGNITION",
        (35, 50),
        0.75,
        2
    )


    # --------------------------------------------------------
    # TOP 3
    # --------------------------------------------------------

    y = 85

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        confidence = (
            probabilities[index] * 100
        )

        name = class_names[index]


        text = (
            f"{rank}. {name}: "
            f"{confidence:.1f}%"
        )

        put_text(
            frame,
            text,
            (35, y),
            0.62,
            2
        )

        y += 32


    # --------------------------------------------------------
    # STABLE RESULT
    # --------------------------------------------------------

    if stable_index is not None:

        stable_confidence = (
            probabilities[stable_index] * 100
        )

        stable_name = class_names[
            stable_index
        ]

        put_text(
            frame,
            f"STABLE: {stable_name}",
            (35, 175),
            0.65,
            2
        )

    return frame


# ============================================================
# MAIN WEBCAM
# ============================================================

print("\nStarting webcam...")

print("\nControls:")
print("  Q = Quit")
print("  R = Reset prediction")
print("  +/- = Change confidence threshold")


cap = cv2.VideoCapture(
    CAMERA_INDEX
)


if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


# ------------------------------------------------------------
# CAMERA SETTINGS
# ------------------------------------------------------------

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)


# ============================================================
# SEQUENCE BUFFER
# ============================================================

sequence = deque(
    maxlen=SEQUENCE_LENGTH
)


# ============================================================
# FPS VARIABLES
# ============================================================

previous_time = time.time()

fps = 0.0

frame_counter = 0


# ============================================================
# CURRENT PREDICTION
# ============================================================

current_probabilities = None

current_index = None


# ============================================================
# MEDIAPIPE HOLISTIC
# ============================================================

with mp_holistic.Holistic(

    static_image_mode=False,

    model_complexity=1,

    smooth_landmarks=True,

    enable_segmentation=False,

    refine_face_landmarks=False

) as holistic:


    while True:

        # ----------------------------------------------------
        # READ FRAME
        # ----------------------------------------------------

        ret, frame = cap.read()


        if not ret:

            print(
                "\nERROR: Could not read webcam frame."
            )

            break


        # Mirror webcam
        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # MEDIAPIPE
        # ----------------------------------------------------

        results = holistic.process(
            rgb
        )


        # ----------------------------------------------------
        # DRAW LANDMARKS
        # ----------------------------------------------------

        draw_landmarks(
            frame,
            results
        )


        # ----------------------------------------------------
        # EXTRACT FEATURES
        # ----------------------------------------------------

        features = extract_landmarks(
            results
        )


        # ----------------------------------------------------
        # ADD TO SEQUENCE
        # ----------------------------------------------------

        sequence.append(
            features
        )


        frame_counter += 1


        # ----------------------------------------------------
        # PREDICT WHEN BUFFER IS FULL
        # ----------------------------------------------------

        if (

            len(sequence) == SEQUENCE_LENGTH

            and

            frame_counter % PREDICT_EVERY == 0

        ):


            X = np.asarray(
                sequence,
                dtype=np.float32
            )


            X = np.expand_dims(
                X,
                axis=0
            )


            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            probabilities = model.predict(
                X,
                verbose=0
            )[0]


            # ------------------------------------------------
            # SAVE RAW PREDICTION
            # ------------------------------------------------

            current_probabilities = (
                probabilities.copy()
            )


            raw_index = int(
                np.argmax(probabilities)
            )


            raw_confidence = (
                probabilities[raw_index]
            )


            # ------------------------------------------------
            # CONFIDENCE FILTER
            # ------------------------------------------------

            if raw_confidence >= CONFIDENCE_THRESHOLD:

                prediction_history.append(
                    raw_index
                )


                # --------------------------------------------
                # MAJORITY VOTE
                # --------------------------------------------

                counts = np.bincount(
                    prediction_history,
                    minlength=NUM_CLASSES
                )


                smoothed_index = int(
                    np.argmax(counts)
                )


                # --------------------------------------------
                # STABILITY
                # --------------------------------------------

                current_index = (
                    update_stable_prediction(
                        smoothed_index
                    )
                )

            else:

                current_index = None


        # ====================================================
        # UI
        # ====================================================

        height, width = frame.shape[:2]


        # ----------------------------------------------------
        # TOP PREDICTIONS
        # ----------------------------------------------------

        if (

            current_probabilities is not None

            and

            len(sequence) == SEQUENCE_LENGTH

        ):

            frame = draw_predictions(
                frame,
                current_probabilities,
                stable_index
            )


        # ----------------------------------------------------
        # BUFFER STATUS
        # ----------------------------------------------------

        buffer_count = len(sequence)

        buffer_percent = (
            buffer_count /
            SEQUENCE_LENGTH *
            100
        )


        # ----------------------------------------------------
        # BUFFER BAR
        # ----------------------------------------------------

        bar_x = 20
        bar_y = height - 65

        bar_w = 400
        bar_h = 22


        cv2.rectangle(
            frame,
            (
                bar_x,
                bar_y
            ),
            (
                bar_x + bar_w,
                bar_y + bar_h
            ),
            (80, 80, 80),
            2
        )


        filled = int(
            bar_w *
            buffer_percent /
            100
        )


        if filled > 0:

            cv2.rectangle(
                frame,
                (
                    bar_x,
                    bar_y
                ),
                (
                    bar_x + filled,
                    bar_y + bar_h
                ),
                (255, 255, 255),
                -1
            )


        put_text(
            frame,
            (
                f"Sequence: "
                f"{buffer_count}/{SEQUENCE_LENGTH}"
            ),
            (
                bar_x,
                bar_y - 10
            ),
            0.55,
            2
        )


        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        current_time = time.time()

        elapsed = (
            current_time -
            previous_time
        )


        if elapsed > 0:

            instant_fps = (
                1.0 / elapsed
            )

            fps = (
                0.9 * fps +
                0.1 * instant_fps
            )


        previous_time = current_time


        put_text(
            frame,
            f"FPS: {fps:.1f}",
            (
                width - 140,
                35
            ),
            0.55,
            2
        )


        # ----------------------------------------------------
        # THRESHOLD
        # ----------------------------------------------------

        put_text(
            frame,
            (
                f"Threshold: "
                f"{CONFIDENCE_THRESHOLD * 100:.0f}%"
            ),
            (
                width - 240,
                65
            ),
            0.50,
            2
        )


        # ----------------------------------------------------
        # CURRENT STABLE RESULT
        # ----------------------------------------------------

        if stable_index is not None:

            name = class_names[
                stable_index
            ]

            if current_probabilities is not None:

                confidence = (
                    current_probabilities[
                        stable_index
                    ] * 100
                )

            else:

                confidence = 0.0


            put_text(
                frame,
                f"RESULT: {name}",
                (
                    width - 420,
                    height - 105
                ),
                0.65,
                2
            )


            put_text(
                frame,
                f"CONFIDENCE: {confidence:.2f}%",
                (
                    width - 420,
                    height - 75
                ),
                0.60,
                2
            )


        # ----------------------------------------------------
        # WAITING MESSAGE
        # ----------------------------------------------------

        if len(sequence) < SEQUENCE_LENGTH:

            put_text(
                frame,
                "Collecting frames...",
                (
                    width // 2 - 120,
                    height - 30
                ),
                0.55,
                2
            )


        # ----------------------------------------------------
        # SHOW WINDOW
        # ----------------------------------------------------

        cv2.imshow(
            "INCLUDE-50 V2 - Improved ISL Recognition",
            frame
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if key == ord("q"):

            break


        # ----------------------------------------------------
        # RESET
        # ----------------------------------------------------

        elif key == ord("r"):

            sequence.clear()

            prediction_history.clear()

            stable_index = None

            candidate_index = None

            candidate_count = 0

            current_probabilities = None

            current_index = None

            print(
                "\nPrediction reset."
            )


        # ----------------------------------------------------
        # INCREASE THRESHOLD
        # ----------------------------------------------------

        elif key in [
            ord("+"),
            ord("=")
        ]:

            CONFIDENCE_THRESHOLD = min(
                0.95,
                CONFIDENCE_THRESHOLD + 0.05
            )

            print(
                f"\nConfidence threshold: "
                f"{CONFIDENCE_THRESHOLD * 100:.0f}%"
            )


        # ----------------------------------------------------
        # DECREASE THRESHOLD
        # ----------------------------------------------------

        elif key in [
            ord("-"),
            ord("_")
        ]:

            CONFIDENCE_THRESHOLD = max(
                0.30,
                CONFIDENCE_THRESHOLD - 0.05
            )

            print(
                f"\nConfidence threshold: "
                f"{CONFIDENCE_THRESHOLD * 100:.0f}%"
            )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("\n" + "=" * 75)
print("WEBCAM STOPPED")
print("=" * 75)