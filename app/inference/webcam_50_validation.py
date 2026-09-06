import json
import csv
import shutil
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "include50_lstm_v2_best.keras"
CLASS_PATH = BASE_DIR / "models" / "include50_classes_v2.json"

OUTPUT_DIR = BASE_DIR / "data" / "webcam50_validation"
RESULTS_FILE = OUTPUT_DIR / "webcam_results.csv"

SEQUENCE_LENGTH = 45
FEATURES = 258

CAMERA_INDEX = 0

# This is only a warning threshold.
# It does NOT mean the prediction is correct.
CONFIDENCE_THRESHOLD = 0.70

SMOOTHING_WINDOW = 5

# Number of samples to record for each sign
SAMPLES_PER_SIGN = 5


# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# COLORS
# ============================================================

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 150, 0)
BLACK = (0, 0, 0)
CYAN = (255, 255, 0)


# ============================================================
# LOAD CLASSES
# ============================================================

def load_classes():

    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        # dictionary: {"0": "1. Dog", ...}
        try:
            return [
                value
                for key, value in sorted(
                    data.items(),
                    key=lambda x: int(x[0])
                )
            ]
        except Exception:
            pass

        for key in ["classes", "class_names", "labels"]:

            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError("Invalid class mapping")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_keypoints(results):

    # ----------------------------
    # Pose
    # ----------------------------

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
            33 * 4,
            dtype=np.float32
        )


    # ----------------------------
    # Left hand
    # ----------------------------

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
            21 * 3,
            dtype=np.float32
        )


    # ----------------------------
    # Right hand
    # ----------------------------

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
            21 * 3,
            dtype=np.float32
        )


    features = np.concatenate(
        [
            pose,
            left_hand,
            right_hand
        ]
    )


    if features.shape[0] != FEATURES:

        features = np.resize(
            features,
            FEATURES
        )


    return features.astype(np.float32)


# ============================================================
# HAND DETECTION
# ============================================================

def hands_present(results):

    left = results.left_hand_landmarks is not None
    right = results.right_hand_landmarks is not None

    return left or right


# ============================================================
# DRAW
# ============================================================

def draw_text(
    frame,
    text,
    x,
    y,
    color=WHITE,
    size=0.7,
    thickness=2
):

    cv2.putText(
        frame,
        str(text),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thickness,
        cv2.LINE_AA
    )


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
# SAFE SAVE
# ============================================================

def safe_save(array, path):

    try:

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Check free space
        usage = shutil.disk_usage(
            path.parent
        )

        required = array.nbytes + 1024 * 1024

        if usage.free < required:

            print()
            print("ERROR: NOT ENOUGH DISK SPACE")
            print("Required :", required)
            print("Available:", usage.free)
            print()

            return False


        temp_path = Path(
            str(path) + ".tmp.npy"
        )

        np.save(
            temp_path,
            array,
            allow_pickle=False
        )

        # Replace final file
        temp_path.replace(path)

        return True

    except Exception as e:

        print()
        print("SAVE ERROR:")
        print(type(e).__name__)
        print(str(e))
        print()

        return False


# ============================================================
# SAVE CSV RESULT
# ============================================================

def save_result(
    expected,
    predicted,
    confidence,
    passed,
    sample_number,
    sequence_file
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    file_exists = RESULTS_FILE.exists()

    with open(
        RESULTS_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        if not file_exists:

            writer.writerow(
                [
                    "expected",
                    "predicted",
                    "confidence",
                    "passed",
                    "sample",
                    "sequence_file"
                ]
            )

        writer.writerow(
            [
                expected,
                predicted,
                round(confidence * 100, 2),
                passed,
                sample_number,
                str(sequence_file)
            ]
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ISL INCLUDE-50 WEBCAM VALIDATION")
    print("=" * 70)

    print()
    print("Loading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded.")

    classes = load_classes()

    print()
    print("Classes:", len(classes))

    if len(classes) != 50:

        raise RuntimeError(
            f"Expected 50 classes, found {len(classes)}"
        )


    print()
    for i, name in enumerate(classes):

        print(
            f"{i:02d} -> {name}"
        )


    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    print()
    print("Opening webcam...")

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Cannot open webcam"
        )


    # Stable resolution
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


    print()
    print("=" * 70)
    print("WEBCAM READY")
    print("=" * 70)

    print()
    print("CONTROLS")
    print()
    print("UP/DOWN : select sign")
    print("SPACE   : record sample")
    print("R       : reset prediction")
    print("ESC     : exit")
    print()
    print("Each sign:", SAMPLES_PER_SIGN, "samples")
    print()


    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    selected_index = 0

    prediction_history = deque(
        maxlen=SMOOTHING_WINDOW
    )

    last_prediction = "Waiting..."
    last_confidence = 0.0
    last_pass = False

    sample_counts = {
        name: 0
        for name in classes
    }


    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:


        while True:

            ret, frame = cap.read()

            if not ret:

                print(
                    "Could not read webcam frame."
                )

                break


            # Mirror camera
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


            # Draw landmarks
            draw_landmarks(
                frame,
                results
            )


            expected = classes[
                selected_index
            ]


            # ------------------------------------------------
            # UI
            # ------------------------------------------------

            h, w = frame.shape[:2]

            cv2.rectangle(
                frame,
                (0, 0),
                (w, 175),
                BLACK,
                -1
            )


            draw_text(
                frame,
                f"EXPECTED: {expected}",
                20,
                35,
                YELLOW,
                0.85,
                2
            )


            draw_text(
                frame,
                f"PREDICTED: {last_prediction}",
                20,
                70,
                WHITE,
                0.75,
                2
            )


            draw_text(
                frame,
                f"CONFIDENCE: {last_confidence * 100:.1f}%",
                20,
                105,
                CYAN,
                0.70,
                2
            )


            if last_pass:

                draw_text(
                    frame,
                    "RESULT: PASS",
                    20,
                    140,
                    GREEN,
                    0.75,
                    2
                )

            else:

                draw_text(
                    frame,
                    "RESULT: FAIL",
                    20,
                    140,
                    RED,
                    0.75,
                    2
                )


            draw_text(
                frame,
                f"SAMPLES: {sample_counts[expected]}/{SAMPLES_PER_SIGN}",
                20,
                h - 45,
                WHITE,
                0.65,
                2
            )


            if not hands_present(results):

                draw_text(
                    frame,
                    "SHOW YOUR HAND",
                    w - 300,
                    h - 45,
                    RED,
                    0.65,
                    2
                )


            draw_text(
                frame,
                "UP/DOWN=SIGN  SPACE=TEST  R=RESET  ESC=EXIT",
                20,
                h - 15,
                WHITE,
                0.50,
                1
            )


            cv2.imshow(
                "ISL INCLUDE-50 Webcam Validation",
                frame
            )


            key = cv2.waitKey(1) & 0xFF


            # =================================================
            # ESC
            # =================================================

            if key == 27:

                break


            # =================================================
            # UP
            # =================================================

            elif key == 82 or key == 0:

                selected_index -= 1

                if selected_index < 0:

                    selected_index = len(classes) - 1

                last_prediction = "Waiting..."
                last_confidence = 0
                last_pass = False

                prediction_history.clear()

                print()
                print(
                    "SELECTED:",
                    classes[selected_index]
                )


            # =================================================
            # DOWN
            # =================================================

            elif key == 84:

                selected_index += 1

                if selected_index >= len(classes):

                    selected_index = 0

                last_prediction = "Waiting..."
                last_confidence = 0
                last_pass = False

                prediction_history.clear()

                print()
                print(
                    "SELECTED:",
                    classes[selected_index]
                )


            # =================================================
            # R
            # =================================================

            elif key in [
                ord("r"),
                ord("R")
            ]:

                last_prediction = "Waiting..."
                last_confidence = 0
                last_pass = False

                prediction_history.clear()


            # =================================================
            # SPACE
            # =================================================

            elif key == 32:

                print()
                print("=" * 70)
                print(
                    "RECORDING:",
                    expected
                )
                print(
                    "SAMPLE:",
                    sample_counts[expected] + 1,
                    "/",
                    SAMPLES_PER_SIGN
                )
                print("=" * 70)

                print(
                    "Prepare your sign..."
                )

                # Countdown
                for countdown in [3, 2, 1]:

                    start = cv2.getTickCount()

                    while (
                        cv2.getTickCount() - start
                    ) / cv2.getTickFrequency() < 0.7:

                        ret2, frame2 = cap.read()

                        if not ret2:
                            continue

                        frame2 = cv2.flip(
                            frame2,
                            1
                        )

                        draw_text(
                            frame2,
                            f"GET READY: {countdown}",
                            30,
                            80,
                            YELLOW,
                            1.2,
                            3
                        )

                        cv2.imshow(
                            "ISL INCLUDE-50 Webcam Validation",
                            frame2
                        )

                        if (
                            cv2.waitKey(1) & 0xFF
                        ) == 27:

                            cap.release()
                            cv2.destroyAllWindows()
                            return


                print(
                    "Recording 45 frames..."
                )


                sequence = []

                frame_counter = 0


                while len(sequence) < SEQUENCE_LENGTH:

                    ret2, frame2 = cap.read()

                    if not ret2:
                        continue


                    frame2 = cv2.flip(
                        frame2,
                        1
                    )


                    rgb2 = cv2.cvtColor(
                        frame2,
                        cv2.COLOR_BGR2RGB
                    )


                    results2 = holistic.process(
                        rgb2
                    )


                    keypoints = extract_keypoints(
                        results2
                    )

                    sequence.append(
                        keypoints
                    )


                    # Recording UI

                    progress = len(sequence)

                    draw_landmarks(
                        frame2,
                        results2
                    )


                    cv2.rectangle(
                        frame2,
                        (0, 0),
                        (
                            frame2.shape[1],
                            100
                        ),
                        BLACK,
                        -1
                    )


                    draw_text(
                        frame2,
                        f"RECORDING {progress}/{SEQUENCE_LENGTH}",
                        25,
                        40,
                        YELLOW,
                        0.8,
                        2
                    )


                    draw_text(
                        frame2,
                        expected,
                        25,
                        75,
                        WHITE,
                        0.7,
                        2
                    )


                    cv2.imshow(
                        "ISL INCLUDE-50 Webcam Validation",
                        frame2
                    )


                    if (
                        cv2.waitKey(1) & 0xFF
                    ) == 27:

                        cap.release()
                        cv2.destroyAllWindows()
                        return


                sequence = np.asarray(
                    sequence,
                    dtype=np.float32
                )


                # ------------------------------------------------
                # Prediction
                # ------------------------------------------------

                X = np.expand_dims(
                    sequence,
                    axis=0
                )


                probabilities = model.predict(
                    X,
                    verbose=0
                )[0]


                prediction_index = int(
                    np.argmax(probabilities)
                )


                confidence = float(
                    probabilities[
                        prediction_index
                    ]
                )


                predicted = classes[
                    prediction_index
                ]


                passed = (
                    predicted == expected
                )


                # ------------------------------------------------
                # Save
                # ------------------------------------------------

                sample_number = (
                    sample_counts[expected] + 1
                )


                safe_name = (
                    f"{selected_index:02d}_"
                    + expected
                    .replace(" ", "_")
                    .replace("/", "_")
                )


                output_folder = (
                    OUTPUT_DIR /
                    safe_name
                )


                sequence_file = (
                    output_folder /
                    f"sample_{sample_number:03d}.npy"
                )


                saved = safe_save(
                    sequence,
                    sequence_file
                )


                if saved:

                    sample_counts[
                        expected
                    ] += 1

                else:

                    print(
                        "WARNING: Sample was NOT saved."
                    )


                # ------------------------------------------------
                # Display result
                # ------------------------------------------------

                last_prediction = predicted
                last_confidence = confidence
                last_pass = passed

                print()
                print("=" * 70)
                print(
                    "EXPECTED   :",
                    expected
                )
                print(
                    "PREDICTED  :",
                    predicted
                )
                print(
                    "CONFIDENCE :",
                    f"{confidence * 100:.2f}%"
                )
                print(
                    "RESULT     :",
                    "PASS" if passed else "FAIL"
                )

                if saved:

                    print(
                        "SAVED      :",
                        sequence_file
                    )

                else:

                    print(
                        "SAVED      : NO"
                    )

                print("=" * 70)


                save_result(
                    expected,
                    predicted,
                    confidence,
                    passed,
                    sample_number,
                    sequence_file
                )


    cap.release()

    cv2.destroyAllWindows()

    print()
    print("=" * 70)
    print("WEBCAM VALIDATION COMPLETE")
    print("=" * 70)

    print()
    print(
        "Results:",
        RESULTS_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()