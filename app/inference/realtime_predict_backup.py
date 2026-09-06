"""
INCLUDE-50 ROBUST REAL-TIME ISL RECOGNITION

Model:
    include50_lstm_v2_best.keras

Input:
    45 frames x 258 features

Features:
    Pose      = 33 x 4 = 132
    Left hand = 21 x 3 = 63
    Right hand= 21 x 3 = 63

Total:
    258 features

IMPORTANT:
    The realtime preprocessing below keeps the same landmark
    representation as the training extractor.

Controls:
    Q = Quit
    R = Reset
    SPACE = Select current prediction
    H = Show/hide information
"""

import json
from pathlib import Path
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

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

SIGN_INFO_PATH = (
    BASE_DIR
    / "app"
    / "data"
    / "sign_info.json"
)

SEQUENCE_LENGTH = 45
FEATURES = 258
NUM_CLASSES = 50

# ------------------------------------------------------------
# Prediction settings
# ------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.50

SMOOTHING_WINDOW = 7

PREDICT_EVERY = 2

# Number of consecutive frames without a hand
# before resetting the sequence.
NO_HAND_RESET_FRAMES = 12

# Minimum detection quality required to enter
# a frame into the prediction buffer.
MIN_QUALITY_FOR_PREDICTION = 0.85

# ------------------------------------------------------------
# Camera
# ------------------------------------------------------------

CAMERA_WIDTH = 960
CAMERA_HEIGHT = 720


# ============================================================
# COLORS
# ============================================================

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 150, 0)
CYAN = (255, 255, 0)


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
    # List format
    # --------------------------------------------------------

    if isinstance(data, list):

        classes = [
            str(x)
            for x in data
        ]

        return classes


    # --------------------------------------------------------
    # Dictionary format
    # --------------------------------------------------------

    if isinstance(data, dict):

        # {"classes": [...]}
        for key in [
            "classes",
            "class_names",
            "labels"
        ]:

            if key in data:

                value = data[key]

                if isinstance(value, list):

                    return [
                        str(x)
                        for x in value
                    ]


        # {"0": "1. Dog", "1": "1. loud", ...}
        try:

            ordered = []

            for key, value in sorted(
                data.items(),
                key=lambda x: int(x[0])
            ):

                ordered.append(
                    str(value)
                )

            if len(ordered) == NUM_CLASSES:

                return ordered

        except Exception:

            pass


        # {"Dog": 0, "Bird": 1}
        try:

            if all(
                isinstance(v, int)
                for v in data.values()
            ):

                ordered = [
                    None
                ] * len(data)

                for name, index in data.items():

                    ordered[index] = str(name)

                return ordered

        except Exception:

            pass


    raise ValueError(
        "Invalid classes JSON format."
    )


# ============================================================
# LOAD SIGN INFORMATION
# ============================================================

def load_sign_info():

    if not SIGN_INFO_PATH.exists():

        return {}


    try:

        with open(
            SIGN_INFO_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return {}


# ============================================================
# GET SIGN INFORMATION
# ============================================================

def get_sign_info(
    sign,
    sign_info
):

    if sign in sign_info:

        return sign_info[sign]


    # Remove numeric prefix
    clean_name = sign

    if "." in sign:

        clean_name = (
            sign.split(".", 1)[1]
            .strip()
        )


    for key, value in sign_info.items():

        key_clean = key

        if "." in key:

            key_clean = (
                key.split(".", 1)[1]
                .strip()
            )


        if (
            key_clean.lower()
            ==
            clean_name.lower()
        ):

            return value


    return {
        "title": sign,
        "category": "ISL",
        "instruction":
            "Perform the sign shown in the reference video."
    }


# ============================================================
# EXTRACT EXACTLY 258 FEATURES
# ============================================================

def extract_keypoints(results):

    # --------------------------------------------------------
    # Pose
    # 33 landmarks x 4
    # x, y, z, visibility
    # --------------------------------------------------------

    if results.pose_landmarks:

        pose = np.array(
            [
                [
                    lm.x,
                    lm.y,
                    lm.z,
                    lm.visibility
                ]
                for lm
                in results.pose_landmarks.landmark
            ],
            dtype=np.float32
        ).flatten()

    else:

        pose = np.zeros(
            132,
            dtype=np.float32
        )


    # --------------------------------------------------------
    # Left hand
    # 21 landmarks x 3
    # --------------------------------------------------------

    if results.left_hand_landmarks:

        left_hand = np.array(
            [
                [
                    lm.x,
                    lm.y,
                    lm.z
                ]
                for lm
                in results.left_hand_landmarks.landmark
            ],
            dtype=np.float32
        ).flatten()

    else:

        left_hand = np.zeros(
            63,
            dtype=np.float32
        )


    # --------------------------------------------------------
    # Right hand
    # 21 landmarks x 3
    # --------------------------------------------------------

    if results.right_hand_landmarks:

        right_hand = np.array(
            [
                [
                    lm.x,
                    lm.y,
                    lm.z
                ]
                for lm
                in results.right_hand_landmarks.landmark
            ],
            dtype=np.float32
        ).flatten()

    else:

        right_hand = np.zeros(
            63,
            dtype=np.float32
        )


    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    keypoints = np.concatenate(
        [
            pose,
            left_hand,
            right_hand
        ]
    ).astype(
        np.float32
    )


    # --------------------------------------------------------
    # STRICT validation
    # --------------------------------------------------------

    if keypoints.shape != (
        FEATURES,
    ):

        raise ValueError(
            "Wrong keypoint shape: "
            f"{keypoints.shape}. "
            f"Expected ({FEATURES},)"
        )


    if np.isnan(keypoints).any():

        raise ValueError(
            "NaN detected in keypoints."
        )


    return keypoints


# ============================================================
# DETECTION QUALITY
# ============================================================

def get_detection_quality(results):

    pose_detected = (
        results.pose_landmarks
        is not None
    )

    left_detected = (
        results.left_hand_landmarks
        is not None
    )

    right_detected = (
        results.right_hand_landmarks
        is not None
    )


    hand_count = (
        int(left_detected)
        +
        int(right_detected)
    )


    if hand_count == 2:

        quality = 1.0

    elif hand_count == 1:

        quality = 0.85

    elif pose_detected:

        quality = 0.45

    else:

        quality = 0.0


    return (
        quality,
        pose_detected,
        left_detected,
        right_detected
    )


# ============================================================
# LANDMARK SMOOTHER
# ============================================================

class LandmarkSmoother:

    def __init__(
        self,
        alpha=0.65
    ):

        self.alpha = alpha
        self.previous = None


    def update(
        self,
        current
    ):

        current = np.asarray(
            current,
            dtype=np.float32
        )


        if self.previous is None:

            self.previous = (
                current.copy()
            )

            return current


        smoothed = (
            self.alpha * current
            +
            (1.0 - self.alpha)
            * self.previous
        )


        self.previous = (
            smoothed.copy()
        )


        return smoothed


    def reset(self):

        self.previous = None


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
# DRAW PANEL
# ============================================================

def draw_panel(
    frame,
    x1,
    y1,
    x2,
    y2
):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        BLACK,
        -1
    )

    frame[:] = cv2.addWeighted(
        overlay,
        0.72,
        frame,
        0.28,
        0
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        WHITE,
        1
    )


# ============================================================
# DRAW PREDICTION
# ============================================================

def draw_prediction(
    frame,
    predicted_sign,
    confidence,
    selected_sign
):

    h, w = frame.shape[:2]


    cv2.rectangle(
        frame,
        (10, 50),
        (min(w - 10, 620), 150),
        BLACK,
        -1
    )


    draw_text(
        frame,
        "PREDICTION",
        (25, 80),
        YELLOW,
        0.65,
        2
    )


    color = (
        GREEN
        if confidence >= CONFIDENCE_THRESHOLD
        else RED
    )


    draw_text(
        frame,
        predicted_sign,
        (25, 115),
        color,
        0.85,
        2
    )


    draw_text(
        frame,
        f"Confidence: "
        f"{confidence * 100:.1f}%",
        (25, 140),
        WHITE,
        0.55,
        2
    )


    if selected_sign:

        draw_text(
            frame,
            f"Selected: {selected_sign}",
            (25, h - 25),
            BLUE,
            0.60,
            2
        )


# ============================================================
# DRAW SIGN INFORMATION
# ============================================================

def draw_sign_information(
    frame,
    selected_sign,
    sign_info
):

    if not selected_sign:

        return


    info = get_sign_info(
        selected_sign,
        sign_info
    )


    h, w = frame.shape[:2]


    x1 = max(
        10,
        w - 390
    )

    y1 = 50

    x2 = w - 10

    y2 = 220


    overlay = frame.copy()


    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        BLACK,
        -1
    )


    frame[:] = cv2.addWeighted(
        overlay,
        0.75,
        frame,
        0.25,
        0
    )


    draw_text(
        frame,
        info.get(
            "title",
            selected_sign
        ),
        (x1 + 15, y1 + 35),
        YELLOW,
        0.65,
        2
    )


    draw_text(
        frame,
        "Category: "
        + info.get(
            "category",
            "ISL"
        ),
        (x1 + 15, y1 + 65),
        WHITE,
        0.48,
        1
    )


    instruction = info.get(
        "instruction",
        "Perform the sign shown in the reference video."
    )


    words = instruction.split()

    lines = []

    current = ""


    for word in words:

        test = (
            current
            + " "
            + word
        ).strip()


        if len(test) > 42:

            if current:

                lines.append(
                    current
                )

            current = word

        else:

            current = test


    if current:

        lines.append(
            current
        )


    y = y1 + 95


    for line in lines[:4]:

        draw_text(
            frame,
            line,
            (x1 + 15, y),
            WHITE,
            0.43,
            1
        )

        y += 22


# ============================================================
# DRAW STATUS
# ============================================================

def draw_status(
    frame,
    sequence_size,
    quality,
    pose,
    left,
    right,
    good_frames
):

    h, w = frame.shape[:2]


    # --------------------------------------------------------
    # Sequence status
    # --------------------------------------------------------

    if sequence_size < SEQUENCE_LENGTH:

        status = (
            f"Building sequence: "
            f"{sequence_size}/"
            f"{SEQUENCE_LENGTH}"
        )

        status_color = YELLOW

    elif good_frames < 20:

        status = (
            f"Improving landmarks: "
            f"{good_frames}/20"
        )

        status_color = YELLOW

    else:

        status = (
            "READY - PERFORM ONE SIGN"
        )

        status_color = GREEN


    draw_text(
        frame,
        status,
        (20, h - 105),
        status_color,
        0.55,
        2
    )


    # --------------------------------------------------------
    # Hands
    # --------------------------------------------------------

    if left and right:

        hands_text = "BOTH HANDS"

    elif left:

        hands_text = "LEFT HAND"

    elif right:

        hands_text = "RIGHT HAND"

    else:

        hands_text = "NO HAND"


    draw_text(
        frame,
        f"Hands: {hands_text}",
        (20, h - 78),
        WHITE,
        0.48,
        1
    )


    # --------------------------------------------------------
    # Pose
    # --------------------------------------------------------

    pose_text = (
        "DETECTED"
        if pose
        else
        "NOT DETECTED"
    )


    draw_text(
        frame,
        f"Pose: {pose_text}",
        (20, h - 52),
        WHITE,
        0.48,
        1
    )


    draw_text(
        frame,
        f"Quality: {quality:.2f}",
        (20, h - 27),
        WHITE,
        0.45,
        1
    )


# ============================================================
# RESET
# ============================================================

def reset_state(
    sequence,
    prediction_history,
    smoother
):

    sequence.clear()

    prediction_history.clear()

    smoother.reset()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "INCLUDE-50 ROBUST REAL-TIME ISL RECOGNITION"
    )

    print("=" * 70)


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print("\nLoading model...")


    if not MODEL_PATH.exists():

        print(
            "ERROR: Model not found:"
        )

        print(
            MODEL_PATH
        )

        return


    model = load_model(
        MODEL_PATH
    )


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


    # --------------------------------------------------------
    # Validate model
    # --------------------------------------------------------

    if (
        model.input_shape[-2]
        != SEQUENCE_LENGTH
        or
        model.input_shape[-1]
        != FEATURES
        or
        model.output_shape[-1]
        != NUM_CLASSES
    ):

        print(
            "\nERROR:"
        )

        print(
            "Model shape does not match configuration."
        )

        return


    # ========================================================
    # LOAD CLASSES
    # ========================================================

    print(
        "\nLoading classes..."
    )


    class_names = load_classes()


    if len(class_names) != NUM_CLASSES:

        print(
            "ERROR: Expected 50 classes."
        )

        print(
            "Found:",
            len(class_names)
        )

        return


    print(
        "50 classes loaded."
    )


    # --------------------------------------------------------
    # Print mapping
    # --------------------------------------------------------

    print(
        "\nMODEL INDEX -> CLASS"
    )


    for i, name in enumerate(
        class_names
    ):

        print(
            f"{i:02d} -> {name}"
        )


    # ========================================================
    # SIGN INFORMATION
    # ========================================================

    sign_info = load_sign_info()


    print(
        "\nSign information:",
        len(sign_info)
    )


    # ========================================================
    # CAMERA
    # ========================================================

    print(
        "\nOpening webcam..."
    )


    cap = cv2.VideoCapture(0)


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


    print(
        "Webcam started."
    )


    print(
        "\nControls:"
    )

    print(
        "Q = Quit"
    )

    print(
        "R = Reset"
    )

    print(
        "SPACE = Select prediction"
    )

    print(
        "H = Show/hide information"
    )


    # ========================================================
    # STATE
    # ========================================================

    sequence = deque(
        maxlen=SEQUENCE_LENGTH
    )


    prediction_history = deque(
        maxlen=SMOOTHING_WINDOW
    )


    smoother = LandmarkSmoother(
        alpha=0.65
    )


    predicted_sign = "Waiting..."

    predicted_confidence = 0.0

    selected_sign = None

    show_info = True

    frame_counter = 0

    no_hand_counter = 0

    good_frame_history = deque(
        maxlen=SEQUENCE_LENGTH
    )


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    with mp_holistic.Holistic(

        static_image_mode=False,

        model_complexity=1,

        smooth_landmarks=True,

        enable_segmentation=False,

        refine_face_landmarks=False,

        min_detection_confidence=0.5,

        min_tracking_confidence=0.5

    ) as holistic:


        while cap.isOpened():

            # =================================================
            # READ FRAME
            # =================================================

            ret, frame = cap.read()


            if not ret:

                print(
                    "Failed to read webcam."
                )

                break


            frame_counter += 1


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
            # DETECTION QUALITY
            # =================================================

            (
                quality,
                pose_detected,
                left_detected,
                right_detected
            ) = get_detection_quality(
                results
            )


            usable_hand = (
                left_detected
                or
                right_detected
            )


            # =================================================
            # DRAW LANDMARKS
            # =================================================

            draw_landmarks(
                frame,
                results
            )


            # =================================================
            # NO HAND HANDLING
            # =================================================

            if usable_hand:

                no_hand_counter = 0

            else:

                no_hand_counter += 1


            if (
                no_hand_counter
                >= NO_HAND_RESET_FRAMES
            ):

                reset_state(
                    sequence,
                    prediction_history,
                    smoother
                )

                predicted_sign = (
                    "Waiting..."
                )

                predicted_confidence = 0.0

                no_hand_counter = 0


            # =================================================
            # EXTRACT KEYPOINTS
            # =================================================

            keypoints = extract_keypoints(
                results
            )


            # =================================================
            # IMPORTANT:
            # ONLY GOOD HAND FRAMES ENTER BUFFER
            # =================================================

            if (
                usable_hand
                and
                quality
                >= MIN_QUALITY_FOR_PREDICTION
            ):

                # Smooth exactly once
                keypoints = smoother.update(
                    keypoints
                )


                sequence.append(
                    keypoints
                )


                good_frame_history.append(
                    True
                )

            else:

                good_frame_history.append(
                    False
                )


            # =================================================
            # PREDICTION
            # =================================================

            good_frames = sum(
                good_frame_history
            )


            if (
                len(sequence)
                ==
                SEQUENCE_LENGTH
                and
                good_frames
                >= 20
                and
                frame_counter
                % PREDICT_EVERY
                == 0
            ):

                input_data = np.asarray(
                    sequence,
                    dtype=np.float32
                )


                # Final shape:
                # (1, 45, 258)

                input_data = np.expand_dims(
                    input_data,
                    axis=0
                )


                # ------------------------------------------------
                # Safety checks
                # ------------------------------------------------

                if input_data.shape != (
                    1,
                    SEQUENCE_LENGTH,
                    FEATURES
                ):

                    print(
                        "WARNING: Wrong input shape:",
                        input_data.shape
                    )

                    continue


                if np.isnan(
                    input_data
                ).any():

                    print(
                        "WARNING: NaN in input."
                    )

                    continue


                # =================================================
                # MODEL
                # =================================================

                try:

                    prediction = model.predict(
                        input_data,
                        verbose=0
                    )[0]


                    index = int(
                        np.argmax(
                            prediction
                        )
                    )


                    confidence = float(
                        prediction[index]
                    )


                    if index < len(
                        class_names
                    ):

                        current_sign = (
                            class_names[index]
                        )


                        # ------------------------------------------------
                        # Store prediction
                        # ------------------------------------------------

                        prediction_history.append(
                            (
                                index,
                                confidence
                            )
                        )


                        # ------------------------------------------------
                        # Majority vote
                        # ------------------------------------------------

                        indices = [
                            item[0]
                            for item
                            in prediction_history
                        ]


                        if indices:

                            most_common_index = (
                                Counter(
                                    indices
                                )
                                .most_common(1)[0][0]
                            )


                            # Average confidence for
                            # the selected majority class

                            matching_confidences = [
                                conf
                                for idx, conf
                                in prediction_history
                                if idx
                                ==
                                most_common_index
                            ]


                            if matching_confidences:

                                predicted_confidence = (
                                    sum(
                                        matching_confidences
                                    )
                                    /
                                    len(
                                        matching_confidences
                                    )
                                )


                            predicted_sign = (
                                class_names[
                                    most_common_index
                                ]
                            )


                except Exception as e:

                    print(
                        "Prediction error:",
                        e
                    )


            # =================================================
            # DRAW PREDICTION
            # =================================================

            draw_prediction(
                frame,
                predicted_sign,
                predicted_confidence,
                selected_sign
            )


            # =================================================
            # SIGN INFORMATION
            # =================================================

            if (
                show_info
                and
                selected_sign
            ):

                draw_sign_information(
                    frame,
                    selected_sign,
                    sign_info
                )


            # =================================================
            # STATUS
            # =================================================

            draw_status(
                frame,
                len(sequence),
                quality,
                pose_detected,
                left_detected,
                right_detected,
                good_frames
            )


            # =================================================
            # TOP STATUS
            # =================================================

            if len(sequence) < SEQUENCE_LENGTH:

                draw_text(
                    frame,
                    f"Collecting: "
                    f"{len(sequence)}/"
                    f"{SEQUENCE_LENGTH}",
                    (20, 35),
                    YELLOW,
                    0.55,
                    2
                )

            else:

                draw_text(
                    frame,
                    "READY - PERFORM SIGN",
                    (20, 35),
                    GREEN,
                    0.55,
                    2
                )


            # =================================================
            # WINDOW
            # =================================================

            cv2.imshow(
                "INCLUDE-50 Robust ISL Recognition",
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
                    prediction_history,
                    smoother
                )


                predicted_sign = (
                    "Waiting..."
                )

                predicted_confidence = 0.0

                good_frame_history.clear()

                no_hand_counter = 0

                print(
                    "Prediction reset."
                )


            # ------------------------------------------------
            # SPACE
            # ------------------------------------------------

            elif key == 32:

                if (
                    predicted_sign
                    !=
                    "Waiting..."
                ):

                    selected_sign = (
                        predicted_sign
                    )


                    print(
                        "Selected sign:",
                        selected_sign
                    )


            # ------------------------------------------------
            # H
            # ------------------------------------------------

            elif key == ord("h"):

                show_info = (
                    not show_info
                )


    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()


    print(
        "\nWebcam stopped."
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()