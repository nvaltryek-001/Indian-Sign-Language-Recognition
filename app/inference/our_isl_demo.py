import cv2
import json
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pyttsx3
import time

# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "app/models/isl_14class.keras"
LABELS = "app/models/labels.json"

# ============================================================
# LOAD LABELS
# ============================================================

with open(LABELS, "r") as f:
    ACTIONS = json.load(f)

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading ISL LSTM model...")

model = tf.keras.models.load_model(MODEL)

print("Model loaded successfully.")

# ============================================================
# TEXT TO SPEECH
# ============================================================

engine = pyttsx3.init("sapi5")

engine.setProperty("rate", 145)
engine.setProperty("volume", 1.0)

voices = engine.getProperty("voices")

if voices:
    engine.setProperty("voice", voices[0].id)

# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def extract_keypoints(results):

    # Pose = 33 × 4 = 132
    pose = (
        np.array([
            [r.x, r.y, r.z, r.visibility]
            for r in results.pose_landmarks.landmark
        ]).flatten()
        if results.pose_landmarks
        else np.zeros(132)
    )

    # Left hand = 21 × 3 = 63
    left = (
        np.array([
            [r.x, r.y, r.z]
            for r in results.left_hand_landmarks.landmark
        ]).flatten()
        if results.left_hand_landmarks
        else np.zeros(63)
    )

    # Right hand = 21 × 3 = 63
    right = (
        np.array([
            [r.x, r.y, r.z]
            for r in results.right_hand_landmarks.landmark
        ]).flatten()
        if results.right_hand_landmarks
        else np.zeros(63)
    )

    # Total = 132 + 63 + 63 = 258
    return np.concatenate(
        [pose, left, right]
    ).astype(np.float32)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam could not be opened")

# ============================================================
# START MESSAGE
# ============================================================

print()
print("=" * 65)
print("             ISL AI TRANSLATOR")
print("=" * 65)

print()
print("Supported signs:")

for i, action in enumerate(ACTIONS, 1):
    print(f"  {i}. {action.replace('_', ' ')}")

print()
print("SPACE = record sign")
print("Q     = quit")

print("=" * 65)

# ============================================================
# MEDIAPIPE LOOP
# ============================================================

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while True:

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.flip(frame, 1)

        # ----------------------------------------------------
        # MAIN SCREEN
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "ISL AI TRANSLATOR",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "SPACE = Record Sign | Q = Quit",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "ISL AI Translator",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if key == ord("q"):
            break

        # ----------------------------------------------------
        # RECORD ONLY WHEN SPACE IS PRESSED
        # ----------------------------------------------------

        if key != 32:
            continue

        print()
        print("Recording sign...")

        sequence = []

        start_time = time.time()

        # ----------------------------------------------------
        # RECORD 5 SECOND SIGN
        # ----------------------------------------------------

        while time.time() - start_time < 5:

            success, frame = cap.read()

            if not success:
                continue

            frame = cv2.flip(frame, 1)

            # Convert BGR → RGB
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            rgb.flags.writeable = False

            results = holistic.process(rgb)

            rgb.flags.writeable = True

            # Extract landmarks
            keypoints = extract_keypoints(
                results
            )

            sequence.append(keypoints)

            # ------------------------------------------------
            # DRAW LANDMARKS
            # ------------------------------------------------

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS
            )

            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

            remaining = max(
                0,
                5 - (time.time() - start_time)
            )

            cv2.putText(
                frame,
                "RECORDING",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                "Time: "
                + str(round(remaining, 1))
                + " sec",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "PERFORM YOUR SIGN",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.imshow(
                "ISL AI Translator",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):

                cap.release()
                cv2.destroyAllWindows()

                raise SystemExit

        # ====================================================
        # CHECK FRAMES
        # ====================================================

        if len(sequence) < 45:

            print(
                "Not enough frames:",
                len(sequence)
            )

            continue

        # ====================================================
        # CONVERT TO EXACTLY 45 FRAMES
        # ====================================================

        indices = np.linspace(
            0,
            len(sequence) - 1,
            45
        ).astype(int)

        data = np.array(
            [
                sequence[i]
                for i in indices
            ],
            dtype=np.float32
        )

        print(
            "Input shape:",
            data.shape
        )

        # Expected:
        # (45, 258)

        # ====================================================
        # LSTM PREDICTION
        # ====================================================

        prediction = model.predict(
            np.expand_dims(
                data,
                axis=0
            ),
            verbose=0
        )[0]

        # Highest probability
        index = int(
            np.argmax(prediction)
        )

        confidence = float(
            prediction[index]
        )

        word = ACTIONS[index]

        spoken_word = word.replace(
            "_",
            " "
        )

        # ====================================================
        # TERMINAL OUTPUT
        # ====================================================

        print()
        print("=" * 55)

        print(
            "PREDICTED:",
            spoken_word
        )

        print(
            "CONFIDENCE:",
            round(
                confidence * 100,
                2
            ),
            "%"
        )

        print(
            "SPEAKING:",
            spoken_word
        )

        print("=" * 55)

        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        result = frame.copy()

        cv2.rectangle(
            result,
            (0, 0),
            (900, 190),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            result,
            "PREDICTION: "
            + spoken_word.upper(),
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            result,
            "CONFIDENCE: "
            + str(
                round(
                    confidence * 100,
                    2
                )
            )
            + "%",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            result,
            "SPEAKING: "
            + spoken_word.upper(),
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            result,
            "SPACE = Next Sign | Q = Quit",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "ISL AI Translator",
            result
        )

        # ====================================================
        # SPEAK EVERY PREDICTION
        # ====================================================

        try:

            engine.stop()

            engine.say(
                spoken_word
            )

            engine.runAndWait()

        except Exception as error:

            print(
                "Speech error:",
                error
            )

        # ====================================================
        # WAIT FOR NEXT SIGN
        # ====================================================

        while True:

            key = cv2.waitKey(0) & 0xFF

            if key == ord("q"):

                cap.release()
                cv2.destroyAllWindows()

                raise SystemExit

            if key == 32:

                break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

engine.stop()

print()
print("ISL AI Translator stopped.")
