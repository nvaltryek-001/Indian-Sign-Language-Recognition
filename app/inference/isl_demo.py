import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyttsx3
import os
import time

MODEL_PATH = os.path.join(
    "baseline",
    "lstm-model",
    "170-0.83.hdf5"
)

ACTIONS = np.array([
    "Hello",
    "How are you",
    "Thank you"
])

SEQUENCE_LENGTH = 45
RECORD_SECONDS = 5
CAPTURE_FPS = 10

CONFIDENCE_THRESHOLD = 0.60

# ============================================================
# MODEL
# ============================================================

print("Loading pretrained LSTM...")

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(
        64,
        return_sequences=True,
        activation="relu",
        input_shape=(45, 258)
    ),
    tf.keras.layers.LSTM(
        128,
        return_sequences=True,
        activation="relu"
    ),
    tf.keras.layers.LSTM(
        256,
        return_sequences=True,
        activation="relu"
    ),
    tf.keras.layers.LSTM(
        64,
        return_sequences=False,
        activation="relu"
    ),
    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),
    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),
    tf.keras.layers.Dense(
        3,
        activation="softmax"
    )
])

model.compile(
    optimizer="Adam",
    loss="categorical_crossentropy"
)

model.load_weights(MODEL_PATH)

print("Pretrained LSTM loaded successfully.")

# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def extract_keypoints(results):

    pose = (
        np.array([
            [
                res.x,
                res.y,
                res.z,
                res.visibility
            ]
            for res in results.pose_landmarks.landmark
        ]).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )

    left_hand = (
        np.array([
            [
                res.x,
                res.y,
                res.z
            ]
            for res in results.left_hand_landmarks.landmark
        ]).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )

    right_hand = (
        np.array([
            [
                res.x,
                res.y,
                res.z
            ]
            for res in results.right_hand_landmarks.landmark
        ]).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )

    return np.concatenate([
        pose,
        left_hand,
        right_hand
    ]).astype(np.float32)


# ============================================================
# SPEECH
# ============================================================

engine = pyttsx3.init()

engine.setProperty("rate", 150)


def speak(text):

    print()
    print("SPEAKING:", text)

    engine.say(text)
    engine.runAndWait()


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


print()
print("=" * 65)
print("             ISL AI TRANSLATOR")
print("=" * 65)
print()
print("Supported signs:")
print("  1. Hello")
print("  2. How are you")
print("  3. Thank you")
print()
print("SPACE = record sign")
print("Q     = quit")
print("=" * 65)


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
        # Live display
        # ----------------------------------------------------

        display = frame.copy()

        cv2.rectangle(
            display,
            (0, 0),
            (900, 170),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            display,
            "ISL AI TRANSLATOR",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            "Press SPACE to record a sign",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "Press Q to quit",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "Ready",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "ISL AI Translator",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key != 32:
            continue

        # ====================================================
        # RECORDING
        # ====================================================

        print()
        print("Recording started...")
        print("Perform ONE sign continuously.")

        frames = []

        start_time = time.time()

        while time.time() - start_time < RECORD_SECONDS:

            success, frame = cap.read()

            if not success:
                continue

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            rgb.flags.writeable = False

            results = holistic.process(rgb)

            rgb.flags.writeable = True

            # Extract 258 features
            keypoints = extract_keypoints(results)

            frames.append(
                keypoints
            )

            # Draw landmarks
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

            elapsed = time.time() - start_time

            remaining = max(
                0,
                RECORD_SECONDS - elapsed
            )

            cv2.rectangle(
                frame,
                (0, 0),
                (850, 110),
                (20, 20, 20),
                -1
            )

            cv2.putText(
                frame,
                "RECORDING SIGN",
                (20, 40),
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
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
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

        print(
            "Captured",
            len(frames),
            "frames."
        )

        # ====================================================
        # SAMPLE EXACTLY 45 FRAMES
        # ====================================================

        if len(frames) < 45:

            print(
                "Not enough frames. Try again."
            )

            continue

        indices = np.linspace(
            0,
            len(frames) - 1,
            SEQUENCE_LENGTH
        ).astype(int)

        sequence = np.array(
            [frames[i] for i in indices],
            dtype=np.float32
        )

        print(
            "Input shape:",
            sequence.shape
        )

        # ====================================================
        # PREDICTION
        # ====================================================

        input_data = np.expand_dims(
            sequence,
            axis=0
        )

        prediction = model.predict(
            input_data,
            verbose=0
        )[0]

        predicted_index = np.argmax(
            prediction
        )

        predicted_word = ACTIONS[
            predicted_index
        ]

        confidence = float(
            prediction[predicted_index]
        )

        print()
        print("=" * 50)
        print(
            "Prediction:",
            predicted_word
        )
        print(
            "Confidence:",
            round(confidence * 100, 2),
            "%"
        )
        print(
            "All probabilities:",
            prediction
        )
        print("=" * 50)

        # ====================================================
        # RESULT SCREEN
        # ====================================================

        result_frame = frame.copy()

        cv2.rectangle(
            result_frame,
            (0, 0),
            (950, 210),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            result_frame,
            "PREDICTION",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            result_frame,
            predicted_word,
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            3
        )

        cv2.putText(
            result_frame,
            "Confidence: "
            + str(round(confidence * 100, 2))
            + "%",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if confidence >= CONFIDENCE_THRESHOLD:

            cv2.putText(
                result_frame,
                "CONFIDENT PREDICTION",
                (20, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            speak(
                predicted_word
            )

        else:

            cv2.putText(
                result_frame,
                "LOW CONFIDENCE - TRY AGAIN",
                (20, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "ISL AI Translator",
            result_frame
        )

        print()
        print("Press SPACE for another sign.")
        print("Press Q to quit.")

        while True:

            key = cv2.waitKey(0) & 0xFF

            if key == ord("q"):

                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit

            if key == 32:

                break


cap.release()
cv2.destroyAllWindows()
