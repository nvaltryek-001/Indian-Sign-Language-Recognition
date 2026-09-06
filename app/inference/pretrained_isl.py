import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyttsx3
import os
from collections import deque

# ============================================================
# REAL-TIME INDIAN SIGN LANGUAGE RECOGNITION
# Pretrained LSTM baseline
# ============================================================

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
FEATURES = 258

CONFIDENCE_THRESHOLD = 0.70
PREDICTION_STABILITY = 5

# ------------------------------------------------------------
# Build the exact architecture used by the pretrained model
# ------------------------------------------------------------

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
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"]
)

print("Loading pretrained LSTM weights...")

model.load_weights(MODEL_PATH)

print("Pretrained model loaded successfully.")


# ------------------------------------------------------------
# MediaPipe
# ------------------------------------------------------------

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

    keypoints = np.concatenate([
        pose,
        left_hand,
        right_hand
    ])

    return keypoints.astype(np.float32)


# ------------------------------------------------------------
# Text-to-speech
# ------------------------------------------------------------

engine = pyttsx3.init()

last_spoken = ""
last_prediction = ""
stable_count = 0


def speak(text):

    global last_spoken

    if text != last_spoken:

        print("Speech:", text)

        engine.say(text)
        engine.runAndWait()

        last_spoken = text


# ------------------------------------------------------------
# Camera
# ------------------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


sequence = deque(
    maxlen=SEQUENCE_LENGTH
)

sentence = []

print()
print("=" * 60)
print("       ISL AI TRANSLATOR")
print("=" * 60)
print()
print("Recognized signs:")
print("1. Hello")
print("2. How are you")
print("3. Thank you")
print()
print("Perform a sign in front of the camera.")
print("Press S = speak sentence")
print("Press C = clear sentence")
print("Press Q = quit")
print("=" * 60)


with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.flip(frame, 1)

        # --------------------------------------------
        # MediaPipe
        # --------------------------------------------

        image = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image.flags.writeable = False

        results = holistic.process(image)

        image.flags.writeable = True

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        # --------------------------------------------
        # Draw landmarks
        # --------------------------------------------

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )

        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

        # --------------------------------------------
        # Extract 258 features
        # --------------------------------------------

        keypoints = extract_keypoints(results)

        if len(keypoints) == FEATURES:

            sequence.append(keypoints)

        prediction_text = "Collecting frames..."
        confidence = 0.0

        # --------------------------------------------
        # Prediction after 45 frames
        # --------------------------------------------

        if len(sequence) == SEQUENCE_LENGTH:

            input_data = np.expand_dims(
                np.array(sequence),
                axis=0
            )

            prediction = model.predict(
                input_data,
                verbose=0
            )[0]

            predicted_index = np.argmax(prediction)

            confidence = float(
                prediction[predicted_index]
            )

            predicted_word = ACTIONS[
                predicted_index
            ]

            prediction_text = predicted_word

            # ----------------------------------------
            # Stability filter
            # ----------------------------------------

            if (
                confidence >= CONFIDENCE_THRESHOLD
                and predicted_word == last_prediction
            ):

                stable_count += 1

            elif confidence >= CONFIDENCE_THRESHOLD:

                last_prediction = predicted_word
                stable_count = 1

            else:

                stable_count = 0

            # ----------------------------------------
            # Add stable prediction to sentence
            # ----------------------------------------

            if stable_count == PREDICTION_STABILITY:

                if (
                    len(sentence) == 0
                    or sentence[-1] != predicted_word
                ):

                    sentence.append(
                        predicted_word
                    )

                    print(
                        "Detected:",
                        predicted_word,
                        "Confidence:",
                        round(confidence * 100, 2),
                        "%"
                    )

                stable_count = 0

        # --------------------------------------------
        # UI
        # --------------------------------------------

        cv2.rectangle(
            image,
            (0, 0),
            (900, 210),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            image,
            "ISL AI TRANSLATOR",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            "Prediction: " + prediction_text,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            "Confidence: "
            + str(round(confidence * 100, 1))
            + "%",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        sentence_text = " ".join(sentence)

        if len(sentence_text) == 0:
            sentence_text = "Waiting for signs..."

        cv2.putText(
            image,
            "Sentence: " + sentence_text,
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            "S: Speak | C: Clear | Q: Quit",
            (20, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1
        )

        cv2.imshow(
            "ISL AI Translator",
            image
        )

        # --------------------------------------------
        # Keyboard
        # --------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

        elif key == ord("c"):

            sentence.clear()

            last_prediction = ""
            stable_count = 0

            print("Sentence cleared.")

        elif key == ord("s"):

            if sentence:

                speak(
                    " ".join(sentence)
                )


cap.release()
cv2.destroyAllWindows()

print()
print("Final sentence:")
print(" ".join(sentence))
