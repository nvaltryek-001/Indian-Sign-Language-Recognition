import cv2
import mediapipe as mp
import numpy as np
import os
import time

DATA_DIR = "data/sequences"

ACTIONS = [
    "hello",
    "thank_you",
    "yes",
    "no",
    "please",
    "sorry",
    "help",
    "good",
    "bad",
    "welcome",
    "love",
    "friend",
    "family",
    "mother",
    "father",
    "eat",
    "drink",
    "water",
    "help_me"
]

SEQUENCES = 10
FRAMES = 45

os.makedirs(DATA_DIR, exist_ok=True)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def extract_keypoints(results):

    pose = (
        np.array([
            [r.x, r.y, r.z, r.visibility]
            for r in results.pose_landmarks.landmark
        ]).flatten()
        if results.pose_landmarks
        else np.zeros(132)
    )

    left = (
        np.array([
            [r.x, r.y, r.z]
            for r in results.left_hand_landmarks.landmark
        ]).flatten()
        if results.left_hand_landmarks
        else np.zeros(63)
    )

    right = (
        np.array([
            [r.x, r.y, r.z]
            for r in results.right_hand_landmarks.landmark
        ]).flatten()
        if results.right_hand_landmarks
        else np.zeros(63)
    )

    return np.concatenate([
        pose,
        left,
        right
    ]).astype(np.float32)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam could not be opened")


print("=" * 60)
print("ISL DATASET COLLECTION FROM SCRATCH")
print("=" * 60)
print("1. HELLO")
print("2. HOW ARE YOU")
print("3. THANK YOU")
print()
print("20 examples per sign")
print("45 frames per example")
print()
print("Press Q to stop")
print("=" * 60)


with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    for action in ACTIONS:

        folder = os.path.join(
            DATA_DIR,
            action
        )

        os.makedirs(folder, exist_ok=True)

        existing = len([
            f for f in os.listdir(folder)
            if f.endswith(".npy")
        ])

        for sequence in range(existing, SEQUENCES):

            print()
            print(
                "SIGN:",
                action.upper(),
                "| EXAMPLE:",
                sequence + 1,
                "/",
                SEQUENCES
            )

            # Countdown
            for count in [3, 2, 1]:

                start = time.time()

                while time.time() - start < 1:

                    success, frame = cap.read()

                    if not success:
                        continue

                    frame = cv2.flip(frame, 1)

                    cv2.putText(
                        frame,
                        "SIGN: " + action.upper(),
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        "STARTING " + str(count),
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        3
                    )

                    cv2.imshow(
                        "ISL Training Data",
                        frame
                    )

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        raise SystemExit

            sequence_data = []

            # Capture exactly 45 frames
            for frame_number in range(FRAMES):

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

                keypoints = extract_keypoints(
                    results
                )

                sequence_data.append(
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

                cv2.putText(
                    frame,
                    action.upper(),
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Example "
                    + str(sequence + 1)
                    + "/"
                    + str(SEQUENCES),
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Frame "
                    + str(frame_number + 1)
                    + "/45",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(
                    "ISL Training Data",
                    frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    raise SystemExit

            data = np.array(
                sequence_data,
                dtype=np.float32
            )

            filename = os.path.join(
                folder,
                str(sequence) + ".npy"
            )

            np.save(
                filename,
                data
            )

            print(
                "Saved:",
                filename,
                "Shape:",
                data.shape
            )


cap.release()
cv2.destroyAllWindows()

print()
print("DATA COLLECTION COMPLETE")

