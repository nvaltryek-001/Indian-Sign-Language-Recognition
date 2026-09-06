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
    "welcome"
]

SEQUENCES_PER_ACTION = 30
SEQUENCE_LENGTH = 45

os.makedirs(DATA_DIR, exist_ok=True)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def extract_keypoints(results):

    points = []

    # Four pose landmarks
    if results.pose_landmarks:
        pose = results.pose_landmarks.landmark

        for idx in [0, 11, 12, 23]:
            points.append([pose[idx].x, pose[idx].y])
    else:
        points.extend([[0.0, 0.0]] * 4)

    # Ten left hand landmarks
    if results.left_hand_landmarks:
        hand = results.left_hand_landmarks.landmark

        for idx in range(10):
            points.append([hand[idx].x, hand[idx].y])
    else:
        points.extend([[0.0, 0.0]] * 10)

    # Ten right hand landmarks
    if results.right_hand_landmarks:
        hand = results.right_hand_landmarks.landmark

        for idx in range(10):
            points.append([hand[idx].x, hand[idx].y])
    else:
        points.extend([[0.0, 0.0]] * 10)

    return np.array(points, dtype=np.float32)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam could not be opened")


print("=" * 60)
print("INDIAN SIGN LANGUAGE DATASET COLLECTOR")
print("=" * 60)

for i, action in enumerate(ACTIONS, 1):
    print(str(i) + ". " + action)

print("=" * 60)
print("Sequences per sign:", SEQUENCES_PER_ACTION)
print("Frames per sequence:", SEQUENCE_LENGTH)
print("Press Q to stop")
print("=" * 60)


with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    for action_number, action in enumerate(ACTIONS, 1):

        action_dir = os.path.join(DATA_DIR, action)
        os.makedirs(action_dir, exist_ok=True)

        existing = len([
            f for f in os.listdir(action_dir)
            if f.endswith(".npy")
        ])

        print()
        print("SIGN", action_number, "/ 10:", action.upper())
        print("Already collected:", existing, "/", SEQUENCES_PER_ACTION)

        if existing >= SEQUENCES_PER_ACTION:
            print("Already complete. Skipping.")
            continue

        for sequence in range(existing, SEQUENCES_PER_ACTION):

            print(
                "Collecting",
                action.upper(),
                "example",
                sequence + 1,
                "/",
                SEQUENCES_PER_ACTION
            )

            # Countdown
            for countdown in [3, 2, 1]:

                start_time = time.time()

                while time.time() - start_time < 1:

                    success, frame = cap.read()

                    if not success:
                        continue

                    frame = cv2.flip(frame, 1)

                    cv2.putText(
                        frame,
                        "SIGN: " + action.upper(),
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        "EXAMPLE: " + str(sequence + 1) + "/30",
                        (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "STARTING: " + str(countdown),
                        (20, 125),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        3
                    )

                    cv2.imshow(
                        "ISL Dataset Collector",
                        frame
                    )

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        raise SystemExit

            sequence_data = []

            # Collect 45 frames
            for frame_num in range(SEQUENCE_LENGTH):

                success, frame = cap.read()

                if not success:
                    continue

                frame = cv2.flip(frame, 1)

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

                keypoints = extract_keypoints(results)

                sequence_data.append(keypoints)

                # Draw landmarks
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

                # Display information
                cv2.putText(
                    image,
                    "SIGN: " + action.upper(),
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    "EXAMPLE: " + str(sequence + 1) + "/30",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    image,
                    "FRAME: " + str(frame_num + 1) + "/45",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    image,
                    "Perform sign",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

                cv2.imshow(
                    "ISL Dataset Collector",
                    image
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    raise SystemExit

            # Save sequence
            sequence_array = np.array(
                sequence_data,
                dtype=np.float32
            )

            filename = os.path.join(
                action_dir,
                str(sequence) + ".npy"
            )

            np.save(
                filename,
                sequence_array
            )

            print(
                "Saved:",
                filename,
                "Shape:",
                sequence_array.shape
            )

            time.sleep(0.3)


cap.release()
cv2.destroyAllWindows()

print()
print("=" * 60)
print("DATA COLLECTION COMPLETE")
print("=" * 60)
print("10 signs x 30 sequences x 45 frames")
print("Total sequences: 300")
print("Total frames: 13500")
