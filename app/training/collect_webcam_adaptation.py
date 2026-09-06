import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import time

CLASS_NAME = "1. loud"
OUTPUT_DIR = Path("data/include50/webcam_sequences") / CLASS_NAME

SEQUENCE_LENGTH = 45
CAPTURE_FRAMES = 90
FEATURES = 258
SAMPLES = 20

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mp_holistic = mp.solutions.holistic


def extract_features(results):

    features = []

    # Pose: 33 x 4 = 132
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            features.extend([
                lm.x,
                lm.y,
                lm.z,
                lm.visibility
            ])
    else:
        features.extend([0.0] * 132)

    # Left hand: 21 x 3 = 63
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            features.extend([
                lm.x,
                lm.y,
                lm.z
            ])
    else:
        features.extend([0.0] * 63)

    # Right hand: 21 x 3 = 63
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            features.extend([
                lm.x,
                lm.y,
                lm.z
            ])
    else:
        features.extend([0.0] * 63)

    return features


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")


print("=" * 70)
print("WEBCAM DOMAIN ADAPTATION - LOUD")
print("=" * 70)
print()
print(f"Collecting {SAMPLES} webcam samples.")
print()
print("For every sample:")
print("  1. Wait for the countdown")
print("  2. Perform LOUD naturally")
print("  3. Keep your upper body and hands visible")
print()
print("Press Q to stop.")
print("=" * 70)


with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False
) as holistic:

    for sample_number in range(1, SAMPLES + 1):

        # Countdown
        start = time.time()

        while time.time() - start < 2:

            ret, frame = cap.read()

            if not ret:
                continue

            remaining = 2 - (time.time() - start)

            display = frame.copy()

            cv2.putText(
                display,
                f"Sample {sample_number}/{SAMPLES}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"GET READY: {remaining:.1f}",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            cv2.imshow(
                "LOUD Webcam Collection",
                display
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit

        frames = []

        while len(frames) < CAPTURE_FRAMES:

            ret, frame = cap.read()

            if not ret:
                continue

            image = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = holistic.process(image)

            feature = extract_features(results)

            if len(feature) == FEATURES:
                frames.append(feature)

            display = frame.copy()

            cv2.putText(
                display,
                f"LOUD | Sample {sample_number}/{SAMPLES}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"Frames: {len(frames)}/{CAPTURE_FRAMES}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.imshow(
                "LOUD Webcam Collection",
                display
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit

        # Uniformly sample 45 frames
        indices = np.linspace(
            0,
            CAPTURE_FRAMES - 1,
            SEQUENCE_LENGTH
        ).astype(int)

        sequence = np.asarray(
            [frames[i] for i in indices],
            dtype=np.float32
        )

        if sequence.shape != (45, 258):
            print("SKIPPED - invalid shape:", sequence.shape)
            continue

        filename = (
            OUTPUT_DIR /
            f"webcam_loud_{sample_number:03d}.npy"
        )

        np.save(filename, sequence)

        print(
            f"Saved {sample_number}/{SAMPLES}: {filename}"
        )

cap.release()
cv2.destroyAllWindows()

print()
print("=" * 70)
print("COLLECTION COMPLETE")
print("=" * 70)
print("Samples:", SAMPLES)
print("Output :", OUTPUT_DIR)
print("=" * 70)
