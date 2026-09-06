import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

OUTPUT = Path("data/include50/debug_webcam.npy")

SEQUENCE_LENGTH = 45
FEATURES = 258

mp_holistic = mp.solutions.holistic


def extract_landmarks(results):

    features = []

    # EXACTLY SAME AS TRAINING
    # Pose = 33 * 4 = 132
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

    # Left hand = 21 * 3 = 63
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            features.extend([
                lm.x,
                lm.y,
                lm.z
            ])
    else:
        features.extend([0.0] * 63)

    # Right hand = 21 * 3 = 63
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            features.extend([
                lm.x,
                lm.y,
                lm.z
            ])
    else:
        features.extend([0.0] * 63)

    features = np.asarray(features, dtype=np.float32)

    if features.shape != (FEATURES,):
        raise RuntimeError(
            f"Expected {FEATURES} features, got {features.shape}"
        )

    return features


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("=" * 70)
print("WEBCAM - TRAINING PIPELINE MATCH")
print("=" * 70)
print()
print("Perform the LOUD sign.")
print()
print("Capture exactly 45 frames.")
print("Try to keep the complete sign inside these 45 frames.")
print()
print("Press Q to cancel.")
print()

frames = []

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False
) as holistic:

    while len(frames) < SEQUENCE_LENGTH:

        ret, frame = cap.read()

        if not ret:
            continue

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = holistic.process(rgb)

        feature = extract_landmarks(results)

        frames.append(feature)

        display = frame.copy()

        cv2.putText(
            display,
            f"Frames: {len(frames)}/{SEQUENCE_LENGTH}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            "PERFORM LOUD",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "ISL Webcam Diagnostic",
            display
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

if len(frames) != SEQUENCE_LENGTH:
    raise RuntimeError(
        f"Only captured {len(frames)} frames"
    )

sequence = np.asarray(
    frames,
    dtype=np.float32
)

if sequence.shape != (45, 258):
    raise RuntimeError(
        f"Wrong shape: {sequence.shape}"
    )

if not np.isfinite(sequence).all():
    raise RuntimeError(
        "NaN or infinite values detected"
    )

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

np.save(
    OUTPUT,
    sequence
)

print()
print("=" * 70)
print("CAPTURE COMPLETE")
print("=" * 70)
print("Shape :", sequence.shape)
print("Mean  :", float(sequence.mean()))
print("Std   :", float(sequence.std()))
print("Min   :", float(sequence.min()))
print("Max   :", float(sequence.max()))
print("Nonzero:", int(np.count_nonzero(sequence)))
print()
print("Saved:", OUTPUT)
print("=" * 70)
