import cv2
import mediapipe as mp
import numpy as np
import os

SEQUENCE_LENGTH = 45
CAPTURE_FRAMES = 90
FEATURES = 258

OUT = "data/include50/webcam_compatible_test.npy"

mp_holistic = mp.solutions.holistic


def extract_features(results):
    """
    EXACT feature layout:

    Pose       : 33 * 4 = 132
    Left hand  : 21 * 3 = 63
    Right hand : 21 * 3 = 63

    TOTAL = 258

    IMPORTANT:
    No normalization.
    No z-score.
    No clipping.
    No coordinate scaling.
    """

    row = []

    # ---------------------------------------------------------
    # POSE: x, y, z, visibility
    # ---------------------------------------------------------
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            row.extend([
                lm.x,
                lm.y,
                lm.z,
                lm.visibility
            ])
    else:
        row.extend([0.0] * (33 * 4))

    # ---------------------------------------------------------
    # LEFT HAND: x, y, z
    # ---------------------------------------------------------
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            row.extend([
                lm.x,
                lm.y,
                lm.z
            ])
    else:
        row.extend([0.0] * (21 * 3))

    # ---------------------------------------------------------
    # RIGHT HAND: x, y, z
    # ---------------------------------------------------------
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            row.extend([
                lm.x,
                lm.y,
                lm.z
            ])
    else:
        row.extend([0.0] * (21 * 3))

    if len(row) != FEATURES:
        raise ValueError(
            f"Feature mismatch: {len(row)} != {FEATURES}"
        )

    return np.asarray(row, dtype=np.float32)


def resample_sequence(sequence, target_length=45):
    sequence = np.asarray(sequence, dtype=np.float32)

    if len(sequence) == target_length:
        return sequence

    indices = np.linspace(
        0,
        len(sequence) - 1,
        target_length
    )

    output = []

    for idx in indices:
        left = int(np.floor(idx))
        right = int(np.ceil(idx))

        if left == right:
            output.append(sequence[left])
        else:
            alpha = idx - left
            frame = (
                sequence[left] * (1.0 - alpha)
                + sequence[right] * alpha
            )
            output.append(frame)

    return np.asarray(output, dtype=np.float32)


print("=" * 80)
print("DATASET-COMPATIBLE WEBCAM EXTRACTION")
print("=" * 80)

print()
print("Feature layout : Pose(132) + LeftHand(63) + RightHand(63)")
print("Total features : 258")
print("Raw frames     :", CAPTURE_FRAMES)
print("Final frames   :", SEQUENCE_LENGTH)
print()
print("IMPORTANT: NO NORMALIZATION / NO CLIPPING")
print()
print("Press SPACE to start recording.")
print("Keep the COMPLETE sign visible.")
print("Perform: 1. loud")
print("Recording will automatically collect 90 frames.")
print()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam. Try changing VideoCapture(0) to VideoCapture(1)."
    )

sequence = []
started = False

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        display = frame.copy()

        cv2.putText(
            display,
            "SPACE = START | ESC = CANCEL",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        if started:

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = holistic.process(rgb)

            features = extract_features(results)

            sequence.append(features)

            cv2.putText(
                display,
                f"Recording: {len(sequence)}/{CAPTURE_FRAMES}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            if len(sequence) >= CAPTURE_FRAMES:
                break

        cv2.imshow(
            "ISL Dataset-Compatible Extraction",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 32:
            started = True
            sequence = []
            print("Recording started...")

        elif key == 27:
            print("Cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit

cap.release()
cv2.destroyAllWindows()

if len(sequence) < CAPTURE_FRAMES:
    raise RuntimeError(
        f"Only captured {len(sequence)} frames."
    )

raw = np.asarray(
    sequence,
    dtype=np.float32
)

final = resample_sequence(
    raw,
    SEQUENCE_LENGTH
)

print()
print("=" * 80)
print("EXTRACTION RESULT")
print("=" * 80)

print("Raw shape :", raw.shape)
print("Final shape:", final.shape)

print()
print("FINAL STATISTICS")
print("Mean     :", float(final.mean()))
print("Std      :", float(final.std()))
print("Min      :", float(final.min()))
print("Max      :", float(final.max()))
print("Finite   :", bool(np.isfinite(final).all()))
print("Nonzero  :", int(np.count_nonzero(final)))

if final.shape != (45, 258):
    raise RuntimeError(
        f"BAD SHAPE: {final.shape}"
    )

if not np.isfinite(final).all():
    raise RuntimeError(
        "NaN or Inf detected!"
    )

os.makedirs(
    os.path.dirname(OUT),
    exist_ok=True
)

np.save(
    OUT,
    final.astype(np.float32)
)

print()
print("Saved:", OUT)

print()
print("=" * 80)
print("TRAINING DISTRIBUTION REFERENCE")
print("=" * 80)

stats = np.load(
    "models/include50_training_stats.npz"
)

print(
    "Training mean:",
    float(stats["mean"])
    if "mean" in stats
    else "available in feature statistics"
)

print()
print("Expected training feature range:")
print("approximately -1.01 to +1.56")

print()
print("=" * 80)
print("DONE")
print("=" * 80)
