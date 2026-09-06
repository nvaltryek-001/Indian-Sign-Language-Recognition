import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

# ============================================================
# INCLUDE-50 CANONICAL FEATURE EXTRACTOR
# ============================================================

FEATURES = 258

mp_holistic = mp.solutions.holistic


def safe_clip(x, low=-3.0, high=3.0):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )
    return np.clip(x, low, high)


def extract_raw(results):
    features = []

    # --------------------------------------------------------
    # POSE: 33 * 4 = 132
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # LEFT HAND: 21 * 3 = 63
    # --------------------------------------------------------

    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            features.extend([
                lm.x,
                lm.y,
                lm.z
            ])
    else:
        features.extend([0.0] * 63)

    # --------------------------------------------------------
    # RIGHT HAND: 21 * 3 = 63
    # --------------------------------------------------------

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            features.extend([
                lm.x,
                lm.y,
                lm.z
            ])
    else:
        features.extend([0.0] * 63)

    x = np.asarray(features, dtype=np.float32)

    if x.shape != (FEATURES,):
        raise RuntimeError(
            f"Expected 258 features, got {x.shape}"
        )

    return x


def normalize_pose(x):
    """
    Canonical body normalization.

    Pose coordinates are centered around the body and scaled
    using shoulder/hip geometry.

    Hand coordinates are kept in the same camera coordinate
    system but clipped to prevent extreme values.
    """

    x = x.copy()

    # --------------------------------------------------------
    # POSE ARRAYS
    # --------------------------------------------------------

    pose = x[:132].reshape(33, 4)

    # Nose
    nose = pose[0, :3]

    # Left shoulder = 11
    # Right shoulder = 12
    # Left hip = 23
    # Right hip = 24

    left_shoulder = pose[11, :3]
    right_shoulder = pose[12, :3]

    left_hip = pose[23, :3]
    right_hip = pose[24, :3]

    # --------------------------------------------------------
    # BODY CENTER
    # --------------------------------------------------------

    valid_points = []

    for p in [
        left_shoulder,
        right_shoulder,
        left_hip,
        right_hip
    ]:
        if np.isfinite(p).all():
            valid_points.append(p)

    if len(valid_points) >= 2:

        center = np.mean(
            np.asarray(valid_points),
            axis=0
        )

    else:

        center = nose

    # --------------------------------------------------------
    # BODY SCALE
    # --------------------------------------------------------

    shoulder_width = np.linalg.norm(
        left_shoulder - right_shoulder
    )

    hip_width = np.linalg.norm(
        left_hip - right_hip
    )

    scale_candidates = [
        shoulder_width,
        hip_width
    ]

    scale_candidates = [
        s for s in scale_candidates
        if np.isfinite(s) and s > 1e-4
    ]

    if scale_candidates:

        scale = float(
            np.mean(scale_candidates)
        )

    else:

        scale = 1.0

    scale = max(scale, 1e-3)

    # --------------------------------------------------------
    # NORMALIZE POSE XYZ
    # --------------------------------------------------------

    pose[:, :3] = (
        pose[:, :3] - center
    ) / scale

    pose[:, :3] = np.clip(
        pose[:, :3],
        -3.0,
        3.0
    )

    x[:132] = pose.reshape(-1)

    # --------------------------------------------------------
    # HANDS
    # --------------------------------------------------------

    left = x[132:195].reshape(21, 3)
    right = x[195:258].reshape(21, 3)

    # Convert hand coordinates relative to body center
    left = (left - center) / scale
    right = (right - center) / scale

    left = np.clip(left, -3.0, 3.0)
    right = np.clip(right, -3.0, 3.0)

    x[132:195] = left.reshape(-1)
    x[195:258] = right.reshape(-1)

    return safe_clip(x)


def extract_features(results):
    raw = extract_raw(results)
    normalized = normalize_pose(raw)

    if normalized.shape != (258,):
        raise RuntimeError(
            f"Feature mismatch: {normalized.shape}"
        )

    return normalized.astype(np.float32)


# ============================================================
# TEST WITH WEBCAM
# ============================================================

OUTPUT = Path(
    "data/include50/canonical_test.npy"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")


frames = []

print("=" * 70)
print("CANONICAL FEATURE EXTRACTION TEST")
print("=" * 70)
print()
print("Perform LOUD.")
print("Capturing 90 frames.")
print()
print("Keep your complete upper body and both hands visible.")
print("Press Q to cancel.")
print()

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while len(frames) < 90:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = holistic.process(rgb)

        feature = extract_features(results)

        frames.append(feature)

        display = frame.copy()

        cv2.putText(
            display,
            f"Frames: {len(frames)}/90",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            "Perform LOUD",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "Canonical Extraction Test",
            display
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit


cap.release()
cv2.destroyAllWindows()


# ============================================================
# SAMPLE 90 -> 45
# ============================================================

frames = np.asarray(
    frames,
    dtype=np.float32
)

indices = np.linspace(
    0,
    89,
    45
).astype(int)

sequence = frames[indices]


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 70)
print("CANONICAL EXTRACTION COMPLETE")
print("=" * 70)

print("Raw shape :", frames.shape)
print("Final shape:", sequence.shape)
print("Mean      :", float(sequence.mean()))
print("Std       :", float(sequence.std()))
print("Min       :", float(sequence.min()))
print("Max       :", float(sequence.max()))
print("Nonzero   :", int(np.count_nonzero(sequence)))
print("Finite    :", bool(np.isfinite(sequence).all()))

if sequence.shape != (45, 258):
    raise RuntimeError(
        f"WRONG SHAPE: {sequence.shape}"
    )

if not np.isfinite(sequence).all():
    raise RuntimeError(
        "NaN/Inf detected"
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
print("Saved:", OUTPUT)
print("=" * 70)
