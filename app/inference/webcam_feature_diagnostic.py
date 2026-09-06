import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

OUTPUT = Path("data/include50/debug_webcam.npy")
SEQUENCE_LENGTH = 45
FEATURES = 258

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


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

    return np.asarray(features, dtype=np.float32)


print("=" * 70)
print("WEBCAM -> MEDIAPIPE -> 258 FEATURE DIAGNOSTIC")
print("=" * 70)
print()
print("Perform the LOUD sign clearly in front of the camera.")
print("Keep your upper body and both hands visible.")
print("Capturing 45 frames...")
print()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera 0 cannot be opened.")

frames = []

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
) as holistic:

    while len(frames) < SEQUENCE_LENGTH:

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

        pose = results.pose_landmarks is not None
        left = results.left_hand_landmarks is not None
        right = results.right_hand_landmarks is not None

        nonzero = int(np.count_nonzero(feature))

        print(
            f"Frame {len(frames)+1:02d}/45 | "
            f"Pose={pose} | "
            f"Left={left} | "
            f"Right={right} | "
            f"NonZero={nonzero}"
        )

        if feature.shape != (FEATURES,):
            print("ERROR: Wrong feature shape:", feature.shape)
            continue

        frames.append(feature)

        cv2.putText(
            frame,
            f"Frame: {len(frames)}/45",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Pose: {pose} Left: {left} Right: {right}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"NonZero: {nonzero}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

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

        cv2.imshow(
            "Webcam Feature Diagnostic - Q to quit",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()


if len(frames) != 45:
    raise RuntimeError(
        f"Only captured {len(frames)} frames."
    )


sequence = np.asarray(
    frames,
    dtype=np.float32
)

np.save(
    OUTPUT,
    sequence
)

print()
print("=" * 70)
print("DIAGNOSTIC RESULT")
print("=" * 70)

print("Shape    :", sequence.shape)
print("NonZero  :", int(np.count_nonzero(sequence)))
print("Mean     :", float(sequence.mean()))
print("Std      :", float(sequence.std()))
print("Min      :", float(sequence.min()))
print("Max      :", float(sequence.max()))

print()
print("Saved to :", OUTPUT)
print("=" * 70)
