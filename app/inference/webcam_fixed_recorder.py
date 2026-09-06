import cv2
import mediapipe as mp
import numpy as np

FRAMES = 90
WARMUP_REQUIRED = 15
OUT = "data/include50/webcam_fixed_test.npy"

mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Let the camera choose its real supported resolution.
print("=" * 75)
print("FIXED WEBCAM RECORDER")
print("=" * 75)
print("First: keep BOTH hands visible.")
print("The program will wait until MediaPipe detects both hands.")
print("Then recording starts automatically.")
print()

features = []
warmup_both = 0
started = False

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
) as holistic:

    while True:
        ok, frame = cap.read()

        if not ok:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = holistic.process(rgb)
        rgb.flags.writeable = True

        pose = result.pose_landmarks is not None
        left = result.left_hand_landmarks is not None
        right = result.right_hand_landmarks is not None
        both = left and right

        # -------------------------
        # WARMUP
        # -------------------------
        if not started:

            if both:
                warmup_both += 1
            else:
                warmup_both = 0

            text = f"WARMUP: both hands {warmup_both}/{WARMUP_REQUIRED}"

            if warmup_both >= WARMUP_REQUIRED:
                started = True
                print("Both hands stable -> RECORDING STARTED")

            cv2.putText(
                frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # -------------------------
        # RECORDING
        # -------------------------
        else:

            pose_vec = (
                np.array(
                    [[lm.x, lm.y, lm.z, lm.visibility]
                     for lm in result.pose_landmarks.landmark],
                    dtype=np.float32
                ).flatten()
                if result.pose_landmarks
                else np.zeros(132, dtype=np.float32)
            )

            left_vec = (
                np.array(
                    [[lm.x, lm.y, lm.z]
                     for lm in result.left_hand_landmarks.landmark],
                    dtype=np.float32
                ).flatten()
                if result.left_hand_landmarks
                else np.zeros(63, dtype=np.float32)
            )

            right_vec = (
                np.array(
                    [[lm.x, lm.y, lm.z]
                     for lm in result.right_hand_landmarks.landmark],
                    dtype=np.float32
                ).flatten()
                if result.right_hand_landmarks
                else np.zeros(63, dtype=np.float32)
            )

            feat = np.concatenate(
                [pose_vec, left_vec, right_vec]
            ).astype(np.float32)

            features.append(feat)

            text = (
                f"RECORDING {len(features)}/{FRAMES} | "
                f"Pose:{'Y' if pose else 'N'} "
                f"Left:{'Y' if left else 'N'} "
                f"Right:{'Y' if right else 'N'}"
            )

            cv2.putText(
                frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            if len(features) >= FRAMES:
                break

        # Draw hands
        if left:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                result.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        if right:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                result.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        cv2.imshow("Fixed Webcam Recorder", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

x = np.asarray(features, dtype=np.float32)

if len(x) < FRAMES:
    raise RuntimeError(
        f"Recording incomplete: {len(x)}/{FRAMES} frames"
    )

# EXACT training temporal sampling
idx = np.linspace(0, len(x) - 1, 45).astype(int)
final = x[idx]

np.save(OUT, final)

# Detection statistics from the captured 90 frames
left_frames = np.sum(np.abs(x[:,132:195]).sum(axis=1) > 0)
right_frames = np.sum(np.abs(x[:,195:258]).sum(axis=1) > 0)
both_frames = np.sum(
    (np.abs(x[:,132:195]).sum(axis=1) > 0) &
    (np.abs(x[:,195:258]).sum(axis=1) > 0)
)

print()
print("=" * 75)
print("FIXED RECORDING RESULT")
print("=" * 75)
print("Raw shape       :", x.shape)
print("Final shape     :", final.shape)
print("Left frames     :", int(left_frames), "/", FRAMES)
print("Right frames    :", int(right_frames), "/", FRAMES)
print("Both frames     :", int(both_frames), "/", FRAMES)
print("Mean            :", float(final.mean()))
print("Std             :", float(final.std()))
print("Min             :", float(final.min()))
print("Max             :", float(final.max()))
print("Finite          :", bool(np.isfinite(final).all()))
print("Saved           :", OUT)
