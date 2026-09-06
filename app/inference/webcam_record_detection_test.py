import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic

FRAMES = 90
OUT = "data/include50/webcam_detection_debug.npy"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

features = []
left_count = 0
right_count = 0
both_count = 0
pose_count = 0

print("=" * 75)
print("WEBCAM RECORDING + HAND DETECTION CHECK")
print("=" * 75)
print("Perform ONE sign continuously for the entire recording.")
print("Keep both hands visible.")
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

    while len(features) < FRAMES:
        ok, frame = cap.read()

        if not ok:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True

        pose = results.pose_landmarks is not None
        left = results.left_hand_landmarks is not None
        right = results.right_hand_landmarks is not None

        if pose:
            pose_count += 1
        if left:
            left_count += 1
        if right:
            right_count += 1
        if left and right:
            both_count += 1

        pose_vec = (
            np.array(
                [[lm.x, lm.y, lm.z, lm.visibility]
                 for lm in results.pose_landmarks.landmark],
                dtype=np.float32
            ).flatten()
            if results.pose_landmarks
            else np.zeros(132, dtype=np.float32)
        )

        left_vec = (
            np.array(
                [[lm.x, lm.y, lm.z]
                 for lm in results.left_hand_landmarks.landmark],
                dtype=np.float32
            ).flatten()
            if results.left_hand_landmarks
            else np.zeros(63, dtype=np.float32)
        )

        right_vec = (
            np.array(
                [[lm.x, lm.y, lm.z]
                 for lm in results.right_hand_landmarks.landmark],
                dtype=np.float32
            ).flatten()
            if results.right_hand_landmarks
            else np.zeros(63, dtype=np.float32)
        )

        feat = np.concatenate([
            pose_vec,
            left_vec,
            right_vec
        ]).astype(np.float32)

        features.append(feat)

        text = (
            f"{len(features)}/{FRAMES}  "
            f"Pose:{'Y' if pose else 'N'}  "
            f"Left:{'Y' if left else 'N'}  "
            f"Right:{'Y' if right else 'N'}"
        )

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        if left:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        if right:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        cv2.imshow("Recording Detection Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

x = np.asarray(features, dtype=np.float32)

np.save(OUT, x)

print()
print("=" * 75)
print("RESULT")
print("=" * 75)
print("Frames captured :", len(x))
print("Pose detected   :", pose_count, "/", len(x))
print("Left detected   :", left_count, "/", len(x))
print("Right detected  :", right_count, "/", len(x))
print("Both detected   :", both_count, "/", len(x))
print()
print("Feature shape   :", x.shape)
print("Feature mean    :", float(x.mean()))
print("Feature std     :", float(x.std()))
print("Saved           :", OUT)
