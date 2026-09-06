from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np

OUT = Path("data/include50/webcam_exact_test.npy")
FRAMES = 90
SEQUENCE = 45

mp_holistic = mp.solutions.holistic

def extract(results):
    pose = np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
        dtype=np.float32
    ).flatten() if results.pose_landmarks else np.zeros(132, dtype=np.float32)

    left = np.array(
        [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
        dtype=np.float32
    ).flatten() if results.left_hand_landmarks else np.zeros(63, dtype=np.float32)

    right = np.array(
        [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
        dtype=np.float32
    ).flatten() if results.right_hand_landmarks else np.zeros(63, dtype=np.float32)

    return np.concatenate([pose, left, right]).astype(np.float32)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frames = []

print("=" * 75)
print("EXACT TRAINING-COMPATIBLE WEBCAM CAPTURE")
print("=" * 75)
print("Show your upper body + both hands clearly.")
print("Perform ONE clear sign continuously.")
print("Press Q to stop early.")
print()

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False
) as holistic:

    while len(frames) < FRAMES:
        ok, frame = cap.read()

        if not ok:
            print("Camera frame failed.")
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        feat = extract(results)
        frames.append(feat)

        cv2.putText(
            frame,
            f"Recording: {len(frames)}/{FRAMES}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("ISL Exact Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

raw = np.asarray(frames, dtype=np.float32)

if len(raw) < FRAMES:
    raise RuntimeError(f"Only captured {len(raw)} frames.")

# EXACT SAME TEMPORAL SAMPLING AS TRAINING
idx = np.linspace(0, len(raw) - 1, SEQUENCE).astype(int)
x = raw[idx]

np.save(OUT, x)

print()
print("=" * 75)
print("SAVED")
print("=" * 75)
print("Raw shape :", raw.shape)
print("Final shape:", x.shape)
print("Mean      :", float(x.mean()))
print("Std       :", float(x.std()))
print("Min       :", float(x.min()))
print("Max       :", float(x.max()))
print("Finite    :", bool(np.isfinite(x).all()))
print("Nonzero   :", int(np.count_nonzero(x)))
print("Saved to  :", OUT)
