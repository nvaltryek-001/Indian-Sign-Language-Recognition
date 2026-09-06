import os
import re

files = [
    "app/training/extract_landmarks.py",
    "app/training/prepare_include50.py",
    "app/training/prepare_dataset.py",
    "app/inference/realtime_predict.py",
    "app/inference/predict_video.py",
    "app/inference/debug_webcam_sequence.py",
    "app/inference/canonical_model_diagnostic.py",
]

print("=" * 90)
print("ISL EXTRACTION PIPELINE AUDIT")
print("=" * 90)

patterns = [
    "Holistic",
    "mp.solutions.holistic",
    "pose_landmarks",
    "left_hand_landmarks",
    "right_hand_landmarks",
    "model_complexity",
    "static_image_mode",
    "smooth_landmarks",
    "np.linspace",
    "45",
    "258",
    "visibility",
    "x.flatten",
    "y.flatten",
    "z.flatten",
    "np.zeros",
    "normalize",
    "standard",
    "scaler",
    "clip",
    "mediapipe_detection",
    "extract_keypoints",
]

for path in files:
    print()
    print("=" * 90)
    print(path)
    print("=" * 90)

    if not os.path.exists(path):
        print("NOT FOUND")
        continue

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        s = line.strip()

        if any(p.lower() in s.lower() for p in patterns):
            print(f"{i:04d}: {s}")

print()
print("=" * 90)
print("AUDIT COMPLETE")
print("=" * 90)
