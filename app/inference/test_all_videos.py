import csv
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf


# ============================================================
# INCLUDE-50 V2 — TEST ALL 943 VIDEOS
# ============================================================

MODEL_PATH = Path("models/include50_lstm_v2_best.keras")
CLASS_MAP_PATH = Path("models/include50_classes_v2.json")

VIDEO_ROOT = Path("data/include50/videos")
OUTPUT_DIR = Path("models/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_CSV = OUTPUT_DIR / "all_943_video_predictions_v2.csv"
SUMMARY_JSON = OUTPUT_DIR / "all_943_video_test_summary_v2.json"

SEQUENCE_LENGTH = 45
FEATURES = 258


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("INCLUDE-50 V2 — ALL VIDEO TEST")
print("=" * 70)

print("\nLoading V2 model...")

model = tf.keras.models.load_model(str(MODEL_PATH))

print("V2 model loaded successfully.")


# ============================================================
# CLASS MAP
# ============================================================

with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
    class_mapping = json.load(f)

class_names = [
    class_mapping[str(i)]
    for i in range(len(class_mapping))
]

label_to_id = {
    name: i
    for i, name in enumerate(class_names)
}

print(f"Classes loaded: {len(class_names)}")


# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic


# ============================================================
# EXACT 258 FEATURES
# ============================================================

def extract_landmarks(results):

    # Pose = 33 x 4 = 132
    pose = []

    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            pose.extend([
                lm.x,
                lm.y,
                lm.z,
                lm.visibility
            ])
    else:
        pose = [0.0] * 132


    # Left hand = 21 x 3 = 63
    left_hand = []

    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            left_hand.extend([
                lm.x,
                lm.y,
                lm.z
            ])
    else:
        left_hand = [0.0] * 63


    # Right hand = 21 x 3 = 63
    right_hand = []

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            right_hand.extend([
                lm.x,
                lm.y,
                lm.z
            ])
    else:
        right_hand = [0.0] * 63


    features = (
        pose +
        left_hand +
        right_hand
    )

    features = np.asarray(
        features,
        dtype=np.float32
    )

    if features.shape != (FEATURES,):
        raise RuntimeError(
            f"Expected 258 features, got {features.shape}"
        )

    return features


# ============================================================
# SAMPLE 45 FRAMES
# ============================================================

def sample_frames(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        raise RuntimeError(
            f"No frames found: {video_path}"
        )

    indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH
    ).astype(int)

    frames = []

    for index in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index)
        )

        ret, frame = cap.read()

        if ret:
            frames.append(frame)

    cap.release()

    return frames


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(video_path, holistic):

    frames = sample_frames(video_path)

    sequence = []

    for frame in frames:

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = holistic.process(rgb)

        features = extract_landmarks(results)

        sequence.append(features)


    # Padding if necessary
    while len(sequence) < SEQUENCE_LENGTH:

        sequence.append(
            np.zeros(
                FEATURES,
                dtype=np.float32
            )
        )


    sequence = np.asarray(
        sequence[:SEQUENCE_LENGTH],
        dtype=np.float32
    )


    if sequence.shape != (
        SEQUENCE_LENGTH,
        FEATURES
    ):
        raise RuntimeError(
            f"Wrong shape: {sequence.shape}"
        )


    return sequence


# ============================================================
# FIND VIDEOS
# ============================================================

video_extensions = {
    ".mov",
    ".mp4",
    ".avi",
    ".mkv"
}

videos = sorted([
    p
    for p in VIDEO_ROOT.rglob("*")
    if p.is_file()
    and p.suffix.lower() in video_extensions
])

print()
print(f"Videos found: {len(videos)}")

if len(videos) != 943:

    print(
        f"WARNING: Expected 943 videos, "
        f"but found {len(videos)}"
    )


# ============================================================
# TEST ALL VIDEOS
# ============================================================

results_csv = []

correct = 0
wrong = 0
failed = 0

confidence_values = []

print()
print("=" * 70)
print("STARTING ALL VIDEO TEST")
print("=" * 70)


with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False
) as holistic:

    for number, video_path in enumerate(
        videos,
        start=1
    ):

        # Actual class = parent folder
        actual_label = video_path.parent.name

        print()
        print("-" * 70)
        print(
            f"[{number}/{len(videos)}] "
            f"{video_path.relative_to(VIDEO_ROOT)}"
        )

        try:

            sequence = process_video(
                video_path,
                holistic
            )

            X = np.expand_dims(
                sequence,
                axis=0
            )

            probabilities = model.predict(
                X,
                verbose=0
            )[0]

            predicted_id = int(
                np.argmax(probabilities)
            )

            predicted_label = class_names[
                predicted_id
            ]

            confidence = float(
                probabilities[predicted_id] * 100
            )

            is_correct = (
                predicted_label == actual_label
            )

            if is_correct:
                correct += 1
                status = "CORRECT"
            else:
                wrong += 1
                status = "WRONG"

            confidence_values.append(
                confidence
            )

            print(
                f"Actual     : {actual_label}"
            )

            print(
                f"Predicted  : {predicted_label}"
            )

            print(
                f"Confidence : {confidence:.2f}%"
            )

            print(
                f"Status     : {status}"
            )

            results_csv.append([
                str(video_path.relative_to(VIDEO_ROOT)),
                actual_label,
                predicted_label,
                f"{confidence:.4f}",
                status
            ])

        except Exception as e:

            failed += 1

            print(
                f"ERROR: {e}"
            )

            results_csv.append([
                str(video_path.relative_to(VIDEO_ROOT)),
                actual_label,
                "ERROR",
                "0.0000",
                "FAILED"
            ])


# ============================================================
# SAVE CSV
# ============================================================

with open(
    RESULT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "video",
        "actual_sign",
        "predicted_sign",
        "confidence_percent",
        "status"
    ])

    writer.writerows(results_csv)


# ============================================================
# SUMMARY
# ============================================================

total = len(videos)

tested = correct + wrong

accuracy = (
    correct / tested * 100
    if tested > 0
    else 0
)

average_confidence = (
    float(np.mean(confidence_values))
    if confidence_values
    else 0
)


summary = {
    "total_videos": total,
    "tested": tested,
    "correct": correct,
    "wrong": wrong,
    "failed": failed,
    "accuracy_percent": round(accuracy, 2),
    "average_confidence_percent": round(
        average_confidence,
        2
    ),
    "model": str(MODEL_PATH),
    "sequence_shape": [
        SEQUENCE_LENGTH,
        FEATURES
    ],
    "classes": len(class_names)
}


with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("ALL VIDEO TEST COMPLETE")
print("=" * 70)

print(
    f"Total videos       : {total}"
)

print(
    f"Successfully tested: {tested}"
)

print(
    f"Correct predictions: {correct}"
)

print(
    f"Wrong predictions  : {wrong}"
)

print(
    f"Failed videos      : {failed}"
)

print(
    f"Accuracy            : {accuracy:.2f}%"
)

print(
    f"Average confidence  : "
    f"{average_confidence:.2f}%"
)

print()
print("Results CSV:")
print(RESULT_CSV)

print()
print("Summary JSON:")
print(SUMMARY_JSON)

print()
print("STATUS: ALL VIDEO TEST COMPLETE")