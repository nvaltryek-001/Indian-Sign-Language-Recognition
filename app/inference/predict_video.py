import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from pathlib import Path
import json
import sys

MODEL_PATH = Path("models/include50_lstm_v2_best.keras")
CLASS_MAP_PATH = Path("models/include50_classes_v2.json")

SEQUENCE_LENGTH = 45
FEATURES = 258


print("=" * 70)
print("INCLUDE-50 V2 VIDEO PREDICTION")
print("=" * 70)

print("\nLoading V2 model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"V2 model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(str(MODEL_PATH))

print("V2 model loaded successfully.")


if not CLASS_MAP_PATH.exists():
    raise FileNotFoundError(
        f"V2 class mapping not found: {CLASS_MAP_PATH}"
    )

with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
    class_mapping = json.load(f)

class_names = [
    class_mapping[str(i)]
    for i in range(len(class_mapping))
]

print(f"Classes loaded: {len(class_names)}")


mp_holistic = mp.solutions.holistic


def extract_landmarks(results):

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


    features = pose + left_hand + right_hand

    features = np.asarray(
        features,
        dtype=np.float32
    )

    if features.shape != (FEATURES,):
        raise ValueError(
            f"Expected {FEATURES} features, got {features.shape}"
        )

    return features


def sample_frames(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Frames: {total_frames}")
    print(f"FPS: {fps:.2f}")

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


def process_video(video_path):

    print("\nVideo:")
    print(video_path)

    frames = sample_frames(video_path)

    print(
        f"Sampled frames: {len(frames)}"
    )

    sequence = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False
    ) as holistic:

        for frame in frames:

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = holistic.process(rgb)

            features = extract_landmarks(results)

            sequence.append(features)


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


    print(
        f"Extracted sequence: {sequence.shape}"
    )


    if sequence.shape != (
        SEQUENCE_LENGTH,
        FEATURES
    ):
        raise RuntimeError(
            f"Wrong sequence shape: {sequence.shape}"
        )


    if np.isnan(sequence).any():
        raise RuntimeError("NaN values detected")

    if np.isinf(sequence).any():
        raise RuntimeError("Inf values detected")


    return sequence


def predict(video_path):

    sequence = process_video(video_path)

    X = np.expand_dims(
        sequence,
        axis=0
    )

    print("\nRunning V2 LSTM prediction...")

    probabilities = model.predict(
        X,
        verbose=0
    )[0]

    top_indices = np.argsort(
        probabilities
    )[::-1][:5]


    print("\n" + "=" * 70)
    print("TOP 5 V2 PREDICTIONS")
    print("=" * 70)


    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        print(
            f"{rank}. "
            f"{class_names[index]:25s} "
            f"{probabilities[index] * 100:.2f}%"
        )


    best_index = top_indices[0]

    confidence = float(
        probabilities[best_index]
    )


    print("\n" + "-" * 70)

    print(
        f"PREDICTED SIGN : "
        f"{class_names[best_index]}"
    )

    print(
        f"CONFIDENCE     : "
        f"{confidence * 100:.2f}%"
    )

    print("-" * 70)


def main():

    if len(sys.argv) < 2:

        print("\nUsage:")

        print(
            'python app\\inference\\predict_video.py '
            '"path\\to\\video.MOV"'
        )

        sys.exit(1)


    video = Path(sys.argv[1])


    if not video.exists():

        print(
            "\nERROR: Video not found:"
        )

        print(video)

        sys.exit(1)


    predict(video)


if __name__ == "__main__":
    main()
