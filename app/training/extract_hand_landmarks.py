"""Extract normalized 45-frame, hand-only INCLUDE-50 sequences."""

import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


VIDEO_DIR = Path("data/include50/videos")
OUTPUT_DIR = Path("data/include50/hand_sequences")
ERROR_LOG = Path("data/include50/hand_landmark_extraction_errors.csv")
SEQUENCE_LENGTH = 45
LANDMARKS_PER_HAND = 21
FEATURES_PER_HAND = LANDMARKS_PER_HAND * 3
FEATURES_PER_FRAME = FEATURES_PER_HAND * 2

mp_hands = mp.solutions.hands


def normalize_hand_landmarks(landmarks):
    """Return wrist-centered, scale-normalized hand coordinates."""

    coordinates = np.asarray(
        [[landmark.x, landmark.y, landmark.z] for landmark in landmarks.landmark],
        dtype=np.float32,
    )
    if coordinates.shape != (LANDMARKS_PER_HAND, 3):
        raise ValueError(f"Expected hand landmarks shaped (21, 3), got {coordinates.shape}")

    coordinates = coordinates - coordinates[0]
    scale = float(np.linalg.norm(coordinates, axis=1).max())
    if scale > 0.0:
        coordinates /= scale
    return coordinates.reshape(-1).astype(np.float32)


def _hand_features(landmarks):
    if landmarks is None:
        return np.zeros(FEATURES_PER_HAND, dtype=np.float32)
    return normalize_hand_landmarks(landmarks)


def extract_hand_landmarks(results):
    """Extract one normalized frame in left-hand-then-right-hand order.

    MediaPipe Hands reports handedness for each detected hand. Missing hands
    are represented by exactly 63 zero values.
    """

    hands = {"Left": None, "Right": None}
    detected = getattr(results, "multi_hand_landmarks", None) or []
    handedness = getattr(results, "multi_handedness", None) or []

    for landmarks, classification in zip(detected, handedness):
        label = classification.classification[0].label
        if label in hands and hands[label] is None:
            hands[label] = landmarks

    features = np.concatenate(
        [_hand_features(hands["Left"]), _hand_features(hands["Right"])],
    ).astype(np.float32)
    if features.shape != (FEATURES_PER_FRAME,):
        raise ValueError(f"Expected {FEATURES_PER_FRAME} features, got {features.shape}")
    return features, hands["Left"] is not None, hands["Right"] is not None


def sample_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"No frames found: {video_path}")

    indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH).astype(int)
    frames = []
    target_indices = set(int(index) for index in indices)
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index in target_indices:
            frames.append(frame)
        frame_index += 1
        if frame_index > int(indices[-1]):
            break
    cap.release()
    return frames


def process_video(video_path, output_path, hands):
    sequence = []
    left_count = 0
    right_count = 0

    for frame in sample_frames(video_path):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        features, left_present, right_present = extract_hand_landmarks(hands.process(rgb))
        sequence.append(features)
        left_count += int(left_present)
        right_count += int(right_present)

    while len(sequence) < SEQUENCE_LENGTH:
        sequence.append(np.zeros(FEATURES_PER_FRAME, dtype=np.float32))

    sequence = np.asarray(sequence[:SEQUENCE_LENGTH], dtype=np.float32)
    if sequence.shape != (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
        raise ValueError(f"Expected sequence shape (45, 126), got {sequence.shape}")
    if not np.isfinite(sequence).all():
        raise ValueError("Sequence contains NaN or Inf")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, sequence)
    return sequence, left_count, right_count


def find_videos(video_dir):
    return sorted(
        path
        for path in video_dir.rglob("*")
        if path.suffix.lower() in {".mov", ".mp4"}
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", nargs="*", help="Category folders to process")
    parser.add_argument("--videos", nargs="*", help="Video paths relative to the INCLUDE-50 video directory")
    parser.add_argument("--limit", type=int, help="Maximum number of videos to process")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess valid existing outputs")
    args = parser.parse_args(argv)

    video_dir = VIDEO_DIR
    output_dir = OUTPUT_DIR
    videos = find_videos(video_dir)
    if args.videos:
        videos = [video_dir / relative_path for relative_path in args.videos]
    if args.classes:
        selected = {name.casefold() for name in args.classes}
        videos = [path for path in videos if path.relative_to(video_dir).parts[0].casefold() in selected]
    if args.limit is not None:
        videos = videos[:args.limit]

    failures = []
    processed = 0
    skipped = 0
    left_frames = 0
    right_frames = 0
    example_shape = None
    example_dtype = None

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        for video_index, video_path in enumerate(videos, start=1):
            relative = video_path.relative_to(video_dir)
            output_path = output_dir / relative.with_suffix(".npy")
            if output_path.exists() and not args.overwrite:
                try:
                    existing = np.load(output_path)
                    if existing.shape == (SEQUENCE_LENGTH, FEATURES_PER_FRAME) and existing.dtype == np.float32:
                        skipped += 1
                        continue
                except Exception:
                    pass
            try:
                sequence, left, right = process_video(video_path, output_path, hands)
                processed += 1
                left_frames += left
                right_frames += right
                example_shape = sequence.shape
                example_dtype = sequence.dtype
                if processed % 25 == 0:
                    print(f"Processed {video_index}/{len(videos)} videos", flush=True)
            except Exception as error:
                failures.append((str(relative), str(error)))

    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ERROR_LOG.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video", "error"])
        writer.writerows(failures)

    total_frames = processed * SEQUENCE_LENGTH
    print(f"Videos selected: {len(videos)}")
    print(f"Videos processed successfully: {processed}")
    print(f"Videos skipped: {skipped}")
    print(f"Failures: {len(failures)}")
    print(f"Example shape: {example_shape}")
    print(f"Example dtype: {example_dtype}")
    print(f"Left-hand detections: {left_frames}/{total_frames}")
    print(f"Right-hand detections: {right_frames}/{total_frames}")
    print(f"Failure log: {ERROR_LOG}")


if __name__ == "__main__":
    main()