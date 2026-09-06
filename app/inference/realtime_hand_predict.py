"""Hand-only INCLUDE-50 webcam recognition with LSTM and DTW matching.

This is a new pipeline.  It deliberately does not load or call the legacy
258-feature pose-and-hand model or any of the existing realtime scripts.
"""

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.dtw_matcher import DTWReferenceMatcher
from app.training.extract_hand_landmarks import (
    FEATURES_PER_FRAME,
    SEQUENCE_LENGTH,
    extract_hand_landmarks,
)

MODEL_PATH = ROOT / "models" / "include50_hand_lstm_best.keras"
CLASS_MAP_PATH = ROOT / "models" / "include50_hand_classes.json"
WINDOW_NAME = "ISL Hand Recognition (126 features)"
VOTE_WINDOWS = 3
MIN_HAND_FRAMES = 8


def load_class_names(path):
    """Load the strict index-to-label map expected from hand-model training."""

    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        return [str(data[str(index)]) for index in range(50)]
    except (KeyError, TypeError) as error:
        raise ValueError("Hand class map must contain string keys 0 through 49") from error


def load_hand_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Hand-focused model not found: {MODEL_PATH}")
    if not CLASS_MAP_PATH.is_file():
        raise FileNotFoundError(f"Hand class map not found: {CLASS_MAP_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    if model.input_shape[1:] != (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
        raise ValueError(f"Unexpected hand-model input shape: {model.input_shape}")
    if model.output_shape[-1] != 50:
        raise ValueError(f"Unexpected hand-model output shape: {model.output_shape}")
    return model, load_class_names(CLASS_MAP_PATH)


def resolve_expected_label(expected, class_names):
    """Accept either the full class label or its text after the numeric prefix."""

    if not expected:
        return None
    expected = expected.casefold()
    matches = [
        label
        for label in class_names
        if label.casefold() == expected
        or label.partition(". ")[2].casefold() == expected
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected sign is not a unique INCLUDE-50 class: {expected}")
    return matches[0]


def vote_prediction(history):
    """Return a confidence-weighted label only after two agreeing windows."""

    totals = {}
    counts = {}
    for label, confidence in history:
        totals[label] = totals.get(label, 0.0) + confidence
        counts[label] = counts.get(label, 0) + 1
    label = max(totals, key=lambda item: (counts[item], totals[item]))
    stable = counts[label] >= 2
    confidence = totals[label] / counts[label]
    return label, confidence, stable


def classify(model, class_names, matcher, sequence, history, expected_label=None):
    """Classify once per completed window and apply temporal voting."""

    array = np.asarray(sequence, dtype=np.float32)
    if array.shape != (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
        raise ValueError(f"Expected {(SEQUENCE_LENGTH, FEATURES_PER_FRAME)}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Captured hand sequence contains NaN or Inf")

    probabilities = np.asarray(model.predict(array[None, :, :], verbose=0)[0])
    index = int(np.argmax(probabilities))
    raw_label = class_names[index]
    raw_confidence = float(probabilities[index])
    history.append((raw_label, raw_confidence))
    label, confidence, stable = vote_prediction(history)

    # With an expected sign, DTW answers "does this movement match expected?".
    # Otherwise it measures the movement match for the temporally voted label.
    reference_label = expected_label or label
    reference_match = matcher.match(array, candidate_label=reference_label)
    if expected_label:
        status = "MATCHING" if stable and label == expected_label and reference_match["available"] else "NOT MATCHING"
    else:
        status = "MATCHING" if stable and reference_match["available"] else (
            "STABILIZING" if not stable else "REFERENCE UNAVAILABLE"
        )

    if expected_label:
        print(f"\nEXPECTED: {expected_label}")
        print(f"DETECTED: {label}")
    else:
        print(f"\nSIGN: {label}")
    print(f"LSTM CONFIDENCE: {confidence * 100:.1f}%")
    if reference_match["available"]:
        print(f"REFERENCE MATCH: {reference_match['match_percent']:.1f}%")
    else:
        print("REFERENCE MATCH: unavailable")
    print(f"STATUS: {status}")
    return label, confidence, reference_match, status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", help="Optional expected INCLUDE-50 sign, e.g. '1. Dog' or Dog")
    args = parser.parse_args(argv)
    model, class_names = load_hand_model()
    expected_label = resolve_expected_label(args.expected, class_names)
    matcher = DTWReferenceMatcher().load()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam index 0")

    sequence = deque(maxlen=SEQUENCE_LENGTH)
    prediction_history = deque(maxlen=VOTE_WINDOWS)
    capturing = False
    detected_hand_frames = 0
    status = "Press C to capture 45 frames; Q to quit."
    display_label = "Waiting"
    display_confidence = None
    display_match = None
    display_status = "WAITING"

    try:
        with mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Webcam frame read failed")

                # This calls the identical extractor used for dataset creation:
                # MediaPipe handedness -> wrist-centering -> scale normalization
                # -> left-hand 63 values + right-hand 63 values.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                features, left_present, right_present = extract_hand_landmarks(results)

                if capturing:
                    sequence.append(features)
                    detected_hand_frames += int(left_present or right_present)
                    status = f"Capturing {len(sequence)}/{SEQUENCE_LENGTH}"
                    if len(sequence) == SEQUENCE_LENGTH:
                        if detected_hand_frames < MIN_HAND_FRAMES:
                            display_label, display_confidence, display_match = "No sign", None, None
                            display_status = "NO SIGN DETECTED"
                            print("\nSTATUS: NO SIGN DETECTED")
                        else:
                            display_label, display_confidence, match, display_status = classify(
                                model, class_names, matcher, list(sequence), prediction_history, expected_label
                            )
                            display_match = match.get("match_percent") if match["available"] else None
                        capturing = False
                        status = "Complete. Press C to capture again."

                preview = cv2.flip(frame, 1)
                cv2.putText(preview, status, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
                cv2.putText(preview, f"Hands: left={'yes' if left_present else 'no'}, right={'yes' if right_present else 'no'}", (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                prefix = "Detected" if expected_label else "Sign"
                cv2.putText(preview, f"{prefix}: {display_label}", (16, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                if display_confidence is not None:
                    cv2.putText(preview, f"Confidence: {display_confidence * 100:.1f}%", (16, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                match_text = f"Match: {display_match:.1f}%" if display_match is not None else "Match: unavailable"
                cv2.putText(preview, match_text, (16, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(preview, f"Status: {display_status}", (16, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                if expected_label:
                    cv2.putText(preview, f"Expected: {expected_label}", (16, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow(WINDOW_NAME, preview)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c") and not capturing:
                    sequence.clear()
                    capturing = True
                    detected_hand_frames = 0
                    display_label, display_confidence, display_match = "Capturing", None, None
                    display_status = "CAPTURING"
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
