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

# ============================================================
# SOFT ACTION MATCHING CONFIGURATION
# ============================================================
#
# These values are NOT LSTM accuracy thresholds.
# They control the final action-relatedness decision.
#
# The goal:
#   SAME SIGN + natural variation  -> MATCHING
#   PARTIAL SAME SIGN              -> MATCHING
#   COMPLETELY DIFFERENT ACTION    -> NOT MATCHING
#
# DTW match_percent is used as action similarity evidence.
# We do NOT use a single threshold by itself.
# ============================================================

# Strong action similarity.
STRONG_ACTION_SIMILARITY = 65.0

# Soft/variation action similarity.
SOFT_ACTION_SIMILARITY = 50.0

# Partial action similarity.
PARTIAL_ACTION_SIMILARITY = 30.0

# If another sign has very high LSTM confidence,
# don't allow weak expected-sign similarity to override it.
STRONG_WRONG_PREDICTION = 0.75

# If expected action is only slightly worse/better than
# competing action, allow the action evidence to participate.
ACTION_MARGIN_TOLERANCE = 8.0

# Require the action decision to be positive in at least
# two of the latest three completed windows.
MATCH_VOTE_REQUIRED = 2



# ============================================================
# V3.1 ACTION-FIRST MATCHING
# ============================================================

# Action similarity is the PRIMARY evidence.
#
# Natural human variation is allowed:
#   - speed
#   - hand position
#   - hand angle
#   - trajectory
#   - starting position
#   - ending position
#   - signing style

STRONG_ACTION_SIMILARITY = 60.0
MODERATE_ACTION_SIMILARITY = 45.0
PARTIAL_ACTION_SIMILARITY = 30.0

# A competing prediction can reject only when it is very
# confident AND its action similarity is clearly stronger.
STRONG_WRONG_PREDICTION = 0.80
STRONG_CONFLICT_MARGIN = 12.0

VOTE_WINDOWS = 3
MATCH_VOTE_REQUIRED = 2
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


def classify(
    model,
    class_names,
    matcher,
    sequence,
    history,
    match_history,
    expected_label=None,
):
    """Action-first matching for natural human sign variation.

    ACTION SIMILARITY is primary.

    LSTM is supporting evidence and conflict detection.

    The performer does not need to reproduce the reference
    video exactly.
    """

    array = np.asarray(sequence, dtype=np.float32)

    if array.shape != (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
        raise ValueError(
            f"Expected {(SEQUENCE_LENGTH, FEATURES_PER_FRAME)}, "
            f"got {array.shape}"
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "Captured hand sequence contains NaN or Inf"
        )

    # ========================================================
    # LSTM
    # ========================================================

    probabilities = np.asarray(
        model.predict(
            array[None, :, :],
            verbose=0,
        )[0],
        dtype=np.float32,
    )

    index = int(np.argmax(probabilities))

    raw_label = class_names[index]
    raw_confidence = float(probabilities[index])

    history.append(
        (raw_label, raw_confidence)
    )

    label, confidence, stable = vote_prediction(history)

    # ========================================================
    # NO EXPECTED SIGN
    # ========================================================

    if expected_label is None:

        reference_match = matcher.match(
            array,
            candidate_label=label,
        )

        if reference_match["available"]:
            status = (
                "MATCHING"
                if stable
                else "STABILIZING"
            )
        else:
            status = "REFERENCE UNAVAILABLE"

        print(f"\nSIGN: {label}")
        print(
            f"LSTM CONFIDENCE: "
            f"{confidence * 100:.1f}%"
        )

        if reference_match["available"]:
            print(
                f"ACTION SIMILARITY: "
                f"{reference_match['match_percent']:.1f}%"
            )
        else:
            print(
                "ACTION SIMILARITY: unavailable"
            )

        print(f"STATUS: {status}")

        return (
            label,
            confidence,
            reference_match,
            status,
        )

    # ========================================================
    # EXPECTED SIGN
    # ========================================================

    expected_index = class_names.index(
        expected_label
    )

    expected_probability = float(
        probabilities[expected_index]
    )

    top_indices = np.argsort(
        probabilities
    )[-3:][::-1]

    expected_in_top3 = (
        expected_index in top_indices
    )

    # ========================================================
    # EXPECTED ACTION
    # ========================================================

    expected_match = matcher.match(
        array,
        candidate_label=expected_label,
    )

    expected_similarity = (
        float(expected_match["match_percent"])
        if expected_match["available"]
        else 0.0
    )

    # ========================================================
    # COMPETING ACTION
    # ========================================================

    if raw_label == expected_label:

        predicted_match = expected_match

    else:

        predicted_match = matcher.match(
            array,
            candidate_label=raw_label,
        )

    predicted_similarity = (
        float(predicted_match["match_percent"])
        if predicted_match["available"]
        else 0.0
    )

    action_margin = (
        expected_similarity
        - predicted_similarity
    )

    # ========================================================
    # 1. STRONG EXPECTED ACTION
    #
    # IMPORTANT:
    #
    # LSTM confidence is NOT required to be high.
    #
    # If the performed action strongly resembles the expected
    # sign, it can match.
    # ========================================================

    strong_action = (
        expected_match["available"]
        and expected_similarity
        >= STRONG_ACTION_SIMILARITY
    )

    # ========================================================
    # 2. STRONG DIFFERENT ACTION
    #
    # Only reject when ALL are true:
    #
    #   - LSTM strongly predicts another sign
    #   - competing action is clearly stronger
    #   - expected action is not strong
    # ========================================================

    strong_conflict = (
        raw_label != expected_label
        and raw_confidence
        >= STRONG_WRONG_PREDICTION
        and expected_similarity
        < STRONG_ACTION_SIMILARITY
        and predicted_similarity
        >= expected_similarity
        + STRONG_CONFLICT_MARGIN
    )

    # ========================================================
    # 3. MODERATE SAME ACTION
    # ========================================================

    moderate_action = (
        expected_similarity
        >= MODERATE_ACTION_SIMILARITY
        and not strong_conflict
        and (
            expected_in_top3
            or expected_probability >= 0.10
            or action_margin >= 0
            or raw_confidence < 0.50
        )
    )

    # ========================================================
    # 4. PARTIAL ACTION
    # ========================================================

    partial_action = (
        expected_similarity
        >= PARTIAL_ACTION_SIMILARITY
        and expected_similarity
        < MODERATE_ACTION_SIMILARITY
        and not strong_conflict
        and (
            expected_in_top3
            or expected_probability >= 0.10
            or action_margin >= 0
            or raw_confidence < 0.35
        )
    )

    # ========================================================
    # WINDOW DECISION
    # ========================================================

    if strong_conflict:

        window_matching = False
        match_type = "DIFFERENT ACTION"

    elif strong_action:

        window_matching = True
        match_type = "STRONG / SAME ACTION"

    elif moderate_action:

        window_matching = True
        match_type = "VARIATION / SAME ACTION"

    elif partial_action:

        window_matching = True
        match_type = "PARTIAL ACTION"

    else:

        window_matching = False
        match_type = "DIFFERENT ACTION"

    # ========================================================
    # TEMPORAL HISTORY
    # ========================================================

    match_history.append(
        window_matching
    )

    matching_votes = sum(
        match_history
    )

    # Strong expected action is immediately accepted.
    if strong_action and not strong_conflict:

        final_matching = True

    elif strong_conflict:

        final_matching = False

    else:

        final_matching = (
            matching_votes
            >= MATCH_VOTE_REQUIRED
        )

    status = (
        "MATCHING"
        if final_matching
        else "NOT MATCHING"
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print(
        f"\nEXPECTED: {expected_label}"
    )

    print(
        f"DETECTED: {label}"
    )

    print(
        f"LSTM CONFIDENCE: "
        f"{confidence * 100:.1f}%"
    )

    print(
        f"EXPECTED LSTM PROBABILITY: "
        f"{expected_probability * 100:.1f}%"
    )

    if expected_match["available"]:

        print(
            f"EXPECTED ACTION SIMILARITY: "
            f"{expected_similarity:.1f}%"
        )

    else:

        print(
            "EXPECTED ACTION SIMILARITY: "
            "unavailable"
        )

    if predicted_match["available"]:

        print(
            f"DETECTED ACTION SIMILARITY: "
            f"{predicted_similarity:.1f}%"
        )

    else:

        print(
            "DETECTED ACTION SIMILARITY: "
            "unavailable"
        )

    print(
        f"ACTION MARGIN "
        f"(expected - detected): "
        f"{action_margin:+.1f}%"
    )

    print(
        f"EXPECTED IN LSTM TOP-3: "
        f"{'YES' if expected_in_top3 else 'NO'}"
    )

    print(
        f"WINDOW DECISION: "
        f"{match_type}"
    )

    print(
        f"MATCH VOTES: "
        f"{matching_votes}/{len(match_history)}"
    )

    print(
        f"STATUS: {status}"
    )

    display_match = (
        expected_similarity
        if expected_match["available"]
        else None
    )

    return (
        label,
        confidence,
        {
            **expected_match,
            "match_percent":
                display_match,

            "expected_similarity":
                expected_similarity,

            "detected_similarity":
                predicted_similarity,

            "action_margin":
                action_margin,

            "window_matching":
                window_matching,

            "match_type":
                match_type,
        },
        status,
    )

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
    match_history = deque(maxlen=VOTE_WINDOWS)
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
                                model,
                                class_names,
                                matcher,
                                list(sequence),
                                prediction_history,
                                match_history,
                                expected_label,
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
                match_text = (
                    f"Action Similarity: {display_match:.1f}%"
                    if display_match is not None
                    else "Action Similarity: unavailable"
                )
                cv2.putText(
                    preview,
                    match_text,
                    (16, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(preview, f"Status: {display_status}", (16, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                if expected_label:
                    cv2.putText(preview, f"Expected: {expected_label}", (16, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow(WINDOW_NAME, preview)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c") and not capturing:
                    sequence.clear()
                    match_history.clear()
                    prediction_history.clear()
                    capturing = True
                    detected_hand_frames = 0
                    display_label, display_confidence, display_match = "Capturing", None, None
                    display_status = "CAPTURING"
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
