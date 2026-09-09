"""V5 hand-only INCLUDE-50 webcam recognition with LSTM and same-action verification.

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

# ============================================================
# ACTION VERIFIER
# ============================================================

ACTION_VERIFIER_MODEL_PATH = (
    ROOT / "models" / "include50_action_verifier.keras"
)

ACTION_VERIFIER_META_PATH = (
    ROOT / "models" / "include50_action_verifier.json"
)

ACTION_VERIFIER_MATCH_THRESHOLD = 0.90
ACTION_VERIFIER_STRONG_THRESHOLD = 0.90
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



def load_action_verifier():
    """Load the Siamese same-action verifier."""

    if not ACTION_VERIFIER_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Action verifier not found: "
            f"{ACTION_VERIFIER_MODEL_PATH}"
        )

    if not ACTION_VERIFIER_META_PATH.is_file():
        raise FileNotFoundError(
            f"Action verifier metadata not found: "
            f"{ACTION_VERIFIER_META_PATH}"
        )

    model = tf.keras.models.load_model(
        ACTION_VERIFIER_MODEL_PATH,
        safe_mode=False,
    )

    metadata = json.loads(
        ACTION_VERIFIER_META_PATH.read_text(
            encoding="utf-8"
        )
    )

    mean = np.asarray(
        metadata["normalization_mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        metadata["normalization_std"],
        dtype=np.float32,
    )

    std = np.maximum(
        std,
        1e-4,
    )

    if mean.shape != (FEATURES_PER_FRAME,):
        raise ValueError(
            f"Unexpected verifier mean shape: {mean.shape}"
        )

    if std.shape != (FEATURES_PER_FRAME,):
        raise ValueError(
            f"Unexpected verifier std shape: {std.shape}"
        )

    return model, mean, std


def verifier_probability(
    verifier,
    mean,
    std,
    reference_sequence,
    webcam_sequence,
):
    """Return probability that two sequences represent the same action."""

    reference = np.asarray(
        reference_sequence,
        dtype=np.float32,
    )

    webcam = np.asarray(
        webcam_sequence,
        dtype=np.float32,
    )

    if reference.shape != (
        SEQUENCE_LENGTH,
        FEATURES_PER_FRAME,
    ):
        raise ValueError(
            f"Unexpected reference shape: {reference.shape}"
        )

    if webcam.shape != (
        SEQUENCE_LENGTH,
        FEATURES_PER_FRAME,
    ):
        raise ValueError(
            f"Unexpected webcam shape: {webcam.shape}"
        )

    reference = (
        reference - mean
    ) / std

    webcam = (
        webcam - mean
    ) / std

    prediction = verifier.predict(
        [
            reference[None, :, :],
            webcam[None, :, :],
        ],
        verbose=0,
    )

    return float(
        np.asarray(prediction).reshape(-1)[0]
    )


def get_reference_sequences_for_label(
    matcher,
    label,
):
    """Return the reference sequences belonging to one sign."""

    references = matcher.references.get(
        label,
        [],
    )

    return references


def action_verifier_score(
    verifier,
    mean,
    std,
    matcher,
    webcam_sequence,
    expected_label,
):
    """Compare webcam action against multiple references.

    We don't trust one reference video.

    The verifier compares the webcam action against all
    available examples for the expected sign and uses a
    robust top-reference average.
    """

    references = get_reference_sequences_for_label(
        matcher,
        expected_label,
    )

    if not references:
        return {
            "available": False,
            "probability": 0.0,
            "best_probability": 0.0,
            "reference_count": 0,
        }

    scores = []

    for reference in references:

        # DTWReferenceMatcher stores reference arrays.
        if isinstance(reference, dict):
            sequence = reference.get(
                "sequence"
            )
        else:
            sequence = reference

        if sequence is None:
            continue

        try:

            score = verifier_probability(
                verifier,
                mean,
                std,
                sequence,
                webcam_sequence,
            )

            if np.isfinite(score):
                scores.append(score)

        except Exception:
            continue

    if not scores:
        return {
            "available": False,
            "probability": 0.0,
            "best_probability": 0.0,
            "reference_count": 0,
        }

    scores.sort(
        reverse=True
    )

    # Use the strongest few reference examples.
    #
    # This allows the performer to differ from one particular
    # dataset signer while still matching the same sign family.
    top_count = min(
        5,
        len(scores),
    )

    top_scores = scores[:top_count]

    best = float(
        top_scores[0]
    )

    robust = float(
        np.mean(top_scores)
    )

    # Blend best and robust evidence.
    probability = (
        0.60 * best
        + 0.40 * robust
    )

    return {
        "available": True,
        "probability": probability,
        "best_probability": best,
        "top_reference_average": robust,
        "reference_count": len(scores),
    }



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
    verifier,
    verifier_mean,
    verifier_std,
    sequence,
    history,
    match_history,
    expected_label=None,
):
    """Final V4 decision.

    The Action Verifier is the main same-action detector.

    LSTM:
        identifies the likely sign.

    Action Verifier:
        determines whether the webcam performance belongs
        to the expected sign/action family.

    DTW:
        remains a secondary diagnostic signal.
    """

    array = np.asarray(
        sequence,
        dtype=np.float32,
    )

    if array.shape != (
        SEQUENCE_LENGTH,
        FEATURES_PER_FRAME,
    ):
        raise ValueError(
            f"Expected "
            f"{(SEQUENCE_LENGTH, FEATURES_PER_FRAME)}, "
            f"got {array.shape}"
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "Captured hand sequence contains NaN or Inf"
        )

    # ========================================================
    # EXISTING LSTM
    # ========================================================

    probabilities = np.asarray(
        model.predict(
            array[None, :, :],
            verbose=0,
        )[0],
        dtype=np.float32,
    )

    index = int(
        np.argmax(probabilities)
    )

    raw_label = class_names[index]
    raw_confidence = float(
        probabilities[index]
    )

    history.append(
        (raw_label, raw_confidence)
    )

    label, confidence, stable = vote_prediction(
        history
    )

    # ========================================================
    # EXPECTED SIGN MODE
    # ========================================================

    if expected_label is not None:

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

        # ----------------------------------------------------
        # NEW ACTION VERIFIER
        # ----------------------------------------------------

        verifier_result = action_verifier_score(
            verifier,
            verifier_mean,
            verifier_std,
            matcher,
            array,
            expected_label,
        )

        if verifier_result["available"]:

            verifier_probability_value = float(
                verifier_result["probability"]
            )

            verifier_percent = (
                verifier_probability_value * 100.0
            )

        else:

            verifier_probability_value = 0.0
            verifier_percent = 0.0

        # ----------------------------------------------------
        # Existing DTW diagnostic
        # ----------------------------------------------------

        dtw_result = matcher.match(
            array,
            candidate_label=expected_label,
        )

        if dtw_result["available"]:

            dtw_percent = float(
                dtw_result["match_percent"]
            )

        else:

            dtw_percent = 0.0

        # ----------------------------------------------------
        # FINAL DECISION
        #
        # The verifier is now the main decision maker.
        # ----------------------------------------------------
        # ====================================================
        # V5 FINAL EXPECTED-SIGN DECISION
        # ====================================================
        #
        # The dedicated Siamese Action Verifier is the authority
        # for expected-sign matching.
        #
        # IMPORTANT:
        # - If verifier >= 90%, the action is accepted as the
        #   expected sign. We display the EXPECTED sign as DETECTED.
        # - If verifier < 90%, the expected sign is NOT verified.
        #   We display the LSTM prediction, but cap the final
        #   confidence at 10% so the UI does not imply a strong
        #   recognition when the expected action failed.
        #
        # No confidence is artificially raised above the verifier's
        # actual probability.
        # ====================================================

        if (
            verifier_result["available"]
            and verifier_probability_value >= ACTION_VERIFIER_STRONG_THRESHOLD
        ):
            final_matching = True
            final_detected_label = expected_label
            final_confidence = verifier_probability_value
            match_type = "SAME ACTION / VERIFIED"
            status = "MATCHING"
        else:
            final_matching = False
            final_detected_label = label
            final_confidence = min(float(confidence), 0.10)
            match_type = "DIFFERENT ACTION / NOT VERIFIED"
            status = "NOT MATCHING"


        # ====================================================
        # V5 FINAL EXPECTED-SIGN DECISION
        # ====================================================


        print(
            f"DETECTED: {final_detected_label}"
        )

        print(
            f"FINAL CONFIDENCE: "
            f"{final_confidence * 100:.1f}%"
        )

        print(
            f"EXPECTED LSTM PROBABILITY: "
            f"{expected_probability * 100:.1f}%"
        )

        print(
            f"EXPECTED IN LSTM TOP-3: "
            f"{'YES' if expected_in_top3 else 'NO'}"
        )

        print(
            f"ACTION VERIFIER: "
            f"{verifier_percent:.1f}%"
        )

        print(
            f"BEST REFERENCE: "
            f"{verifier_result.get('best_probability', 0.0) * 100:.1f}%"
        )

        print(
            f"REFERENCE COUNT: "
            f"{verifier_result.get('reference_count', 0)}"
        )

        print(
            f"DTW ACTION SIMILARITY: "
            f"{dtw_percent:.1f}%"
        )

        print(
            f"DECISION: {match_type}"
        )

        print(
            f"STATUS: {status}"
        )

        return (
            final_detected_label,
            final_confidence,
            {
                **dtw_result,
                "match_percent":
                    verifier_percent,
                "action_verifier":
                    verifier_percent,
                "best_reference":
                    verifier_result.get(
                        "best_probability",
                        0.0,
                    ) * 100.0,
                "reference_count":
                    verifier_result.get(
                        "reference_count",
                        0,
                    ),
                "match_type":
                    match_type,
            },
            status,
        )

    # ========================================================
    # NORMAL MODE
    # ========================================================

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

        status = (
            "STABILIZING"
            if not stable
            else "REFERENCE UNAVAILABLE"
        )

    print(
        f"\nSIGN: {label}"
    )

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

    print(
        f"STATUS: {status}"
    )

    return (
        label,
        confidence,
        reference_match,
        status,
    )

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", help="Optional expected INCLUDE-50 sign, e.g. '1. Dog' or Dog")
    args = parser.parse_args(argv)
    print("\n=== ISL INCLUDE-50 V5 ACTION VERIFIER ===")
    print("Expected-sign acceptance threshold: 90%")
    print("Press C to capture 45 frames; Q to quit.\n")

    model, class_names = load_hand_model()

    verifier, verifier_mean, verifier_std = (
        load_action_verifier()
    )

    expected_label = resolve_expected_label(
        args.expected,
        class_names,
    )

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
                                verifier,
                                verifier_mean,
                                verifier_std,
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
                    f"Same-Action Confidence: {display_match:.1f}%"
                    if display_match is not None
                    else "Same-Action Confidence: unavailable"
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
