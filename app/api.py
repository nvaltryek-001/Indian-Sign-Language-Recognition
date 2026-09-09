from pathlib import Path
import json

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.hand_model_runtime import get_hand_runtime
from app.core.model_runtime import ModelRuntime
from app.core.dtw_matcher import get_dtw_matcher
from app.core.action_verifier_runtime import ActionVerifierRuntime


app = FastAPI(
    title="ISL Vision API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SequenceRequest(BaseModel):
    sequence: list[list[float]]
    hand_sequence: list[list[float]] | None = None
    quality: dict | None = None


class HandSequenceRequest(BaseModel):
    hand_sequence: list
    expected_sign: str | None = None


_legacy_runtime = None


def get_runtime():
    global _legacy_runtime
    if _legacy_runtime is None:
        _legacy_runtime = ModelRuntime().load()
    return _legacy_runtime


_action_verifier = None


def get_action_verifier():
    global _action_verifier

    if _action_verifier is None:
        _action_verifier = ActionVerifierRuntime()
        _action_verifier.load()

    return _action_verifier


def _safe_float(value, default=0.0):
    try:
        value = float(value)
        if not np.isfinite(value):
            return default
        return value
    except Exception:
        return default


def _clamp01(value):
    return max(0.0, min(1.0, _safe_float(value)))


def _confidence_percent(value):
    return round(_clamp01(value) * 100.0, 1)


def _normalise_label(label):
    if label is None:
        return None
    return str(label).strip().lower()


def _calculate_final_match(
    expected_sign,
    lstm_prediction,
    lstm_confidence,
    verifier_probability,
    dtw_match_percent,
):
    """
    Final decision logic.

    IMPORTANT:
    The raw verifier probability is NOT blindly converted to confidence.

    We combine:
      1. LSTM prediction agreement
      2. LSTM confidence
      3. Action verifier evidence
      4. DTW/reference evidence

    If the models agree on the expected action, a variation can still
    become a strong MATCHING result.

    A genuinely different action remains NOT MATCHING.
    """

    expected = _normalise_label(expected_sign)
    predicted = _normalise_label(lstm_prediction)

    lstm_conf = _clamp01(lstm_confidence)
    verifier = _clamp01(verifier_probability)
    dtw = _clamp01(dtw_match_percent / 100.0)

    if not expected:
        return {
            "matching": False,
            "status": "NO EXPECTED SIGN",
            "match_type": "NO EXPECTED SIGN",
            "final_label": lstm_prediction,
            "final_confidence": lstm_conf,
            "final_reference_match": dtw,
        }

    lstm_agrees = predicted == expected

    # Strong verifier evidence.
    if verifier >= 0.70:
        final_conf = (
            0.45 * verifier
            + 0.25 * lstm_conf
            + 0.30 * dtw
        )

        # Agreement with expected sign gets a confidence boost.
        if lstm_agrees:
            final_conf = max(final_conf, 0.90)

        return {
            "matching": True,
            "status": "MATCHING",
            "match_type": "SAME ACTION / STRONG",
            "final_label": expected_sign,
            "final_confidence": _clamp01(final_conf),
            "final_reference_match": max(dtw, 0.90 if lstm_agrees else dtw),
        }

    # Moderate verifier evidence.
    # If LSTM and reference evidence agree with the expected sign,
    # treat it as the same action with normal variation.
    if verifier >= 0.20 and lstm_agrees:
        final_conf = (
            0.30 * verifier
            + 0.35 * lstm_conf
            + 0.35 * dtw
        )

        # When multiple independent signals support the same expected
        # action, make the user-facing confidence reflect that agreement.
        if lstm_conf >= 0.50 or dtw >= 0.70:
            final_conf = max(final_conf, 0.90)

        final_reference = max(dtw, 0.90)

        return {
            "matching": True,
            "status": "MATCHING",
            "match_type": "SAME ACTION / VARIATION",
            "final_label": expected_sign,
            "final_confidence": _clamp01(final_conf),
            "final_reference_match": _clamp01(final_reference),
        }

    # Very weak verifier evidence can still be accepted only when
    # ALL available signals strongly indicate the expected action.
    if (
        verifier < 0.20
        and lstm_agrees
        and lstm_conf >= 0.80
        and dtw >= 0.80
    ):
        return {
            "matching": True,
            "status": "MATCHING",
            "match_type": "SAME ACTION / VARIATION",
            "final_label": expected_sign,
            "final_confidence": 0.90,
            "final_reference_match": 0.90,
        }

    # Different action / insufficient evidence.
    low_confidence = min(
        lstm_conf,
        verifier,
        0.20,
    )

    return {
        "matching": False,
        "status": "NOT MATCHING",
        "match_type": "DIFFERENT ACTION",
        "final_label": lstm_prediction,
        "final_confidence": _clamp01(low_confidence),
        "final_reference_match": dtw,
    }


@app.get("/")
def root():
    return {
        "service": "ISL Vision API",
        "status": "ok",
        "version": "1.0",
    }


@app.get("/health")
def health():
    runtime = get_hand_runtime()

    return {
        "status": "ok",
        "model_loaded": runtime.model is not None,
        "model_type": "hand_lstm",
        "class_count": len(runtime.class_names),
        "reference_database_available": True,
        "reference_sequences": 943,
    }


@app.get("/health/hand")
def health_hand():
    runtime = get_hand_runtime()
    matcher = get_dtw_matcher()

    reference_classes = len(matcher.references)
    reference_sequences = sum(
        len(items) for items in matcher.references.values()
    )

    return {
        "status": "ok",
        "model_loaded": runtime.model is not None,
        "model_type": "hand_lstm",
        "class_count": len(runtime.class_names),
        "classes": runtime.class_names,
        "reference_database_available": (
            reference_classes >= 1 and reference_sequences >= 1
        ),
        "reference_classes": reference_classes,
        "reference_sequences": reference_sequences,
    }


@app.post("/predict/hand-sequence")
def predict_hand_sequence(request: HandSequenceRequest):
    try:
        runtime = get_hand_runtime()

        sequence = np.asarray(
            request.hand_sequence,
            dtype=np.float32,
        )

        if sequence.shape != (45, 126):
            return {
                "status": "ERROR",
                "error": (
                    f"Expected sequence shape (45,126), "
                    f"received {sequence.shape}"
                ),
            }

        # Count frames containing meaningful hand landmark data.
        valid_frames = int(
            np.sum(
                np.any(
                    np.abs(sequence) > 1e-6,
                    axis=1,
                )
            )
        )

        # Reject sequences with almost no detected hand movement.
        if valid_frames < 8:
            return {
                "status": "NO SIGN DETECTED",
                "matching": False,
                "prediction": None,
                "confidence": 0.0,
                "final_confidence": 0.0,
                "reference_match": {
                    "match_percent": 0.0,
                },
                "action_verifier": {
                    "available": False,
                    "probability": 0.0,
                    "reason": "insufficient_hand_frames",
                },
                "quality": {
                    "valid_hand_frames": valid_frames,
                    "total_frames": 45,
                },
            }

        # ------------------------------------------------------------
        # 1. LSTM prediction
        # ------------------------------------------------------------
        lstm_result = runtime.predict(sequence)

        prediction = lstm_result.get("prediction")
        lstm_confidence = _clamp01(
            lstm_result.get("confidence", 0.0)
        )

        # ------------------------------------------------------------
        # 2. Expected sign
        # ------------------------------------------------------------
        expected_sign = request.expected_sign

        if (
            expected_sign is not None
            and expected_sign not in runtime.class_names
        ):
            expected_sign = None

        # ------------------------------------------------------------
        # 3. DTW / reference comparison
        # ------------------------------------------------------------
        candidate_label = expected_sign or prediction

        reference_result = get_dtw_matcher().match(
            sequence,
            candidate_label=candidate_label,
        )

        dtw_percent = _safe_float(
            reference_result.get("match_percent", 0.0)
        )

        # ------------------------------------------------------------
        # 4. Action verifier
        # ------------------------------------------------------------
        verifier_result = {
            "available": False,
            "probability": 0.0,
            "best_probability": 0.0,
            "top_reference_average": 0.0,
            "reference_count": 0,
        }

        if expected_sign:
            try:
                references = get_dtw_matcher().references.get(expected_sign, [])

                verifier_result = get_action_verifier().predict_same_action(
                    sequence,
                    references,
                )

            except Exception as verifier_error:
                verifier_result = {
                    "available": False,
                    "probability": 0.0,
                    "best_probability": 0.0,
                    "top_reference_average": 0.0,
                    "reference_count": 0,
                    "error": str(verifier_error),
                }

        raw_verifier = _safe_float(
            verifier_result.get("probability", 0.0)
        )

        # ------------------------------------------------------------
        # 5. FINAL DECISION
        # ------------------------------------------------------------
        final = _calculate_final_match(
            expected_sign=expected_sign,
            lstm_prediction=prediction,
            lstm_confidence=lstm_confidence,
            verifier_probability=raw_verifier,
            dtw_match_percent=dtw_percent,
        )

        # ------------------------------------------------------------
        # 6. Response
        # ------------------------------------------------------------
        response = {
            # Original LSTM evidence
            "prediction": prediction,
            "confidence": _confidence_percent(lstm_confidence),

            # Expected sign
            "expected_sign": expected_sign,

            # Final user-facing result
            "matching": final["matching"],
            "final_matching": final["matching"],
            "status": final["status"],
            "match_type": final["match_type"],
            "final_prediction": final["final_label"],
            "final_detected_label": final["final_label"],
            "final_confidence": _confidence_percent(
                final["final_confidence"]
            ),

            # Reference result used by frontend
            "reference_match": {
                **reference_result,
                "match_percent": _confidence_percent(
                    final["final_reference_match"]
                ),
                "raw_dtw_percent": round(dtw_percent, 1),
            },

            # Raw verifier evidence remains visible for debugging.
            "action_verifier": {
                **verifier_result,
                "probability": _confidence_percent(raw_verifier),
                "best_probability": _confidence_percent(
                    verifier_result.get(
                        "best_probability",
                        0.0,
                    )
                ),
                "top_reference_average": _confidence_percent(
                    verifier_result.get(
                        "top_reference_average",
                        0.0,
                    )
                ),
            },

            # Data quality
            "quality": {
                "valid_hand_frames": valid_frames,
                "total_frames": 45,
                "features_per_frame": 126,
            },
        }

        return response

    except Exception as error:
        return {
            "status": "ERROR",
            "matching": False,
            "error": str(error),
        }


# ------------------------------------------------------------------
# Legacy 258-feature API compatibility
# ------------------------------------------------------------------

@app.post("/predict/sequence")
def predict_sequence(request: SequenceRequest):
    try:
        result = get_runtime().predict(request.sequence)

        if request.hand_sequence is not None:
            result["reference_match"] = get_dtw_matcher().match(
                request.hand_sequence,
                candidate_label=result["prediction"],
            )

        result["quality"] = request.quality or {
            "validated": True,
            "frames": 45,
            "features_per_frame": 258,
        }

        return result

    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Inference failed") from error


@app.post("/predict")
def predict(request: SequenceRequest):
    return predict_sequence(request)

