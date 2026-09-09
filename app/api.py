"""FastAPI service for validated Include-50 V2 sequence inference."""

import logging
import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from app.core.dtw_matcher import DTWReferenceMatcher
from app.core.hand_model_runtime import HandModelRuntime
from app.core.model_runtime import ModelRuntime


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOGGER = logging.getLogger(__name__)
app = FastAPI(title="ISL Recognition API", version="2.0")
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ISL_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"]
)


class SequenceRequest(BaseModel):
    sequence: list[list[float]] = Field(..., min_length=45, max_length=45)
    hand_sequence: list[list[float]] | None = Field(default=None, min_length=45, max_length=45)
    quality: dict[str, float | int | bool] | None = None


class HandSequenceRequest(BaseModel):
    hand_sequence: list[list[float]] = Field(..., min_length=45, max_length=45)
    expected_sign: str | None = None

    @field_validator("hand_sequence")
    @classmethod
    def validate_hand_sequence_shape(cls, value):
        if len(value) != 45:
            raise ValueError("hand_sequence must contain exactly 45 frames")
        if any(len(frame) != 126 for frame in value):
            raise ValueError("Each frame in hand_sequence must contain exactly 126 features")
        return value


@lru_cache(maxsize=1)
def get_runtime():
    return ModelRuntime().load()


@lru_cache(maxsize=1)
def get_dtw_matcher():
    return DTWReferenceMatcher().load()


@lru_cache(maxsize=1)
def get_hand_runtime():
    return HandModelRuntime().load()


@app.get("/health")
def health():
    try:
        runtime = get_hand_runtime()
        matcher = get_dtw_matcher()
        class_count = len(runtime.class_names)
        reference_sequences = sum(len(items) for items in matcher.references.values())

        return {
            "status": "ok",
            "model_loaded": runtime.model is not None,
            "model_type": "hand_lstm",
            "class_count": class_count,
            "reference_database_available": reference_sequences > 0,
            "reference_sequences": reference_sequences,
        }
    except Exception as error:
        LOGGER.exception("Health check failed")
        raise HTTPException(
            status_code=503,
            detail={"status": "degraded", "model_loaded": False},
        ) from error


@app.post("/predict")
def predict(request: SequenceRequest):
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
        LOGGER.exception("Inference request failed")
        raise HTTPException(status_code=500, detail="Inference failed") from error


@app.post("/predict/sequence")
def predict_sequence(request: SequenceRequest):
    return predict(request)


@app.get("/health/hand")
def hand_health():
    try:
        runtime = get_hand_runtime()
        matcher = get_dtw_matcher()
        class_count = len(runtime.class_names)
        reference_classes = len(matcher.references)
        reference_sequences = sum(len(items) for items in matcher.references.values())
        return {
            "status": "ok",
            "model_loaded": runtime.model is not None,
            "class_count": class_count,
            "reference_database_available": reference_classes >= 1 and reference_sequences >= 1,
            "reference_classes": reference_classes,
            "reference_sequences": reference_sequences,
            "classes": runtime.class_names,
        }
    except Exception as error:
        LOGGER.exception("Hand-model health check failed")
        raise HTTPException(status_code=503, detail={"status": "degraded", "model_loaded": False}) from error


@app.post("/predict/hand-sequence")
def predict_hand_sequence(request: HandSequenceRequest):
    try:
        runtime = get_hand_runtime()
        result = runtime.predict(request.hand_sequence)
        expected_sign = request.expected_sign if request.expected_sign in runtime.class_names else None
        result["reference_match"] = get_dtw_matcher().match(
            request.hand_sequence,
            candidate_label=expected_sign or result["prediction"],
        )
        result["expected_sign"] = expected_sign
        return result
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        LOGGER.exception("Hand inference request failed")
        raise HTTPException(status_code=500, detail="Hand inference failed") from error

