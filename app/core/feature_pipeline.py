"""Shared feature and quality checks for the Include-50 V2 pipeline."""

from dataclasses import dataclass

import numpy as np


POSE_FEATURES = 33 * 4
HAND_FEATURES = 21 * 3
FEATURES_PER_FRAME = POSE_FEATURES + HAND_FEATURES * 2
SEQUENCE_LENGTH = 45


@dataclass(frozen=True)
class LandmarkQuality:
    """Presence and validity information for one MediaPipe result."""

    pose_present: bool
    left_hand_present: bool
    right_hand_present: bool

    @property
    def both_hands_present(self):
        return self.left_hand_present and self.right_hand_present

    @property
    def all_required_present(self):
        return (
            self.pose_present
            and self.left_hand_present
            and self.right_hand_present
        )


class CaptureQualityGate:
    """Require consecutive complete frames before accepting capture input."""

    def __init__(self, warmup_frames=15):
        if warmup_frames < 1:
            raise ValueError("warmup_frames must be positive")
        self.warmup_frames = warmup_frames
        self.stable_frames = 0
        self.ready = False

    def update(self, quality):
        was_ready = self.ready
        if quality.all_required_present:
            self.stable_frames += 1
            self.ready = self.stable_frames >= self.warmup_frames
            status = "Ready" if self.ready else f"Warming up {self.stable_frames}/{self.warmup_frames}"
        else:
            self.stable_frames = 0
            self.ready = False
            status = "Show both hands and pose"
        return self.ready, status, was_ready and not self.ready

    def reset(self):
        self.stable_frames = 0
        self.ready = False


def assess_landmark_quality(results):
    """Return presence quality without changing or imputing landmarks."""

    quality = LandmarkQuality(
        pose_present=bool(results.pose_landmarks),
        left_hand_present=bool(results.left_hand_landmarks),
        right_hand_present=bool(results.right_hand_landmarks),
    )

    if quality.pose_present and len(results.pose_landmarks.landmark) != 33:
        raise ValueError("MediaPipe pose landmark count must be 33")
    if quality.left_hand_present and len(results.left_hand_landmarks.landmark) != 21:
        raise ValueError("MediaPipe left-hand landmark count must be 21")
    if quality.right_hand_present and len(results.right_hand_landmarks.landmark) != 21:
        raise ValueError("MediaPipe right-hand landmark count must be 21")

    return quality


def _landmark_values(landmarks, dimensions):
    if landmarks is None:
        return [0.0] * dimensions

    values = []
    for landmark in landmarks.landmark:
        values.extend((landmark.x, landmark.y, landmark.z))
        if dimensions == POSE_FEATURES:
            values.append(landmark.visibility)
    return values


def extract_landmarks(results):
    """Extract raw MediaPipe features in the training order."""

    assess_landmark_quality(results)
    features = np.asarray(
        _landmark_values(results.pose_landmarks, POSE_FEATURES)
        + _landmark_values(results.left_hand_landmarks, HAND_FEATURES)
        + _landmark_values(results.right_hand_landmarks, HAND_FEATURES),
        dtype=np.float32,
    )

    if features.shape != (FEATURES_PER_FRAME,):
        raise ValueError(
            f"Expected {FEATURES_PER_FRAME} features, got {features.shape}"
        )
    if not np.isfinite(features).all():
        raise ValueError("Landmark features contain NaN or Inf")
    return features


def validate_sequence(sequence):
    """Validate a model-ready sequence without modifying it."""

    array = np.asarray(sequence, dtype=np.float32)
    expected_shape = (SEQUENCE_LENGTH, FEATURES_PER_FRAME)
    if array.shape != expected_shape:
        raise ValueError(f"Expected sequence shape {expected_shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Sequence contains NaN or Inf")
    return array