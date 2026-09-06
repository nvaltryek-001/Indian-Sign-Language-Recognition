from types import SimpleNamespace

import numpy as np

from app.core.feature_pipeline import (
    CaptureQualityGate,
    FEATURES_PER_FRAME,
    assess_landmark_quality,
    extract_landmarks,
    validate_sequence,
)


def _landmarks(count):
    return SimpleNamespace(
        landmark=[
            SimpleNamespace(x=0.1, y=0.2, z=0.3, visibility=0.9)
            for _ in range(count)
        ]
    )


def _results(left=True, right=True):
    return SimpleNamespace(
        pose_landmarks=_landmarks(33),
        left_hand_landmarks=_landmarks(21) if left else None,
        right_hand_landmarks=_landmarks(21) if right else None,
    )


def test_extractor_matches_raw_training_contract():
    features = extract_landmarks(_results())
    assert features.shape == (FEATURES_PER_FRAME,)
    assert features.dtype == np.float32
    assert np.allclose(features[:4], [0.1, 0.2, 0.3, 0.9])
    assert assess_landmark_quality(_results()).all_required_present


def test_missing_hand_is_measured_and_zero_filled():
    results = _results(left=False)
    features = extract_landmarks(results)
    quality = assess_landmark_quality(results)
    assert not quality.all_required_present
    assert np.count_nonzero(features[132:195]) == 0


def test_sequence_validator_rejects_wrong_shape():
    with np.testing.assert_raises(ValueError):
        validate_sequence(np.zeros((44, FEATURES_PER_FRAME), dtype=np.float32))


def test_quality_gate_requires_consecutive_complete_frames_and_recovers():
    gate = CaptureQualityGate(warmup_frames=2)
    complete = assess_landmark_quality(_results())
    incomplete = assess_landmark_quality(_results(right=False))
    assert gate.update(complete) == (False, "Warming up 1/2", False)
    assert gate.update(complete) == (True, "Ready", False)
    assert gate.update(incomplete) == (False, "Show both hands and pose", True)
    assert gate.update(complete) == (False, "Warming up 1/2", False)