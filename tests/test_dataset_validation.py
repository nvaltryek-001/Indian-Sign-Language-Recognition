from pathlib import Path

import pytest

from app.core.dataset_validation import validate_sequence_directory


ROOT = Path(__file__).resolve().parents[1]


def test_active_sequences_are_model_compatible():
    sequence_dir = ROOT / "data" / "include50" / "sequences"

    # The training sequence dataset is intentionally excluded from Git.
    # Run this validation when the local dataset is available.
    if not sequence_dir.exists():
        pytest.skip("Local INCLUDE-50 sequence dataset is not included in CI")

    summary = validate_sequence_directory(sequence_dir)

    assert summary["files"] == 918
    assert summary["shape"] == (45, 258)
