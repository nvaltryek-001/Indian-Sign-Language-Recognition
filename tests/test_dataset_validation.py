from pathlib import Path

from app.core.dataset_validation import validate_sequence_directory


ROOT = Path(__file__).resolve().parents[1]


def test_active_sequences_are_model_compatible():
    summary = validate_sequence_directory(ROOT / "data" / "include50" / "sequences")
    assert summary["files"] == 918
    assert summary["shape"] == (45, 258)