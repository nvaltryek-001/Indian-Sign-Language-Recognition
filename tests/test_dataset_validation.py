from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = ROOT / "data" / "include50" / "reference_hand_sequences"


def test_reference_hand_sequences_are_present_and_model_compatible():
    assert REFERENCE_ROOT.is_dir(), f"Missing reference directory: {REFERENCE_ROOT}"

    files = sorted(REFERENCE_ROOT.rglob("*.npy"))

    assert len(files) >= 50, (
        f"Expected reference hand sequences for INCLUDE-50, found {len(files)}"
    )

    checked = 0

    for path in files:
        array = np.load(path, allow_pickle=False)

        assert array.shape == (45, 126), (
            f"{path} has shape {array.shape}; expected (45, 126)"
        )

        assert array.dtype == np.float32, (
            f"{path} has dtype {array.dtype}; expected float32"
        )

        assert np.isfinite(array).all(), (
            f"{path} contains NaN or Inf"
        )

        checked += 1

        # Enough coverage for CI without unnecessarily loading every file.
        if checked >= min(25, len(files)):
            break
