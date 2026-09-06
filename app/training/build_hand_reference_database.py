"""Build the INCLUDE-50 reference database from verified hand sequences."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("data/include50")
SOURCE_DIR = BASE / "hand_sequences"
REFERENCE_DIR = BASE / "reference_hand_sequences"
INDEX = BASE / "dataset_index.csv"


def main():
    dataset = pd.read_csv(INDEX)
    failures = []
    copied = 0
    for row in dataset.itertuples(index=False):
        relative = Path(row.video_path).with_suffix(".npy")
        source = SOURCE_DIR / relative
        target = REFERENCE_DIR / relative
        try:
            sequence = np.load(source)
            if sequence.shape != (45, 126) or sequence.dtype != np.float32:
                raise ValueError(f"invalid sequence {sequence.shape} {sequence.dtype}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied += 1
        except Exception as error:
            failures.append((str(relative), str(error)))

    error_log = BASE / "hand_reference_database_errors.csv"
    error_log.write_text("video,error\n" + "\n".join(f"{video},{error}" for video, error in failures), encoding="utf-8")
    print(f"Reference sequences copied: {copied}")
    print(f"Failures: {len(failures)}")
    print(f"Error log: {error_log}")
    if failures:
        raise RuntimeError(f"Reference database has {len(failures)} failures")


if __name__ == "__main__":
    main()