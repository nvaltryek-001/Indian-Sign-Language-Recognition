import pandas as pd
from pathlib import Path

files = [
    Path("data/include50/train_augmented.csv"),
    Path("data/include50/val.csv"),
    Path("data/include50/test.csv")
]

for csv_file in files:

    df = pd.read_csv(csv_file)

    before = len(df)

    exists = df["sequence_path"].apply(
        lambda p: Path(p).exists()
    )

    removed = df.loc[~exists, "sequence_path"].tolist()

    df = df.loc[exists].copy()

    df.to_csv(
        csv_file,
        index=False
    )

    print("=" * 70)
    print("FILE:", csv_file)
    print("Before :", before)
    print("Removed:", len(removed))
    print("After  :", len(df))

    for path in removed:
        print("  REMOVED:", path)

print()
print("=" * 70)
print("CSV CLEANUP COMPLETE")
print("=" * 70)
