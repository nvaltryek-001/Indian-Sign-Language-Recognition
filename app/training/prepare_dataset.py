from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# INCLUDE-50 DATASET PREPARATION
# ============================================================

BASE = Path("data/include50")
VIDEO_DIR = BASE / "videos"
SEQ_DIR = BASE / "sequences"
MAPPING = BASE / "video_archive_mapping.csv"

INDEX_OUT = BASE / "dataset_index.csv"
TRAIN_OUT = BASE / "train.csv"
VAL_OUT = BASE / "val.csv"
TEST_OUT = BASE / "test.csv"

RANDOM_STATE = 42

print("=" * 70)
print("INCLUDE-50 DATASET PREPARATION")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load mapping
# ------------------------------------------------------------

df = pd.read_csv(MAPPING)

print(f"Mapping rows: {len(df)}")

# ------------------------------------------------------------
# 2. Use folder name as ground-truth label
# ------------------------------------------------------------

df["folder_label"] = df["video_path"].apply(
    lambda x: Path(x).parent.name
)

# ------------------------------------------------------------
# 3. Remove duplicate video paths
# ------------------------------------------------------------

before = len(df)

df = df.drop_duplicates(
    subset=["video_path"],
    keep="first"
).copy()

print(f"Duplicate mapping rows removed: {before - len(df)}")
print(f"Unique videos: {len(df)}")

# ------------------------------------------------------------
# 4. Create sequence paths
# ------------------------------------------------------------

def make_sequence_path(video_path):
    return str(
        SEQ_DIR / Path(video_path).with_suffix(".npy")
    ).replace("\\", "/")

df["sequence_path"] = df["video_path"].apply(
    make_sequence_path
)

# ------------------------------------------------------------
# 5. Verify sequence files
# ------------------------------------------------------------

print("\nChecking sequence files...")

df["sequence_exists"] = df["sequence_path"].apply(
    lambda x: Path(x).exists()
)

missing = df[~df["sequence_exists"]]

print(f"Sequences found: {df['sequence_exists'].sum()}")
print(f"Sequences missing: {len(missing)}")

if len(missing) > 0:
    print("\nMissing sequences:")
    print(missing["sequence_path"].to_string(index=False))
    raise RuntimeError(
        "Some sequence files are missing."
    )

# ------------------------------------------------------------
# 6. Verify 50 classes
# ------------------------------------------------------------

classes = sorted(df["folder_label"].unique())

print(f"\nUnique classes: {len(classes)}")

if len(classes) != 50:
    raise RuntimeError(
        f"Expected 50 classes, found {len(classes)}."
    )

# ------------------------------------------------------------
# 7. Assign class IDs
# ------------------------------------------------------------

class_to_id = {
    label: idx
    for idx, label in enumerate(classes)
}

df["class_id"] = df["folder_label"].map(class_to_id)

# ------------------------------------------------------------
# 8. Final clean dataset
# ------------------------------------------------------------

dataset = df[
    [
        "video_path",
        "sequence_path",
        "folder_label",
        "class_id",
        "archive",
    ]
].copy()

# Save complete index
dataset.to_csv(INDEX_OUT, index=False)

print(f"\nDataset index saved:")
print(INDEX_OUT)

# ------------------------------------------------------------
# 9. Stratified train/val/test split
# ------------------------------------------------------------

from sklearn.model_selection import train_test_split

# First split:
# 70% train
# 30% temporary

train_df, temp_df = train_test_split(
    dataset,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=dataset["class_id"],
)

# Split temporary:
# 15% validation
# 15% test

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["class_id"],
)

# ------------------------------------------------------------
# 10. Save splits
# ------------------------------------------------------------

train_df = train_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

val_df = val_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

test_df = test_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

train_df.to_csv(TRAIN_OUT, index=False)
val_df.to_csv(VAL_OUT, index=False)
test_df.to_csv(TEST_OUT, index=False)

# ------------------------------------------------------------
# 11. Summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET PREPARATION COMPLETE")
print("=" * 70)

print(f"Total samples : {len(dataset)}")
print(f"Classes       : {len(classes)}")
print(f"Train         : {len(train_df)}")
print(f"Validation    : {len(val_df)}")
print(f"Test          : {len(test_df)}")

print("\nClass mapping:")

for class_id, label in enumerate(classes):
    print(f"{class_id:2d} -> {label}")

print("\nFiles created:")
print(f"  {INDEX_OUT}")
print(f"  {TRAIN_OUT}")
print(f"  {VAL_OUT}")
print(f"  {TEST_OUT}")

print("\nSTATUS: READY FOR LSTM TRAINING")