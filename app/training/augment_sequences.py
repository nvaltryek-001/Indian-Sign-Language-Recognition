from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# INCLUDE-50 V3 TARGETED LANDMARK AUGMENTATION
# ============================================================

BASE = Path("data/include50")

TRAIN_CSV = BASE / "train.csv"
OUTPUT_DIR = BASE / "v3_augmented"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = [
    "34. Pen",
    "78. Girl",
    "84. small little",
]

AUGMENTATIONS_PER_SEQUENCE = 4

SEED = 42
rng = np.random.default_rng(SEED)

SEQUENCE_LENGTH = 45
FEATURES = 258


def augment_sequence(x, mode):
    """
    Controlled augmentation of 45 x 258 landmark sequence.

    We modify only training data.
    Validation and test data are never touched.
    """

    y = x.copy()

    # --------------------------------------------------------
    # 1. Small coordinate noise
    # --------------------------------------------------------

    if mode == 0:
        noise = rng.normal(
            0.0,
            0.003,
            size=y.shape
        ).astype(np.float32)

        y = y + noise

    # --------------------------------------------------------
    # 2. Slight temporal variation
    # --------------------------------------------------------

    elif mode == 1:

        shift = rng.integers(-2, 3)

        y = np.roll(
            y,
            shift,
            axis=0
        )

        noise = rng.normal(
            0.0,
            0.002,
            size=y.shape
        ).astype(np.float32)

        y = y + noise

    # --------------------------------------------------------
    # 3. Mild feature scaling
    # --------------------------------------------------------

    elif mode == 2:

        scale = rng.uniform(
            0.985,
            1.015
        )

        y = y * scale

        noise = rng.normal(
            0.0,
            0.002,
            size=y.shape
        ).astype(np.float32)

        y = y + noise

    # --------------------------------------------------------
    # 4. Combined mild augmentation
    # --------------------------------------------------------

    elif mode == 3:

        scale = rng.uniform(
            0.99,
            1.01
        )

        y = y * scale

        shift = rng.integers(
            -1,
            2
        )

        y = np.roll(
            y,
            shift,
            axis=0
        )

        noise = rng.normal(
            0.0,
            0.003,
            size=y.shape
        ).astype(np.float32)

        y = y + noise

    return y.astype(np.float32)


# ============================================================
# LOAD TRAINING CSV
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)

target_df = train_df[
    train_df["folder_label"].isin(TARGET_CLASSES)
].copy()

print("=" * 70)
print("INCLUDE-50 V3 TARGETED AUGMENTATION")
print("=" * 70)

print("\nOriginal target training samples:")

print(
    target_df["folder_label"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# GENERATE AUGMENTED SEQUENCES
# ============================================================

created = 0

for row in target_df.itertuples(index=False):

    source = Path(row.sequence_path)

    if not source.exists():
        print(
            f"WARNING: Missing sequence: {source}"
        )
        continue

    data = np.load(
        source
    ).astype(np.float32)

    if data.shape != (
        SEQUENCE_LENGTH,
        FEATURES
    ):
        raise RuntimeError(
            f"Wrong shape {data.shape}: {source}"
        )

    class_name = row.folder_label

    safe_class = (
        class_name
        .replace(" ", "_")
        .replace(".", "")
    )

    stem = source.stem

    for aug_id in range(
        AUGMENTATIONS_PER_SEQUENCE
    ):

        augmented = augment_sequence(
            data,
            aug_id
        )

        output_name = (
            f"{safe_class}__"
            f"{stem}__aug{aug_id + 1}.npy"
        )

        output_path = (
            OUTPUT_DIR /
            output_name
        )

        np.save(
            output_path,
            augmented
        )

        created += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("AUGMENTATION COMPLETE")
print("=" * 70)

print(
    f"Original target sequences : "
    f"{len(target_df)}"
)

print(
    f"Augmentations per sequence : "
    f"{AUGMENTATIONS_PER_SEQUENCE}"
)

print(
    f"Augmented sequences created: "
    f"{created}"
)

print(
    f"Output directory           : "
    f"{OUTPUT_DIR}"
)

print("\nTarget classes:")

for c in TARGET_CLASSES:
    count = len(
        list(
            OUTPUT_DIR.glob(
                c.replace(" ", "_")
                 .replace(".", "")
                + "__*.npy"
            )
        )
    )

    print(
        f"  {c}: {count}"
    )

print("\nSTATUS: V3 AUGMENTATION READY")