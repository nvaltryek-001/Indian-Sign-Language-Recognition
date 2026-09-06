import numpy as np
import json
from tensorflow.keras.models import load_model

MODEL = "models/include50_lstm_v2_best.keras"
CLASSES = "models/include50_classes_v2.json"
XFILE = "data/include50/canonical_test.npy"

model = load_model(MODEL)

with open(CLASSES, encoding="utf-8") as f:
    d = json.load(f)

names = [d[str(i)] for i in range(50)]

x = np.load(XFILE).astype("float32")

print("=" * 75)
print("CANONICAL EXTRACTION -> MODEL DIAGNOSTIC")
print("=" * 75)

print("Input shape :", x.shape)
print("Mean        :", float(x.mean()))
print("Std         :", float(x.std()))
print("Min         :", float(x.min()))
print("Max         :", float(x.max()))
print("Finite      :", bool(np.isfinite(x).all()))

# ------------------------------------------------------------
# TEST 1 - RAW CANONICAL
# ------------------------------------------------------------

p1 = model.predict(x[None, ...], verbose=0)[0]

print()
print("=" * 75)
print("TEST 1 - RAW CANONICAL")
print("=" * 75)

for i in np.argsort(p1)[::-1][:10]:
    print(f"{names[i]:25s} : {p1[i]:.6f}")

# ------------------------------------------------------------
# TEST 2 - GLOBAL SCALE TO TRAINING STATISTICS
# ------------------------------------------------------------

TRAIN_MEAN = 0.36022168
TRAIN_STD = 0.41790426

x2 = (x - float(x.mean())) / max(float(x.std()), 1e-6)
x2 = x2 * TRAIN_STD + TRAIN_MEAN

p2 = model.predict(x2[None, ...], verbose=0)[0]

print()
print("=" * 75)
print("TEST 2 - GLOBAL DISTRIBUTION MATCH")
print("=" * 75)

print("New mean :", float(x2.mean()))
print("New std  :", float(x2.std()))

for i in np.argsort(p2)[::-1][:10]:
    print(f"{names[i]:25s} : {p2[i]:.6f}")

# ------------------------------------------------------------
# TEST 3 - CLIP ONLY
# ------------------------------------------------------------

x3 = np.clip(x, -1.5, 1.5)

p3 = model.predict(x3[None, ...], verbose=0)[0]

print()
print("=" * 75)
print("TEST 3 - CLIPPED CANONICAL")
print("=" * 75)

for i in np.argsort(p3)[::-1][:10]:
    print(f"{names[i]:25s} : {p3[i]:.6f}")

# ------------------------------------------------------------
# TEST 4 - TRAINING SAMPLE REFERENCE
# ------------------------------------------------------------

import glob

files = glob.glob(
    "data/include50/sequences/**/*.npy",
    recursive=True
)

loud_files = [
    f for f in files
    if "1. loud" in f
]

print()
print("=" * 75)
print("REFERENCE TRAINING LOUD")
print("=" * 75)

if loud_files:

    train = np.load(loud_files[0]).astype("float32")

    pt = model.predict(train[None, ...], verbose=0)[0]

    print("Training file :", loud_files[0])
    print("Mean          :", float(train.mean()))
    print("Std           :", float(train.std()))
    print("Min           :", float(train.min()))
    print("Max           :", float(train.max()))

    print()
    print("Training prediction:")

    for i in np.argsort(pt)[::-1][:5]:
        print(f"{names[i]:25s} : {pt[i]:.6f}")

print()
print("=" * 75)
print("DIAGNOSTIC COMPLETE")
print("=" * 75)
