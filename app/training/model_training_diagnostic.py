import json
import numpy as np
import glob
import os
from collections import Counter
from tensorflow.keras.models import load_model

MODEL = "models/include50_lstm_v2_best.keras"
CLASSES = "models/include50_classes_v2.json"
DATA = "data/include50/sequences"

model = load_model(MODEL)

with open(CLASSES, encoding="utf-8") as f:
    d = json.load(f)

names = [d[str(i)] for i in range(50)]

files = glob.glob(
    os.path.join(DATA, "**", "*.npy"),
    recursive=True
)

print("=" * 70)
print("MODEL TRAINING-DATA DIAGNOSTIC")
print("=" * 70)
print("Model classes:", len(names))
print("Sequence files:", len(files))
print()

results = []

for f in files:

    x = np.load(f).astype("float32")

    if x.shape != (45, 258):
        continue

    p = model.predict(
        x[None, ...],
        verbose=0
    )[0]

    pred = int(np.argmax(p))

    results.append(
        (
            os.path.relpath(f, DATA),
            pred,
            float(p[pred])
        )
    )

print("Predictions completed:", len(results))
print()

print("FIRST 30 MODEL PREDICTIONS")
print("-" * 70)

for path, pred, confidence in results[:30]:

    print(
        f"{pred:02d} -> "
        f"{names[pred]:25s} "
        f"{confidence:.4f} "
        f"{path}"
    )

print()
print("PREDICTION DISTRIBUTION")
print("-" * 70)

counter = Counter(
    names[pred]
    for _, pred, _ in results
)

for name, count in counter.most_common(20):

    print(
        f"{name:25s} -> {count}"
    )

print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
