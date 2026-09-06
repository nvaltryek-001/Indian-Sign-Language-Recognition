import os
import glob
import json
import numpy as np

ROOT = "data/include50/sequences"
CLASSES = "models/include50_classes_v2.json"

with open(CLASSES, encoding="utf-8") as f:
    d = json.load(f)

names = [d[str(i)] for i in range(50)]

print("=" * 80)
print("INCLUDE50 DATASET AUDIT")
print("=" * 80)

total = 0
bad = 0

for i, name in enumerate(names):
    folder = os.path.join(ROOT, "**", name)
    files = glob.glob(folder + "/*.npy", recursive=True)

    shapes = {}
    valid = 0

    for f in files:
        try:
            x = np.load(f)
            shapes[str(x.shape)] = shapes.get(str(x.shape), 0) + 1

            if x.shape == (45, 258) and np.isfinite(x).all():
                valid += 1
            else:
                bad += 1

        except Exception:
            bad += 1

    total += len(files)

    print(
        f"{i:02d} | {name:25s} | "
        f"files={len(files):4d} | "
        f"valid={valid:4d} | "
        f"shapes={shapes}"
    )

print()
print("=" * 80)
print("TOTAL FILES :", total)
print("BAD FILES   :", bad)
print("=" * 80)
