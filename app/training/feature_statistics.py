import os
import glob
import numpy as np

ROOT = "data/include50/sequences"

files = glob.glob(ROOT + "/**/*.npy", recursive=True)

# Collect statistics for every feature.
sum_x = np.zeros(258, dtype=np.float64)
sum_x2 = np.zeros(258, dtype=np.float64)

minimum = np.full(258, np.inf)
maximum = np.full(258, -np.inf)

count = 0

for f in files:
    x = np.load(f).astype(np.float64)

    if x.shape != (45, 258):
        continue

    x = x.reshape(-1, 258)

    sum_x += x.sum(axis=0)
    sum_x2 += (x * x).sum(axis=0)

    minimum = np.minimum(minimum, x.min(axis=0))
    maximum = np.maximum(maximum, x.max(axis=0))

    count += x.shape[0]

mean = sum_x / count
var = (sum_x2 / count) - mean * mean
var = np.maximum(var, 0)

std = np.sqrt(var)

print("=" * 90)
print("FEATURE-BY-FEATURE TRAINING STATISTICS")
print("=" * 90)

print("Total frames:", count)

print()
print("FEATURE | MEAN | STD | MIN | MAX")

for i in range(258):
    print(
        f"{i:03d} | "
        f"{mean[i]: .6f} | "
        f"{std[i]: .6f} | "
        f"{minimum[i]: .6f} | "
        f"{maximum[i]: .6f}"
    )

np.savez(
    "models/include50_feature_statistics.npz",
    mean=mean.astype(np.float32),
    std=std.astype(np.float32),
    minimum=minimum.astype(np.float32),
    maximum=maximum.astype(np.float32)
)

print()
print("=" * 90)
print("SAVED: models/include50_feature_statistics.npz")
print("=" * 90)
