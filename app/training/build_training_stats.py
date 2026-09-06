import os
import glob
import json
import numpy as np

ROOT = "data/include50/sequences"
OUT = "models/include50_training_stats.npz"

files = glob.glob(ROOT + "/**/*.npy", recursive=True)

print("=" * 80)
print("BUILDING TRAINING-DISTRIBUTION STATISTICS")
print("=" * 80)
print("Files:", len(files))

all_mean_sum = np.zeros(258, dtype=np.float64)
all_sq_sum = np.zeros(258, dtype=np.float64)
total_frames = 0

global_min = np.inf
global_max = -np.inf

valid = 0

for n, f in enumerate(files, 1):

    try:
        x = np.load(f).astype(np.float32)

        if x.shape != (45, 258):
            continue

        if not np.isfinite(x).all():
            continue

        xf = x.reshape(-1, 258).astype(np.float64)

        all_mean_sum += xf.sum(axis=0)
        all_sq_sum += (xf * xf).sum(axis=0)

        total_frames += xf.shape[0]

        global_min = min(global_min, float(x.min()))
        global_max = max(global_max, float(x.max()))

        valid += 1

    except Exception:
        pass

mean = all_mean_sum / total_frames

variance = (all_sq_sum / total_frames) - (mean * mean)

variance = np.maximum(variance, 1e-8)

std = np.sqrt(variance)

np.savez(
    OUT,
    mean=mean.astype(np.float32),
    std=std.astype(np.float32),
    global_min=np.float32(global_min),
    global_max=np.float32(global_max),
    total_frames=np.int64(total_frames),
    valid_files=np.int64(valid)
)

print()
print("=" * 80)
print("TRAINING STATISTICS CREATED")
print("=" * 80)

print("Valid files :", valid)
print("Frames      :", total_frames)

print()
print("Feature statistics:")
print("Mean of means :", float(mean.mean()))
print("Mean of stds  :", float(std.mean()))
print("Min feature   :", float(mean.min()))
print("Max feature   :", float(mean.max()))

print()
print("Dataset range:")
print("Min :", global_min)
print("Max :", global_max)

print()
print("Saved:")
print(OUT)

print("=" * 80)
