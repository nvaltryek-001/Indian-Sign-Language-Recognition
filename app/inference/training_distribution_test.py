import numpy as np

STATS = "models/include50_training_stats.npz"
WEBCAM = "data/include50/canonical_test.npy"

s = np.load(STATS)

mean = s["mean"]
std = s["std"]

x = np.load(WEBCAM).astype(np.float32)

print("=" * 80)
print("WEBCAM vs TRAINING DISTRIBUTION")
print("=" * 80)

print()
print("WEBCAM")
print("Mean :", float(x.mean()))
print("Std  :", float(x.std()))
print("Min  :", float(x.min()))
print("Max  :", float(x.max()))

print()
print("TRAINING")
print("Mean :", float(mean.mean()))
print("Std  :", float(std.mean()))
print("Min  :", float(mean.min()))
print("Max  :", float(mean.max()))

# Standardize webcam feature-by-feature using training distribution
z = (x - mean[None, :]) / np.maximum(std[None, :], 1e-6)

print()
print("AFTER TRAINING-DISTRIBUTION NORMALIZATION")
print("Mean :", float(z.mean()))
print("Std  :", float(z.std()))
print("Min  :", float(z.min()))
print("Max  :", float(z.max()))

# Robust clipping prevents extreme webcam landmarks from dominating
zc = np.clip(z, -3.0, 3.0)

print()
print("AFTER CLIPPING")
print("Mean :", float(zc.mean()))
print("Std  :", float(zc.std()))
print("Min  :", float(zc.min()))
print("Max  :", float(zc.max()))

np.save(
    "data/include50/canonical_test_normalized.npy",
    zc.astype(np.float32)
)

print()
print("Saved:")
print("data/include50/canonical_test_normalized.npy")

print("=" * 80)
