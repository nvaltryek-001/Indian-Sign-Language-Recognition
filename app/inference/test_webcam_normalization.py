import numpy as np
import json
from tensorflow.keras.models import load_model

MODEL = "models/include50_lstm_v2_best.keras"
CLASSES = "models/include50_classes_v2.json"
WEBCAM = "data/include50/debug_webcam.npy"

model = load_model(MODEL)

with open(CLASSES, encoding="utf-8") as f:
    d = json.load(f)

names = [d[str(i)] for i in range(50)]

x = np.load(WEBCAM).astype(np.float32)

print("=" * 70)
print("WEBCAM NORMALIZATION DIAGNOSTIC")
print("=" * 70)

print("Original:")
print("  Mean :", float(x.mean()))
print("  Std  :", float(x.std()))
print("  Min  :", float(x.min()))
print("  Max  :", float(x.max()))

# ------------------------------------------------------------
# TEST 1: Original
# ------------------------------------------------------------

p1 = model.predict(x[None, ...], verbose=0)[0]
i1 = int(np.argmax(p1))

print()
print("TEST 1 - ORIGINAL")
print("Prediction :", names[i1])
print("Confidence :", f"{p1[i1]:.4f}")

# ------------------------------------------------------------
# TEST 2: Per-sequence standardization
# ------------------------------------------------------------

mean = x.mean(axis=0, keepdims=True)
std = x.std(axis=0, keepdims=True)

std = np.where(std < 1e-6, 1.0, std)

x2 = (x - mean) / std

p2 = model.predict(x2[None, ...], verbose=0)[0]
i2 = int(np.argmax(p2))

print()
print("TEST 2 - PER FEATURE STANDARDIZATION")
print("Prediction :", names[i2])
print("Confidence :", f"{p2[i2]:.4f}")

# ------------------------------------------------------------
# TEST 3: Global standardization
# ------------------------------------------------------------

x3 = (x - x.mean()) / max(float(x.std()), 1e-6)

p3 = model.predict(x3[None, ...], verbose=0)[0]
i3 = int(np.argmax(p3))

print()
print("TEST 3 - GLOBAL STANDARDIZATION")
print("Prediction :", names[i3])
print("Confidence :", f"{p3[i3]:.4f}")

print()
print("=" * 70)
print("TOP 5 - ORIGINAL")
print("=" * 70)

for i in np.argsort(p1)[::-1][:5]:
    print(f"{names[i]:25s} : {p1[i]:.4f}")

print()
print("=" * 70)
print("TOP 5 - FEATURE STANDARDIZATION")
print("=" * 70)

for i in np.argsort(p2)[::-1][:5]:
    print(f"{names[i]:25s} : {p2[i]:.4f}")

print()
print("=" * 70)
print("TOP 5 - GLOBAL STANDARDIZATION")
print("=" * 70)

for i in np.argsort(p3)[::-1][:5]:
    print(f"{names[i]:25s} : {p3[i]:.4f}")

print("=" * 70)
