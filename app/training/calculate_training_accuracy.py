import json
import numpy as np
import glob
import os
from tensorflow.keras.models import load_model

MODEL = "models/include50_lstm_v2_best.keras"
CLASSES = "models/include50_classes_v2.json"
DATA = "data/include50/sequences"

model = load_model(MODEL)

with open(CLASSES, encoding="utf-8") as f:
    d = json.load(f)

names = [d[str(i)] for i in range(50)]

name_to_index = {
    name: i for i, name in enumerate(names)
}

files = glob.glob(
    os.path.join(DATA, "**", "*.npy"),
    recursive=True
)

correct = 0
total = 0

print("=" * 70)
print("TRAINING DATA ACCURACY TEST")
print("=" * 70)

for f in files:

    x = np.load(f).astype("float32")

    if x.shape != (45, 258):
        continue

    folder = os.path.basename(os.path.dirname(f))

    if folder not in name_to_index:
        continue

    true_label = name_to_index[folder]

    p = model.predict(
        x[None, ...],
        verbose=0
    )[0]

    predicted = int(np.argmax(p))

    if predicted == true_label:
        correct += 1

    total += 1

accuracy = correct / total if total else 0

print()
print("Total tested :", total)
print("Correct      :", correct)
print("Wrong        :", total - correct)
print(f"Accuracy     : {accuracy * 100:.2f}%")
print()

print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
