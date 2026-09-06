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
name_to_index = {name: i for i, name in enumerate(names)}

files = glob.glob(
    os.path.join(DATA, "**", "*.npy"),
    recursive=True
)

wrong = []

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

    if predicted != true_label:
        wrong.append(
            (
                f,
                folder,
                names[predicted],
                float(p[predicted])
            )
        )

print("=" * 80)
print("WRONG TRAINING FILES")
print("=" * 80)

for i, (path, actual, predicted, confidence) in enumerate(wrong, 1):
    print(
        f"{i:02d}. ACTUAL={actual:25s} "
        f"PREDICTED={predicted:25s} "
        f"CONF={confidence:.4f}"
    )
    print(f"    {path}")

print()
print("TOTAL WRONG:", len(wrong))

with open(
    "data/include50/wrong_files.txt",
    "w",
    encoding="utf-8"
) as f:
    for path, actual, predicted, confidence in wrong:
        f.write(path + "\n")

print()
print("Saved list:")
print("data/include50/wrong_files.txt")
