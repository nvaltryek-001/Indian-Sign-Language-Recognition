import glob
import json
import numpy as np
from tensorflow.keras.models import load_model

MODEL = "models/include50_lstm_v2_best.keras"
CLASSES = "models/include50_classes_v2.json"
DATA_DIR = "data/include50/webcam_sequences/1. loud"

model = load_model(MODEL)

with open(CLASSES, encoding="utf-8") as f:
    d = json.load(f)

names = [d[str(i)] for i in range(50)]

files = sorted(glob.glob(DATA_DIR + "/*.npy"))

correct = 0

print("=" * 70)
print("WEBCAM LOUD - CURRENT MODEL TEST")
print("=" * 70)
print("Samples:", len(files))
print()

for i, f in enumerate(files, start=1):

    x = np.load(f).astype(np.float32)

    prediction = model.predict(
        x[None, ...],
        verbose=0
    )[0]

    index = int(np.argmax(prediction))
    label = names[index]
    confidence = float(prediction[index])

    if label == "1. loud":
        correct += 1
        status = "CORRECT"
    else:
        status = "WRONG"

    print(
        f"{i:02d}. {label:25s} "
        f"{confidence:.4f}  {status}"
    )

print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print("LOUD CORRECT:", correct, "/", len(files))

if files:
    print(
        "ACCURACY:",
        f"{100 * correct / len(files):.2f}%"
    )

print("=" * 70)
