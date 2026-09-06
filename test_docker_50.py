import numpy as np
import requests
from pathlib import Path

API = "http://127.0.0.1:8000/predict/hand-sequence"
ROOT = Path("data/include50/hand_sequences")

files = list(ROOT.rglob("*.npy"))

groups = {}

for p in files:
    label = p.parent.name
    if label not in groups:
        groups[label] = p

print("FOUND CLASSES:", len(groups))
print()

results = []

for i, (label, path) in enumerate(groups.items(), 1):
    print(f"Testing {i:02d}/50: {label}")

    try:
        sequence = np.load(path).astype(float)

        response = requests.post(
            API,
            json={
                "hand_sequence": sequence.tolist(),
                "expected_sign": label
            },
            timeout=120
        )

        data = response.json()

        predicted = data.get("prediction")
        confidence = data.get("confidence", 0) * 100

        reference = data.get("reference_match", {})
        dtw = reference.get("match_percent")

        correct = predicted == label

        results.append({
            "expected": label,
            "predicted": predicted,
            "confidence": confidence,
            "dtw": dtw,
            "correct": correct
        })

    except Exception as e:
        print("ERROR:", e)
        results.append({
            "expected": label,
            "predicted": "ERROR",
            "confidence": 0,
            "dtw": None,
            "correct": False
        })

print()
print("=" * 100)
print("50-CLASS DOCKER TEST RESULTS")
print("=" * 100)

for i, r in enumerate(results, 1):
    status = "PASS" if r["correct"] else "FAIL"

    print(
        f"{i:02d}. "
        f"{r['expected']:<25} -> "
        f"{str(r['predicted']):<25} | "
        f"LSTM: {r['confidence']:6.2f}% | "
        f"DTW: {str(r['dtw']):>6}% | "
        f"{status}"
    )

correct_count = sum(r["correct"] for r in results)

print()
print("=" * 100)
print(f"TOTAL: {correct_count}/{len(results)} CORRECT")
print(f"ACCURACY: {correct_count / len(results) * 100:.2f}%")
print("=" * 100)