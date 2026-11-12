"""Render README charts for the active hand-focused LSTM from its saved evaluation artefacts.

Inputs : models/evaluation/hand_model/{confusion_matrix.npy, summary.json}
Outputs: docs/evaluation/hand_model_confusion_matrix.png, docs/evaluation/hand_model_per_sign_recall.png
Usage  : python scripts/plot_hand_model_eval.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "models" / "evaluation" / "hand_model"
OUT = ROOT / "docs" / "evaluation"
OUT.mkdir(parents=True, exist_ok=True)

cm = np.load(EVAL / "confusion_matrix.npy")
summary = json.loads((EVAL / "summary.json").read_text())
labels = [c["label"].split(". ", 1)[-1] for c in summary["per_class"]]
acc = np.trace(cm) / cm.sum()

# Confusion matrix (row-normalised so each true sign sums to 1)
row = cm.sum(axis=1, keepdims=True)
norm = np.divide(cm, row, out=np.zeros_like(cm, dtype=float), where=row > 0)
fig, ax = plt.subplots(figsize=(11, 10))
im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
ax.set_yticks(range(len(labels)), labels, fontsize=7)
ax.set_xlabel("Predicted sign"); ax.set_ylabel("True sign")
ax.set_title(f"Hand-LSTM confusion matrix — INCLUDE-50 test split\n{int(np.trace(cm))}/{int(cm.sum())} correct ({acc:.2%}); off-diagonal cells are mistakes", fontsize=11)
for i, j in zip(*np.nonzero((cm > 0) & ~np.eye(len(cm), dtype=bool))):
    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#dc2626", lw=1.2))
fig.colorbar(im, ax=ax, fraction=0.035, label="Share of true sign")
fig.tight_layout(); fig.savefig(OUT / "hand_model_confusion_matrix.png", dpi=110); plt.close(fig)

# Per-sign recall, sorted, with test support annotated
pc = [c for c in summary["per_class"] if c["support"] > 0]
pc.sort(key=lambda c: (c["recall"], c["label"]))
fig, ax = plt.subplots(figsize=(8, 11))
names = [c["label"].split(". ", 1)[-1] for c in pc]
vals = [c["recall"] for c in pc]
ax.barh(names, vals, color=["#f97316" if v < 1 else "#2563eb" for v in vals])
for y, c in enumerate(pc):
    ax.text(c["recall"] + .01, y, f'n={c["support"]}', va="center", fontsize=7, color="#475569")
ax.set_xlim(0, 1.12); ax.tick_params(axis="y", labelsize=7)
ax.set_xlabel("Recall on test clips")
ax.set_title(f"Per-sign recall ({len(pc)} signs with test clips; orange = at least one miss)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "hand_model_per_sign_recall.png", dpi=110); plt.close(fig)
print(f"accuracy {acc:.4f}; charts in {OUT}")
