# Final Model Evaluation

Date: 2026-09-06

## Production model

- Model: `models/include50_lstm_v2_best.keras`
- Class map: `models/include50_classes_v2.json`
- Input: `(45, 258)`
- Output classes: 50
- Test samples: 124
- Test split coverage: 49 of 50 classes

## Overall metrics

| Metric | Result |
| --- | ---: |
| Top-1 accuracy | 91.13% |
| Top-3 accuracy | 98.39% |
| Macro precision | 0.887 |
| Macro recall | 0.887 |
| Macro F1 | 0.872 |
| Weighted F1 | 0.899 |

## Per-class metrics

The complete per-class table is generated at `models/evaluation/classification_report.csv`. It contains precision, recall, F1, and support for all 50 class-map entries. The test split contains no `34. Pen` examples, so that class is not measurable in this split.

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| 1. Dog | 1.00 | 0.67 | 0.80 | 3 |
| 1. loud | 1.00 | 1.00 | 1.00 | 3 |
| 11. Car | 0.75 | 1.00 | 0.86 | 3 |
| 2. Death | 1.00 | 0.50 | 0.67 | 2 |
| 2. quiet | 1.00 | 0.67 | 0.80 | 3 |
| 3. happy | 1.00 | 1.00 | 1.00 | 2 |
| 40. Paint | 1.00 | 0.50 | 0.67 | 2 |
| 42. T-Shirt | 0.00 | 0.00 | 0.00 | 2 |
| 44. Shoes | 0.60 | 1.00 | 0.75 | 3 |
| 48. Hello | 1.00 | 0.67 | 0.80 | 3 |
| 55. White | 1.00 | 0.33 | 0.50 | 3 |
| 67. Monday | 0.67 | 1.00 | 0.80 | 2 |
| 77. Boy | 0.00 | 0.00 | 0.00 | 2 |
| 78. Girl | 0.50 | 1.00 | 0.67 | 2 |
| 91. new | 0.75 | 1.00 | 0.86 | 3 |
| Remaining 35 classes | See CSV artifact | See CSV artifact | See CSV artifact | See CSV artifact |

## Confusion findings

The largest errors were `77. Boy -> 78. Girl` (2 samples) and `42. T-Shirt -> 44. Shoes` (2 samples). Other observed confusions include `55. White -> 47. Red`, `55. White -> 34. Pen`, `48. Hello -> 51. Good Morning`, `40. Paint -> 67. Monday`, `2. quiet -> 91. new`, `2. Death -> 86. Time`, and `1. Dog -> 11. Car`.

Artifacts:

- `models/evaluation/evaluation_summary.json`
- `models/evaluation/classification_report.csv`
- `models/evaluation/confusion_matrix.npy`
- `models/evaluation/test_predictions.csv`

## Decision

The V2 baseline is retained. No retraining was promoted. The stored webcam artifact contains only six `1. Dog` trials with 0 correct predictions, which indicates domain shift but is insufficient evidence for one-class adaptation or a model replacement. Representative human trials for `loud`, `quiet`, `happy`, `good`, and `new` remain required before reporting webcam accuracy.
