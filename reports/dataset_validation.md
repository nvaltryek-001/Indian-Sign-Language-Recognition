# Dataset Validation Report

Date: 2026-09-05

## Result

- Active extracted sequences: 918
- Active sequence shape: `(45, 258)`
- Non-finite or malformed active sequences: 0
- Dataset index rows: 943
- Missing sequence references in the full index: 25
- Missing references in train/validation/test splits: 0
- Train/validation overlap: 0
- Train/test overlap: 0
- Validation/test overlap: 0
- Dataset classes: 50
- V2 class-map entries: 50, contiguous indices `0..49`
- Class-label differences between dataset and V2 map: none
- Quarantine directory: `data/include50/wrong_backup`

The 25 missing full-index references are quarantined historical/wrong samples and are excluded from the active train, validation, and test CSVs. No data was deleted or modified during this release verification.

The test split represents 49 of 50 classes; this is recorded as a distribution characteristic and was not changed automatically.
