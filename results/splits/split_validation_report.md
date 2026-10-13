# Master Split Validation Report

**Overall status: PASS**

## Counts

- Total images: 6000
- Total groups: 5985
- Train: 4200
- Validation: 900
- Test: 900

## Checks

- **required_columns**: PASS — All required columns present
- **duplicate_image_paths**: PASS — 0 duplicate image_path rows
- **invalid_labels**: PASS — Labels restricted to expected classes
- **invalid_splits**: PASS — Splits restricted to train/validation/test
- **group_leakage**: PASS — 0 groups appear in multiple splits
- **exact_duplicate_leakage**: PASS — 0 MD5 hashes appear in multiple splits
- **train_test_overlap**: PASS — 0 overlapping image paths
- **train_validation_overlap**: PASS — 0 overlapping image paths
- **validation_test_overlap**: PASS — 0 overlapping image paths
- **missing_classes_overall**: PASS — All classes present in dataset index
- **missing_classes_per_split**: PASS — All classes appear in every split
- **split_proportions**: PASS — train=0.700, validation=0.150, test=0.150 (targets 0.70/0.15/0.15; group constraints apply)
- **split_groups_consistency**: PASS — 0 group_id split mismatches vs split_groups.csv

## Research note

This is a leakage-controlled **proxy** split. Similarity groups are not confirmed subjects.

Fingerprint-to-blood-group prediction is an experimental research task. This system is not a clinical diagnostic tool and must not replace laboratory blood-group testing.
