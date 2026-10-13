# Project Audit Report

Accuracy improvement is **not** the goal of this audit. Scientific validity and reproducibility are.

## 1. Existing architecture

- Four CNN notebooks under `code/`: LeNet, AlexNet, VGG16, ResNet50 (folder named `Resnet34`).
- Best reported model: **ResNet50 + frozen ImageNet backbone + Dense(128)-Dense(128)-Dense(8)**.
- Inference prototype scripts: `predict.py` (ResNet `.h5`) and `run_lenet.py` (training helper).
- System task: 8-class fingerprint image classification into ABO/Rh labels.
- Scientific caveat: the system does **not** detect antigens; it learns visual correlations in this dataset.

### ResNet50 head

```text
ResNet50(ImageNet, include_top=False, pooling='avg', trainable=False)
  -> Dense(128, relu)
  -> Dense(128, relu)
  -> Dense(8, softmax)
```

## 2. Existing preprocessing

- Only `resnet50.preprocess_input` via `ImageDataGenerator`.
- No CLAHE, denoise, ridge enhancement, or fingerprint segmentation.
- No geometric/color augmentation.
- ResNet/VGG/AlexNet resize to **256x256**; LeNet to **224x224**.
- Native images are almost entirely **96x103 RGBA (4-channel)**; a small subset is **241x298 RGB (3-channel)** (`5,956` vs `44`).

## 3. Existing split strategy

- `train_test_split(..., test_size=0.20, random_state=42)`.
- **Not stratified.**
- Preceded by `data.sample(frac=1)` **without a seed** => split is **not fully reproducible** across notebook reruns.
- No dedicated validation set.
- The 20% split is used both as `validation_data` in `fit()` and as the set for final `evaluate()` / classification report.
- **Not subject-aware.**

## 4. Existing leakage risks

### Confirmed

- Image-level random split can place highly similar fingerprints from the same acquisition context into both train and test.
- Test images influence early-stopping-style monitoring because they are the validation stream during training.
- No subject ID metadata exists to prevent person-level leakage.

### Evidence from this audit

- Exact MD5 duplicates: **11** groups covering **26** files (same bytes, different filenames). All observed duplicate groups stay inside one blood-group folder.
- Largest exact-duplicate group: **5** identical `B+` files.
- Average-hash screening found many identical aHash pairs (**11,994**), but only **~24.6%** shared the same label.
- That low same-label rate means 8x8 aHash is **too coarse** for these fingerprints and mostly reflects hash collisions / shared global appearance, not reliable subject grouping.
- Therefore this audit treats exact MD5 duplicates as hard evidence, and aHash only as a weak similarity screen.
- **No verified subject IDs were found.** Near-duplicate hashes must **not** be called subject IDs.

### Filename / subject analysis

- Pattern: `cluster_<class_code>_<image_id>.BMP`.
- Cluster codes map 1:1 to class folders: `{0: 'A+', 1: 'A-', 2: 'B+', 3: 'B-', 4: 'AB+', 5: 'AB-', 6: 'O+', 7: 'O-'}`.
- Unique numeric `image_id`s: **6000** / 6000.
- IDs spanning multiple classes: **0**.
- Subject ID field present: **False**.
- Interpretation: No explicit subject/person ID field was found. Filenames use cluster_<0-7>_<image_id>, where cluster maps 1:1 to blood-group folder. Numeric image_ids are unique across the whole dataset (0..5999 style indexing), so they behave like global sample indices, not verified participant IDs. Near-duplicate aHash grouping is an imperfect proxy and must not be treated as true subjects.

## 5. Existing model parameters

| Item | Value |
|---|---|
| Optimizer | Adam (default LR, typically 1e-3; not set explicitly) |
| Loss | categorical_crossentropy |
| Batch size | 32 |
| Epochs | 20 |
| Class weights | none |
| ImageNet weights | yes (ResNet50/VGG16) |
| Frozen layers | entire ResNet50/VGG16 backbone |
| Trainable layers | classification head only |
| Augmentation | none |

### Source of reported 81.42%

- Notebook: `code/Resnet34/Resnet.ipynb`
- Printed metric: `Test Accuracy: 81.42%`
- Test loss in same output: `0.58375`
- Support: `1200` images
- Notes: Value comes from saved notebook cell output after model.evaluate(valid_gen). valid_gen is built from the 20% test split, which is also passed as validation_data during fit().

Other notebook-reported accuracies:

| Model | Accuracy | Loss |
|---|---:|---:|
| ResNet50 | 81.42% | 0.58375 |
| LeNet | 77.50% | 1.15884 |
| VGG16 | 73.75% | 1.04504 |
| AlexNet | 16.33% | 2.06960 |

## 6. Dataset problems

- Total images: **6000**
- Corrupted/unreadable: **0**
- Class imbalance: A- has **1009**, A+ has **565** (majority baseline ≈ **16.82%**)
- Dominant native size: **96x103** (`5,956` images, 4-channel RGBA)
- Secondary size: **241x298** (`44` images, 3-channel RGB)
- Distinct native sizes: **2**
- Mean brightness range: **86.78 – 224.11**
- Contrast (std) range: **60.02 – 114.05**
- Exact duplicate files exist under different names (**11** MD5 groups / **26** files).
- No participant metadata / demographics / finger-type labels.
- Cluster prefix encodes class, which is fine for labeling but provides no subject grouping.
- Cropping/framing varies across samples; acquisition/artifact bias has not yet been ruled out.

## 7. Recommended next experiment

Do **not** optimize accuracy next.

1. Create a **master reproducible split config** with a fixed global seed and no unseeded `sample()`.
2. Build a **leakage-controlled experiment**:
   - Experiment A: reproduce random image split baseline (target ~81.42%).
   - Experiment B: subject-proxy / near-duplicate-aware split, clearly labeled as a proxy.
3. If true subject IDs become available, replace the proxy with a real subject-aware split.
4. Keep a frozen copy of the current ResNet50 as the official baseline; do not overwrite it while auditing.
5. Only after leakage-controlled numbers exist, improve preprocessing / fine-tuning / EfficientNet.

### Suggested evaluation table for the next phase

| Evaluation | Accuracy | Macro-F1 | Status |
|---|---:|---:|---|
| Random image split (current notebooks) | 81.42% | 0.82 | recorded |
| Reproduced random split (seeded, no unseeded shuffle) | TBD | TBD | next |
| Leakage-controlled / subject-proxy split | TBD | TBD | next |
| External new-subject data | TBD | TBD | later |

## Reproducibility config

See `configs/baseline_experiment.json` created alongside this audit.

## Generated artifacts

- `results/audit/dataset_summary.csv`
- `results/audit/dataset_summary.json`
- `results/audit/class_distribution.png`
- `results/audit/sample_grid.png`
- `results/audit/image_dimensions.png`
- `results/audit/brightness_contrast.png`
- `results/audit/duplicate_hashes.csv`
- `results/audit/near_duplicate_pairs.csv`
- `results/audit/filename_analysis.json`
