# Research reproducibility — Print2Blood

**Project:** Fingerprint-Based ABO Blood Group Prediction (comparative ML/DL study)

**Disclaimer:** Research prototype only. Predictions are not clinical blood-group results.

## Evaluation protocol (current)

| Protocol | File / script | Use |
|----------|---------------|-----|
| Leakage-controlled proxy split | `data/splits/master_split.csv` | **Official** train / validation / test for new experiments |
| Split validation | `scripts/validate_master_split.py` | CI + dashboard checks |
| Historical CNN metric (81.42%) | `code/Resnet34/Resnet.ipynb` | **Legacy** random 80/20 image split — do not compare directly to master-split metrics |

Proxy groups merge exact MD5 duplicates and (at configured thresholds) same-label near-duplicates. **No verified subject IDs** — not guaranteed person-level independence.

## Class label order (softmax index)

`A+`, `A-`, `AB+`, `AB-`, `B+`, `B-`, `O+`, `O-` → indices `0`–`7`.

Must match: `src/data/dataset.py`, `backend/app/core/config.py`, training notebook, deployed `.h5`.

## Offline experiments

### Setup

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt -r requirements-research.txt
```

### Classical ML (HOG + SVM / Random Forest / Gradient Boosting)

```powershell
.\.venv\Scripts\python.exe scripts\run_classical_ml_experiment.py
```

- Reads `configs/classical_ml_experiment.json`
- Fits `StandardScaler` on **train** only
- Selects hyperparameters by **validation macro F1**
- Reports **test** metrics once (accuracy, macro precision/recall/F1, confusion matrices)
- Writes `results/experiments/classical_ml_<timestamp>/` and `results/experiments/latest_classical_ml.json`
- Does **not** change the production API model

Optional:

```powershell
.\.venv\Scripts\python.exe scripts\run_classical_ml_experiment.py --feature hog_lbp
.\.venv\Scripts\python.exe scripts\run_classical_ml_experiment.py --max-samples 200
```

### CNN on master split (existing weights)

```powershell
.\.venv\Scripts\python.exe scripts\eval_cnn_master_split.py
```

Evaluates `test/model_blood_group_detection_resnet.h5` on the **master split test set** only. Saves a new experiment folder; **does not overwrite** weights.

### Pattern association (exploratory)

```powershell
.\.venv\Scripts\python.exe scripts\run_pattern_association_stats.py
```

Chi-square and Cramér's V for **blood group × heuristic pattern** (Arch-like / Loop-like / Whorl-like). Patterns are **not** AFIS ground truth.

## Production inference preprocessing

- API: `resnet50.preprocess_input`, 256×256 (`backend/app/ml/resnet50_service.py`)
- CLI: `predict.py` (aligned with API)

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

## Comparing models fairly

Only compare metrics when:

1. Same split manifest (`master_split.csv`)
2. Same split partition (test rows)
3. Test set used **once** for final numbers
4. Historical 81.42% is labeled as random-split notebook baseline
