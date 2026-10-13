# FingerABO — Implementation Plan

Research prototype web application for experimental fingerprint → ABO/Rh model prediction.

**Disclaimer:** Not a clinical diagnostic system. Predictions are model outputs, not confirmed blood groups.

## Existing assets to reuse (do not overwrite weights)

| Asset | Path | Role |
|---|---|---|
| ResNet50 trained weights | `test/model_blood_group_detection_resnet.h5` | First registered model **ResNet50 v1** |
| Prediction preprocessing | `predict.py` / ResNet notebook | 256×256 + `resnet50.preprocess_input` |
| Class order | A+, A-, AB+, AB-, B+, B-, O+, O- | Softmax index mapping |
| Leakage-controlled split | `data/splits/master_split.csv` | Future training pipeline (Phase 2+) |
| Dataset audit | `results/audit/`, `scripts/` | Future dataset service |

## Architecture

```
React (Vite/TS/Tailwind)
        │ REST
        ▼
FastAPI
  ├── Prediction Service  → ResNet50 v1 adapter
  ├── Dataset Service     (Phase 2)
  ├── Training Service    (Phase 3, background jobs)
  ├── Model Registry
  └── Reports             (Phase 5)
        │
        ├── SQLite (local default) / PostgreSQL
        └── storage/ + models/registry/
```

## Phases

### PHASE 1 — Prediction MVP (this phase)
- FastAPI backend foundation
- React frontend shell + Predict page
- DB schema (SQLAlchemy; SQLite default, Postgres-ready)
- ResNet50 v1 adapter around existing `.h5`
- `POST /api/predict` end-to-end
- Image validation, probabilities, research disclaimer
- Model metadata endpoint
- **No training, no weight changes**

### PHASE 2 — Datasets & leakage
- ZIP dataset upload, audit, explorer, proxy split integration

### PHASE 3 — Training jobs
- Background worker, experiments UI, evaluation artifacts

### PHASE 4 — Registry governance
- Approve / deploy / compare model versions

### PHASE 5 — Explainability & stats
- Grad-CAM, chi-square / Cramér's V, report packs

### PHASE 6 — Hardening
- Auth roles, Docker full stack, tests, production settings

## Local defaults (Phase 1)

- Database: SQLite file `backend/fingerabo.db` unless `DATABASE_URL` points to Postgres
- Active model: ResNet50 v1 from existing weights
- Prediction images: temporary only (deleted after inference); store hash + prediction metadata
