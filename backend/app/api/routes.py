from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.models.entities import ModelVersion, Prediction
from app.schemas.prediction import DashboardSummary, ModelInfo, PredictResponse
from app.services.prediction import run_prediction
from app.services.registry import DISCLAIMER, ensure_resnet50_v1_registered

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "service": "fingerabo-api"}


@router.get("/dashboard/summary", response_model=DashboardSummary)
def dashboard_summary(db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    model = ensure_resnet50_v1_registered(db, settings)
    total_predictions = db.scalar(select(func.count()).select_from(Prediction)) or 0

    class_distribution: dict[str, int] = {c: 0 for c in settings.class_labels}
    dataset_dir = settings.resolve("dataset/dataset_blood_group")
    total_images = 0
    if dataset_dir.exists():
        for label in settings.class_labels:
            n = len(list((dataset_dir / label).glob("*")))
            class_distribution[label] = n
            total_images += n

    leakage_status = "PASS" if settings.resolve("data/splits/master_split.csv").exists() else "Not evaluated yet"
    validation = settings.resolve("results/splits/split_validation.json")
    if validation.exists():
        import json

        try:
            leakage_status = json.loads(validation.read_text(encoding="utf-8")).get("overall", leakage_status)
        except Exception:
            pass

    return DashboardSummary(
        app_name=settings.app_name,
        total_dataset_images=total_images,
        n_classes=len(settings.class_labels),
        deployed_model_name=model.model_name,
        deployed_model_version=model.version,
        deployed_model_accuracy=model.accuracy,
        deployed_model_macro_f1=model.macro_f1,
        total_predictions=int(total_predictions),
        total_experiments=0,
        leakage_status=leakage_status,
        class_distribution=class_distribution,
        disclaimer=DISCLAIMER,
    )


@router.get("/models/active", response_model=ModelInfo)
def active_model(db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    model = ensure_resnet50_v1_registered(db, settings)
    return ModelInfo(
        id=model.id,
        model_name=model.model_name,
        version=model.version,
        architecture=model.architecture,
        status=model.status,
        is_active=model.is_active,
        accuracy=model.accuracy,
        macro_f1=model.macro_f1,
        split_type=model.split_type,
        image_size=model.image_size,
        notes=model.notes,
    )


@router.get("/models", response_model=list[ModelInfo])
def list_models(db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    ensure_resnet50_v1_registered(db, settings)
    rows = db.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc())).all()
    return [
        ModelInfo(
            id=m.id,
            model_name=m.model_name,
            version=m.version,
            architecture=m.architecture,
            status=m.status,
            is_active=m.is_active,
            accuracy=m.accuracy,
            macro_f1=m.macro_f1,
            split_type=m.split_type,
            image_size=m.image_size,
            notes=m.notes,
        )
        for m in rows
    ]


@router.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    return run_prediction(db, settings, file)


@router.get("/predictions")
def list_predictions(limit: int = 50, db: Session = Depends(get_db)):
    rows = db.scalars(
        select(Prediction).order_by(Prediction.created_at.desc()).limit(min(limit, 200))
    ).all()
    return [
        {
            "request_id": r.request_id,
            "timestamp": r.created_at,
            "prediction": r.predicted_label,
            "confidence": r.confidence,
            "model_id": r.model_id,
            "image_hash": r.image_hash,
            "original_filename": r.original_filename,
        }
        for r in rows
    ]
