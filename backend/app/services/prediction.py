from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.ml.resnet50_service import get_active_model_service
from app.models.entities import Prediction, PredictionProbability
from app.schemas.prediction import PredictResponse, ProbabilityItem
from app.services.image_validation import validate_and_save_upload
from app.services.registry import DISCLAIMER, ensure_resnet50_v1_registered


def run_prediction(db: Session, settings: Settings, upload: UploadFile) -> PredictResponse:
    model_row = ensure_resnet50_v1_registered(db, settings)
    service = get_active_model_service()

    request_id = uuid.uuid4().hex
    temp_path = settings.temp_path / f"{request_id}{Path(upload.filename or '.bmp').suffix.lower()}"
    meta = validate_and_save_upload(
        upload,
        temp_path,
        max_bytes=settings.max_upload_mb * 1024 * 1024,
    )

    try:
        result = service.predict(str(meta["path"]))
    finally:
        if not settings.store_prediction_images:
            Path(meta["path"]).unlink(missing_ok=True)

    probs = result["probabilities"]
    ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top3 = [ProbabilityItem(label=k, probability=v) for k, v in ordered[:3]]

    pred = Prediction(
        id=uuid.uuid4().hex,
        request_id=request_id,
        image_hash=meta["image_hash"],
        predicted_label=result["prediction"],
        confidence=float(result["confidence"]),
        model_id=model_row.id,
        preprocessing_version=service.preprocessing_version,
        original_filename=meta["filename"],
        created_at=datetime.utcnow(),
    )
    db.add(pred)
    for label, probability in probs.items():
        db.add(
            PredictionProbability(
                prediction_id=pred.id,
                label=label,
                probability=float(probability),
            )
        )
    db.commit()

    return PredictResponse(
        request_id=request_id,
        prediction=result["prediction"],
        confidence=float(result["confidence"]),
        probabilities=probs,
        top3=top3,
        model_name=model_row.model_name,
        model_version=model_row.version,
        model_id=model_row.id,
        preprocessing_version=service.preprocessing_version,
        image_hash=meta["image_hash"],
        timestamp=pred.created_at,
        disclaimer=DISCLAIMER,
    )
