from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.entities import ModelVersion


DISCLAIMER = (
    "Research prototype only. Fingerprint-based blood-group prediction is experimental "
    "and must not replace laboratory blood-group testing."
)


def ensure_resnet50_v1_registered(db: Session, settings: Settings) -> ModelVersion:
    model = db.get(ModelVersion, settings.active_model_id)
    if model is None:
        model = ModelVersion(
            id=settings.active_model_id,
            model_name="ResNet50",
            version="v1",
            architecture="ResNet50",
            weights_path=str(settings.model_path),
            image_size=settings.image_size,
            status="DEPLOYED",
            is_active=True,
            # Existing notebook random-split result; not a leakage-controlled benchmark.
            accuracy=0.8142,
            macro_f1=0.82,
            split_type="random_image_split_notebook",
            notes=(
                "Imported existing trained weights without modification. "
                "Reported 81.42% accuracy comes from the original random 80/20 notebook split "
                "and is NOT a leakage-controlled evaluation."
            ),
        )
        db.add(model)
        db.commit()
        db.refresh(model)
    else:
        # Keep active flag aligned with config for Phase 1.
        if not model.is_active:
            model.is_active = True
            model.status = "DEPLOYED"
            db.commit()
            db.refresh(model)

    # Write registry metadata sidecar (does not copy/overwrite weights).
    registry_dir = settings.registry_path / settings.active_model_id
    registry_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": model.id,
        "model_name": model.model_name,
        "version": model.version,
        "architecture": model.architecture,
        "weights_path": model.weights_path,
        "image_size": model.image_size,
        "status": model.status,
        "is_active": model.is_active,
        "accuracy": model.accuracy,
        "macro_f1": model.macro_f1,
        "split_type": model.split_type,
        "class_labels": settings.class_labels,
        "preprocessing_version": "resnet50.preprocess_input@256",
        "disclaimer": DISCLAIMER,
        "notes": model.notes,
    }
    (registry_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    metrics = {
        "accuracy": model.accuracy,
        "macro_f1": model.macro_f1,
        "split_type": model.split_type,
        "source": "code/Resnet34/Resnet.ipynb saved output",
        "warning": "Not evaluated under leakage-controlled proxy split yet.",
    }
    (registry_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return model
