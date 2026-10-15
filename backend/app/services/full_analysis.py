from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import UploadFile
from PIL import Image
from sqlalchemy.orm import Session


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)

from app.core.config import Settings
from app.ml.explainability import compute_gradcam, overlay_heatmap_b64
from app.ml.resnet50_service import get_active_model_service
from app.models.entities import Prediction, PredictionProbability
from app.services.fingerprint_analysis import analyze_quality, make_enhanced_preview_b64
from app.services.image_validation import validate_and_save_upload
from app.services.registry import DISCLAIMER, ensure_resnet50_v1_registered


def _confidence_band(confidence: float) -> dict:
    if confidence >= 0.85:
        return {
            "level": "High",
            "meaning": "Model is strongly peaked on one class — still not medical certainty.",
        }
    if confidence >= 0.60:
        return {
            "level": "Medium",
            "meaning": "Top class leads, but alternatives remain plausible.",
        }
    return {
        "level": "Low",
        "meaning": "Uncertain prediction. Recapture or treat as inconclusive.",
    }


def run_full_analysis(db: Session, settings: Settings, upload: UploadFile) -> dict:
    model_row = ensure_resnet50_v1_registered(db, settings)
    service = get_active_model_service()
    service.load()

    request_id = uuid.uuid4().hex
    temp_path = settings.temp_path / f"{request_id}{Path(upload.filename or '.jpg').suffix.lower()}"
    meta = validate_and_save_upload(
        upload,
        temp_path,
        max_bytes=settings.max_upload_mb * 1024 * 1024,
    )
    path = Path(meta["path"])

    try:
        quality = analyze_quality(path)
        enhanced_b64 = make_enhanced_preview_b64(path)

        # Prediction
        from tensorflow.keras.applications.resnet50 import preprocess_input
        from tensorflow.keras.preprocessing import image as kimage

        img = kimage.load_img(path, target_size=(service.image_size, service.image_size))
        arr = kimage.img_to_array(img)
        batch = preprocess_input(np.expand_dims(arr, axis=0))
        probs = np.asarray(service._model.predict(batch, verbose=0)[0], dtype=np.float64)
        probs = probs / float(probs.sum())
        idx = int(np.argmax(probs))
        prediction = service.class_labels[idx]
        confidence = float(probs[idx].item())
        probability_map = {
            label: float(np.asarray(probs[i]).item()) for i, label in enumerate(service.class_labels)
        }
        ordered = sorted(probability_map.items(), key=lambda kv: kv[1], reverse=True)

        # Explainability
        heatmap, method = compute_gradcam(service._model, batch, idx)
        overlay_b64 = overlay_heatmap_b64(Image.open(path).convert("RGB"), heatmap, size=service.image_size)

        # Persist prediction metadata (not the image)
        pred = Prediction(
            id=uuid.uuid4().hex,
            request_id=request_id,
            image_hash=meta["image_hash"],
            predicted_label=prediction,
            confidence=confidence,
            model_id=model_row.id,
            preprocessing_version=service.preprocessing_version,
            original_filename=meta["filename"],
            created_at=datetime.utcnow(),
        )
        db.add(pred)
        for label, probability in probability_map.items():
            db.add(
                PredictionProbability(
                    prediction_id=pred.id,
                    label=label,
                    probability=float(probability),
                )
            )
        db.commit()

        return _json_safe(
            {
                "request_id": request_id,
                "prediction": prediction,
                "confidence": confidence,
                "confidence_band": _confidence_band(confidence),
                "probabilities": probability_map,
                "top3": [{"label": k, "probability": v} for k, v in ordered[:3]],
                "model_name": model_row.model_name,
                "model_version": model_row.version,
                "model_id": model_row.id,
                "timestamp": pred.created_at.isoformat(),
                "image_hash": meta["image_hash"],
                "quality": quality,
                "explainability": {
                    "method": method,
                    "heatmap_overlay_jpeg_base64": overlay_b64,
                    "enhanced_fingerprint_jpeg_base64": enhanced_b64,
                    "interpretation": (
                        "Warm/red regions are where the CNN attended most for this prediction. "
                        "This explains model focus, not a biological proof of blood group."
                    ),
                },
                "ai_insights": [
                    f"Estimated dermatoglyphic cue: {quality['pattern']['pattern_estimate']}.",
                    f"Capture quality: {quality['verdict']} ({quality['scores']['overall']}/100).",
                    f"Model confidence band: {_confidence_band(confidence)['level']}.",
                    "Prediction uses learned visual correlations from labelled fingerprint images, not antigen testing.",
                ],
                "disclaimer": DISCLAIMER,
            }
        )
    finally:
        if not settings.store_prediction_images:
            path.unlink(missing_ok=True)
