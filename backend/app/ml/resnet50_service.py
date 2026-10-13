"""
Adapter around the existing trained ResNet50 weights.

Does NOT modify weights. Preprocessing matches predict.py / ResNet notebook:
- resize to 256x256
- resnet50.preprocess_input
- class order: A+, A-, AB+, AB-, B+, B-, O+, O-
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from app.ml.base import BaseModelService


class ResNet50Service(BaseModelService):
    def __init__(
        self,
        weights_path: Path,
        class_labels: List[str],
        image_size: int = 256,
        model_id: str = "resnet50_v1",
        version: str = "v1",
    ):
        self.weights_path = Path(weights_path)
        self.class_labels = class_labels
        self.image_size = image_size
        self.model_id = model_id
        self.model_name = "ResNet50"
        self.version = version
        self.architecture = "ResNet50"
        self.preprocessing_version = "resnet50.preprocess_input@256"
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.weights_path.exists():
            raise FileNotFoundError(f"ResNet50 weights not found: {self.weights_path}")
        # Lazy import so API can boot even if TF import is slow.
        from tensorflow.keras.models import load_model

        self._model = load_model(self.weights_path)

    def predict_proba(self, image_path: str) -> np.ndarray:
        self.load()
        from tensorflow.keras.applications.resnet50 import preprocess_input
        from tensorflow.keras.preprocessing import image

        img = image.load_img(image_path, target_size=(self.image_size, self.image_size))
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        probs = self._model.predict(arr, verbose=0)[0]
        return np.asarray(probs, dtype=np.float64)


_ACTIVE_SERVICE: Optional[ResNet50Service] = None


def get_active_model_service() -> ResNet50Service:
    global _ACTIVE_SERVICE
    if _ACTIVE_SERVICE is None:
        from app.core.config import get_settings

        settings = get_settings()
        _ACTIVE_SERVICE = ResNet50Service(
            weights_path=settings.model_path,
            class_labels=settings.class_labels,
            image_size=settings.image_size,
            model_id=settings.active_model_id,
            version="v1",
        )
        _ACTIVE_SERVICE.load()
    return _ACTIVE_SERVICE
