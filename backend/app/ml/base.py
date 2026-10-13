from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


class BaseModelService(ABC):
    model_id: str
    model_name: str
    version: str
    architecture: str
    image_size: int
    class_labels: List[str]
    preprocessing_version: str

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, image_path: str) -> np.ndarray:
        raise NotImplementedError

    def predict(self, image_path: str) -> Dict:
        probs = self.predict_proba(image_path)
        if abs(float(probs.sum()) - 1.0) > 1e-3:
            probs = probs / probs.sum()
        idx = int(np.argmax(probs))
        return {
            "prediction": self.class_labels[idx],
            "confidence": float(probs[idx]),
            "probabilities": {
                label: float(probs[i]) for i, label in enumerate(self.class_labels)
            },
        }

    def metadata(self) -> Dict:
        return {
            "id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "architecture": self.architecture,
            "image_size": self.image_size,
            "class_labels": self.class_labels,
            "preprocessing_version": self.preprocessing_version,
        }
