"""Offline CNN inference aligned with production ResNet50 preprocessing."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


def load_resnet_model(weights_path: Path):
    from tensorflow.keras.models import load_model

    return load_model(weights_path)


def predict_batch(
    model,
    image_paths: List[Path],
    root: Path,
    class_labels: List[str],
    image_size: int = 256,
    batch_size: int = 32,
) -> np.ndarray:
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing import image as kimage

    y_pred: List[int] = []
    for i in range(0, len(image_paths), batch_size):
        chunk = image_paths[i : i + batch_size]
        batch = []
        for rel in chunk:
            path = root / rel if not Path(rel).is_absolute() else Path(rel)
            img = kimage.load_img(path, target_size=(image_size, image_size))
            arr = kimage.img_to_array(img)
            batch.append(arr)
        x = preprocess_input(np.stack(batch, axis=0))
        probs = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
    return np.asarray(y_pred, dtype=np.int64)
