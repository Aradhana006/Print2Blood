"""Grad-CAM / saliency helpers for the deployed CNN."""
from __future__ import annotations

import base64
import io
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def _find_last_conv_layer(model) -> Optional[object]:
    import tensorflow as tf

    last = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            for sub in layer.layers:
                if isinstance(sub, tf.keras.layers.Conv2D):
                    last = sub
        elif isinstance(layer, tf.keras.layers.Conv2D):
            last = layer
    return last


def compute_gradcam(
    model,
    preprocessed_batch: np.ndarray,
    class_index: int,
) -> Tuple[np.ndarray, str]:
    """
    Returns (heatmap 0..1 HxW, method_name).
    Falls back to input-gradient saliency if Grad-CAM wiring fails.
    """
    import tensorflow as tf

    try:
        last_conv = _find_last_conv_layer(model)
        if last_conv is None:
            raise RuntimeError("No conv layer found")

        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv.output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(preprocessed_batch, training=False)
            loss = preds[:, class_index]
        grads = tape.gradient(loss, conv_out)
        if grads is None:
            raise RuntimeError("Null gradients")
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)
        cam = tf.nn.relu(cam)
        cam = cam / (tf.reduce_max(cam) + 1e-8)
        return cam.numpy(), "gradcam"
    except Exception:
        # Input saliency fallback
        x = tf.convert_to_tensor(preprocessed_batch)
        with tf.GradientTape() as tape:
            tape.watch(x)
            preds = model(x, training=False)
            loss = preds[:, class_index]
        grads = tape.gradient(loss, x)
        sal = tf.reduce_mean(tf.abs(grads[0]), axis=-1)
        sal = sal / (tf.reduce_max(sal) + 1e-8)
        return sal.numpy(), "input_saliency"


def overlay_heatmap_b64(original_rgb: Image.Image, heatmap: np.ndarray, size: int = 256) -> str:
    base = original_rgb.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    heat = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L").resize(
        (size, size), Image.Resampling.BILINEAR
    )
    # Pseudo-color: red=high importance
    heat_rgb = Image.merge(
        "RGB",
        (
            heat,
            Image.eval(heat, lambda p: int(p * 0.25)),
            Image.eval(heat, lambda p: int(255 - p * 0.6)),
        ),
    )
    blend = Image.blend(base, heat_rgb, alpha=0.45)
    buf = io.BytesIO()
    blend.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")
