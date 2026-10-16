"""Predict a blood group from a fingerprint image and save the result."""
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "test", "model_blood_group_detection_resnet.h5")
DEFAULT_IMAGE = os.path.join(ROOT, "test", "O- blood group.BMP")
RESULT_IMAGE = os.path.join(ROOT, "test", "prediction_result.png")
RESULT_TEXT = os.path.join(ROOT, "test", "prediction_result.txt")

LABELS = {0: "A+", 1: "A-", 2: "AB+", 3: "AB-", 4: "B+", 5: "B-", 6: "O+", 7: "O-"}


def predict(img_path):
    model = load_model(MODEL_PATH)
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    result = model.predict(x, verbose=0)[0]
    predicted_class = int(np.argmax(result))
    predicted_label = LABELS[predicted_class]
    confidence = float(result[predicted_class] * 100)

    order = np.argsort(result)[::-1]
    lines = [
        f"Image: {img_path}",
        f"Prediction: {predicted_label}",
        f"Confidence: {confidence:.2f}%",
        "",
        "All class scores:",
    ]
    for idx in order:
        lines.append(f"  {LABELS[int(idx)]:>3}  {result[idx] * 100:6.2f}%")
    report = "\n".join(lines)
    print(report)

    with open(RESULT_TEXT, "w", encoding="utf-8") as handle:
        handle.write(report + "\n")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.asarray(img))
    ax.axis("off")
    ax.set_title(f"Prediction: {predicted_label}  ({confidence:.2f}%)", fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULT_IMAGE, dpi=120)
    plt.close(fig)
    print(f"\nSaved image: {RESULT_IMAGE}")
    print(f"Saved text:  {RESULT_TEXT}")
    return predicted_label, confidence


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    predict(path)
