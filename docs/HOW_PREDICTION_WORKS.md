# How fingerprint → blood-group prediction works in this project

## Honest summary

This system does **not** measure blood. It shows a fingerprint image to a neural network that was trained on labelled fingerprint photos. The network returns which blood-group label looks most similar to patterns it saw during training.

That is **experimental pattern recognition**, not antigen typing.

## Pipeline

```text
Fingerprint image (upload / camera / future USB scanner)
        │
        ▼
Image validation (type, size, readable)
        │
        ▼
Quality analysis (sharpness, contrast, brightness, ridge activity)
        │
        ▼
Pattern cue (heuristic loop / whorl / arch estimate)
        │
        ▼
Resize to 256×256 + ResNet preprocessing
        │
        ▼
ResNet50 CNN (trained weights)
        │
        ▼
8 probabilities → top class = model prediction
        │
        ▼
Explainability heatmap (Grad-CAM / saliency)
```

## What patterns are involved?

Fingerprint research (dermatoglyphics) talks about:

- **Arch / Loop / Whorl** pattern types
- Ridge flow / orientation
- Ridge density
- Minutiae (ridge endings, bifurcations) — used in identity systems, not deeply used here yet

This project’s CNN mostly learns **texture and ridge-structure features** automatically. The app also computes a lightweight orientation heuristic for a human-readable pattern cue.

## What is prediction based on?

- Training labels from the dataset folders (`A+`, `O-`, …)
- Visual features extracted by ResNet50
- Softmax probabilities over 8 classes

It is **not** based on:

- ABO antigens
- Rh factor lab reagents
- DNA
- Proven biological causation

## Scanner plan (your idea)

1. Buy a fingerprint scanner
2. Capture live prints (or use camera until the scanner SDK is integrated)
3. Feed captures into `/api/analyze`
4. If you know your true lab blood group, compare prediction vs truth

That is a valid demo path when you lack a big external labelled test set.
