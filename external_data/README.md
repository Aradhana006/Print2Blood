# Testing with your own data — Goodbye to Blood Test

## A) Quick test (prediction only) — do this now

You do **not** need labels to try your own fingerprints.

1. Start the app (API + frontend).
2. Open **Scan fingerprint**.
3. Upload any new image: `.bmp`, `.png`, `.jpg`, `.jpeg`.
4. Click **Analyze fingerprint**.

Tips for better images:
- Clear ridge detail, not blurry
- Finger roughly centered
- Avoid heavy filters / screenshots of screenshots
- Prefer a real scan or a sharp close-up photo

Your image is processed temporarily and not kept by default.

---

## B) New labelled dataset (for later training)

Use this when you collect fingerprints from **new people** with known lab blood groups.

Create folders like this:

```text
external_data/
├── A+/
├── A-/
├── AB+/
├── AB-/
├── B+/
├── B-/
├── O+/
└── O-/
```

Rules:
- Put each image in the folder matching its **lab-confirmed** blood group.
- Prefer people who were **not** in the original 6,000-image training set.
- Keep filenames simple (no special path characters).
- One person can have multiple finger images, but for research you should track person IDs separately if possible.

Example:

```text
external_data/O-/me_right_thumb.png
external_data/A+/friend_index.bmp
```

Today the web app’s Dataset upload page is still “Soon” (Phase 2). Until then:
1. Organize images in `external_data/` as above.
2. Use **Scan fingerprint** to test individual new images against the current model.
3. Later we will wire this folder into audit → leakage-controlled split → training → model comparison.

---

## Important

This system is experimental. Even if the model predicts your blood group correctly, that is **not** a clinical result.
