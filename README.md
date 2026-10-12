# Print2Blood

**Fingerprint-Based ABO/Rh Blood Group Prediction Using Machine Learning and Deep Learning**

## Overview

Print2Blood is a machine learning research project that investigates eight-class ABO/Rh blood group classification from fingerprint images. It compares classical machine learning algorithms with deep learning using handcrafted fingerprint features, CNN transfer learning, statistical analysis, and leakage-aware evaluation.

## Objectives

- Develop an eight-class fingerprint image classification pipeline.
- Compare classical ML models with CNN-based transfer learning.
- Analyse the association between fingerprint pattern types and blood groups.
- Evaluate model performance using subject-aware validation and data leakage analysis.
- Develop a research prototype for experimental predictions.

## Methodology

**Classical Machine Learning:** SVM, Random Forest, and Gradient Boosting trained on handcrafted fingerprint features, including ridge patterns, minutiae descriptors, and optional HOG/LBP texture features.

**Deep Learning:** Transfer learning using a pretrained ResNet18 CNN for eight-class image classification.

**Statistical Analysis:** Chi-square test of independence and Cramér's V effect size to examine the association between fingerprint pattern types and blood groups.

**Evaluation:** Accuracy, macro-precision, macro-recall, macro-F1, confusion matrices, and majority-class baseline comparison. Subject-aware splitting and near-duplicate checks are used where feasible to investigate data leakage.

## Tech Stack

- Python
- OpenCV and scikit-image
- scikit-learn
- PyTorch and torchvision
- SciPy
- Matplotlib and Seaborn
- Streamlit
- Git and GitHub

## Project Structure

```text
Print2Blood/
├── data/
├── splits/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── split_data.py
│   ├── classical_ml.py
│   ├── deep_learning.py
│   ├── evaluation.py
│   └── statistics.py
├── models/
├── outputs/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/print2blood.git
cd print2blood
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The project uses a publicly available, anonymised fingerprint dataset containing approximately 6,000 images across eight ABO/Rh blood group classes. The dataset must be obtained separately, and its source and licence should be verified before use or redistribution.

## Running the Application

Launch the Streamlit research prototype:

```bash
streamlit run app.py
```

The interface is intended to accept a fingerprint image and display an experimental model prediction after the trained model has been integrated.

## Evaluation

The classical ML models and CNN are evaluated using a shared held-out test set and consistent metrics. The evaluation includes a majority-class baseline, confusion matrices, macro-averaged metrics, statistical association analysis, and random-split versus subject-aware comparison where feasible.

Actual experimental results will be documented after model training and evaluation.

## Limitations

The relationship between fingerprint characteristics and ABO/Rh blood groups is not established as a reliable predictive mechanism. Dataset bias, repeated impressions, subject overlap, and population differences may affect performance and generalisability.

## Disclaimer

Print2Blood is an academic research prototype, not a clinical diagnostic tool. Its predictions must not be used for medical decisions, emergency care, or blood transfusion. Blood groups must be confirmed through appropriate laboratory testing.

## License

The original project code is licensed under the MIT License, subject to the terms in the repository's `LICENSE` file. Third-party datasets, images, and pretrained model weights remain subject to their respective licences.
