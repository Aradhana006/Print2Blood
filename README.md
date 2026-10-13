
# Goodbye to Blood Test

> **Research prototype only.** Fingerprint-based blood-group prediction is experimental and must not replace laboratory blood-group testing. Outputs are **model predictions**, not confirmed blood groups.

Web app + research notebooks for experimental fingerprint → ABO/Rh prediction.

- **Backend:** FastAPI + existing ResNet50 v1 weights
- **Frontend:** React + Vite + TypeScript + Tailwind
- **Database:** SQLite by default (PostgreSQL-ready via `DATABASE_URL`)

## Run the app

```powershell
# Terminal 1 — API (from project root)
$env:PYTHONPATH="backend"
.\.venv\Scripts\uvicorn.exe app.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 — UI
cd frontend
npm install
npm run dev
```

Open:

- Frontend: http://localhost:5173
- API docs: http://localhost:8000/docs

Primary workflow: **Scan fingerprint → upload image → Analyze → model prediction + probabilities**.

### Test with your own images
- Prediction now: just upload on **Scan fingerprint** (no labels needed).
- Future labelled dataset: put images under `external_data/A+/`, `A-/`, … (see `external_data/README.md`).

---

# Fingerprint-Based Blood Group Detection (research notebooks)

This repository also contains code, datasets, and results related to fingerprint-based blood group detection. The project explores using various machine learning models (ResNet, VGG16, AlexNet, and LeNet) to classify blood groups based on fingerprint images.

The project is organized into various folders that store datasets, models, performance metrics, and visualization graphs.

## Table of Contents

- [Folder Structure](#folder-structure)
- [Getting Started](#getting-started)
- [Leakage-Controlled Master Split](#leakage-controlled-master-split)
- [Research Dashboard](#research-dashboard)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Sample Dataset](#sample-dataset)
- [Usage](#usage)

## Folder Structure

The main directories are organized as follows:

```plaintext
Fingerprint-Based-Blood-Group-Detection/
├── code/                    # Existing CNN notebooks (LeNet, AlexNet, VGG16, ResNet50)
├── configs/                 # baseline_experiment.json, split_config.json
├── dashboard/               # Streamlit research audit dashboard
├── data/splits/             # master_split.csv, split_groups.csv
├── dataset/dataset_blood_group/
├── results/audit/           # Dataset audit artifacts
├── results/splits/          # Split summaries, similarity, validation
├── scripts/                 # build_master_split.py, validate_master_split.py, ...
├── src/                     # data / similarity / splitting / validation modules
├── graphs/
├── performance metrix/
├── sample dataset/
└── test/
```

### Folder Details

- **code/**: Contains subfolders for each model (ResNet, VGG16, AlexNet, LeNet), with Jupyter notebooks files for training and evaluation.
- **dataset/**: Contains the main dataset with images of fingerprints categorized by blood group types (A+, A-, B+, B-, AB+, AB-, O+, O-).
- **graphs/**: Stores accuracy, validation accuracy, loss, and validation loss graphs for each model.
- **performance_metrix/**: Contains bar graphs comparing the performance metrics across different models.
- **sample dataset/**: Includes a sample fingerprint image to preview the dataset.
- **test/**: Placeholder for any testing or additional data as needed.

## Getting Started

### Prerequisites

- Python 3.11 recommended (3.8+ may work for older notebooks)
- Virtual environment with TensorFlow / scientific stack
- Streamlit for the research dashboard

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install tensorflow pandas matplotlib seaborn scikit-learn pillow jupyter streamlit
```

> Note: the pinned versions in `requirements.txt` are outdated for modern Python. Prefer the packages above for the current audit/split/dashboard workflow.

## Leakage-Controlled Master Split

True participant IDs are **not available** in this dataset. The project therefore builds a
**leakage-controlled proxy split** (not a confirmed subject-aware split):

1. Exact MD5 duplicates always share one `group_id`.
2. Same-label highly similar pairs may merge via pHash + SSIM confirmation.
3. Whole groups are assigned to train / validation / test (target 70% / 15% / 15%).

```bash
# 1) Build proxy groups + master split
.\.venv\Scripts\python.exe scripts\build_master_split.py

# 2) Validate leakage constraints (non-zero exit on critical failure)
.\.venv\Scripts\python.exe scripts\validate_master_split.py
```

Outputs:

- `data/splits/master_split.csv`
- `data/splits/split_groups.csv`
- `results/splits/split_summary.json`
- `results/splits/split_validation_report.md`

## Research Dashboard

```bash
.\.venv\Scripts\streamlit.exe run dashboard\app.py
```

Open the local URL printed by Streamlit (typically `http://localhost:8501`).

The dashboard is **read-only**: it does not train models and does not silently regenerate the split.
Use the **Run Split Validation** button on the Leakage Validation page to re-check constraints.

### Cloning the Repository

```bash
git clone https://github.com/yourusername/Fingerprint-Based-Blood-Group-Detection.git
cd Fingerprint-Based-Blood-Group-Detection
```

## Dataset

The dataset contains fingerprint images organized into folders by blood group type. There are approximately 6,000–7,000 images spread across the eight blood group categories.

- Each subfolder within `dataset/datasetbloodgroup/` represents a blood group and contains fingerprint images labeled accordingly.
  
Ensure that you have sufficient storage space before downloading or expanding the dataset.

## Models

This project explores the following CNN architectures:

- **ResNet**
- **VGG16**
- **AlexNet**
- **LeNet**

Each model has a dedicated folder under `code/`, with:
- A Jupyter notebook (`.ipynb`) that contains code for training and testing.

## Results

Performance metrics and training graphs are stored in separate folders for easy access and visualization.

- **graphs/**: Includes accuracy, validation accuracy, loss, and validation loss plots for each model, which help visualize the model's performance over time.
- **performance_metrix/**: Contains bar graphs comparing the performance of different models.

## Sample Dataset

The `sample dataset/` folder includes a sample fingerprint image to give users an idea of the dataset's format.

## Usage

1. **Dataset Preparation**:
   Ensure that the `dataset/` folder is populated with the required fingerprint images.

2. **Running the Models**:
   Navigate to the respective model folder in `code/`, open the Jupyter notebook, and run it step-by-step. Each notebook is designed to:
   - Load the dataset
   - Train the model on the training dataset
   - Evaluate the model on the test dataset

3. **Viewing Results**:
   After training, check the `graphs/` and `performance_metrix/` folders for visualizations of the training process and model comparisons.

4. **Testing on Sample Data**:
   You can test the model using images from the `sample dataset/` or by adding additional test images to the `test/` folder.

## Contributing

Feel free to fork this repository, make improvements, and submit a pull request. Contributions to improve model performance, add additional blood group types, or optimize the codebase are welcome.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

Special thanks to the authors of the research paper that served as the foundation for this work. For details on the methodology and findings, refer to the paper included in this repository.
