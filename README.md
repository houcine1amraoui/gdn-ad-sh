# GDN-AD-SH: Graph Neural Network-Based Anomaly Detection for Smart Homes

This project implements a **Graph Neural Network-based anomaly detection
framework** for smart home environments using:

- GDN (Graph Deviation Network)
- MTAD-GAT
- Smart home datasets (BRE, CU, or merged BRE+CU)
- Full pipeline: preprocessing → training → evaluation → visualization

The system detects behavioral anomalies and distribution shifts in smart
home sensor data.

---

## Project Structure

    gdn-ad-sh/
    │
    ├── configs/
    │   └── config.yaml
    ├── data/
    │   ├── raw/
    │   ├── processed/
    ├── eval_results/
    ├── notebooks/
    ├── src/
    │   ├── preprocessing/
    │   ├── models/
    │   ├── training/
    │   ├── evaluation/
    │   ├── utils/
    │   └── visualization/
    ├── main_preprocess.py
    ├── main_train.py
    ├── main_evaluation.py
    ├── main.py
    └── README.md

---

## Pipeline Overview

Raw Data → Preprocessing → Processed Data → Training → Experiments →
Evaluation → Results

---

## Dataset Setup

Create:

    data/raw/

Place inside:

    BREMaster.csv
    CUMaster.csv
    BRE_filtered_columns.txt
    CU_filtered_columns.txt

---

## Configuration

Edit:

    configs/config.yaml

Key parameters:

- dataset_name: BRE or CU
- merge_bre_cu: true/false
- window_size
- training parameters
- model selection

---

## Installation

### Clone

    git clone https://github.com/houcine1amraoui/gdn-ad-sh.git
    cd gdn-ad-sh

### Create environment

    python -m venv venv
    source venv/bin/activate

Windows:

    venv\Scripts\activate

### Install dependencies

    pip install torch numpy pandas scikit-learn matplotlib scipy tqdm pyyaml joblib

---

## Running the Project

### Step 1: Preprocessing

    python main_preprocess.py

Output:

    data/processed/

Files:

- train_array.npy
- val_array.npy
- actor1_test_array.npy
- actor2_test_array.npy
- scaler.joblib
- timestamps.json

---

### Step 2: Training

    python main_train.py

Output:

    experiments/

Each run contains:

- config.yaml
- model.pt
- logs
- metrics

---

### Step 3: Evaluation

    python main_evaluation.py

Output:

    eval_results/

Contains:

- errors
- scores
- metrics
- plots

---

## Full Pipeline

    python main.py

Runs:

preprocess → train → evaluate

---

## Supported Models

### GDN

Graph Deviation Network based forecasting anomaly detection.

### MTAD-GAT

Graph Attention based forecasting and reconstruction anomaly detection.

Final error:

    final_error = α * forecast + (1-α) * reconstruction

---

## Visualization

Available in:

    src/visualization/

Includes:

- anomaly scores
- distributions
- learned graph
- plots

---

## Colab Support

Notebook:

    notebooks/colab_launcher.ipynb

---

## Workflow

1. Put raw data
2. Configure config.yaml
3. Run preprocessing
4. Run training
5. Run evaluation
6. Analyze results

---

## Author

Noureddine Amraoui

Graph Neural Networks & Smart Home Anomaly Detection

---

## License

MIT License
