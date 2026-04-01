# Graph Neural Network-Based Anomaly Detection for Smart Homes

This project implements a **Graph Neural Network-based anomaly detection
framework** for smart home environments using:

- GDN (Graph Deviation Network)
- MTAD-GAT (Multivariate Timeseries Anomaly Detection-Graph Attention Network)
- Smart home datasets (BRE, CU, or merged BRE+CU)
- Full pipeline: preprocessing → training → evaluation → visualization

The system detects behavioral anomalies and distribution shifts in smart
home sensor data.

---

## 📁 Project Structure Before Running Full Pipeline

    gdn-ad-sh/
    │
    ├── configs/
    │   └── BRE_filtered_columns.txt
    │   └── config.yaml
    │   └── CU_filtered_columns.txt
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
    ├── main_visualization.py
    ├── main.py
    ├── environment.yaml
    └── README.md

### 📌 Description

- **configs/** → Configuration files and filtered sensor columns
- **notebooks/** → Experimental and analysis notebooks
- **src/** → Core implementation (preprocessing, models, training, evaluation, utils, visualization)
- **main_preprocess.py** → Runs full preprocessing pipeline
- **main_train.py** → Runs model training
- **main_evaluation.py** → Runs evaluation and metrics generation
- **main_visualization.py** → Runs visualization and plots generation
- **main.py** → Full pipeline execution (preprocess → train → evaluate → visualize)
- **README.md** → Project documentation
- **environment_cpu.yaml** → Contains all required dependencies and package versions used to run the project (for CPU users).
- **environment_gpu.yaml** → Contains all required dependencies and package versions used to run the project (for GPU users).

## 📁 Automatically Created Folders After Running Full Pipeline

    gdn-ad-sh/
    │
    ├── data/
    │ └── processed/
    ├── eval_results/
    ├── train_experiments/
    ├── visualizations/

### 📌 Automatically Created Folders Description

- **data/** → You should **manually** create this folder and place BRE and CU datasets inside.
- **data/processed/** → Contains preprocessed data ready for training for each dataset (BRE, CU, BRE+CU merged)
- **eval_results/** → Contains final evaluation outputs (errors and scores) for each dataset (BRE, CU, BRE+CU merged) and model (GDN, MTAD-GAT)
- **train_experiments/** → Contains training outputs for each dataset (BRE, CU, BRE+CU merged) and model (GDN, MTAD-GAT)
- **visualizations/** →Contains visualization outputs for each dataset (BRE, CU, BRE+CU merged) and model (GDN, MTAD-GAT)

---

## Installation

### Clone

    git clone https://github.com/houcine1amraoui/gdn-ad-sh.git
    cd gdn-ad-sh

### Create Conda environment (for CPU users)

    conda env create -f environment_cpu.yml
    conda activate gnn-env
    pip install torch-cluster -f <https://data.pyg.org/whl/torch-2.5.1+cpu.html>

### Create Conda environment (for GPU users)

    conda env create -f environment_gpu.yml
    conda activate gnn-env
    pip install torch-cluster -f <https://data.pyg.org/whl/torch-2.5.1+cu124.html>

---

## Running the Project

### Step 1: Data Setup

In the root directory create a folder:

    data

Place inside:

    BREMaster.csv
    CUMaster.csv

---

### Step 2: Configuration

You can edit default preprocessing/training/evaluation parameters inside:

    configs/config.yaml

Key parameters:

- dataset_name: BRE or CU
- merge_bre_cu: true/false
- window_size
- training parameters
- model selection

You can also edit default selected columns to filter BRE and CU dataset inside the following text files:

    configs/BRE_filtered_columns.txt
    configs/CU_filtered_columns.txt

---

### Step 3: Preprocessing

    python -m main_preprocess

Output:

    data/processed/

Files:

- arrays.npz
- timestamps.npz
- devices.json
- scaler.joblib

---

### Step 4: Training

    python -m main_train

Output:

    train_experiments/

Each run (per dataset/model) contains:

- best.pth
- last.pth
- best_config.yaml
- last_config.yaml
- metrics

---

### Step 5: Evaluation

    python -m main_evaluation

Output:

    eval_results/

Contains:

- errors
- scores
- metrics

---

### Step 6: Visualization

    python -m main_visualization

Output:

    visualization/

Contains:

- anomaly scores distributions
- learned graph

---

## Full Pipeline

    python -m main

Runs:

preprocess → train → evaluatation → visualization

---

## 🧠 Supported Models

### GDN: Graph Deviation Network for anomaly detection in multivariate time-series

#### GDN Features

- Graph-based feature relationships
- Sensor dependency modeling
- Reconstruction error-based anomaly detection

### MTAD-GAT: Multivariate Time-series Anomaly Detection with Graph Attention Network

#### MTAD-GAT Features

- Graph attention
- Temporal modeling
- Forecasting + reconstruction

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
