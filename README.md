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
    ├── data/
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

### 📌 Description

- **configs/** → Configuration files and filtered sensor columns
- **notebooks/** → Experimental and analysis notebooks
- **src/** → Core implementation (preprocessing, models, training, evaluation, utils, visualization)
- **main_preprocess.py** → Runs full preprocessing pipeline
- **main_train.py** → Runs model training
- **main_evaluation.py** → Runs evaluation and metrics generation
- **main.py** → Full pipeline execution (preprocess → train → evaluate)
- **README.md** → Project documentation

## 📁 Created folders/files After Running Full Pipeline

    gdn-ad-sh/
    │
    ├── data/
    │ └── processed/
    │ ├── BRE/
    │ ├── CU/
    │ └── merged/
    │
    ├── eval_results/
    │ ├── BRE/
    │ │ ├── errors/
    │ │ └── plots/
    │ │
    │ ├── CU/
    │ │ ├── errors/
    │ │ └── plots/
    │ │
    │ └── merged/
    │ ├── errors/
    │ └── plots/
    │
    ├── train_experiments/
    │ ├── BRE/
    │ │ ├── gdn/
    │ │ │ ├── errors/
    │ │ │ └── plots/
    │ │ │
    │ │ └── mtad-gat/
    │ │ ├── errors/
    │ │ └── plots/
    │ │
    │ ├── CU/
    │ │ ├── gdn/
    │ │ │ ├── errors/
    │ │ │ └── plots/
    │ │ │
    │ │ └── mtad-gat/
    │ │ ├── errors/
    │ │ └── plots/
    │ │
    │ └── merged/
    │ ├── gdn/
    │ │ ├── errors/
    │ │ └── plots/
    │ │
    │ └── mtad-gat/
    │ ├── errors/
    │ └── plots/

### 📌 Created Folders/Files Description

- **data/processed/** → Preprocessed data ready for training for each dataset (BRE, CU, BRE+CU merged)
- **eval_results/** → Final evaluation outputs (errors and plots) for each dataset and model (GDN, MTAD-GAT)
- **train_experiments/** → Training outputs for each dataset and model (GDN, MTAD-GAT)

## Pipeline Overview

Raw Data → Preprocessing → Processed Data → Training → Experiments →
Evaluation → Results

---

## Dataset Setup

Create a folder:

    data

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

## 🧠 Models

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

## Installation

### Clone

    git clone https://github.com/houcine1amraoui/gdn-ad-sh.git
    cd gdn-ad-sh

### Create environment

    conda env create -f environment.yml
    conda activate gnn-env

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
