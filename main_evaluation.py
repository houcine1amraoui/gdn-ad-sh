import yaml
import numpy as np
import json
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

from src.preprocessing.TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import DataLoader
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.experiment import create_experiment_folder
from src.models.builders import build_gdn_model
from src.evaluation.load_checkpoint import load_checkpoint
from src.utils.io import save_scores
from src.evaluation.pipeline import evaluate_pipeline
from src.utils.visualization import plot_actor_comparison
from src.evaluation.anomaly import compute_errors, anomaly_score
from src.visualization.plot_scores import plot_scores

def main_evaluation():
    # 1. Set configuration
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    set_seed(config["seed"])
    device = get_device()
    exp_dir = create_experiment_folder(config)
    with open("data/processed/sensors.json") as f:
        sensors = json.load(f)
    window_size = config["dataset"]["window_size"]

    # 2. Model Initialization
    model = build_gdn_model(len(sensors), config, device)

    # 3. load checkpoint
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])
    model, optimizer, _ = load_checkpoint(
        model,
        config["checkpoint"]["path"],
        optimizer
    )

    # 4. Dataset/Loaders Create
    train_array = np.load("data/processed/train_array.npy")
    test_array_actor2 = np.load("data/processed/test_array_actor2.npy")
    train_dataset = TimeSeriesDataset(train_array, window_size)
    test_dataset_actor2 = TimeSeriesDataset(test_array_actor2, window_size)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader_actor2 = DataLoader(test_dataset_actor2, batch_size=64, shuffle=True)

    # Compute raw errors
    train_errors = compute_errors(model, dataloader=train_loader)
    test_errors_actor2 = compute_errors(model, dataloader=test_loader_actor2)

    # computer normalized anomaly score
    median = np.median(train_errors, axis=0)
    iqr = np.percentile(train_errors, 75, axis=0) - np.percentile(train_errors, 25, axis=0)
    iqr[iqr == 0] = 1e-6  # avoid division by zero

    train_score = anomaly_score(train_errors, median, iqr)
    test_score_actor2 = anomaly_score(test_errors_actor2, median, iqr)
    print(train_score)
    print(test_score_actor2)

    # Aply smoothing
    train_scores_series = pd.Series(train_score)
    smoothed_train_scores = train_scores_series.rolling(window=5).mean()
    test_score_actor2_series = pd.Series(test_score_actor2)
    smoothed_actor2_scores = test_score_actor2_series.rolling(window=5).mean()

    # Visualization
    plt.figure(figsize=(15,5))
    plt.plot(smoothed_train_scores, label="Actor 1 (Normal)")
    plt.plot(smoothed_actor2_scores, label="Actor 2 (Test)")
    plt.legend()
    plt.title("Anomaly Scores")
    plt.show()
    # save_scores(test_errors_actor1, exp_dir)
    # save_scores(test_errors_actor2, exp_dir)

    # 6. Visualization
    # plot_scores(test_errors_actor1, exp_dir)
    # plot_scores(test_errors_actor1, test_errors_actor2, exp_dir)

if __name__ == "__main__":
    main_evaluation()

