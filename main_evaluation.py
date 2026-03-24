import yaml
import numpy as np
import json
import torch.optim as optim

from src.preprocessing.TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import DataLoader
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.experiment import create_experiment_folder
from src.models.builders import build_gdn_model
from src.evaluation.load_checkpoint import load_checkpoint
from src.evaluation.pipeline import evaluate_pipeline
from src.evaluation.pipeline import compute_prediction_errors
from src.evaluation.pipeline import compute_anomaly_score
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

def main_evaluation():
    # 1. Set configuration
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    set_seed(config["seed"])
    device = get_device()
    

    with open("data/processed/devices.json") as f:
        devices = json.load(f)
    window_size = config["dataset"]["window_size"]

    processed_data_folder = config["dataset"]["processed_folder"]
    # experiments_folder = config["experiments_folder"]
    # exp_dir = create_experiment_folder(config)
    
    # 2. Model Initialization
    model_arch = build_gdn_model(len(devices), config, device)

    # 3. load checkpoint
    optimizer = optim.Adam(model_arch.parameters(), lr=config["training"]["lr"])
    model, optimizer, _ = load_checkpoint(
        model_arch,
        "best_model_minmax.pth",
        optimizer
    )

    # 4. Dataset/Loaders Create
    train_array = np.load(f"{processed_data_folder}/train_array.npy")
    test_array = np.load(f"{processed_data_folder}/test_array.npy")
    train_dataset = TimeSeriesDataset(train_array, window_size)
    test_dataset = TimeSeriesDataset(test_array, window_size)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # 5. Compute raw prediction errors
    # train_errors = compute_prediction_errors(model, dataloader=train_loader)
    # test_errors = compute_prediction_errors(model, dataloader=test_loader)
    # np.save("train_errors.npy", train_errors)
    # np.save("test_errors.npy", test_errors)

    # 6. Compute a global (normalized) anomaly score per timestamp
    train_errors = np.load("train_errors.npy")
    test_errors = np.load("test_errors.npy")
    train_scores, test_scores = compute_anomaly_score(train_errors, test_errors)

    plt.figure(figsize=(8, 5))
    plt.hist(train_scores, bins=100, alpha=0.6, label="Train", density=True)
    plt.hist(test_scores, bins=100, alpha=0.6, label="Test", density=True)
    plt.legend()
    plt.title("Score Distribution (Train vs Test)")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.show()

    print("Train mean:", train_scores.mean())
    print("Test mean:", test_scores.mean())

    print("Train 95th percentile:", np.percentile(train_scores, 95))
    print("Test 95th percentile:", np.percentile(test_scores, 95))

    # # Visualizae scores distribution
    # plt.figure(figsize=(15,5))
    # plt.plot(train_scores, label="Actor 1 (Normal)")
    # plt.plot(test_scores, label="Actor 2 (Test)")
    # plt.legend()
    # plt.title("Anomaly Scores")
    # plt.show()

    # 
    # stat, p = ks_2samp(train_scores, test_scores)
    # print("KS statistic:", stat)
    # print("p-value:", p)
    # # 6. Visualization
    # # plot_scores(test_errors_actor1, exp_dir)
    # # plot_scores(test_errors_actor1, test_errors_actor2, exp_dir)

if __name__ == "__main__":
    main_evaluation()

