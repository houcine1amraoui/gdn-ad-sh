import yaml
import numpy as np
import json

from src.dataset.TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import DataLoader
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.experiment import create_experiment_folder
from src.models.builders import build_gdn_model
from src.evaluation.load_checkpoint import load_checkpoint
from src.utils.io import save_scores
from src.evaluation.pipeline import evaluate_pipeline
from src.utils.visualization import plot_actor_comparison
from src.evaluation.anomaly import compute_errors
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
    model, optimizer, _ = load_checkpoint(
        model,
        config["checkpoint"]["path"],
        optimizer
    )

    # 4. Dataset/Loaders Create
    test_array_actor1 = np.load("data/processed/test_array_actor1.npy")
    test_array_actor2 = np.load("data/processed/test_array_actor2.npy")
    test_dataset_actor1 = TimeSeriesDataset(test_array_actor1, window_size)
    test_dataset_actor2 = TimeSeriesDataset(test_array_actor2, window_size)
    test_loader_actor1 = DataLoader(test_dataset_actor1, batch_size=64, shuffle=True)
    test_loader_actor2 = DataLoader(test_dataset_actor2, batch_size=64, shuffle=True)

    test_errors_actor1 = compute_errors(model, test_loader_actor1, device)
    test_errors_actor2 = compute_errors(model, test_loader_actor2, device)
    save_scores(test_errors_actor1, exp_dir)
    save_scores(test_errors_actor2, exp_dir)

    # 6. Visualization
    # plot_scores(test_errors_actor1, exp_dir)
    # plot_scores(test_errors_actor1, test_errors_actor2, exp_dir)

if __name__ == "main":
    main_evaluation()

