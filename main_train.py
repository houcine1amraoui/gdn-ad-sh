import numpy as np
import yaml
import json
import torch.optim as optim

from src.preprocessing.TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import DataLoader
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.experiment import create_experiment_folder
from src.models.builders import build_gdn_model
from src.training.trainer import train


def main_train():
    # 1. Set configuration
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    set_seed(config["seed"])
    device = get_device()
    exp_dir = create_experiment_folder(config)
    with open("data/processed/sensors.json") as f:
        sensors = json.load(f)
    window_size = config["dataset"]["window_size"]

    # 2. Dataset/DataLoader creation
    train_array = np.load("data/processed/train_array.npy")
    dataset = TimeSeriesDataset(train_array, window_size)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 3. Model Initialization
    model = build_gdn_model(len(sensors), config, device)

    # 4. Train
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])
    train(model, loader, optimizer, config["training"]["epochs"], exp_dir, device)
  
if __name__ == "__main__":
    main_train()