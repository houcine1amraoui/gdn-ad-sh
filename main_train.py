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
import argparse


def main_train():
    # 1. Set configuration
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    # parse CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--processed_folder", type=str)
    parser.add_argument("--exp_dir", type=str)
    args = parser.parse_args()

    # override dataset path
    if args.data_path:
        config["dataset"]["path"] = args.data_path

    # override processed data folder path
    if args.processed_folder:
        config["dataset"]["processed_folder"] = args.processed_folder

    # override processed data folder path
    if args.processed_folder:
        config["dataset"]["exp_dir"] = args.exp_dir

    set_seed(config["seed"])
    device = get_device()
    exp_dir = create_experiment_folder(config)
    processed_data_folder = config["dataset"]["processed_folder"]
    with open(f"{processed_data_folder}/devices.json") as f:
        devices = json.load(f)
    window_size = config["dataset"]["window_size"]

    # 2. Dataset/DataLoader creation
    train_array = np.load(f"{processed_data_folder}/train_array.npy")
    val_array = np.load(f"{processed_data_folder}/val_array.npy")

    train_dataset = TimeSeriesDataset(train_array, window_size)
    # train_loader = DataLoader(train_dataset)
    
    val_dataset = TimeSeriesDataset(val_array, window_size)
    # val_loader = DataLoader(val_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    # 3. Model Initialization
    model = build_gdn_model(len(devices), config, device)

    # 4. Train
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])
    epochs = config["training"]["epochs"]
    train(model, train_loader, val_loader, optimizer, epochs, exp_dir, device)
    # train(model, train_loader, optimizer, epochs, exp_dir, device)
  
if __name__ == "__main__":
    main_train()