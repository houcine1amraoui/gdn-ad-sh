import numpy as np
import yaml
import json
import torch.optim as optim
import os

from src.preprocessing.TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import DataLoader
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.experiment import create_experiment_folder
from src.models.builder import build_model
from src.training.trainer import train
import argparse


def main_train():
    # 1. Set configuration
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    # parse CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_folder", type=str)
    parser.add_argument("--train_experiments_main_folder", type=str)
    args = parser.parse_args()

    # override processed data folder path
    if args.processed_data_folder:
        config["dataset"]["processed_data_folder"] = args.processed_data_folder

    # override experiments_folder
    if args.train_experiments_main_folder:
        config["training"]["train_experiments_main_folder"] = args.train_experiments_main_folder

    set_seed(config["seed"])
    device = get_device()
    
    processed_data_folder = config["dataset"]["processed_data_folder"]
    train_experiments_main_folder = config["training"]["train_experiments_main_folder"]

    # Create a folder if it doesn't exist
    os.makedirs(train_experiments_main_folder, exist_ok=True)

    train_experiments_sub_folder = create_experiment_folder(config, train_experiments_main_folder)

    with open(f"{processed_data_folder}/devices.json") as f:
        devices = json.load(f)
    window_size = config["dataset"]["window_size"]

    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    patience = config["training"]["patience"]

    # 2. Dataset/DataLoader creation
    train_array = np.load(f"{processed_data_folder}/train_array.npy")
    val_array = np.load(f"{processed_data_folder}/val_array.npy")
    
    train_dataset = TimeSeriesDataset(train_array, window_size)
    val_dataset = TimeSeriesDataset(val_array, window_size)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=0)

    # 3. Model Initialization
    model = build_model("mtad_gat", config)

    # 4. Train
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])
    
    train(model, train_loader, val_loader, optimizer, epochs, train_experiments_sub_folder, patience)
  
if __name__ == "__main__":
    main_train()