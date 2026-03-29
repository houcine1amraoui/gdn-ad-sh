import numpy as np
import yaml
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
        
    set_seed(config["seed"])
    
    project_root_dir = config["project_root_dir"]
    dataset_folder = config["dataset"]["dataset_folder"]

    processed_data_folder = f"{project_root_dir}/{dataset_folder}/processed"

    train_experiments_main_folder = f"{project_root_dir}/train_experiments"
    # Create a folder if it doesn't exist
    os.makedirs(train_experiments_main_folder, exist_ok=True)

    model_name = config["training"]["model"]
    train_exeriments_per_model_folder = f"{train_experiments_main_folder}/{model_name}"
    # Create a folder if it doesn't exist
    os.makedirs(train_exeriments_per_model_folder, exist_ok=True)    

    train_experiments_sub_folder = create_experiment_folder(config, train_exeriments_per_model_folder)
    window_size = config["dataset"]["window_size"]
    batch_size = config["training"]["batch_size"]

    # 2. Dataset/DataLoader creation
    train_array = np.load(f"{processed_data_folder}/train_array.npy")
    val_array = np.load(f"{processed_data_folder}/val_array.npy")
    
    train_dataset = TimeSeriesDataset(train_array, window_size)
    val_dataset = TimeSeriesDataset(val_array, window_size)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=0)

    # 3. Model Initialization
    model = build_model(model_name, config)

    # Train
    train(model, train_loader, val_loader, train_experiments_sub_folder, config)
  
if __name__ == "__main__":
    main_train()