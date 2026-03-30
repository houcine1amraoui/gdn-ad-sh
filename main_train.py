import numpy as np
import yaml
import os

from src.utils.create_train_val_loaders import create_train_val_loaders
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
    
    # parse CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root_dir", type=str)
    parser.add_argument("--dataset_folder", type=str)
    parser.add_argument("--dataset_name", type=str)
    args = parser.parse_args()

    # override project_root_directory
    if args.project_root_dir:
        config["project_root_dir"] = args.project_root_dir

    # override data folder
    if args.dataset_folder:
        config["dataset"]["dataset_folder"] = args.dataset_folder

    project_root_dir = config["project_root_dir"]
    dataset_folder = config["dataset"]["dataset_folder"]
    dataset_name = config["dataset"]["dataset_name"]
    processed_data_folder = f"{project_root_dir}/{dataset_folder}/{dataset_name}_processed"

    train_experiments_main_folder = f"{project_root_dir}/{dataset_name}_train_experiments"
    # Create a folder if it doesn't exist
    os.makedirs(train_experiments_main_folder, exist_ok=True)

    model_name = config["training"]["model"]
    train_exeriments_per_model_folder = f"{train_experiments_main_folder}/{model_name}"
    # Create a folder if it doesn't exist
    os.makedirs(train_exeriments_per_model_folder, exist_ok=True)    

    train_experiments_sub_folder = create_experiment_folder(config, train_exeriments_per_model_folder)
    

    # 2. Dataset/DataLoader creation
    train_loader, val_loader = create_train_val_loaders(config)

    # 3. Model Initialization
    model = build_model(processed_data_folder, model_name, config)

    # 3. Start training
    train(model, train_loader, val_loader, train_experiments_sub_folder, config)
  
if __name__ == "__main__":
    main_train()