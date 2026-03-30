import numpy as np
from torch.utils.data import DataLoader

from src.preprocessing.TimeSeriesDataset import TimeSeriesDataset

def create_train_val_loaders(config):
    project_root_dir = config["project_root_dir"]
    dataset_folder = config["dataset"]["dataset_folder"]
    dataset_name = config["dataset"]["dataset_name"]
    merge_bre_cu = config["dataset"]["merge_bre_cu"]

    name = ""
    if merge_bre_cu: name = "merged"
    else: name = dataset_name
    
    processed_data_folder = f"{project_root_dir}/{dataset_folder}/processed/{name}"

    window_size = config["training"]["window_size"]
    batch_size = config["training"]["batch_size"]

    train_array = np.load(f"{processed_data_folder}/train_array.npy")
    val_array = np.load(f"{processed_data_folder}/val_array.npy")

    train_loader, val_loader = create_train_val_loaders(config)
    train_dataset = TimeSeriesDataset(train_array, window_size)
    val_dataset = TimeSeriesDataset(val_array, window_size)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader

