import numpy as np
from torch.utils.data import DataLoader

from src.preprocessing.TimeSeriesDataset import TimeSeriesDataset
from src.utils.get_folders_utils import get_processed_folder
    
def create_train_val_loaders(config):
    processed_data_folder = get_processed_folder(config)
    
    window_size = config["training"]["window_size"]
    batch_size = config["training"]["batch_size"]

    train_array = np.load(f"{processed_data_folder}/train_array.npy")
    val_array = np.load(f"{processed_data_folder}/val_array.npy")

    train_dataset = TimeSeriesDataset(train_array, window_size)
    val_dataset = TimeSeriesDataset(val_array, window_size)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader

