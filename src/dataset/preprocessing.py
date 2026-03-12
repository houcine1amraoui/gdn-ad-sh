import pandas as pd
import numpy as np
import joblib
import json

from sklearn.preprocessing import StandardScaler
from src.dataset.load_data import load_data
from src.dataset.TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import DataLoader

def split_actor_periods(df):
    """
    We split based on timestamp boundary between actors.
    """
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    actor2_start = pd.Timestamp("2022-11-08 00:00:00")
    actor2_end = pd.Timestamp("2022-11-10 23:59:59")
    actor2_mask = (df["Timestamp"] >= actor2_start) & (df["Timestamp"] < actor2_end)
    actor1_mask = ~actor2_mask
    actor1_df = df[actor1_mask].copy()
    actor2_df = df[actor2_mask].copy()
    return actor1_df, actor2_df

def normalize(actor1_df, actor2_df, sensors):
    """
    We normalize using ONLY Actor 1.
    This prevents data leakage.
    """
    scaler = StandardScaler()
    train_array = scaler.fit_transform(actor1_df[sensors].values)
    test_array_actor1 = scaler.transform(actor1_df[sensors].values)
    test_array_actor2 = scaler.transform(actor2_df[sensors].values)
    return train_array, test_array_actor1, test_array_actor2, scaler

def data_preprocessing(path):
    # 1. Load data
    df, sensors = load_data(path)
    # 2. Split actors
    actor1_df, actor2_df = split_actor_periods(df)
    # 3. Normalization
    train_array, test_array_actor1, test_array_actor2, scaler = normalize(actor1_df, 
                                                                          actor2_df, sensors)
    return train_array, test_array_actor1, test_array_actor2, scaler, sensors
    """
    It is not recommended to save dataset/loader:    
        ❌ class path must be identical
        ❌ code changes break loading
        ❌ less portable
    """

def create_dataloaders(window_size):
    """
    The dataset alone only describes how to access one sample.
    A DataLoader is a PyTorch utility that:
        - Reads data from a Dataset
        - Groups samples into mini-batches
        - Shuffles data if needed
        - Loads data efficiently (parallel workers)
    The DataLoader manages how samples are delivered during training.
    """
    train_array = np.load("data/processed/train.npy")
    test_array_actor1 = np.load("data/processed/test_actor1.npy")
    test_array_actor2 = np.load("data/processed/test_actor2.npy")
    train_dataset = TimeSeriesDataset(train_array, window_size)
    test_dataset_actor1 = TimeSeriesDataset(test_array_actor1, window_size)
    test_dataset_actor2 = TimeSeriesDataset(test_array_actor2, window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader_actor1 = DataLoader(test_dataset_actor1, batch_size=64, shuffle=False)
    test_loader_actor2 = DataLoader(test_dataset_actor2, batch_size=64, shuffle=False)
    return train_loader, test_loader_actor1, test_loader_actor2