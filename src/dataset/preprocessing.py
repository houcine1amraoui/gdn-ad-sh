import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.dataset.load_data import load_data
from src.dataset.TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import DataLoader

def split_actor_periods(df):
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    actor2_start = pd.Timestamp("2022-11-08 00:00:00")
    actor2_end = pd.Timestamp("2022-11-10 23:59:59")
    actor2_mask = (df["Timestamp"] >= actor2_start) & (df["Timestamp"] < actor2_end)
    actor1_mask = ~actor2_mask
    actor1_df = df[actor1_mask].copy()
    actor2_df = df[actor2_mask].copy()
    return actor1_df, actor2_df

def normalize(actor1_df, actor2_df):
    sensor_columns = [c for c in actor1_df.columns if c != "Timestamp"]
    scaler = StandardScaler()
    train_array = scaler.fit_transform(actor1_df[sensor_columns].values)
    test_array_actor1 = scaler.transform(actor1_df[sensor_columns].values)
    test_array_actor2 = scaler.transform(actor2_df[sensor_columns].values)
    return train_array, test_array_actor1, test_array_actor2

def prepare_data(path):
    df, sensor_columns = load_data(path)
    actor1_df, actor2_df = split_actor_periods(df)
    train_array, test_array_actor1, test_array_actor2 = normalize(actor1_df, actor2_df)
    return train_array, test_array_actor1, test_array_actor2, sensor_columns

def create_dataloaders(train_array, test_array_actor1, test_array_actor2, window_size):
    train_dataset = TimeSeriesDataset(train_array, window_size)
    test_dataset_actor1 = TimeSeriesDataset(test_array_actor1, window_size)
    test_dataset_actor2 = TimeSeriesDataset(test_array_actor2, window_size)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader_actor1 = DataLoader(test_dataset_actor1, batch_size=64, shuffle=False)
    test_loader_actor2 = DataLoader(test_dataset_actor2, batch_size=64, shuffle=False)
    return train_loader, test_loader_actor1, test_loader_actor2

