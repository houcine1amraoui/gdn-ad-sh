import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.utils.load_data import load_data

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
    print("hhhhhhhhhh", path)
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