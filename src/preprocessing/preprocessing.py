import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils.load_data import load_data

def split_actor_periods(df, val_ratio=0.2):
    """
    Split dataset into:
    - actor1_train (normal training)
    - actor1_val (normal validation)
    - actor2_test (attack test)

    Validation is taken ONLY from actor1 to avoid leakage.
    """

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    actor2_start = pd.Timestamp("2022-11-08 00:00:00")
    actor2_end = pd.Timestamp("2022-11-10 23:59:59")

    actor2_mask = (df["Timestamp"] >= actor2_start) & (df["Timestamp"] < actor2_end)
    actor1_mask = ~actor2_mask

    actor1_df = df[actor1_mask].copy().sort_values("Timestamp")
    actor2_df = df[actor2_mask].copy().sort_values("Timestamp")

    # --- Time-based split for validation ---
    split_idx = int(len(actor1_df) * (1 - val_ratio))

    actor1_train_df = actor1_df.iloc[:split_idx].copy()
    actor1_val_df = actor1_df.iloc[split_idx:].copy()

    return actor1_train_df, actor1_val_df, actor2_df


def normalize(actor1_train_df, actor1_val_df, actor2_test_df, sensors):
    """
    Normalize data using ONLY actor1_train (to avoid leakage).

    Returns:
    - train_array
    - val_array
    - test_array
    - scaler
    """

    scaler = StandardScaler()

    # Fit ONLY on training data
    train_array = scaler.fit_transform(actor1_train_df[sensors].values)

    # Transform validation and test using same scaler
    val_array = scaler.transform(actor1_val_df[sensors].values)
    test_array = scaler.transform(actor2_test_df[sensors].values)

    return train_array, val_array, test_array, scaler

def data_preprocessing(path):
    # 1. Load data
    df, sensors = load_data(path)
    print("Data loding done.")

    # 2. Split actors
    actor1_train_df, actor1_val_df, actor2_df = split_actor_periods(df)
    print("Actors split done.")
    
    # 3. Normalization
    train_array, test_array_actor1, test_array_actor2, scaler = (
        normalize(actor1_train_df, actor1_val_df, actor2_df, sensors)
    )
    print("Normalization done.")
    return train_array, test_array_actor1, test_array_actor2, scaler, sensors
    """
    It is not recommended to save dataset/loader:    
        ❌ class path must be identical
        ❌ code changes break loading
        ❌ less portable
    """