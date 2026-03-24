import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def clean_data(df):
    """
    clean data 
    CU dataset has the column "light.philips_hue_lightstrip_pid_146 " 
    with white sapce at the end
    CU dataset has the column "light.philips_hue_lightstrip_pid_146 " with all NaN
    By default, df.dropna() removes any row that has at least one NaN. (we dont want that)
    """
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1)  # or use fillna()
    return df

import pandas as pd

def split_actor_periods(df, val_ratio=0.2):
    """
    Split dataset into:
    - actor1_train (normal training from Actor 1 timeline 1 only)
    - actor1_val (normal validation from Actor 1 timeline 1 only)
    - actor2_test (test from Actor 2 timeline)

    Actor 1 timeline 2 is EXCLUDED to avoid leakage.
    """

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # --- Define periods ---
    actor1_start = pd.Timestamp("2022-10-18 00:00:00")
    actor1_end   = pd.Timestamp("2022-11-07 23:59:59")

    actor2_start = pd.Timestamp("2022-11-08 00:00:00")
    actor2_end   = pd.Timestamp("2022-11-10 23:59:59")

    # --- Masks ---
    actor1_mask = (df["Timestamp"] >= actor1_start) & (df["Timestamp"] <= actor1_end)
    actor2_mask = (df["Timestamp"] >= actor2_start) & (df["Timestamp"] <= actor2_end)

    # --- Filter ---
    actor1_df = df[actor1_mask].copy().sort_values("Timestamp")
    actor2_df = df[actor2_mask].copy().sort_values("Timestamp")

    # --- Time-based split (no shuffle!) ---
    split_idx = int(len(actor1_df) * (1 - val_ratio))

    actor1_train_df = actor1_df.iloc[:split_idx].copy()
    actor1_val_df   = actor1_df.iloc[split_idx:].copy()

    return actor1_train_df, actor1_val_df, actor2_df

def split_actor_periods_included(df, val_ratio=0.2):
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

    scaler = MinMaxScaler()

    # Fit ONLY on training data
    # .to_numpy() is safer than .values() which removes column structure
    train_array = scaler.fit_transform(actor1_train_df[sensors].to_numpy())

    # Transform validation and test using same scaler
    val_array = scaler.transform(actor1_val_df[sensors].to_numpy())
    test_array = scaler.transform(actor2_test_df[sensors].to_numpy())

    return train_array, val_array, test_array, scaler

def data_preprocessing(path):
    # 1. Load data
    df = pd.read_csv(path)
    print("Data loding done.")

    # Data Cleaning
    df = clean_data(df)
    print("Data cleaning done.")

    # Get the list of devices
    devices = [c for c in df.columns if c != "Timestamp"]

    # 2. Split actors
    actor1_train_df, actor1_val_df, actor2_df = split_actor_periods(df)
    print("Actors split done.")
    
    # 3. Normalization
    train_array, test_array_actor1, test_array_actor2, scaler = (
        normalize(actor1_train_df, actor1_val_df, actor2_df, devices)
    )
    print("Normalization done.")
    return train_array, test_array_actor1, test_array_actor2, scaler, devices
    """
    It is not recommended to save dataset/loader:    
        ❌ class path must be identical
        ❌ code changes break loading
        ❌ less portable
    """