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
    df.set_index('Timestamp', inplace=True)
    df = df.dropna(axis=1)  # or use fillna()
    return df

def downsample_data(df, freq="3s"):
    df_downsampled = df.resample(freq).mean()
    return df_downsampled

def split_actor_periods(df, val_ratio=0.2):
    """
    Split dataset into:
    - actor1_train (normal training from Actor 1 timeline 1 only)
    - actor1_val (normal validation from Actor 1 timeline 1 only)
    - actor2_test (test from Actor 2 timeline)

    Actor 1 timeline 2 is created into seperate test set
    """

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # --- Define ranges ---
    actor1_t1_start = "2022-10-18 00:00:00"
    actor1_t1_end   = "2022-11-07 23:59:59"

    actor2_start    = "2022-11-08 00:00:00"
    actor2_end      = "2022-11-10 23:59:59"

    actor1_t2_start = "2022-11-11 00:00:00"
    actor1_t2_end   = "2022-11-17 23:59:59"

    # --- Masks ---
    actor1_t1 = df[(df["Timestamp"] >= actor1_t1_start) & (df["Timestamp"] <= actor1_t1_end)]
    actor2    = df[(df["Timestamp"] >= actor2_start) & (df["Timestamp"] <= actor2_end)]
    actor1_t2 = df[(df["Timestamp"] >= actor1_t2_start) & (df["Timestamp"] <= actor1_t2_end)]

    actor1_t1 = actor1_t1.sort_values("Timestamp")
    actor2_test_df    = actor2.sort_values("Timestamp")
    actor1_test_df = actor1_t2.sort_values("Timestamp")

    # --- Train/Val split ---
    split_idx = int(len(actor1_t1) * (1 - val_ratio))

    train_df = actor1_t1.iloc[:split_idx]
    val_df   = actor1_t1.iloc[split_idx:]

    return train_df, val_df, actor2_test_df, actor1_test_df

def normalize(train_df, val_df, actor2_test_df, actor1_test_df, devices):
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
    train_array = scaler.fit_transform(train_df[devices].to_numpy())

    # Transform validation and test using same scaler
    val_array = scaler.transform(val_df[devices].to_numpy())
    actor2_test_array = scaler.transform(actor2_test_df[devices].to_numpy())
    actor1_test_array = scaler.transform(actor1_test_df[devices].to_numpy())

    return train_array, val_array, actor2_test_array, actor1_test_array, scaler

def data_preprocessing(raw_data_path):
    # 1. Load data
    df = pd.read_csv(raw_data_path)
    print("Data loding done.")

    # Data Cleaning
    df = clean_data(df)
    print("Data cleaning done.")

    # Get the list of devices
    devices = [c for c in df.columns if c != "Timestamp"]

    # Data downsampling
    df = downsample_data(df)

    # 2. Split actors
    train_df, val_df, actor2_test_df, actor1_test_df = split_actor_periods(df)
    print("Actors split done.")
    
    # 3. Normalization
    train_array, val_array, actor2_test_array, actor1_test_array, scaler = (
        normalize(train_df, val_df, actor2_test_df, actor1_test_df, devices)
    )
    print("Normalization done.")

    return train_array, val_array, actor2_test_array, actor1_test_array, scaler, devices
    """
    It is not recommended to save dataset/loader:    
        ❌ class path must be identical
        ❌ code changes break loading
        ❌ less portable
    """