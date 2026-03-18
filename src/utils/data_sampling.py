import pandas as pd
from tqdm import tqdm

def data_sampling(path, save_path):
    """
    Actor 1: 
    2022-10-18 00:00:00 → 2022-11-07 23:59:59
    2022-11-11 00:00:00 → 2022-11-17 21:59:46
    Actor 2: 
    2022-11-08 00:00:00 → 2022-11-10 23:59:59
    """

    # Load CSV
    df = pd.read_csv(path)

    # Convert timestamp column
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Define ranges to remove (inclusive)
    # 1 day actor 1, 1 day actor 2
    ranges = [
        ("2022-10-19 00:00:00", "2022-11-07 23:59:59"),
        ("2022-11-09 00:00:00", "2022-11-17 23:59:59")]

    # Build mask
    mask = pd.Series(False, index=df.index)

    for start, end in tqdm(ranges):
        mask |= df["Timestamp"].between(start, end)

    # Remove ranges
    df_truncated = df[~mask]

    # Save
    # df_truncated = df_truncated.drop(columns=["timestamp"])
    df_truncated.to_csv(save_path, index=False)

