import pandas as pd
from tqdm import tqdm

def truncate_dataset(path, save_path):
    """
    Actor 1: Oct 18 → Nov 7 and Nov 11 → Nov 17
    Actor 2: Nov 08 → Nov 10
    """
    # Load CSV
    df = pd.read_csv(path)

    # Convert timestamp column
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Define ranges to remove (inclusive)
    ranges = [
        ("2022-10-19 00:00:00", "2022-11-07 23:59:59"),
        ("2022-11-09 00:00:00", "2022-11-17 23:59:59")
    ]

    # Build mask
    mask = pd.Series(False, index=df.index)

    for start, end in tqdm(ranges):
        mask |= df["Timestamp"].between(start, end)

    # Remove ranges
    df_truncated = df[~mask]

    # Save
    # df_truncated = df_truncated.drop(columns=["timestamp"])
    df_truncated.to_csv(save_path, index=False)

