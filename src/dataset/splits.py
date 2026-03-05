import pandas as pd

def split_actor_periods(df):
    """
    Actor 1: 
    2022-10-18 00:00:00 → 2022-11-07 00:00:00
    2022-11-11 00:00:00 → 2022-11-17 21:59:46
    Actor 2: 
    2022-11-08 00:00:00 → 2022-11-10 00:00:00
    """

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    actor2_start = pd.Timestamp("2022-11-08")
    actor2_end = pd.Timestamp("2022-11-11")

    actor2_mask = (df["Timestamp"] >= actor2_start) & (df["Timestamp"] < actor2_end)
    actor1_mask = ~actor2_mask

    actor1_df = df[actor1_mask].copy()
    actor2_df = df[actor2_mask].copy()

    return actor1_df, actor2_df
