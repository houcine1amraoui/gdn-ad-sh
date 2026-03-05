import pandas as pd
import numpy as np
from src.dataset.truncate import truncate_dataset
from src.dataset.splits import split_actor_periods

# 0. Truncate Data
# truncate_dataset("data/raw/BREMaster.csv", "data/sample/BREMaster-sample2.csv")

# 1. Load Dataset
df = pd.read_csv("data/sample/BREMaster-sample2.csv")
sensor_columns = [c for c in df.columns if c != "Timestamp"]
print("Number of sensors:", len(sensor_columns))  # should be 94

# # 2. Split Actor 1 and Actor 2
actor1_df, actor2_df = split_actor_periods(df)
print(actor1_df.shape)
print(actor2_df.shape)