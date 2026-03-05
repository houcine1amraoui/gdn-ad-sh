import pandas as pd
from src.dataset.splits import split_actor_periods
import matplotlib.pyplot as plt
from exploration.exploration import average_sensor_activation_per_day, corr_heatmap

# 1. Load Dataset
df = pd.read_csv("data/sample/BREMaster-sample2.csv")

# 2. Split Actor 1 and Actor 2
actor1_df, actor2_df = split_actor_periods(df)

# 3. Plot sensor activation
average_sensor_activation_per_day(actor1_df)
average_sensor_activation_per_day(actor2_df)

# 4. Correlation Heatmap (1 Day Per Actor)
corr_heatmap(actor1_df, "2022-10-18 00:00:00", "2022-10-18 23:59:59")
corr_heatmap(actor2_df, "2022-11-08 00:00:00", "2022-11-08 23:59:59")
