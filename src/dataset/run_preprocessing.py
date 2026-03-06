from src.dataset.truncate import truncate_dataset
from src.dataset.load_data import load_data
from src.dataset.splits import split_actor_periods
from src.dataset.preprocessing import normalize
from src.dataset.TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import DataLoader

# 0. Truncate Data
truncate_dataset("data/raw/BREMaster.csv", "data/sample/BREMaster-sample2.csv")

# 1. Load Dataset
df, sensor_columns = load_data("data/sample/BREMaster-sample2.csv")
print("Number of sensors:", len(sensor_columns))  # should be 94

# 2. Split Actor 1 and Actor 2
actor1_df, actor2_df = split_actor_periods(df)
print(actor1_df.shape, actor2_df.shape)

# 3. Normalization (CRITICAL)
train_array, test_array = normalize(actor1_df, actor2_df)

# 4. Create Data Loaders
window_size = 10
train_dataset = TimeSeriesDataset(train_array, window_size)
test_dataset = TimeSeriesDataset(test_array, window_size)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)