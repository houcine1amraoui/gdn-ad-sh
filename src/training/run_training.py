import torch

from src.dataset.load_data import load_data
from src.dataset.splits import split_actor_periods
from src.dataset.preprocessing import normalize
from src.dataset.TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import DataLoader
from src.models.gdn import GDN
from src.training.trainer import train

# 1. Load Dataset
df, sensor_columns = load_data("data/sample/BREMaster-sample2.csv")

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

# 5. Initialize GDN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GDN(
    number_nodes=94,
    in_dim=window_size,
    hid_dim=64,
    topk=15,
    heads=1
).to(device)

# 6. Training Loop
if __name__ == "main":
    train(model, train_loader, device)