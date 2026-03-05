import torch.optim as optim
import torch
import matplotlib.pyplot as plt

from src.dataset.load_data import load_data
from src.dataset.splits import split_actor_periods
from src.dataset.preprocessing import normalize
from src.dataset.TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import DataLoader
from src.models.gdn import GDN
from src.evaluation.anomaly import compute_errors
from src.evaluation.load_checkpoint import load_checkpoint

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

# 5. Initialize GDN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GDN(
    number_nodes=94,
    in_dim=window_size,
    hid_dim=64,
    topk=15,
    heads=1
).to(device)

# Load Chackpoint
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model, optimizer, start_epoch = load_checkpoint(model, "gdn_checkpoint50.pth", optimizer)

# 7. Compute Anomaly Scores
test_errors = compute_errors(model, test_loader, device)

# Visualization
plt.figure(figsize=(15,5))
plt.plot(test_errors, label="Actor 1 (Normal)")
# plt.plot(score_actor2, label="Actor 2 (Test)")
plt.legend()
plt.title("Anomaly Scores")
plt.show()