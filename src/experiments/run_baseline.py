import torch
from torch.utils.data import DataLoader, TensorDataset

from src.dataset.graph_builder import fully_connected_graph
from src.dataset.preprocessing import create_sliding_windows, reshape_for_gdn
from src.models.gdn import GDN
from src.training.trainer import train
from src.utils.device import get_device

device = get_device()
# Dummy example data (replace with IoT Garage)
data = torch.randn(1000, 10)  # 10 sensors

window_size = 12
windows = create_sliding_windows(data.numpy(), window_size)
windows = reshape_for_gdn(windows)

tensor_data = torch.tensor(windows, dtype=torch.float32)
dataset = TensorDataset(tensor_data)
dataloader = DataLoader(tensor_data, batch_size=32, shuffle=True)

node_num = windows.shape[1]
edge_index = fully_connected_graph(node_num)

model = GDN(
    edge_index_sets=[edge_index],
    node_num=node_num,
    input_dim=window_size,
    hidden_dim=64
)

train(model, dataloader, device=device)