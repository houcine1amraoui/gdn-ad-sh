import torch

from src.dataset.load_data import load_data
from src.models.gdn import GDN
from src.evaluation.load_checkpoint import load_checkpoint
from src.visualization.plot_learned_graph import plot_learned_graph, get_important_nodes

# 1. Load Dataset
# df, sensor_columns = load_data("data/sample/BREMaster-sample2.csv")
# print("Number of sensors:", len(sensor_columns))  # should be 94

window_size = 10
# 5. Initialize GDN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_arch = GDN(
    number_nodes=94,
    in_dim=window_size,
    hid_dim=64,
    topk=15,
    heads=1
).to(device)

# plot_learned_graph(model_arch)
get_important_nodes(model_arch)
