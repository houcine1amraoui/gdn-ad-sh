import torch

from src.models.gdn import GDN
from src.visualization.plot_learned_graph import create_graph_from_embeddings, plot_learned_graph, get_important_nodes, viz_embedding_space

window_size = 10
# Initialize GDN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_arch = GDN(
    number_nodes=94,
    in_dim=window_size,
    hid_dim=64,
    topk=15,
    heads=1
).to(device)

# create_graph_from_embeddings(model_arch)
# plot_learned_graph("graph.pkl")
# get_important_nodes("graph.pkl")
viz_embedding_space(model_arch)

