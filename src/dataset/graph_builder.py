import torch
from itertools import permutations

def fully_connected_graph(node_num):
    edges = list(permutations(range(node_num), 2))
    edge_index = torch.tensor(edges).t().contiguous()
    return edge_index