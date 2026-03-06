import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.nn import knn_graph
import networkx as nx

from src.evaluation.load_checkpoint import load_checkpoint

def plot_learned_graph(model_arch):
    # Load Chackpoint
    optimizer = optim.Adam(model_arch.parameters(), lr=1e-3)
    model, optimizer, start_epoch = load_checkpoint(model_arch, "gdn_checkpoint50.pth", optimizer)

    # Extract Sensor Embeddings From the Model
    embeddings = model.embedding.weight.detach().cpu()

    # Rebuild the Learned Graph
    topk = model.topk
    edge_index = knn_graph(embeddings, topk)

    # 4️⃣ Convert to NetworkX Graph
    G = nx.Graph()
    num_nodes = embeddings.shape[0]
    # add nodes
    for i in range(num_nodes):
        G.add_node(i)
    # add edges
    edges = edge_index.cpu().numpy()
    for src, dst in zip(edges[0], edges[1]):
        G.add_edge(int(src), int(dst))

    # OPTION 1: Plot all nodes
    # plt.figure(figsize=(10,8))
    # pos = nx.spring_layout(G)
    # nx.draw(
    #     G,
    #     pos,
    #     node_size=300,
    #     node_color="skyblue",
    #     edge_color="gray",
    #     with_labels=True
    # )
    # plt.title("Learned Sensor Dependency Graph (GDN)")
    # plt.show()

    # OPTION 2: label only important nodes, show only strongest edges
    G_small = nx.Graph()
    for src, dst in zip(edges[0], edges[1]):
        if src < dst:
            G.add_edge(src, dst)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12,10))
    nx.draw(
        G,
        pos,
        node_size=200,
        node_color="lightblue",
        edge_color="gray",
        width=0.5
    )
    plt.title("GDN Learned Sensor Graph")
    plt.show()

    
def get_important_nodes(model_arch):
    """
    Identify Important Sensors
    """
        # Load Chackpoint
    optimizer = optim.Adam(model_arch.parameters(), lr=1e-3)
    model, optimizer, start_epoch = load_checkpoint(model_arch, "gdn_checkpoint50.pth", optimizer)

    # Extract Sensor Embeddings From the Model
    embeddings = model.embedding.weight.detach().cpu()

    # Rebuild the Learned Graph
    topk = model.topk
    edge_index = knn_graph(embeddings, topk)

    # 4️⃣ Convert to NetworkX Graph
    G = nx.Graph()
    num_nodes = embeddings.shape[0]
    # add nodes
    for i in range(num_nodes):
        G.add_node(i)
    # add edges
    edges = edge_index.cpu().numpy()
    for src, dst in zip(edges[0], edges[1]):
        G.add_edge(int(src), int(dst))

    centrality = nx.degree_centrality(G)

    sorted_sensors = sorted(
        centrality.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print(sorted_sensors[:10])