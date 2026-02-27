import torch
import torch.nn as nn
import math

from src.models.graph_layer import GraphLayer


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i * edge_num:(i + 1) * edge_num] += i * node_num

    return batch_edge_index.long()


class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1):
        super().__init__()
        self.gnn = GraphLayer(
            in_channel,
            out_channel,
            inter_dim=inter_dim,
            heads=heads,
            concat=False
        )
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, embedding, node_num):
        out, _ = self.gnn(
            x,
            edge_index,
            embedding,
            return_attention_weights=True
        )
        out = self.bn(out)
        return self.relu(out)


class GDN(nn.Module):
    def __init__(
        self,
        edge_index_sets,
        node_num,
        input_dim,
        hidden_dim=64,
        out_layer_inter_dim=256,
        out_layer_num=1,
        topk=20,
    ):
        super().__init__()

        self.edge_index_sets = edge_index_sets
        self.node_num = node_num
        self.topk = topk

        embed_dim = hidden_dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_out = nn.BatchNorm1d(embed_dim)

        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, hidden_dim, inter_dim=hidden_dim + embed_dim)
            for _ in edge_index_sets
        ])

        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim * len(edge_index_sets), out_layer_inter_dim),
            nn.ReLU(),
            nn.Linear(out_layer_inter_dim, 1)
        )

        self.dropout = nn.Dropout(0.2)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, x):
        device = x.device
        batch_size, node_num, feat_dim = x.shape
        x = x.view(-1, feat_dim)

        gnn_outputs = []

        for edge_index in self.edge_index_sets:
            batch_edge_index = get_batch_edge_index(
                edge_index,
                batch_size,
                node_num
            ).to(device)

            node_embeddings = self.embedding(
                torch.arange(node_num).to(device)
            )
            node_embeddings = node_embeddings.repeat(batch_size, 1)

            out = self.gnn_layers[0](
                x,
                batch_edge_index,
                embedding=node_embeddings,
                node_num=batch_size * node_num
            )

            gnn_outputs.append(out)

        x = torch.cat(gnn_outputs, dim=1)
        x = x.view(batch_size, node_num, -1)

        x = self.dropout(x)
        out = self.out_layer(x)

        return out.squeeze(-1)