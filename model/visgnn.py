import torch
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([GCNConv(hidden_size, hidden_size) for i in range(num_layers)])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.layers:
            x = layer(x, edge_index)

        return x
