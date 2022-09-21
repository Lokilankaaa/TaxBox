import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([GCNConv(hidden_size[i], hidden_size[i+1]) for i in range(num_layers)])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.layers:
            x = layer(x, edge_index)
            # x[:, x.shape[-1] // 2:] = F.relu(x[:, x.shape[-1] // 2:] )

        return x
