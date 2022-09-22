import torch
import clip
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class NodeEncoder(torch.nn.Module):
    def __init__(self, reduced_dims):
        super(NodeEncoder, self).__init__()
        self.clip, self.preprocess = clip.load('ViT-B/32')
        self.project_text = torch.nn.Linear(512, reduced_dims)
        self.project_image = torch.nn.Linear(512, reduced_dims)

        self.fusion_module = GATv2Conv(reduced_dims, reduced_dims, heads=8, negative_slope=0)

    def forward(self, text_embedding, img_features):
        tx = self.clip.encode_image(img_features)
        tx = F.relu(self.project_image(tx))

        ix = self.clip.encode_text(text_embedding)
        ix = F.relu(self.project_text(ix))

        nodes = torch.cat([ix, tx])
        edges = torch.Tensor([list(range(len(nodes) - 1)), [len(nodes)] * (len(nodes) - 1)])

        box_embed = self.fusion_module(x=nodes, edge_index=edges)

        return box_embed[-1]
