import torch
import clip
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import numpy as np


class NodeEncoder(torch.nn.Module):
    def __init__(self, reduced_dims):
        super(NodeEncoder, self).__init__()
        self.clip, self.preprocess = clip.load('ViT-B/32')
        self.project_text = torch.nn.Linear(512, reduced_dims, dtype=torch.float32)
        self.project_image = torch.nn.Linear(512, reduced_dims, dtype=torch.float32)
        self.project_box = torch.nn.Linear(512 * 8, reduced_dims * 2, dtype=torch.float32)

        self.fusion_module = GATv2Conv(2 * reduced_dims, 2 * reduced_dims, heads=8, negative_slope=0)
        self.to(torch.float32)

    def forward(self, text_embedding, img_features):
        b = img_features.shape[0]
        img_features = torch.cat(img_features.chunk(50, 1))
        tx = self.clip.encode_image(img_features)
        tx = F.relu(self.project_image(tx))
        tx = torch.cat([tx, torch.Tensor(tx.shape[0] * [[1e-6] * tx.shape[-1]]).to(tx.device)], dim=-1).to(
            torch.float32)

        ix = self.clip.encode_text(text_embedding)
        ix = F.relu(self.project_text(ix))

        nodes, edges = [], []
        for i, (_ix, _tx) in enumerate(zip(ix, tx.chunk(b, 0))):
            _ix = _ix.unsqueeze(0)
            _ix = torch.cat([_ix, torch.from_numpy(
                np.random.multivariate_normal(np.zeros(256),
                                              np.cov(_tx.transpose(1, 0).chunk(2)[0].detach().cpu().numpy()))).to(
                ix.device).unsqueeze(0)], dim=-1).to(torch.float32)
            nodes.append(torch.cat([_ix, _tx]).to(torch.float32))
            edges.append(
                torch.Tensor([list(range(1 + i * 51, len(nodes[0]) + i * 51)), [i * 51] * 50]).to(nodes[0].device).type(
                    torch.long))
        nodes = torch.cat(nodes, dim=0)
        edges = torch.cat(edges, dim=1)

        box_embed = F.relu(self.fusion_module(x=nodes, edge_index=edges))

        out = F.relu(self.project_box(box_embed[::51]))

        return out
