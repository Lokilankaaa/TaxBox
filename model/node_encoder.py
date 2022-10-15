import torch
import clip
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import numpy as np
from .module import Transformer


class NodeEncoder(torch.nn.Module):
    def __init__(self, reduced_dims, num_heads=8):
        super(NodeEncoder, self).__init__()
        self.clip, self.preprocess = clip.load('ViT-B/32')
        # self.project_text = torch.nn.Linear(512, reduced_dims, dtype=torch.float32)
        # self.project_image = torch.nn.Linear(512, reduced_dims, dtype=torch.float32)
        self.project_box = torch.nn.Sequential(
            torch.nn.Linear(512, 256, dtype=torch.float32),
            torch.nn.Linear(256, reduced_dims, dtype=torch.float32)
        )
        # self.fusion_module = GATv2Conv(reduced_dims, 2 * reduced_dims, heads=num_heads, negative_slope=0)
        self.fusion_module = Transformer(
            width=512,
            layers=6,
            heads=8,
            attn_mask=self.build_attention_mask()
        )
        self.activation = torch.nn.Sigmoid()
        self.type_embedding = torch.nn.Embedding(2, 512)
        self.to(torch.float32)
        self.init_params()

    def init_params(self):
        proj_std = (self.fusion_module.width ** -0.5) * ((2 * self.fusion_module.layers) ** -0.5)
        attn_std = self.fusion_module.width ** -0.5
        fc_std = (2 * self.fusion_module.width) ** -0.5
        for block in self.fusion_module.resblocks:
            torch.nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            torch.nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            torch.nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            torch.nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(51, 51)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text_embedding, img_features):
        b = text_embedding.shape[0]
        img_features = torch.cat(img_features.chunk(50, 1))
        tx = self.clip.encode_image(img_features).unsqueeze(0)
        tx = tx + self.type_embedding(torch.Tensor([1]).to(torch.int).to(tx.device))
        # tx = self.activation(self.project_image(tx))
        # tx = torch.cat([tx, torch.Tensor(tx.shape[0] * [[1e-6] * tx.shape[-1]]).to(tx.device)], dim=-1).to(
        #     torch.float32)

        ix = self.clip.encode_text(text_embedding).unsqueeze(1)
        ix = ix + self.type_embedding(torch.Tensor([0]).to(torch.int).to(ix.device))
        # ix = self.activation(self.project_text(ix)).unsqueeze(1)

        cat_embed = torch.cat([ix, torch.cat(tx.chunk(b, 1))], dim=1)  # N L D // B 51 128

        # nodes, edges = [], []
        # for i, (_ix, _tx) in enumerate(zip(ix, tx.chunk(b, 0))):
        #     _ix = _ix.unsqueeze(0)
        # nodes.append(torch.cat([_ix, _tx]).to(torch.float32))
        # edges.append(
        #     torch.Tensor([list(range(1 + i * 51, len(nodes[0]) + i * 51)), [i * 51] * 50]).to(nodes[0].device).type(
        #         torch.long))
        # nodes = torch.cat(nodes, dim=0)
        # edges = torch.cat(edges, dim=1)

        # box_embed = self.fusion_module(x=nodes, edge_index=edges)
        x = cat_embed  # + self.positional_embedding
        x = x.permute(1, 0, 2)
        fused_embed = self.fusion_module(x)
        fused_embed = fused_embed.permute(1, 0, 2)
        out = self.activation(self.project_box(fused_embed[:, 0, :]))

        return out


class vlTransformer(torch.nn.Module):
    def __init__(self, reduced_dims):
        super(vlTransformer, self).__init__()
        self.project_box = torch.nn.Sequential(
            torch.nn.Linear(512, 256, dtype=torch.float32),
            torch.nn.Linear(256, reduced_dims, dtype=torch.float32)
        )
        self.fusion_module = Transformer(
            width=512,
            layers=6,
            heads=8,
            attn_mask=self.build_attention_mask()
        )
        self.activation = torch.nn.Sigmoid()
        self.type_embedding = torch.nn.Embedding(2, 512)
        self.to(torch.float32)
        self.init_params()

    def init_params(self):
        proj_std = (self.fusion_module.width ** -0.5) * ((2 * self.fusion_module.layers) ** -0.5)
        attn_std = self.fusion_module.width ** -0.5
        fc_std = (2 * self.fusion_module.width) ** -0.5
        for block in self.fusion_module.resblocks:
            torch.nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            torch.nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            torch.nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            torch.nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(51, 51)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text_embedding, img_features):
        t = text_embedding + self.type_embedding(torch.Tensor([1]).to(torch.int).to(text_embedding.device))
        i = img_features + self.type_embedding(torch.Tensor([0]).to(torch.int).to(img_features.device))

        cat_embed = torch.cat([t, i], dim=1)
        x = cat_embed
        x = x.permute(1, 0, 2)
        fused_embed = self.fusion_module(x)
        fused_embed = fused_embed.permute(1, 0, 2)
        out = self.activation(self.project_box(fused_embed[:, 0, :]))

        return out


class twinTransformer(torch.nn.Module):
    def __init__(self, reduced_dims):
        super(twinTransformer, self).__init__()

        self.pos_transformer = vlTransformer(reduced_dims)
        self.neg_transformer = vlTransformer(reduced_dims)

    def forward(self, text_embedding, image_features):
        pass
