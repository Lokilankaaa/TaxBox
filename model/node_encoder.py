from typing import Optional

import torch
import clip
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

from utils.utils import hard_intersection
from .module import Transformer
from transformers import ViltModel, ViltConfig, BertConfig, BertModel
from utils.graph_operate import transitive_closure_mat, adj_mat
from utils.loss import adaptive_BCE
from torch import nn


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
    def __init__(self, reduced_dims, seq_len):
        super(vlTransformer, self).__init__()
        self.project_box = torch.nn.Sequential(
            torch.nn.Linear(512, 256, dtype=torch.float32),
            torch.nn.Linear(256, reduced_dims, dtype=torch.float32)
        )
        self.fusion_module = Transformer(
            width=512,
            layers=6,
            heads=8,
            attn_mask=self.build_attention_mask(seq_len)
        )
        self.activation = torch.nn.Sigmoid()
        self.type_embedding = torch.nn.Embedding(2, 512)

        self.init_params()
        self.to(torch.float32)

    def init_params(self, init_clip=True):
        proj_std = (self.fusion_module.width ** -0.5) * ((2 * self.fusion_module.layers) ** -0.5)
        attn_std = self.fusion_module.width ** -0.5
        fc_std = (2 * self.fusion_module.width) ** -0.5
        for block in self.fusion_module.resblocks:
            torch.nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            torch.nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            torch.nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            torch.nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if init_clip:
            _clip, _ = clip.load('ViT-B/32')
            for m, c in zip(self.fusion_module.parameters(), _clip.transformer.parameters()):
                m.data = c.data
            del _clip

    def build_attention_mask(self, seq_len):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(seq_len, seq_len)
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
    def __init__(self, reduced_dims, seq_len):
        super(twinTransformer, self).__init__()
        self.box_dim = reduced_dims
        self.query_transformer = vlTransformer(reduced_dims, seq_len)
        self.key_transformer = vlTransformer(reduced_dims, seq_len)
        self._init_params()

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        for q, k in zip(self.query_transformer.parameters(), self.key_transformer.parameters()):
            k.data = k.data * m + (1. - m) * q.data

    def _init_params(self):
        for param_q, param_k in zip(self.query_transformer.parameters(), self.key_transformer.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def forward(self, text_embedding, image_features, m):
        q = self.query_transformer(text_embedding, image_features)
        with torch.no_grad():
            self._update_momentum_encoder(m)
            k = self.key_transformer(text_embedding, image_features)

        return q, k


class MultimodalNodeEncoder(torch.nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(MultimodalNodeEncoder, self).__init__()
        self.config = ViltConfig(hidden_size=hidden_size, num_attention_heads=hidden_size // 64,
                                 num_hidden_layers=num_layers)
        self.vilt = ViltModel(self.config)
        self.cls_token = torch.nn.Parameter((torch.zeros(1, 1, 512)))

    def forward(self, text_embeds, imgs_embeds):
        cls_token = self.cls_token.expand(text_embeds.shape[0], -1, -1)
        img_masks = torch.ones((text_embeds.shape[0], 1, imgs_embeds.shape[1] + 1)).to(text_embeds.device)
        text_embeds = torch.cat([cls_token, text_embeds.unsqueeze(1)], dim=1)
        imgs_embeds = torch.cat([cls_token, imgs_embeds], dim=1)
        res = self.vilt(inputs_embeds=text_embeds, image_embeds=imgs_embeds, return_dict=True,
                        pixel_mask=img_masks)
        return res['pooler_output']


class StructuralFusionModule(torch.nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(StructuralFusionModule, self).__init__()
        self.cfg = BertConfig(hidden_size=hidden_size, num_attention_heads=hidden_size // 64,
                              num_hidden_layers=num_layers, max_position_embeddings=512)
        self.fusion = BertModel(self.cfg)
        self.is_training = True

    def change_mode(self):
        self.is_training = not self.is_training

    def generate_mask(self, mat):
        return torch.Tensor(mat).unsqueeze(0)

    def forward(self, node_feature_list, paired_nodes, node_tree):
        # generate matching tree
        # node_feature_list: n * feature_dim

        attention_mask = []
        batch_nodes = []
        fs_pairs = []
        if self.is_training:
            # construct batch, the order is same as paired node
            for n in paired_nodes:
                _c_tree = deepcopy(node_tree)
                f = list(_c_tree.predecessors(n))
                if len(f) != 0:
                    for s in _c_tree.successors(n):
                        _c_tree.add_edge(f[0], s)
                _c_tree.remove_node(n)
                selected_tree_nodes = torch.cat(
                    [node_feature_list[:n, ], node_feature_list[n + 1:, :]])
                mat = transitive_closure_mat(_c_tree)  # (n-1) * (n-1)
                attention_mask.append(self.generate_mask(mat))

                edge = torch.Tensor(list(_c_tree.edges()))
                edge_dummy = torch.Tensor(list([[n, -1] for n in _c_tree.nodes()]))
                edge = torch.cat([edge, edge_dummy])
                edge[edge > n] -= 1
                fs_pairs.append(edge.type(torch.long))

                batch_nodes.append(selected_tree_nodes.unsqueeze(0))

            batch_nodes = torch.cat(batch_nodes)
            # attention_mask shape: n * n * n
            attention_mask = torch.cat(attention_mask).to(batch_nodes.device)
            res = self.fusion(inputs_embeds=batch_nodes, attention_mask=attention_mask, return_dict=True)
            fused = torch.cat([node_feature_list.unsqueeze(1), res['last_hidden_state']],
                              dim=1)
            return fused, fs_pairs  # fused: sep node and other nodes
            # fused shape: n * n * feature_dim

        else:
            mat = transitive_closure_mat(node_tree)
            attention_mask = self.generate_mask(mat).to(node_feature_list.device)
            batch_nodes = node_feature_list.unsqueeze(0)
            res = self.fusion(inputs_embeds=batch_nodes, attention_mask=attention_mask, return_dict=True)
            return res['last_hidden_state']


class HighwayNetwork(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 n_layers: int,
                 activation: Optional[nn.Module] = None):
        super(HighwayNetwork, self).__init__()
        self.n_layers = n_layers
        self.nonlinear = nn.ModuleList(
            [nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
        self.gate = nn.ModuleList(
            [nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
        for layer in self.gate:
            layer.bias = torch.nn.Parameter(0. * torch.ones_like(layer.bias))
        self.final_linear_layer = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU() if activation is None else activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer_idx in range(self.n_layers):
            gate_values = self.sigmoid(self.gate[layer_idx](inputs))
            nonlinear = self.activation(self.nonlinear[layer_idx](inputs))
            inputs = gate_values * nonlinear + (1. - gate_values) * inputs
        return self.final_linear_layer(inputs)


class BoxDecoder(torch.nn.Module):
    def __init__(self, box_dim, hidden_size):
        super(BoxDecoder, self).__init__()
        self.box_dim = box_dim
        self.project_box = [
            torch.nn.Linear(hidden_size, hidden_size // 2, dtype=torch.float32),
            torch.nn.Linear(hidden_size // 2, self.box_dim, dtype=torch.float32)
        ]
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        for module in self.project_box:
            x = self.act(module(x))

        return x


class Scorer(torch.nn.Module):
    def __init__(self, box_dim, out_dim):
        super(Scorer, self).__init__()
        # self.w = torch.nn.Bilinear(box_dim, 2 * box_dim, out_dim)
        self.v = torch.nn.Linear(3 * box_dim, out_dim)
        self.v2 = torch.nn.Linear(2 * box_dim, out_dim)
        self.v3 = torch.nn.Linear(2 * box_dim, out_dim)
        self.act = torch.nn.ReLU()
        self.classifier = torch.nn.Linear(3 * out_dim, 1, bias=False)

    def forward(self, q, f, s):
        q = q.expand(f.shape)
        # intersect = hard_intersection(q, f, True).expand(f.shape)
        # out = self.w(q, torch.cat([f, s], dim=-1)) + self.v(torch.cat([q, f, s], dim=-1))
        out1 = self.act(self.v(torch.cat([q, f, s], dim=-1)))
        out2 = self.act(self.v2(torch.cat([q, s], dim=-1)))
        out3 = self.act(self.v3(torch.cat([f, q], dim=-1)))
        out = torch.nn.Sigmoid()(self.classifier(torch.cat([out1, out2, out3], dim=-1))).squeeze(-1)
        return out


class TreeKiller(torch.nn.Module):
    def __init__(self, box_dim, hidden_size, out_dim=None):
        super(TreeKiller, self).__init__()
        self.is_training = True
        # self.box_decoder = BoxDecoder(box_dim, hidden_size)
        self.box_decoder = HighwayNetwork(hidden_size, box_dim, 3)
        self.node_encoder = MultimodalNodeEncoder(hidden_size, 3)
        self.struct_encoder = StructuralFusionModule(hidden_size, 6)
        self.scorer = Scorer(box_dim, out_dim if out_dim is not None else box_dim // 4)

    def change_mode(self):
        self.is_training = not self.is_training
        self.struct_encoder.change_mode()

    def decode_box(self, features):
        return self.box_decoder(features)

    def forward(self, node_feature_list, leaves_embeds, paired_nodes, tree):
        node_features = self.node_encoder(node_feature_list[:, 0, :], node_feature_list[:, 1:, :])
        if len(leaves_embeds) != 0:
            for i, emb in leaves_embeds.items():
                node_features[i] = emb

        # noted that the first index of fused_features is not fused
        if self.is_training:
            fused_features, fs_pairs = self.struct_encoder(node_features, paired_nodes, tree)
        else:
            fused_features = self.struct_encoder(node_features, paired_nodes, tree)
            return fused_features

        boxes = self.box_decoder(fused_features)

        dummy_box = torch.zeros(boxes.shape[0], 1, boxes.shape[-1]).to(boxes.device)
        _boxes = torch.cat([boxes, dummy_box], dim=1)  # boxes: n * (n+1) * d
        q = _boxes[:, 0, :].unsqueeze(1)  # n * 1 * d
        k = _boxes[:, 1:, :]  # n * n * d

        ignore = -1
        for i, p in enumerate(paired_nodes):
            if len(list(tree.predecessors(p))) == 0:
                ignore = i
                break

        fs_pairs = fs_pairs[:ignore] + fs_pairs[ignore + 1:]
        q = torch.cat([q[:ignore], q[ignore + 1:]])
        k = torch.cat([k[:ignore], k[ignore + 1:]])

        f = torch.cat([b[i[:, 0], :].unsqueeze(0) for b, i in zip(k, fs_pairs)])
        s = torch.cat([b[i[:, 1], :].unsqueeze(0) for b, i in zip(k, fs_pairs)])

        scores = self.scorer(q, f, s)

        return boxes, scores, fs_pairs


if __name__ == "__main__":
    pass
