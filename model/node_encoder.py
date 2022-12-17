from typing import Optional

import torch
import clip
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

from utils.utils import hard_intersection, conditional_prob, center_of
from .module import Transformer
from transformers import ViltModel, ViltConfig, BertConfig, BertModel
from utils.graph_operate import transitive_closure_mat, adj_mat
from utils.loss import adaptive_BCE
from torch import nn


class TMN(nn.Module):
    def __init__(self, l_dim, r_dim, k=5, non_linear=nn.LeakyReLU(0.2)):
        # def __init__(self, l_dim, r_dim, k=5, non_linear=nn.Tanh()):
        super(TMN, self).__init__()

        self.u = nn.Linear(k * 3, 1, bias=False)
        self.u_l = nn.Linear(k, 1, bias=False)
        self.u_r = nn.Linear(k, 1, bias=False)
        self.u_e = nn.Linear(k, 1, bias=False)
        self.f = non_linear  # if GNN/LSTM encoders are used, tanh should not, because they are not compatible
        self.W_l = nn.Bilinear(l_dim, r_dim, k, bias=True)
        self.W_r = nn.Bilinear(l_dim, r_dim, k, bias=True)
        self.W = nn.Bilinear(l_dim * 2, r_dim, k, bias=True)
        self.V_l = nn.Linear(l_dim + r_dim, k, bias=False)
        self.V_r = nn.Linear(l_dim + r_dim, k, bias=False)
        self.V = nn.Linear(l_dim * 2 + r_dim, k, bias=False)

        self.control = nn.Sequential(nn.Linear(l_dim * 2 + r_dim, l_dim * 2, bias=False), nn.Sigmoid())
        self.control_l = nn.Sequential(nn.Linear(l_dim + r_dim, l_dim, bias=False), nn.Sigmoid())
        self.control_r = nn.Sequential(nn.Linear(l_dim + r_dim, l_dim, bias=False), nn.Sigmoid())

    def forward(self, q, e1, e2):
        q = q.expand(e1.shape)
        ec1 = e1 * self.control_l(torch.cat((e1, q), -1))
        ec2 = e2 * self.control_r(torch.cat((e2, q), -1))
        e = torch.cat((e1, e2), -1)
        ec = e * self.control(torch.cat((e, q), -1))
        l = self.W_l(ec1, q) + self.V_l(torch.cat((ec1, q), -1))
        r = self.W_r(ec2, q) + self.V_r(torch.cat((ec2, q), -1))
        e = self.W(ec, q) + self.V(torch.cat((ec, q), -1))
        l_scores = self.u_l(self.f(l))
        r_scores = self.u_r(self.f(r))
        e_scores = self.u_e(self.f(e))
        scores = self.u(self.f(torch.cat((e.detach(), l.detach(), r.detach()), -1)))
        if self.training:
            return nn.Sigmoid()(scores).squeeze(-1), nn.Sigmoid()(l_scores).squeeze(-1), nn.Sigmoid()(r_scores).squeeze(
                -1)  # , nn.Sigmoid()(e_scores)
        else:
            return nn.Sigmoid()(scores), None, None


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
        # self.fusion = BertModel(self.cfg)
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
                # edge = edge_dummy
                edge[edge > n] -= 1
                fs_pairs.append(edge.type(torch.long))

                batch_nodes.append(selected_tree_nodes.unsqueeze(0))

            batch_nodes = torch.cat(batch_nodes)
            # attention_mask shape: n * n * n
            attention_mask = torch.cat(attention_mask).to(batch_nodes.device)
            # res = self.fusion(inputs_embeds=batch_nodes, attention_mask=attention_mask, return_dict=True)
            fused = torch.cat([node_feature_list.unsqueeze(1), batch_nodes],
                              dim=1)
            return fused, fs_pairs  # fused: sep node and other nodes
            # fused shape: n * n * feature_dim

        else:
            mat = transitive_closure_mat(node_tree)
            attention_mask = self.generate_mask(mat).to(node_feature_list.device)
            batch_nodes = node_feature_list.unsqueeze(0)
            # res = self.fusion(inputs_embeds=batch_nodes, attention_mask=attention_mask, return_dict=True)
            return batch_nodes  # res['last_hidden_state']


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


def InsertionScorer(q, f, s):
    # joint_fs = hard_intersection(f, s, True)
    # prob_q_given_fs = conditional_prob(q, joint_fs, True)
    prob_f_given_q = conditional_prob(f, q, True)
    prob_q_given_s = conditional_prob(q, s, True)
    # cen_q = torch.nn.functional.normalize(center_of(q, True), dim=-1)
    # cen_f = torch.nn.functional.normalize(center_of(f, True), dim=-1)
    # sim = (cen_q * cen_f).sum(-1).abs()
    return prob_f_given_q * prob_q_given_s


def AttachScorer(q, f):
    prob_f_given_q = conditional_prob(f, q, True)
    prob_q_given_f = conditional_prob(q, f, True)
    # cen_q = torch.nn.functional.normalize(center_of(q, True), dim=-1)
    # cen_f = torch.nn.functional.normalize(center_of(f, True), dim=-1)
    # cen_q = torch.nn.functional.normalize(center_of(q, True), dim=-1)
    # cen_f = torch.nn.functional.normalize(center_of(f, True), dim=-1)
    # sim = (cen_q * cen_f).sum(-1).abs()
    return prob_f_given_q  # * sim#, prob_f_given_q - prob_q_given_f


class Scorer(torch.nn.Module):
    def __init__(self, box_dim, inter_dim=20):
        super(Scorer, self).__init__()
        k = box_dim
        # self.w = torch.nn.Bilinear(box_dim, 2 * box_dim, out_dim)
        self.p0 = torch.nn.Linear(box_dim, k)
        self.p1 = torch.nn.Linear(3 * box_dim, k)
        self.p2 = torch.nn.Linear(2 * box_dim, k)
        self.p3 = torch.nn.Linear(2 * box_dim, k)

        self.w1 = torch.nn.Bilinear(k, k, k // 2)
        self.w2 = torch.nn.Bilinear(k, k, k // 2)
        self.w3 = torch.nn.Bilinear(k, k, k // 2)
        self.v1 = torch.nn.Linear(2 * k, k // 2)
        self.v2 = torch.nn.Linear(2 * k, k // 2)
        self.v3 = torch.nn.Linear(2 * k, k // 2)

        self.act = torch.nn.ReLU()
        self.classifier1 = torch.nn.Linear(3 * k // 2, 1, bias=False)
        self.classifier2 = torch.nn.Linear(k // 2, 1, bias=False)
        self.classifier3 = torch.nn.Linear(k // 2, 1, bias=False)

    def forward(self, q, f, s):
        q = q.expand(f.shape)
        # f_in_q = conditional_prob(q, f, True).unsqueeze(-1)
        # q_in_f = conditional_prob(f, q, True).unsqueeze(-1)
        # s_in_q = conditional_prob(q, s, True).unsqueeze(-1)
        # q_in_s = conditional_prob(s, q, True).unsqueeze(-1)

        # _q = self.act(self.p0(q))
        # _all = self.act(self.p1(torch.cat([q, f, s], dim=-1)))
        _f = self.act(self.p2(torch.cat([q, f], dim=-1)))
        _s = self.act(self.p3(torch.cat([s, q], dim=-1)))

        # e1 = torch.cat([_all], dim=-1)
        # e2 = torch.cat([_f], dim=-1)
        # e3 = torch.cat([_s], dim=-1)
        # _all = self.w1(_q, e1) + self.v1(torch.cat([_q, e1], dim=-1))
        _f = self.w2(_f, _s) + self.v2(torch.cat([_f, _s], dim=-1))
        # _s = self.w3(_f, e3) + self.v3(torch.cat([_q, e3], dim=-1))

        # s1 = torch.nn.Sigmoid()(self.classifier1(torch.cat([_all, _f, _s], dim=-1)))
        s2 = torch.nn.Sigmoid()(self.classifier2(_f))
        # s3 = torch.nn.Sigmoid()(self.classifier3(_s))

        # intersect = hard_intersection(q, f, True).expand(f.shape)
        # out = self.w(q, torch.cat([f, s], dim=-1)) + self.v(torch.cat([q, f, s], dim=-1))
        # f_all = self.act(self.v1(torch.cat([q, f, s], dim=-1)))
        # f_s = self.act(self.v2(torch.cat([q, s], dim=-1)))
        # f_f = self.act(self.v3(torch.cat([q, f], dim=-1)))
        #
        # out = torch.nn.Sigmoid()(self.classifier(torch.cat([out1, out2, out3], dim=-1))).squeeze(-1)
        return s2.squeeze(-1), s2.squeeze(-1), s2.squeeze(-1)


class TreeKiller(torch.nn.Module):
    def __init__(self, box_dim, hidden_size, out_dim=20):
        super(TreeKiller, self).__init__()
        self.is_training = True
        # self.box_decoder = BoxDecoder(box_dim, hidden_size)
        self.box_decoder_q = HighwayNetwork(hidden_size, box_dim, 3)
        self.box_decoder_k = HighwayNetwork(hidden_size, box_dim, 3)
        self.node_encoder = MultimodalNodeEncoder(hidden_size, 9)
        self.struct_encoder = StructuralFusionModule(hidden_size, 3)
        # self.scorer = Scorer(box_dim, out_dim if out_dim is not None else box_dim // 5)
        self.scorer = InsertionScorer
        # self.scorer = TMN(box_dim, box_dim)

    def change_mode(self):
        self.is_training = not self.is_training
        self.struct_encoder.change_mode()

    def decode_box(self, features):
        return self.box_decoder(features)

    def forward(self, node_feature_list, leaves_embeds, paired_nodes, tree):
        node_features = self.node_encoder(node_feature_list[:, 0, :], node_feature_list[:, 1:, :])
        # if len(leaves_embeds) != 0:
        #     for i, emb in leaves_embeds.items():
        #         node_features[i] = emb

        # noted that the first index of fused_features is not fused
        if self.is_training:
            fused_features, fs_pairs = self.struct_encoder(node_features, paired_nodes, tree)
        else:
            fused_features = self.struct_encoder(node_features, paired_nodes, tree)
            return fused_features

        q_box = self.box_decoder_q(fused_features[:, 0, :].unsqueeze(1))
        k_boxes = self.box_decoder_k(fused_features[:, 1:, :])
        boxes = torch.cat([q_box, k_boxes], dim=1)

        dummy_box = torch.zeros_like(q_box).to(q_box.device)
        _boxes = torch.cat([k_boxes, dummy_box], dim=1)  # boxes: n * (n+1) * d
        q = q_box  # n * 1 * d
        k = _boxes  # n * n * d

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


class TMNModel(torch.nn.Module):
    def __init__(self, box_dim, hidden_size):
        super(TMNModel, self).__init__()
        self.embedding = None
        # self.proj = HighwayNetwork(hidden_size, box_dim, 3)
        # self.proj = MultimodalNodeEncoder(hidden_size, 9)

        self.match = TMN(hidden_size, hidden_size)

    def set_embedding(self, embs):
        embs = torch.nn.functional.normalize(embs, p=2, dim=-1)
        self.embedding = torch.nn.Embedding.from_pretrained(embs, freeze=True)

    def forward(self, x, f, s):
        if self.embedding is not None:
            x = self.embedding(x)
            f = self.embedding(f)
            s = self.embedding(s)
        # x = self.proj(x[:, 0, :], x[:, 1:, :])
        # f = self.proj(f[:, 0, :], f[:, 1:, :])
        # s = self.proj(s[:, 0, :], s[:, 1:, :])
        return self.match(x, f, s)


class BoxTax(torch.nn.Module):
    def __init__(self, hidden_size, box_dim):
        super(BoxTax, self).__init__()
        self.box_decoder = HighwayNetwork(hidden_size, box_dim, 2)
        self.i_scorer = InsertionScorer
        self.a_scorer = AttachScorer

    def mul_sim(self, scores, q, p, c, i_idx):
        # scores: b * l, q,p,c: b*l*d
        cen_q = center_of(q, True)
        cen_p = center_of(p, True)
        cen_c = center_of(c, True)
        dis_qp = torch.nn.Softmax(dim=-1)(1 / torch.norm(cen_p - cen_q, p=2, dim=-1))
        dis_qc = torch.nn.Softmax(dim=-1)(1 / torch.norm(cen_c - cen_q, p=2, dim=-1))
        scores[i_idx] *= dis_qc[i_idx]
        scores *= dis_qp

        return scores

    def forward(self, x, i_idx):
        # x: b * 3 * hidden_size

        boxes = self.box_decoder(x)

        q, p, c = boxes.chunk(3, -2)
        q, p, c = q.squeeze(-2), p.squeeze(-2), c.squeeze(-2)

        i_scores = self.i_scorer(q[i_idx], p[i_idx], c[i_idx])
        a_scores = self.a_scorer(q[torch.logical_not(i_idx)], p[torch.logical_not(i_idx)])

        scores = torch.zeros(boxes.shape[:2]).to(q.device)
        scores[i_idx] += i_scores
        scores[torch.logical_not(i_idx)] += a_scores

        scores = self.mul_sim(scores, q, p, c, i_idx)

        return boxes, scores


if __name__ == "__main__":
    pass
