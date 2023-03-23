from typing import Optional

import torch
import clip
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

from utils.utils import hard_intersection, conditional_prob, center_of
from .module import Transformer
from transformers import ViltModel, ViltConfig, BertConfig, BertModel, RobertaModel
from transformers import AutoTokenizer, CLIPTextModel
from utils.graph_operate import transitive_closure_mat, adj_mat
from utils.loss import adaptive_BCE
from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv
import torch_geometric.utils as gutils
from torch_geometric.data import Data, Batch
from itertools import chain


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

    return prob_f_given_q, prob_q_given_s


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


class TaxBox(torch.nn.Module):
    def __init__(self, hidden_size, box_dim, graph_embed=True):
        super(TaxBox, self).__init__()
        self.graph_embed = graph_embed

        if graph_embed:
            self.fusion_module = nn.ModuleList([
                GATv2Conv(hidden_size, hidden_size, dropout=0.1, heads=4, concat=False),
            ])
        self.activation = nn.LeakyReLU(0.1)
        self.box_decoder_k = HighwayNetwork(hidden_size, box_dim, 1)
        self.box_decoder_q = HighwayNetwork(hidden_size, box_dim, 4)
        self.scorer = InsertionScorer
        self.device = None
        # self.a_scorer = AttachScorer

    def set_device(self, device):
        self.device = device

    def mul_sim(self, scores, q, p, c, i_idx):
        # scores: 2 * b * l, q,p,c: b*l*d
        s1, s2 = scores

        s1 = s1.squeeze(-1)
        s2 = s2.squeeze(-1)

        if s1.dim() == 1:
            s1 = s1.unsqueeze(0)
            s2 = s2.unsqueeze(0)

        cen_q = center_of(q, True)
        cen_p = center_of(p, True)
        cen_c = center_of(c, True)
        dis_qp = torch.nn.Softmax(dim=-1)(1 / torch.norm(cen_p - cen_q, p=2, dim=-1).clamp(1e-10))
        dis_qc = torch.nn.Softmax(dim=-1)(1 / torch.norm(cen_c - cen_q, p=2, dim=-1).clamp(1e-10))
        # scores[i_idx] *= dis_qc[i_idx]
        # scores *= dis_qp
        #
        # return scores
        s1 *= dis_qp
        s2[torch.logical_not(i_idx)] = 1
        s2[i_idx] *= dis_qc[i_idx]
        if self.training:
            return s1 * s2, s1, s2
        else:
            return s1 * s2

    def form_batch_graph(self, g, device):
        g = list(chain.from_iterable(g))
        idx = [0]
        for i, _g in enumerate(g):
            idx.append(_g.x.shape[0] + idx[i])
        return Batch.from_data_list(g).to(device), torch.Tensor(idx[:-1]).long().to(device)

    def forward_graph(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.fusion_module:
            x = layer(x, edge_index)
            x = self.activation(x)
        return x

    def forward(self, query, p_datas, c_datas, i_idx):
        p_batch_graph, p_idx = self.form_batch_graph(p_datas, query.device)
        c_batch_graph, c_idx = self.form_batch_graph(c_datas, query.device)

        if self.graph_embed:
            fused_p = self.forward_graph(p_batch_graph)
            fused_c = self.forward_graph(c_batch_graph)
        else:
            fused_p = p_batch_graph.x
            fused_c = c_batch_graph.x

        b, d = query.shape
        p = fused_p[p_idx].reshape(b, -1, d).unsqueeze(2)
        c = fused_c[c_idx].reshape(b, -1, d).unsqueeze(2)
        q = query.unsqueeze(1).unsqueeze(1).expand(p.shape)
        fused = torch.cat([p, c], dim=-2)
        #####
        boxes = self.box_decoder_k(fused)
        q = self.box_decoder_q(q)
        boxes = torch.cat([q, boxes], dim=-2)
        q, p, c = boxes.chunk(3, -2)
        q, p, c = q.squeeze(-2), p.squeeze(-2), c.squeeze(-2)

        scores = self.scorer(q, p, c)
        scores = self.mul_sim(scores, q, p, c, i_idx)

        return boxes, scores


class TaxBoxWithPLM(TaxBox):
    def __init__(self, hidden_size, box_dim, graph_embed=True):
        super(TaxBoxWithPLM, self).__init__(hidden_size, box_dim, graph_embed)

        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base", device=self.device)
        self.text_encoder = RobertaModel.from_pretrained("roberta-base").to(self.device)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def form_batch_graph(self, g, device):
        g = list(chain.from_iterable(g))
        l = [_g['data'] for _g in g]
        t = []
        for _g in g:
            t += _g['text']
        idx = [0]
        for i, _g in enumerate(l):
            idx.append(_g.x.shape[0] + idx[i])
        return Batch.from_data_list(l).to(device), torch.Tensor(idx[:-1]).long().to(device), t

    def embedding(self, text_list, pool='cls'):
        inputs = self.tokenizer(text_list, padding=True, return_tensors='pt', truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)

        if pool == 'cls':
            embeddings = outputs['pooler_output']
            return embeddings
        elif pool == 'keyword':
            last_stats = outputs['last_hidden_state']
            embeddings = []
            for t, e in zip(text_list, last_stats):
                embeddings.append(e[1:t.index(',') + 1, :].mean(0))
            return torch.Tensor(embeddings)

    def forward(self, query, p_datas, c_datas, i_idx):

        p_batch_graph, p_idx, p_t = self.form_batch_graph(p_datas, self.device)
        c_batch_graph, c_idx, c_t = self.form_batch_graph(c_datas, self.device)

        p_x = self.embedding(p_t).to(self.device)
        c_x = self.embedding(c_t).to(self.device)
        query_embed = self.embedding(query).to(self.device)

        if self.graph_embed:
            fused_p = self.forward_graph(p_x, p_batch_graph.edge_index)
            fused_c = self.forward_graph(c_x, c_batch_graph.edge_index)
        else:
            fused_p = p_x
            fused_c = c_x

        b, d = query_embed.shape
        p = fused_p[p_idx].reshape(b, -1, d).unsqueeze(2)
        c = fused_c[c_idx].reshape(b, -1, d).unsqueeze(2)
        q = query_embed.unsqueeze(1).unsqueeze(1).expand(p.shape)
        fused = torch.cat([p, c], dim=-2)
        #####
        boxes = self.box_decoder_k(fused)
        q = self.box_decoder_q(q)
        boxes = torch.cat([q, boxes], dim=-2)
        q, p, c = boxes.chunk(3, -2)
        q, p, c = q.squeeze(-2), p.squeeze(-2), c.squeeze(-2)

        scores = self.scorer(q, p, c)
        scores = self.mul_sim(scores, q, p, c, i_idx)

        return boxes, scores


if __name__ == "__main__":
    pass
