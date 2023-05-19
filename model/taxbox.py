from typing import Optional

import torch
from copy import deepcopy
import numpy as np

from utils.utils import hard_intersection, conditional_prob, center_of, log_conditional_prob
from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv, GraphConv
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


def InsertionScorer(q, f, s):
    # joint_fs = hard_intersection(f, s, True)
    # prob_q_given_fs = conditional_prob(q, joint_fs, True)
    prob_f_given_q = conditional_prob(f, q, True)
    prob_q_given_s = conditional_prob(q, s, True)
    # cen_q = torch.nn.functional.normalize(center_of(q, True), dim=-1)
    # cen_f = torch.nn.functional.normalize(center_of(f, True), dim=-1)
    # sim = (cen_q * cen_f).sum(-1).abs()

    return prob_f_given_q, prob_q_given_s  # / prob_q_given_s.max(-1)[0].unsqueeze(-1).expand(prob_q_given_s.shape)


def AttachScorer(q, f):
    prob_f_given_q = conditional_prob(f, q, True)
    prob_q_given_f = conditional_prob(q, f, True)
    # cen_q = torch.nn.functional.normalize(center_of(q, True), dim=-1)
    # cen_f = torch.nn.functional.normalize(center_of(f, True), dim=-1)
    # cen_q = torch.nn.functional.normalize(center_of(q, True), dim=-1)
    # cen_f = torch.nn.functional.normalize(center_of(f, True), dim=-1)
    # sim = (cen_q * cen_f).sum(-1).abs()
    return prob_f_given_q  # * sim#, prob_f_given_q - prob_q_given_f


class TaxBox(torch.nn.Module):
    def __init__(self, hidden_size, box_dim, graph_embed=True):
        super(TaxBox, self).__init__()
        self.graph_embed = graph_embed

        if graph_embed:
            self.fusion_module = nn.ModuleList([
                GATv2Conv(hidden_size, hidden_size, heads=4, dropout=0.1, concat=False, add_self_loops=False),
            ])
        self.activation = nn.LeakyReLU(0.1)
        self.lin = nn.Linear(hidden_size, hidden_size)
        self.id = nn.Identity()
        self.box_decoder_k = HighwayNetwork(hidden_size, box_dim, 2)
        self.box_decoder_q = HighwayNetwork(hidden_size, box_dim, 2)
        self.scorer = InsertionScorer
        self.device = None

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
        dis_qp = torch.nn.Softmax(dim=-1)(1 / torch.norm(cen_p - cen_q, p=2, dim=-1).clamp(min=1e-20))
        dis_qc = torch.nn.Softmax(dim=-1)(1 / torch.norm(cen_c - cen_q, p=2, dim=-1).clamp(min=1e-20))
        # scores[i_idx] *= dis_qc[i_idx]
        # scores *= dis_qp
        #
        # return scores
        s1 *= dis_qp
        s2[torch.logical_not(i_idx)] = 1
        s2[i_idx] *= dis_qc[i_idx]

        return s1 * s2

    def form_batch_graph(self, g, device):
        g = list(chain.from_iterable(g))
        idx = [0]
        for i, _g in enumerate(g):
            idx.append(_g.x.shape[0] + idx[i])
        return Batch.from_data_list(g).to(device), torch.Tensor(idx[:-1]).long().to(device)

    def forward_graph(self, data, root_idx):
        x, edge_index = data.x, data.edge_index
        ori_x = self.id(x[root_idx])
        for layer in self.fusion_module:
            x = layer(x, edge_index)[root_idx]
            # x = torch.cat([x, ori_x], dim=-1)
            x += ori_x
            x = self.lin(x)
            x = self.activation(x)
        return x

    def forward(self, query, p_datas, c_datas, i_idx):
        p_batch_graph, p_idx = self.form_batch_graph(p_datas, query.device)
        c_batch_graph, c_idx = self.form_batch_graph(c_datas, query.device)

        if self.graph_embed:
            fused_p = self.forward_graph(p_batch_graph, p_idx)
            fused_c = self.forward_graph(c_batch_graph, c_idx)
        else:
            fused_p = p_batch_graph.x[p_idx]
            fused_c = c_batch_graph.x[c_idx]

        b, d = query.shape
        p = fused_p.view(b, -1, d).unsqueeze(2)
        c = fused_c.view(b, -1, d).unsqueeze(2)
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


if __name__ == "__main__":
    pass
