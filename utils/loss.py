import random
import torch
from copy import deepcopy

from .utils import log_conditional_prob, hard_volume, soft_volume, conditional_prob, center_of
from .utils import sample_path, sample_pair
import numpy as np
import torch.nn.functional as F
from .graph_operate import transitive_closure_mat, adj_mat
from functorch import vmap
from itertools import combinations

def regularization_loss(q):
    vol_max = 10
    vol_q = soft_volume(q, box_mode=True).prod(-1)
    res = vol_q[vol_q > vol_max].mean()
    return 0 if torch.isnan(res) else res
    # qz, qZ = q.chunk(2, -1)
    # abnormal = qZ - qz
    # abnormal = abnormal[abnormal <= 0]
    # if abnormal.sum() == 0:
    #     return 0
    # return abnormal.abs().mean(-1).mean()


def contrastive_loss(q, k, connect_m, batch, regular=False):
    # b * box_dim, b * box_dim, b * b
    loss = []
    test_prob = 0
    count = 0
    for i, _sample in enumerate(q):
        _loss = []
        pos_son = np.where(connect_m[i] == 1)[0]
        pos_fa = np.where(connect_m[:, i] == 1)[0]
        neg = np.intersect1d(np.where(connect_m[i] == 0)[0], np.where(connect_m[:, i] == 0)[0])
        for s in pos_son:
            _loss.append(- log_conditional_prob(_sample, k[s]))
            if batch[i] != batch[s]:
                test_prob += conditional_prob(_sample, k[s])
                count += 1
        for f in pos_fa:
            _loss.append(- log_conditional_prob(k[f], _sample))
            if batch[i] != batch[f]:
                test_prob += conditional_prob(k[f], _sample)
                count += 1
        for n in neg:
            if batch[i] == batch[n]:
                continue
            n_prob1 = conditional_prob(_sample, k[n])
            n_prob2 = conditional_prob(k[n], _sample)
            _loss.append((-torch.log(1 - n_prob1) - torch.log(1 - n_prob2)) / 2)
        loss.append(torch.cat(_loss).mean().unsqueeze(0))
    loss = torch.cat(loss).mean()
    if regular:
        loss += regularization_loss(q)
    # if math.isinf(loss.item()) or math.isnan(loss.item()):
    # print(batch)
    # print(q[0])
    return loss, test_prob / count


def adaptive_BCE(pair_nodes, b_boxes, tree, path_sim):
    path_sim = path_sim / 1
    epsilon = torch.Tensor([1e-8]).to(b_boxes.device)
    trans_clo_maj = torch.Tensor(transitive_closure_mat(tree)).type(torch.bool).to(b_boxes.device)

    reaches = []
    reaches_inv = []
    sims = []
    sims_inv = []
    for q in pair_nodes:
        reaches.append(torch.cat([trans_clo_maj[:q, q], trans_clo_maj[q + 1:, q]]).unsqueeze(0))
        reaches_inv.append(torch.cat([trans_clo_maj[q, :q], trans_clo_maj[q, q + 1:]]).unsqueeze(0))
        sims.append(torch.cat([path_sim[:q, q], path_sim[q + 1:, q]]).unsqueeze(0))
        sims_inv.append(torch.cat([path_sim[q, :q], path_sim[q, q + 1:]]).unsqueeze(0))
    reaches = torch.cat(reaches).to(b_boxes.device)
    reaches_inv = torch.cat(reaches_inv).to(b_boxes.device)
    sims = torch.cat(sims).to(b_boxes.device)
    sims_inv = torch.cat(sims_inv).to(b_boxes.device)

    def node_graph_loss(boxes, reach, reach_inv, sim, sim_inv):
        query_box = boxes[0]
        key_boxes = boxes[1:, ]
        loss = torch.zeros(key_boxes.shape[0]).to(key_boxes.device)

        # q in k
        loss = loss - (reach * log_conditional_prob(key_boxes, query_box, True))

        # k in q
        loss = loss - (reach_inv * log_conditional_prob(query_box, key_boxes, True))

        # q not in k
        probs = conditional_prob(key_boxes, query_box, True)
        margin = torch.log(1 - sim) - torch.log(1 - probs + epsilon)
        loss = loss + torch.logical_not(reach) * (margin > 0) * margin
        # cnts = probs > sim / 1.5
        # loss = loss - torch.logical_not(reach) * cnts * torch.log(1 - probs + epsilon)

        # k not in q
        probs = conditional_prob(query_box, key_boxes, True)
        margin = torch.log(1 - sim_inv) - torch.log(1 - probs + epsilon)
        loss = loss + torch.logical_not(reach_inv) * (margin > 0) * margin
        # cnts = probs > sim_inv / 1.5
        # loss = loss - torch.logical_not(reach_inv) * cnts * torch.log(1 - probs + epsilon)

        return loss.sum() / (loss != 0).sum()

    v_f = vmap(node_graph_loss)

    res = v_f(b_boxes, reaches, reaches_inv, sims, sims_inv)
    return res.mean() + regularization_loss(b_boxes.reshape(-1, b_boxes.shape[-1]))


def insertion_loss(scores, labels, i_idx):
    s1, s2, s3 = scores
    if s1.dim() == 2:
        s1 = s1.squeeze(-1)
        s2 = s2.squeeze(-1)
        s3 = s3.squeeze(-1)

    w1 = labels[:, :, 0] * 10
    w1[w1 == 0] = 1
    w2 = labels[:, :, 1] * 10
    w2[w2 == 0] = 1
    w3 = labels[:, :, 2] * 10
    w3[w3 == 0] = 1
    loss_fn = torch.nn.BCELoss(weight=w1)
    loss_fn2 = torch.nn.BCELoss(weight=w2)
    loss_fn3 = torch.nn.BCELoss(weight=w3[i_idx])

    return loss_fn(s1, labels[:, :, 0].float()) #+ 0.5 * loss_fn2(s2, labels[:, :, 1].float()) +  loss_fn3(
        #0.5 * s3[i_idx], labels[:, :, 2][i_idx].float())


def attachment_loss(scores, labels):
    s1 = scores
    if s1.dim() == 2:
        s1 = s1.squeeze(-1)
        # r1 = r1.squeeze(-1)

    weights = labels[:, 0] * 10
    weights[weights == 0] = 1
    loss_fn = torch.nn.BCELoss(weight=weights)

    return loss_fn(s1, labels[:, 0])  # + (labels[:, 0] * (r1 < 0) * r1.abs()).mean()


def box_constraint_loss(boxes, sims, reaches):
    # box: b * 3 * d, labels: b * 3, sims: b * 2
    sims = sims / 2
    q, p, c = boxes.chunk(3, 1)
    q, p, c = q.squeeze(1), p.squeeze(1), c.squeeze(1)
    loss = torch.zeros_like(q[:, 0])

    def _not_in_(a, b, sim, com=False):
        epsilon = torch.Tensor([1e-8]).to(a.device)
        probs = conditional_prob(b, a, True)
        return torch.max(torch.zeros_like(a[:, 0]),
                         -torch.log(1 - probs + epsilon) + torch.log(1 - sim if not com else (1 - epsilon)))

    def _in_(a, b):
        return -log_conditional_prob(b, a, True)

    def push_away_sim_center(a, b, sim, com=False):
        epsilon = torch.Tensor([1e-8]).to(a.device)
        cen_a = torch.nn.functional.normalize(center_of(a, True), dim=-1)
        cen_b = torch.nn.functional.normalize(center_of(b, True), dim=-1)
        probs = (cen_a * cen_b).sum(-1).abs()

        return torch.max(torch.zeros_like(a[:, 0]),
                         -torch.log(1 - probs + epsilon) + torch.log(1 - sim if not com else (1 - epsilon)))

    # def push_away_sim_center(a, b, sim, com=False):
    #     epsilon = torch.Tensor([1e-8]).to(a.device)
    #     cen_a = center_of(a, True)
    #     cen_b = center_of(b, True)
    #     probs = torch.norm(cen_a - cen_b, p=2, dim=-1)
    #
    #     return 0.1 * torch.max(torch.zeros_like(a[:, 0]),
    #                            sim - probs)

    # q in p
    _idx = reaches[:, 0].bool()
    loss[_idx] += _in_(q[_idx], p[_idx]) + _not_in_(p[_idx], q[_idx], sims[_idx, 0])
    # c in q
    _idx = reaches[:, 2].bool()
    loss[_idx] += _in_(c[_idx], q[_idx]) + _not_in_(q[_idx], c[_idx], sims[_idx, 1])
    # p in q
    _idx = reaches[:, 1].bool()
    loss[_idx] += _in_(p[_idx], q[_idx]) + _not_in_(q[_idx], p[_idx], sims[_idx, 0])
    # q in c
    _idx = reaches[:, 3].bool()
    loss[_idx] += _in_(q[_idx], c[_idx]) + _not_in_(c[_idx], q[_idx], sims[_idx, 1])
    # p not in q, q not in p
    _idx = torch.logical_not((reaches[:, 0] * reaches[:, 1]).bool())
    loss[_idx] = _not_in_(p[_idx], q[_idx], sims[_idx, 0], True) + _not_in_(q[_idx], p[_idx], sims[_idx, 0],
                                                                            True) + push_away_sim_center(q[_idx],
                                                                                                         p[_idx],
                                                                                                         sims[_idx, 0])
    # c not in q, q not in c
    _idx = torch.logical_not((reaches[:, 2] * reaches[:, 3]).bool())
    loss[_idx] = _not_in_(c[_idx], q[_idx], sims[_idx, 1], True) + _not_in_(q[_idx], c[_idx], sims[_idx, 1],
                                                                            True) + push_away_sim_center(q[_idx],
                                                                                                         c[_idx],
                                                                                                         sims[_idx, 0])

    return loss.mean() + regularization_loss(boxes.reshape(-1, boxes.shape[-1]))


def ranking_loss(scores, labels, i_idx, rank_sim, margin=0.5):
    # scores = torch.zeros_like(i_idx).to(i_scores.device).float()
    # scores[i_idx] = i_scores.squeeze(-1)
    # scores[torch.logical_not(i_idx)] = a_scores.squeeze(-1)

    gt = labels[:, :, 0].long()
    b, s = gt.shape[:2]

    scores = scores.reshape(-1, s)

    p_idx = torch.where(gt == 1)
    n_idx = torch.where(gt == 0)

    neg = scores[n_idx].reshape(b, -1)
    pos = scores[p_idx].reshape(b, -1).expand(neg.shape)
    sim = 1 - rank_sim.reshape(b, -1).to(neg.device)

    loss = torch.max(torch.zeros_like(neg),
                     neg - pos + sim.reshape(b, -1) / 2).mean()
    return loss
