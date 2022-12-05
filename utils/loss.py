import random
import torch
from copy import deepcopy

from .utils import log_conditional_prob, hard_volume, soft_volume, conditional_prob
from .utils import sample_path, sample_pair
import numpy as np
import torch.nn.functional as F
from .graph_operate import transitive_closure_mat, adj_mat
from functorch import vmap
from itertools import combinations


# def contrastive_loss(x, num_path, num_samples, raw_graph, margin=2000):
#     paths = sample_path(raw_graph, num_path)
#     pos = [sample_pair(random.choice(paths)) for _ in range(num_samples)]
#     neg = [sample_pair(tuple(random.sample(paths, k=2))) for _ in range(num_samples)]
#     pos = np.array(pos)
#     neg = np.array(neg)
#     p = log_conditional_prob(x[pos[:, 0], :], x[pos[:, 1], :]).mean()
#     n = log_conditional_prob(x[neg[:, 0], :], x[neg[:, 1], :]).mean()
#     print('p_loss:{}'.format(-p.item()),
#           'prob_son_in_father{}'.format(conditional_prob(x[pos[:, 0], :], x[pos[:, 1], :]).mean().item()))
#     return -p + 0.01 * (margin + n), -p.item(), (margin + n).item()


def triplet_loss(pairs, batch):
    margin = 5
    sample_num = len(pairs) // batch
    loss = 0

    p = log_conditional_prob(pairs[1::sample_num, ], pairs[0::sample_num, ])
    for i in range(sample_num - 2):
        n = log_conditional_prob(pairs[0::sample_num, ], pairs[2 + i::sample_num, ])
        n2 = log_conditional_prob(pairs[2 + i::sample_num, ], pairs[0::sample_num, ])

    loss += F.margin_ranking_loss(-p, -(n + n2) / 2, torch.Tensor([1] * p.shape[0]).to(p.device),
                                  margin=margin)
    return loss / batch  # , -p.item(), (margin + (n + n2) / 2).item()


# def contrastive_loss(pairs, batch):
#     t = 1
#     sample_num = len(pairs) // batch
#     logits = conditional_prob(pairs[1::sample_num, ], pairs[0::sample_num, ], box_mode=True).unsqueeze(1)
#     for i in range(sample_num - 2):
#         n = conditional_prob(pairs[0::sample_num, ], pairs[2 + i::sample_num, ], box_mode=True).unsqueeze(1)
#         n2 = conditional_prob(pairs[2 + i::sample_num, ], pairs[0::sample_num, ], box_mode=True).unsqueeze(1)
#         logits = torch.cat([logits, n, n2], dim=-1)
#
#     # b x 9
#     logits = logits / t
#     exp = torch.exp(logits)
#     loss = -torch.log(exp[:, 0] / exp.sum(1)).mean()
#     return loss

import math


# def volume_pen(q):
#     vq = soft_volume(q, box_mode=True).prod(-1)
#     res = (vq - 1e-8)
#     return (res[res < 0]).abs().mean()


def regularization_loss(q):
    qz, qZ = q.chunk(2, -1)
    abnormal = qZ - qz
    abnormal = abnormal[abnormal <= 0]
    if abnormal.sum() == 0:
        return 0
    return abnormal.abs().mean(-1).mean()


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


def tmnloss(scores, l):
    s1, s2, s3, s4 = scores
    s1 = s1.squeeze(-1)
    s2 = s2.squeeze(-1)
    s3 = s3.squeeze(-1)
    s4 = s4.squeeze(-1)
    loss_fn = torch.nn.BCELoss()
    return loss_fn(s1, l[:, 0]) + loss_fn(s2, l[:, 1]) + loss_fn(s3, l[:, 2]) + loss_fn(s4, l[:, 0])


def cls_loss(scores, paired_nodes, fs_pairs, tree):
    sample_num = 10
    # the order of the batch of scores is same as paired_nodes
    s1 = scores
    if s1.dim() == 3:
        s1 = s1.squeeze(-1)
        # s2 = s2.squeeze(-1)
        # s3 = s3.squeeze(-1)
        # s4 = s4.squeeze(-1)

    trans_mat = transitive_closure_mat(tree)
    trans_mat -= np.diag([1] * trans_mat.shape[0])

    gt_labels = []
    reserved = []
    ignore = -1
    for i in paired_nodes:
        if len(list(tree.predecessors(i))) == 0:
            ignore = i
            continue
        else:
            reach = np.where(trans_mat[:, i] == 1)[0]
            reach_inv = np.where(trans_mat[i, :] == 1)[0]
            reach[reach > i] -= 1
            reach_inv[reach_inv > i] -= 1
            _f = list(tree.predecessors(i))[0]
            _s = list(tree.successors(i))
            edge = torch.Tensor(list([[_f, s] for s in _s] + [[_f, -1]]) if len(_s) != 0 else torch.Tensor([[_f, -1]]))
            edge[edge > i] -= 1
            gt_label = torch.Tensor([0] * s1.shape[-1]).type(torch.float)

            if ignore != -1 and i > ignore:
                i -= 1
            for e in edge:
                id0 = (fs_pairs[i][:, 0] - e[0]) == 0
                id1 = (fs_pairs[i][:, 1] - e[1]) == 0
                gt_label[id0 * id1] = 1
            # for e in reach:
            #     id0 = fs_pairs[i][:, 0] == e
            #     gt_label[id0, 1] = 1
            # for e in reach_inv:
            #     id0 = fs_pairs[i][:, 1] == e
            #     gt_label[id0, 2] = 1
            #
            if gt_label.shape[0] > sample_num:
                drop = gt_label == 0
                reserve = torch.zeros_like(drop).bool()
                reserve_idx = random.choice(torch.where(torch.logical_not(drop))[0])
                reserve[reserve_idx] = True
                drop_idx = torch.where(drop)[0].numpy().tolist()
                num_drop = len(drop_idx)
                remain_num = 1
                if sample_num > remain_num:
                    rest_sample = random.sample(drop_idx, sample_num - remain_num)
                else:
                    reserve_idx = torch.where(torch.logical_not(drop))[0].numpy().tolist()
                    rest_sample = random.sample(reserve_idx, sample_num)

                reserve[rest_sample] = True
                reserved.append(reserve.unsqueeze(0))

            gt_labels.append(gt_label.unsqueeze(0))
    gt_labels = torch.cat(gt_labels).to(s1.device)  # n-1 * 2n - 3
    if len(reserved) > 0:
        reserved = torch.cat(reserved).to(s1.device).type(torch.bool)
    else:
        reserved = torch.ones_like(s1).to(s1.device).type(torch.bool)
    not_labels = torch.logical_not(gt_labels)

    # exp_scores = torch.exp(scores)
    exp_scores = scores

    pos = (exp_scores[reserved] * gt_labels[reserved]).sum(-1)
    neg = (exp_scores[reserved] * not_labels[reserved]).sum(-1)

    loss = -torch.log(pos / (pos + neg))
    # loss = -torch.log(pos / exp_scores.sum(-1))
    return loss.mean()
    # loss_fn = torch.nn.BCELoss()
    # return loss_fn(s1[reserved], gt_labels.float()[:, :, 0][reserved])
           # loss_fn(s2[reserved], gt_labels.float()[:, :, 1][reserved]) + \
           # loss_fn(s3[reserved], gt_labels.float()[:, :, 2][reserved])
           # loss_fn(s4[reserved], gt_labels.float()[:, :, 0][reserved])

    #     f = list(tree.predecessors(i))[0]
    #     gt_labels.append(f if f < i else f - 1)
    # scores = torch.cat([scores[:ignore], scores[ignore + 1:]])
    # gt_labels = torch.Tensor(gt_labels).type(torch.long).to(scores.device)
    # return torch.nn.CrossEntropyLoss()(scores, gt_labels)


def adaptive_BCE(pair_nodes, b_boxes, tree, path_sim):
    path_sim = path_sim / 1.5
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
