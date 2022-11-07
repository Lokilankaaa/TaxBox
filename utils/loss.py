import random
import torch
from copy import deepcopy

from .utils import log_conditional_prob, hard_volume, soft_volume, conditional_prob
from .utils import sample_path, sample_pair
import numpy as np
import torch.nn.functional as F
from .graph_operate import transitive_closure_maj


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


def adaptive_BCE(pair_nodes, b_boxes, tree, sim_F, new_to_old_label_lookup):
    trans_clo_maj = transitive_closure_maj(tree)
    for query, boxes in zip(pair_nodes, b_boxes):
        query_box = boxes[0]
        key_boxs = boxes[1:, ]

        def pair_loss(_idx):
            key_box = key_boxs[_idx]
            sim = sim_F(new_to_old_label_lookup[query], new_to_old_label_lookup[_idx])

            loss = 0
            # k in q
            if trans_clo_maj[new_to_old_label_lookup[query]][new_to_old_label_lookup[_idx]] == 1:
                loss += - log_conditional_prob(query_box, key_box, True)
            # q in k
            if trans_clo_maj[new_to_old_label_lookup[_idx]][new_to_old_label_lookup[query]] == 1:
                loss += - log_conditional_prob(key_box, query_box, True)

            # k not in q
            if trans_clo_maj[new_to_old_label_lookup[query]][new_to_old_label_lookup[_idx]] == 0:
                prob = conditional_prob(query_box, key_box, True)
                loss += - sim * torch.log(1 - prob)

            # q not in k
            if trans_clo_maj[new_to_old_label_lookup[_idx]][new_to_old_label_lookup[query]] == 0:
                prob = conditional_prob(key_box, query_box, True)
                loss += - sim * torch.log(1 - prob)

            return loss



