import random
import torch

from .utils import log_conditional_prob, hard_volume, soft_volume, conditional_prob, center_of


def regularization_loss(q):
    vol_max = 10
    vol_q = soft_volume(q, box_mode=True).prod(-1)
    res = vol_q[vol_q > vol_max].mean()
    return 0 if torch.isnan(res) else res


def cls_loss(scores, labels):
    # s1, s2, s3 = scores
    # if s1.dim() == 2:
    #     s1 = s1.squeeze(-1)
        # s2 = s2.squeeze(-1)
        # s3 = s3.squeeze(-1)

    w1 = labels * 30
    w1[w1 == 0] = 1
    loss_fn = torch.nn.BCELoss(weight=w1)

    return loss_fn(scores, labels.float())


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
    #     probs = torch.nn.Softmax(dim=-1)(1 / torch.norm(cen_a - cen_b, p=2, dim=-1))
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
    _idx = ((1 - reaches[:, 0]) * (1 - reaches[:, 1])).bool()
    loss[_idx] = _not_in_(p[_idx], q[_idx], sims[_idx, 0], True) + _not_in_(q[_idx], p[_idx], sims[_idx, 0],
                                                                            True) + push_away_sim_center(q[_idx],
                                                                                                         p[_idx],
                                                                                                         sims[_idx, 0])
    # c not in q, q not in c
    _idx = ((1 - reaches[:, 2]) * (1 - reaches[:, 3])).bool()
    loss[_idx] = _not_in_(c[_idx], q[_idx], sims[_idx, 1], True) + _not_in_(q[_idx], c[_idx], sims[_idx, 1],
                                                                            True) + push_away_sim_center(q[_idx],
                                                                                                         c[_idx],
                                                                                                         sims[_idx, 0])

    return loss.mean()  # + regularization_loss(boxes.reshape(-1, boxes.shape[-1]))


def ranking_loss(scores, labels, rank_sim, margin=0.5):

    b, s = labels.shape
    assert labels.sum() == b
    assert (labels <= 1).sum() == b * s


    try:
        neg = scores[labels == 0].view(b, -1)
        pos = scores[labels == 1].view(b, -1).expand(neg.shape)
        sim = 1 - rank_sim.view(b, -1).to(neg.device)
    except:
        print('labels', labels)
        print('scores', scores)
        exit()

    loss = torch.max(torch.zeros_like(neg), neg - pos + sim / 2).mean()
    return loss
