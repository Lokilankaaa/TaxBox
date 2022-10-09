import random
import torch
from .utils import log_conditional_prob, hard_volume, soft_volume, conditional_prob
from .utils import sample_path, sample_pair
import numpy as np


def contrastive_loss(x, num_path, num_samples, raw_graph, margin=2000):
    paths = sample_path(raw_graph, num_path)
    pos = [sample_pair(random.choice(paths)) for _ in range(num_samples)]
    neg = [sample_pair(tuple(random.sample(paths, k=2))) for _ in range(num_samples)]
    pos = np.array(pos)
    neg = np.array(neg)
    p = log_conditional_prob(x[pos[:, 0], :], x[pos[:, 1], :]).mean()
    n = log_conditional_prob(x[neg[:, 0], :], x[neg[:, 1], :]).mean()
    print('p_loss:{}'.format(-p.item()),
          'prob_son_in_father{}'.format(conditional_prob(x[pos[:, 0], :], x[pos[:, 1], :]).mean().item()))
    return -p + 0.01 * (margin + n), -p.item(), (margin + n).item()


def contrastive_loss(pairs):
    margin = 5
    step = len(pairs) // 4
    p = log_conditional_prob(torch.cat(pairs[:step]), torch.cat(pairs[step: 2 * step])).mean()
    n = log_conditional_prob(torch.cat(pairs[2 * step: 3 * step]), torch.cat(pairs[3 * step:])).mean()
    # print('p_loss:{}'.format(-p.item()))
    loss = -p
    if margin + n > 0:
        loss += 1 * (margin + n)
    return loss, -p.item(), (margin + n).item()
