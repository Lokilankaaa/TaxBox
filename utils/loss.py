import random
import torch
from .utils import log_conditional_prob, hard_volume, soft_volume
from .utils import sample_path, sample_pair
import numpy as np


def contrastive_loss(x, num_path, num_samples, raw_graph, margin=10000):
    paths = sample_path(raw_graph, num_path)
    pos = [sample_pair(random.choice(paths)) for _ in range(num_samples)]
    neg = [sample_pair(tuple(random.sample(paths, k=2))) for _ in range(num_samples)]
    pos = np.array(pos)
    neg = np.array(neg)
    p = log_conditional_prob(x[pos[:, 0], :], x[pos[:, 1], :]).mean()
    n = log_conditional_prob(x[neg[:, 0], :], x[neg[:, 1], :]).mean()
    print(-p.item(), n.item())
    return (-p) / (margin + n - p)
