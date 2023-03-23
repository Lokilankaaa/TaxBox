import numpy as np
import networkx as nx
import itertools

from utils.utils import calculate_ranks_from_similarities, obtain_ranks


class TreeMetric:
    def __init__(self):
        self.hit1 = []
        self.ranks = []
        self.wp = []
        self.hit10 = []
        self.hit5 = []

    def clear(self):
        self.hit1 = []
        self.hit10 = []
        self.hit5 = []
        self.ranks = []
        self.wp = []

    def lca(self, a, b):
        pa = np.array(a)
        pb = np.array(b)
        if len(pa) > len(pb):
            pa = pa[:len(pb)]
        elif len(pb) > len(pa):
            pb = pb[:len(pa)]

        c_len = ((pa - pb) == 0).sum()

        return c_len

    def show_results(self):
        rank_positions = np.array(list(itertools.chain(*self.ranks)))
        return {
            'hit1': np.sum(rank_positions <= 1) / len(rank_positions),
            'hit5': np.sum(rank_positions <= 5) / len(rank_positions),
            'hit10': np.sum(rank_positions <= 10) / len(rank_positions),
            'pre1': np.sum(rank_positions <= 1) / len(self.ranks),
            'pre5': np.sum(rank_positions <= 5) / len(self.ranks) / 5,
            'pre10': np.sum(rank_positions <= 10) / len(self.ranks) / 10,
            'mrr': (1 / np.ceil(np.array(list(itertools.chain(*self.ranks))) / 10)).mean(),
            'macro_mr': np.array([np.array(rank).mean() for rank in self.ranks]).mean(),
            'micro_mr': rank_positions.mean(),
            # 'wup': np.array(self.wp).mean(),
        }

    def update(self, rank):
        # scores, fs_pairs: 2n - 1
        # scores = scores.cpu().numpy()

        # ranks = list([new_to_old[fs_pairs[new_id, 0].item()] for new_id in scores.topk(scores.shape[0])[1]])
        # gt_path = list([i for i in gt_path if i in new_to_old]) + [gt_path[-1]]
        # path = nx.shortest_path(graph, 0, new_to_old[fs_pairs[first, 0].item()])
        # self.wp.append(2 * self.lca(path, gt_path) / (len(gt_path) + len(path)))
        if hasattr(rank, 'tolist'):
            rank = rank.tolist()
        if len(rank) != 0:
            self.ranks.append(rank)

