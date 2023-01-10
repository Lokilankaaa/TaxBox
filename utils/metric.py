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
            'wup': np.array(self.wp).mean(),
        }

    def update(self, scores, labels, gt_path, new_to_old, first, graph, fs_pairs):
        # scores, fs_pairs: 2n - 1
        # scores = scores.cpu().numpy()
        rank = obtain_ranks(scores, labels)
        # ranks = list([new_to_old[fs_pairs[new_id, 0].item()] for new_id in scores.topk(scores.shape[0])[1]])
        gt_path = list([i for i in gt_path if i in new_to_old]) + [gt_path[-1]]
        path = nx.shortest_path(graph, 0, new_to_old[fs_pairs[first, 0].item()])
        self.wp.append(2 * self.lca(path, gt_path) / (len(gt_path) + len(path)))
        self.ranks.append(rank[0])
    # def update(self, res, gt, new_to_old, graph, fs_pairs):
    # novel_in_c, s_in_f = res
    # pred = res[0]
    # gt = list([i for i in gt if i in new_to_old]) + [gt[-1]]
    # # sort_pred_edge = list([fs_pairs[idx, 0].item() for idx in s_in_f.topk(len(s_in_f))[1]])
    # sort_pred_node = list([new_to_old[new] for new in pred.topk(len(pred))[1]])
    # # if novel_in_c[sort_pred_node[0]] > s_in_f[sort_pred_edge[0]]:
    # sort_pred = sort_pred_node
    # # else:
    # # sort_pred = sort_pred_edge
    # top10 = sort_pred[:10]
    # top5 = sort_pred[:5]
    # path = nx.shortest_path(graph, 0, top5[0])
    # self.acc.append(1 if top5[0] == gt[-2] else 0)
    # self.hit5.append(1 if gt[-2] in top5 else 0)
    # self.hit10.append(1 if gt[-2] in top10 else 0)
    # self.wp.append(2 * self.lca(path, gt) / (len(gt) + len(path)))
    # self.ranks.append(sort_pred.index(gt[-2]) + 1)
    # self.mean_prob.append(pred[self.ranks[-1] - 1].cpu().numpy())
