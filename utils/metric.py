import numpy as np
import networkx as nx


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
        return {
            'hit10': np.array(self.hit10).mean(),
            'hit5': np.array(self.hit5).mean(),
            'hit1': np.array(self.hit1).mean(),
            'mrr': (1 / np.array(self.ranks)).mean(),
            'mr': np.array(self.ranks).mean(),
            'wup': np.array(self.wp).mean(),
        }

    def update(self, scores, gt, new_to_old, old_to_new, graph, fs_pairs):
        # scores, fs_pairs: 2n - 1
        if scores.dim() == 2:
            scores = scores.squeeze(1)
        ranks = list([new_to_old[fs_pairs[new_id, 0].item()] for new_id in scores.topk(scores.shape[0])[1]])
        gt = list([i for i in gt if i in new_to_old and i in ranks]) + [gt[-1]]
        top10 = ranks[:10]
        top5 = ranks[:5]
        path = nx.shortest_path(graph, 0, top5[0])
        self.hit1.append(top5[0] == gt[-2])
        self.hit5.append(gt[-2] in top5)
        self.hit10.append(gt[-2] in top10)
        self.wp.append(2 * self.lca(path, gt) / (len(gt) + len(path)))
        self.ranks.append(ranks.index(gt[-2]) + 1)
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


