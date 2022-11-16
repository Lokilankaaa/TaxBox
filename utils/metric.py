import numpy as np
import networkx as nx


class TreeMetric:
    def __init__(self):
        self.acc = []
        self.ranks = []
        self.wp = []

    def clear(self):
        self.acc = []
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
            'mean_acc': np.array(self.acc).mean(),
            'hit10': np.array(self.hit10).mean(),
            'hit5': np.array(self.hit5).mean(),
            'mrr': (1 / np.array(self.ranks)).mean(),
            'mr': np.array(self.ranks).mean(),
            'wup': np.array(self.wp).mean(),
        }

    def update(self, res, gt, new_to_old, graph):
        novel_in_c, c_in_novel = res
        pred = res[0]
        gt = list([i for i in gt if i in new_to_old]) + [gt[-1]]
        sort_pred = list([new_to_old[new] for new in pred.topk(len(pred))[1]])
        top10 = sort_pred[:10]
        top5 = sort_pred[:5]
        path = nx.shortest_path(graph, 0, top5[0])
        self.acc.append(1 if top5[0] == gt[-2] else 0)
        self.hit5.append(1 if gt[-2] in top5 else 0)
        self.hit10.append(1 if gt[-2] in top10 else 0)
        self.wp.append(2 * self.lca(path, gt) / (len(gt) + len(path)))
        self.ranks.append(sort_pred.index(gt[-2]) + 1)
