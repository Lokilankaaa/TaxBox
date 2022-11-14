import numpy as np


class TreeMetric:
    def __init__(self):
        self.acc = []
        self.ranks = []
        self.wp = []

    def clear(self):
        self.acc = []
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
            'mrr': (1 / np.array(self.ranks)).mean(),
            'mr': np.array(self.ranks).mean(),
            'wup': np.array(self.wp).mean(),
        }

    def update(self, pred, gt):
        # pred: a list of node denoting the path from root to the query
        self.acc.append(1 if pred == gt else 0)
        self.wp.append(2 * self.lca(pred, gt) / (len(gt) + len(pred)))
