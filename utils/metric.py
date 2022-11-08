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
        i = 0
        while a[i] == b[i] and i < len(a) and i < len(b):
            i += 1
        return i - 1

    def showResults(self):
        return {
            'mean_acc': np.array(self.acc).mean(),
            'mrr': (1 / np.array(self.ranks)).mean(),
            'mr': np.array(self.ranks).mean(),
            'wup': np.array(self.wp).mean(),
        }

    def update(self, pred, gt):
        # pred: a list of node denoting the path from root to the query
        if pred == gt:
            self.acc.append(1)
        else:
            self.acc.append(0)

        self.wp.append(2 * self.lca(pred, gt) / (len(gt) + len(pred)))
