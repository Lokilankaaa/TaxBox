import networkx
from pyvis.network import Network
import numpy as np
import random
import torch
import matplotlib.pyplot as plt


def vis_box(dataset, boxes, new_to_old, s_f=None):
    # choose = random.choice(list(set(dataset.train).difference(dataset.leaves)))
    # box_f = boxes[new_to_old.index(choose)]
    # sons = []
    # for s in dataset._tree.successors(choose):
    #     sons.append(boxes[new_to_old.index(s)])
    choose = random.choice(dataset.train)
    f = list(dataset._tree.predecessors(choose))[0]
    if s_f is not None:
        choose, f = s_f
    mi, ma = boxes.chunk(2, -1)
    _boxes = torch.cat([mi.unsqueeze(-1), ma.unsqueeze(-1)], dim=-1)
    box_s = _boxes[new_to_old.index(choose)].cpu().numpy()
    box_f = _boxes[new_to_old.index(f)].cpu().numpy()
    fig = plt.figure()
    for i in range(box_s.shape[0]):
        plt.vlines(x=i + 1, ymin=box_f[i][0], ymax=box_f[i][1], colors='red')
        plt.vlines(x=i + 1, ymin=box_s[i][0], ymax=box_s[i][1], colors='blue')
    plt.show()
    return choose, f


def get_adj_matrix(id_to_children):
    num_nodes = len(id_to_children)
    adj_m = []
    for node in id_to_children:
        row = np.zeros(num_nodes)
        if len(node) != 0:
            row[node] = 1
        adj_m.append(row)
    adj_m = np.array(adj_m)
    return adj_m


def vis_graph(adj_m, id_to_name):
    G = networkx.from_numpy_array(adj_m, create_using=networkx.DiGraph)
    for i in range(adj_m.shape[0]):
        G.add_node(i, label=str(i) + id_to_name[i])
    net = Network(directed=True, layout=True)
    net.from_nx(G)
    net.show('example.html')


if __name__ == '__main__':
    pass
