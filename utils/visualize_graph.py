import networkx
from pyvis.network import Network
import numpy as np


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
        G.add_node(i, label=str(i)+id_to_name[i])
    net = Network(directed=True, layout=True)
    net.from_nx(G)
    net.show('example.html')


if __name__ == '__main__':
    pass
