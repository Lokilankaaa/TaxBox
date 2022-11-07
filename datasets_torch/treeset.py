import os
from random import shuffle

import networkx as nx
from copy import deepcopy
from torch.utils.data import Dataset
import torch
from queue import Queue
import numpy as np

from utils.utils import batch_load_img


class TreeSet(Dataset):
    def __init__(self, G, names, descriptions, batch_size=300):
        super(TreeSet, self).__init__()
        self._tree = G
        self._undigraph = G.to_undirected()
        self._c_tree = deepcopy(self._tree)
        self.names = names
        self.descriptions = descriptions
        self.batch_size = batch_size
        self.leaves = []
        self.paths = {}
        self.min_depth = 0
        self.max_depth = 0
        self.mean_depth = 0
        self.mean_order = 0
        self.mini_batches = []
        self.mini_batches_dep = []
        self.decs = None
        self.fetch_order = []
        self._database = {}
        self.box_embeddings = {}

        self._init()
        self._process()
        self.shuffle()

    def _get_leaves(self):
        self.leaves = list([node for node in self._tree.nodes.keys() if self._tree.out_degree(node) == 0])

    def _get_all_paths(self):
        for l in self.leaves:
            self.paths[l] = nx.shortest_path(self._tree, 0, l)

    def _stats(self):
        depths = list(map(lambda x: len(x[1]), self.paths.items()))
        orders = np.array(list(map(lambda x: len(list(self._tree.successors(x))), self._tree.nodes.keys())))
        self.min_depth = min(depths)
        self.max_depth = max(depths)
        self.mean_depth = sum(depths) / len(depths)
        self.mean_order = orders.sum() / (orders != 0).sum()
        self._count_descs()

    def _form_mini_batches(self):
        dfs_depth = int(np.log(self.batch_size) / np.log(self.mean_order))

        def get_roots(tree):
            return list([node for node in tree.nodes.keys() if tree.in_degree(node) == 0])

        q = Queue()
        q.put(0)
        while not q.empty():
            head = q.get()
            d = dfs_depth
            if self.decs[head] >= self.batch_size:
                while nx.traversal.dfs_tree(self._c_tree, head, d).number_of_nodes() > int(self.batch_size * 0.8):
                    d -= 1
                mini_tree = nx.traversal.dfs_tree(self._c_tree, head, d + 1)
                original_node = list(mini_tree.nodes().keys())
                append_node = []
                for c in self._c_tree.successors(head):
                    if self.decs[c] < int(self.batch_size * 0.2):
                        append_node += list(nx.traversal.dfs_tree(self._c_tree, c).nodes.keys())
                original_node = list(set(original_node + append_node))
                mini_tree = deepcopy(self._c_tree.subgraph(original_node))
                for n in get_roots(mini_tree):
                    self._c_tree.remove_node(n)
                    original_node.remove(n)
                for n in append_node:
                    self._c_tree.remove_node(n)
                    original_node.remove(n)
                for n in get_roots(self._c_tree.subgraph(original_node)):
                    q.put(n)
                    self.mini_batches_dep.append([head, n])
            else:
                mini_tree = deepcopy(nx.traversal.dfs_tree(self._c_tree, head))
                for n in mini_tree.nodes.keys():
                    self._c_tree.remove_node(n)
            self.mini_batches.append(mini_tree)

        del self._c_tree

    def _count_descs(self):
        self.decs = [0] * int((self._c_tree.number_of_nodes() + 5) / 0.8)

        def dfs(head):
            self.decs[head] = 1
            if self._c_tree.out_degree(head) != 0:
                for c in self._c_tree.successors(head):
                    self.decs[head] += dfs(c)
            return self.decs[head]

        dfs(0)
        self.decs = np.array(self.decs)

    def _init(self):
        self._get_leaves()
        self._get_all_paths()
        self._stats()
        self._form_mini_batches()

    def _check_saved(self):
        return os.path.exists('tree_data.pt')

    def _process(self):
        mini_batches_dep_g = nx.DiGraph()
        mini_batches_dep_g.add_edges_from(self.mini_batches_dep)

        def get_leaves(tree):
            return list([node for node in tree.nodes.keys() if tree.out_degree(node) == 0])

        level = []
        while mini_batches_dep_g.number_of_nodes() > 0:
            leaves = get_leaves(mini_batches_dep_g)
            level.append(leaves)
            for n in leaves:
                mini_batches_dep_g.remove_node(n)
        self.fetch_order = level

        if self._check_saved():
            self._database = torch.load('tree_data.pt')
        else:
            import clip
            m, prep = clip.load('ViT-B/32')

            img_path = '/data/home10b/xw/imagenet21k/imagenet_images'
            for n in self._tree.nodes():
                name = self.names[n].replace('_', ' ')
                description = self.descriptions[n]
                text = clip.tokenize(','.join([name, description]), truncate=True)
                if self._tree.out_degree(n) == 0:
                    imgs = [os.path.join(img_path, name, i) for i in os.listdir(os.path.join(img_path, name))]
                    imgs = torch.cat(batch_load_img(imgs, prep, 100))
                    with torch.no_grad():
                        text_embedding = m.encode_text(text.cuda()).cpu()
                        imgs_embedding = m.encode_image(imgs.cuda()).cpu()
                    cat_ = torch.cat([text_embedding, imgs_embedding])
                else:
                    with torch.no_grad():
                        text_embedding = m.encode_text(text.cuda()).cpu()
                    cat_ = text_embedding

                self._database[n] = cat_
            torch.save(self._database, 'tree_data.pt')

    def __len__(self):
        return len(self.mini_batches)

    def shuffle(self):
        if len(self.fetch_order) != 0:
            for l in self.fetch_order:
                shuffle(l)

    def distance(self, a, b):
        return nx.shortest_path_length(self._undigraph, a, b)

    def path_sim(self, a, b):
        common = nx.lowest_common_ancestor(a, b)
        common_path = nx.shortest_path(self._tree, 0, common)
        return len(common_path) / (len(common_path) + self.distance(a, b))

    def path_between(self, a, b):
        return nx.shortest_path(self._undigraph, a, b)

    def __getitem__(self, idx):
        l = []
        for _ in self.fetch_order:
            l += _

        g = l[idx]
        node_feature_list = []
        old_to_new_label_lookup = {}
        new_to_old_label_lookup = {}
        for i, node in enumerate(g.nodes()):
            old_to_new_label_lookup[node] = i
            new_to_old_label_lookup[i] = node
            node_feature_list.append(self._database[node])
        new_g = nx.relabel_nodes(g, old_to_new_label_lookup)

        # old: label in original tree, new: range from 0 to len(g)
        # new_g and node_feature_list are one-to-one
        return new_g, node_feature_list, old_to_new_label_lookup, new_to_old_label_lookup


if __name__ == "__main__":
    from utils.mkdataset import split_tree_dataset

    G, names, descriptions, train, test, eva = split_tree_dataset(
        '/data/home10b/xw/visualCon/datasets_json/imagenet_dataset.json')
    t = TreeSet(G, names, descriptions, batch_size=200)
