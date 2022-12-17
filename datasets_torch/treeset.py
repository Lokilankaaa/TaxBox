import os
import random
from collections import deque
from itertools import combinations, chain
from random import shuffle

import networkx as nx
from copy import deepcopy
from torch.utils.data import Dataset
import torch
from queue import Queue
import numpy as np
from tqdm import tqdm

from utils.utils import batch_load_img


class TreeSet(Dataset):
    def __init__(self, whole, G, names, descriptions, train, eva, test, batch_size=250):
        super(TreeSet, self).__init__()
        self.edges = None
        self.mode = 'train'
        self.whole = whole
        self.train = train
        # self.train.remove(0)
        self.eva = eva
        self.test = test
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
        self.mini_batches_root = []
        self.decs = None
        self.fetch_order = []
        self._database = {}
        self.fused_embeddings = {}
        self.path_sim_matrix = None
        self.fs_pairs = []
        self.node2pairs = {}
        self.node2ancestor = {}
        self.node2descendant = {}

        self._init()
        self._process()
        self.shuffle()

    def update_box(self, i, embed):
        self.fused_embeddings[i] = embed

    def update_boxes(self, boxes, new_to_old_map):
        for i, b in enumerate(boxes):
            self.update_box(new_to_old_map[i], b)

    def clear_boxes(self):
        self.fused_embeddings = {}

    def _get_leaves(self, t=None):
        if t is None:
            self.leaves = list([node for node in self._tree.nodes.keys() if self._tree.out_degree(node) == 0])
        else:
            return list([node for node in t.nodes.keys() if t.out_degree(node) == 0])

    def _get_all_paths(self):
        for l in self._tree.nodes():
            self.paths[l] = nx.shortest_path(self._tree, 0, l)

    def _stats(self):
        depths = list(map(lambda x: len(x[1]), self.paths.items()))
        orders = np.array(list(map(lambda x: len(list(self._tree.successors(x))), self._tree.nodes.keys())))
        self.min_depth = min(depths)
        self.max_depth = max(depths)
        self.mean_depth = sum(depths) / len(depths)
        self.mean_order = orders.sum() / (orders != 0).sum()
        self._count_descs()

    def generate_fs_pairs(self):
        list(map(lambda x: self.fs_pairs.append([x[0], x[1]]), self._tree.edges()))
        self.fs_pairs = torch.Tensor(self.fs_pairs).type(torch.long)

    def _form_mini_batches(self):
        dfs_depth = int(np.log(self.batch_size) / np.log(self.mean_order))

        def get_roots(tree):
            return list([node for node in tree.nodes.keys() if tree.in_degree(node) == 0])

        q = Queue()
        q.put(0)
        while not q.empty():
            head = q.get()
            self.mini_batches_root.append(head)
            d = dfs_depth
            if self.decs[head] >= self.batch_size:
                while nx.traversal.dfs_tree(self._c_tree, head, d).number_of_nodes() > int(
                        self.batch_size * 0.5) and d > 0:
                    d -= 1
                mini_tree = nx.traversal.dfs_tree(self._c_tree, head, d + 1)
                original_node = list(mini_tree.nodes().keys())
                append_node = []
                for c in self._c_tree.successors(head):
                    if self.decs[c] < int(10):
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
        # self._form_mini_batches()
        self.generate_fs_pairs()
        self.generate_node2pairs()
        self.generate_node2ancestors()
        self.generate_node2descendants()
        self.generate_edges()

    def generate_edges(self):
        candidates = set(chain.from_iterable([[(n, d) for d in ds] for n, ds in self.node2descendant.items()]))
        self.edges = candidates # list(self._tree.edges()) + list([[n, -1] for n in self._tree.nodes()])

    def generate_node2pairs(self):
        for node in self.whole.nodes():
            if node == 0:
                continue
            parents = set()
            children = set()
            ps = deque(self.whole.predecessors(node))
            cs = deque(self.whole.successors(node))
            while ps:
                p = ps.popleft()
                if p in self._tree:
                    parents.add(p)
                else:
                    ps += list(self.whole.predecessors(p))
            while cs:
                c = cs.popleft()
                if c in self._tree:
                    children.add(c)
                else:
                    cs += list(self.whole.successors(c))
            children.add(-1)
            position = [(p, c) for p in parents for c in children if p != c]
            self.node2pairs[node] = position

        # for n in self._tree.nodes():
        #     if n == 0:
        #         continue
        #     _f = list(self._tree.predecessors(n))[0]
        #     _s = list(self._tree.successors(n)) + [-1]
        #     self.node2pairs[n] = list([[_f, s] for s in _s])

    def generate_node2ancestors(self):
        for n in self._tree.nodes():
            self.node2ancestor[n] = self.paths[n][:-1]

    def generate_node2descendants(self):
        for n in self._tree.nodes():
            self.node2descendant[n] = list(nx.descendants(self._tree, n)) + [-1]

    def _check_saved(self):
        return os.path.exists('tree_data.pt') and os.path.exists('path_sim_maj.pt')

    def _process(self):
        mini_batches_dep_g = nx.DiGraph()
        mini_batches_dep_g.add_edges_from(self.mini_batches_dep)

        def get_leaves(tree):
            return list([node for node in tree.nodes.keys() if tree.out_degree(node) == 0])

        level = []
        while mini_batches_dep_g.number_of_nodes() > 0:
            leaves = get_leaves(mini_batches_dep_g)
            level.append(list(map(self.mini_batches_root.index, leaves)))
            for n in leaves:
                mini_batches_dep_g.remove_node(n)
        self.fetch_order = level

        if self._check_saved():
            self._database = torch.load('tree_data.pt')
            for i in self._database.keys():
                self._database[i] = torch.nn.functional.normalize(self._database[i].float(), p=2, dim=-1)
            self.path_sim_matrix = torch.load('path_sim_maj.pt')

            # import clip
            # m, prep = clip.load('ViT-B/32')
            # img_path = '/data/home10b/xw/imagenet21k/imagenet_images'
            # for i, d in self._database.items():
            #     if d.shape[0] < 101:
            #         name = self.names[i]
            #         imgs = [os.path.join(img_path, name, i) for i in os.listdir(os.path.join(img_path, name))]
            #         imgs = torch.cat(batch_load_img(imgs, prep, 101 - d.shape[0]))
            #         with torch.no_grad():
            #             imgs_embedding = m.encode_image(imgs.cuda()).cpu()
            #         self._database[i] = torch.cat([d, imgs_embedding])
            #         if self._database[i].shape[0] < 101:
            #             print(self.names[i])
            # torch.save(self._database, 'tree_data.pt')

        else:
            import clip
            m, prep = clip.load('ViT-B/32')

            img_path = '/data/home10b/xw/imagenet21k/imagenet_images'
            for n in self.whole.nodes():
                name = self.names[n]
                description = self.descriptions[n]
                text = clip.tokenize(','.join([name.replace('_', ' '), description]), truncate=True)

                if name in os.listdir(img_path):
                    name = name
                elif name.replace('_', ' ') in os.listdir(img_path):
                    name = name.replace('_', ' ')
                elif name + '#' + description in os.listdir(img_path):
                    name = (name + '#' + description)

                imgs = [os.path.join(img_path, name, i) for i in os.listdir(os.path.join(img_path, name))]
                imgs = torch.cat(batch_load_img(imgs, prep, 100))
                with torch.no_grad():
                    text_embedding = m.encode_text(text.cuda()).cpu()
                    imgs_embedding = m.encode_image(imgs.cuda()).cpu()
                cat_ = torch.cat([text_embedding, imgs_embedding])

                self._database[n] = cat_
            torch.save(self._database, 'tree_data.pt')

            self.path_sim_matrix = torch.zeros(len(self._database), len(self._database))

            def cal_sim(comb):
                # path_sim: a matrix containing all train pairs with whole id
                self.path_sim_matrix[comb[0], comb[1]] = self.path_sim(comb[0], comb[1])
                self.path_sim_matrix[comb[1], comb[0]] = self.path_sim_matrix[comb[0], comb[1]]

            for comb in tqdm(combinations(list(self._tree.nodes()), r=2)):
                cal_sim(comb)

            # list(map(cal_sim, combinations(self._tree.nodes(), r=2)))
            torch.save(self.path_sim_matrix, 'path_sim_maj.pt')

    def __len__(self):
        if self.mode == 'train':
            return len(self.train) - 1
        elif self.mode == 'eval':
            return len(self.eva)
        else:
            return len(self.test)

    def shuffle(self):
        if self.mode == 'train':
            if len(self.fetch_order) != 0:
                for l in self.fetch_order:
                    shuffle(l)
        elif self.mode == 'eval':
            shuffle(self.eva)

        elif self.mode == 'test':
            shuffle(self.test)

    def distance(self, a, b):
        return nx.shortest_path_length(self._undigraph, a, b)

    def path_sim(self, a, b):
        aa = nx.shortest_path(self._tree, 0, a)
        bb = nx.shortest_path(self._tree, 0, b)
        pa = np.array(aa)
        pb = np.array(bb)
        if len(pa) > len(pb):
            pa = pa[:len(pb)]
        elif len(pb) > len(pa):
            pb = pb[:len(pa)]
        common_path = ((pa - pb) == 0).sum()

        # common = nx.lowest_common_ancestor(self._tree, a, b)
        # common_path = self.distance(0, common)
        return common_path / (len(aa) + len(bb) - common_path)

    def path_between(self, a, b):
        return nx.shortest_path(self._undigraph, a, b)

    def change_mode(self, mode):
        assert mode in ('train', 'test', 'eval')
        self.mode = mode

    def get_milestone(self):
        res = [0]
        for l in self.fetch_order:
            res.append(res[-1] + len(l))
        return res[1:-1]

    # def generate_struct_text(self, p, q, c):
    #     len()

    def extract_ego_tree(self, n, q):
        s = list(self._tree.successors(n))
        s.remove(q) if q in s else None
        node_features = [self._database[n][0].unsqueeze(0)]

        for _s in s:
            node_features.append(self._database[_s][0].unsqueeze(0))

        edge = [] if len(s) == 0 else [[0, i] for i in range(1, len(s))]

        return node_features, edge

    def sample_train_pairs(self, q, num, pos_num=1):
        pos = random.sample(self.node2pairs[q], k=pos_num)

        # neg = [list(e) for e in self.edges if e[0] != q and e[1] != q and e not in self.node2pairs[q]]
        remain = num - len(pos)
        if remain > 0:
            rest = [e for e in random.sample(self.edges, remain) if
                    e[0] != q and e[1] != q and e not in self.node2pairs[q]]
            while len(rest) < remain:
                rest += [e for e in random.sample(self.edges, remain - len(rest)) if
                         e[0] != q and e[1] != q and e not in self.node2pairs[q] and e not in rest]
        else:
            rest = random.sample(pos, num)
        sims = self.path_sim_matrix[pos[0][0], [r[0] for r in rest]]
        labels = []
        reaches = []  # q in p, p in q, c in q, q in c
        rest.insert(random.randint(0, len(rest) - 1), pos[0])
        for r in rest:
            l = [0, 0, 0]
            rea = [0, 0, 0, 0]
            if r[0] in self.node2ancestor[q]:
                l[1] = 1
                rea[0] = 1
            if r[0] in self.node2descendant[q]:
                rea[1] = 1
            if r[1] in self.node2descendant[q]:
                l[2] = 1
                rea[2] = 1
            if r[1] in self.node2ancestor[q]:
                rea[3] = 1
            l[0] = r in self.node2pairs[q]
            labels.append(l)
            reaches.append(rea)
        return torch.Tensor(rest), torch.Tensor(labels), torch.Tensor(reaches), torch.Tensor(sims)

    # def __getitem__(self, idx):
    #     if self.mode == 'eval':
    #         return self._database[self.eva[idx]].float(), nx.shortest_path(
    #             self.whole, 0, self.eva[idx]), self.eva[idx]
    #     if self.mode == 'test':
    #         return self._database[self.test[idx]].float(), nx.shortest_path(
    #             self.whole, 0, self.test[idx]), self.test[idx]
    #
    #     # q = self.train[idx]
    #     # pairs, labels = self.sample_train_pairs(q, 10)
    #     # eq = self._database[q].unsqueeze(0)
    #     # embeds = []
    #     # for p in pairs:
    #     #     f = self._database[p[0].item()].unsqueeze(0)
    #     #     s = self._database[p[1].item()].unsqueeze(0) if p[1] != -1 else torch.zeros_like(f)
    #     #     embeds.append(torch.cat([eq, f, s]).unsqueeze(0))
    #     #
    #     # embeds = torch.cat(embeds).float()
    #     # return embeds, torch.Tensor(labels)
    #     l = []
    #     for _ in self.fetch_order:
    #         l += _
    #
    #     g = self.mini_batches[l[idx]]
    #     leaves = self._get_leaves(g)
    #     node_feature_list = []
    #     old_to_new_label_lookup = {}
    #     new_to_old_label_lookup = []
    #     for i, node in enumerate(g.nodes()):
    #         old_to_new_label_lookup[node] = i
    #         new_to_old_label_lookup.append(node)
    #         node_feature_list.append(self._database[node].unsqueeze(0))
    #     new_g = nx.relabel_nodes(g, old_to_new_label_lookup)
    #     leaves_embeds = dict(
    #         (old_to_new_label_lookup[k], self.fused_embeddings[k]) for k in leaves if k in self.fused_embeddings.keys())
    #
    #     path_sim = self.path_sim_matrix[new_to_old_label_lookup][:, new_to_old_label_lookup]
    #
    #     # old: label in original tree, new: range from 0 to len(g)
    #     # new_g and node_feature_list are one-to-one
    #     return new_g, torch.cat(
    #         node_feature_list), leaves_embeds, old_to_new_label_lookup, new_to_old_label_lookup, path_sim
    def __getitem__(self, idx):
        if self.mode == 'eval':
            return self._database[self.eva[idx]].float(), nx.shortest_path(
                self.whole, 0, self.eva[idx]), self.node2pairs[self.eva[idx]]
        elif self.mode == 'test':
            return self._database[self.test[idx]].float(), nx.shortest_path(
                self.whole, 0, self.test[idx]), self.node2pairs[self.test[idx]]
        else:
            idx += 1
            anchor = self.train[idx]
            samples, labels, reaches, rank_sims = self.sample_train_pairs(anchor, 32)
            assert labels[:, 0].sum() == 1
            embeds = []
            sims = []
            anchor_embed = self._database[anchor][0].unsqueeze(0)
            i_idx = [False] * 32
            for i, s in enumerate(samples):
                i_idx[i] = s[1] != -1
                _f = self._database[s[0].item()][0].unsqueeze(0)
                _s = self._database[s[1].item()][0].unsqueeze(0) if s[1] != -1 else torch.zeros_like(_f)
                embeds.append(torch.cat([anchor_embed, _f, _s]).unsqueeze(0))
                sims.append(
                    torch.Tensor([self.path_sim_matrix[s[0].int().item(), anchor],
                                  self.path_sim_matrix[anchor, s[1].int().item()]]).unsqueeze(0))

            return torch.cat(embeds), labels, torch.cat(sims), reaches, torch.Tensor(i_idx).bool(), rank_sims


if __name__ == "__main__":
    from utils.mkdataset import split_tree_dataset

    G, names, descriptions, train, test, eva = split_tree_dataset(
        '/data/home10b/xw/visualCon/datasets_json/imagenet_dataset.json')
    t = TreeSet(G, names, descriptions, batch_size=200)
