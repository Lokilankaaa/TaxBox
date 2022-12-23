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

from utils.utils import batch_load_img, grouper
from torch_geometric.data import Data

import multiprocessing as mp


class TreeSet(Dataset):
    # whole, G, names, descriptions, train, eva, test,
    def __init__(self, path, dataset_name, graph_emb=False):
        super(TreeSet, self).__init__()

        self.d_name = dataset_name

        if path.endswith('pt'):
            d = torch.load(path)
        elif path.endswith('bin'):
            import pickle
            with open(path, 'rb') as f:
                d = pickle.load(f)

        self.graph_emb = graph_emb
        self.edges = None
        self.mode = 'train'
        self.whole = d['whole']
        self.train = d['train']
        self.eva = d['eva']
        self.test = d['test']
        self._tree = d['g']
        self._undigraph = self._tree.to_undirected()
        self._c_tree = deepcopy(self._tree)
        self.names = d['names']
        self.descriptions = d['descriptions']
        self.leaves = []
        self.paths = {}
        self.min_depth = 0
        self.max_depth = 0
        self.mean_depth = 0
        self.mean_order = 0
        self._database = {}
        self.fused_embeddings = {}
        self.path_sim_matrix = None
        self.node2pairs = {}
        self.node2ancestor = {}
        self.node2descendant = {}
        self.pyg_data = {}

        self._init()
        self._process()
        # self.generate_pyg_data()
        # self.shuffle()

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

    def _init(self):
        self._get_leaves()
        self._get_all_paths()
        # self._stats()
        # self._form_mini_batches()
        self.generate_node2pairs()
        self.generate_node2ancestors()
        self.generate_node2descendants()
        self.generate_edges()

    def generate_pyg_data(self, samples, q):
        p_datas, c_datas = [], []
        for s in samples:
            p, c = s
            pc = list(self._tree.successors(p))
            cc = list(self._tree.successors(c)) if c != -1 else []
            if q in pc:
                pc.remove(q)
            if q in cc:
                cc.remove(q)
            p_features = [self._database[n][0].unsqueeze(0) for n in [p] + pc]
            p_edge = [[1 + i, 0] for i in range(len(pc))] + [[0, 0]]
            p_data = Data(x=torch.cat(p_features), edge_index=torch.Tensor(p_edge).t().contiguous().int())

            c_features = [self._database[n][0].unsqueeze(0) if n != -1 else torch.zeros_like(self._database[0]) for n in [c] + cc]
            c_edge = [[1 + i, 0] for i in range(len(cc))] + [[0, 0]]
            c_data = Data(x=torch.cat(c_features), edge_index=torch.Tensor(c_edge).t().contiguous().int())

            p_datas.append(p_data), c_datas.append(c_data)
        return p_datas, c_datas

    def generate_edges(self):
        candidates = set(chain.from_iterable([[(n, d) for d in ds] for n, ds in self.node2descendant.items()]))
        self.edges = candidates  # list(self._tree.edges()) + list([[n, -1] for n in self._tree.nodes()])

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
        return os.path.exists(self.d_name + '.feature.pt') and os.path.exists(self.d_name + '.pathsim.pt')

    def _process(self):
        if os.path.exists(self.d_name + '.feature.pt'):
            self._database = torch.load(self.d_name + '.feature.pt')
            for i in self._database.keys():
                self._database[i] = torch.nn.functional.normalize(self._database[i].float(), p=2, dim=-1)
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
            torch.save(self._database, self.d_name + '.feature.pt')

        if os.path.exists(self.d_name + '.pathsim.pt'):
            self.path_sim_matrix = torch.load(self.d_name + '.pathsim.pt')
        else:
            self.path_sim_matrix = torch.zeros(len(self._database), len(self._database))
            print("generating path sim mat")

            def cal_sim(comb):
                r, c = comb
                # path_sim: a matrix containing all train pairs with whole id
                self.path_sim_matrix[r, c] = self.path_sim(r, c)
                self.path_sim_matrix[c, r] = self.path_sim_matrix[r, c]

            for comb in tqdm(combinations(list(self._tree.nodes()), r=2)):
                cal_sim(comb)

            # list(map(cal_sim, combinations(self._tree.nodes(), r=2)))
            torch.save(self.path_sim_matrix, self.d_name + '.pathsim.pt')

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

    def sample_train_pairs(self, q, num, pos_num=1):
        pos = random.sample(self.node2pairs[q], k=pos_num) if pos_num <= len(self.node2pairs[q]) else self.node2pairs[q]

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
        return rest, torch.Tensor(labels), torch.Tensor(reaches), torch.Tensor(sims)

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
            samples, labels, reaches, rank_sims = self.sample_train_pairs(anchor, 16)
            # assert labels[:, 0].sum() == 1
            embeds = []
            sims = []
            anchor_embed = self._database[anchor][0].unsqueeze(0)
            i_idx = [False] * 16
            for i, s in enumerate(samples):
                i_idx[i] = s[1] != -1
                # _f = self._database[s[0].item()][0].unsqueeze(0)
                # _s = self._database[s[1].item()][0].unsqueeze(0) if s[1] != -1 else torch.zeros_like(_f)
                # embeds.append(torch.cat([anchor_embed, _f, _s]).unsqueeze(0))
                sims.append(
                    torch.Tensor([self.path_sim_matrix[s[0], anchor],
                                  self.path_sim_matrix[anchor, s[1]]]).unsqueeze(0))
            p_datas, c_datas = self.generate_pyg_data(samples, anchor)
            return anchor_embed, p_datas, c_datas, labels, torch.cat(
                sims), reaches, torch.Tensor(i_idx).bool(), rank_sims
            # return torch.cat(embeds), labels, torch.cat(sims), reaches, torch.Tensor(i_idx).bool(), rank_sims


if __name__ == "__main__":
    from utils.mkdataset import split_tree_dataset

    G, names, descriptions, train, test, eva = split_tree_dataset(
        '/data/home10b/xw/visualCon/datasets_json/imagenet_dataset.json')
    t = TreeSet(G, names, descriptions, batch_size=200)
