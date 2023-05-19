import os
import random
from collections import deque
from itertools import combinations, chain, product
from rich.progress import track

import networkx as nx
from copy import deepcopy
from torch.utils.data import Dataset
import torch
from queue import Queue
import numpy as np
from tqdm import tqdm
from transformers import RobertaModel, AutoTokenizer, CLIPTextModel

from utils.utils import batch_load_img, grouper
from torch_geometric.data import Data


class TreeSet(Dataset):
    # whole, G, names, descriptions, train, eva, test,
    def __init__(self, path, dataset_name, sample_size=32):
        super(TreeSet, self).__init__()

        self.d_name = dataset_name
        self.sample_size = sample_size

        if path.endswith('pt'):
            with open(path, 'rb') as f:
                d = torch.load(f)
        elif path.endswith('bin'):
            import pickle
            with open(path, 'rb') as f:
                d = pickle.load(f)

        self.edges = None
        self.mode = 'train'
        self.whole = d['whole']
        self.train = d['train']
        self.train.sort()
        self.eva = d['eva']
        self.test = d['test']
        self._tree = d['g']
        self._undigraph = self._tree.to_undirected()
        self._c_tree = deepcopy(self._tree)
        self.names = d['names']
        self.root = [n for n in self._tree if self._tree.in_degree(n) == 0][0]
        self.descriptions = d[
            'descriptions']  # [d.split('\t')[1] for d in d['descriptions']] if d['descriptions'] is not None else []
        self.embeds = {}
        self.fused_embeddings = {}
        self.path_sim_matrix = None
        self.node2pairs = {}
        self.node2ancestor = {}
        self.node2descendant = {}

        self._init()
        self._process()

    def shuffle(self):
        if self.mode == 'train':
            if len(self.fetch_order) != 0:
                for l in self.fetch_order:
                    random.shuffle(l)
        elif self.mode == 'eval':
            random.shuffle(self.eva)

        elif self.mode == 'test':
            random.shuffle(self.test)

    def update_box(self, i, embed):
        self.fused_embeddings[i] = embed

    def update_boxes(self, boxes, new_to_old_map):
        for i, b in enumerate(boxes):
            self.update_box(new_to_old_map[i], b)

    def clear_boxes(self):
        self.fused_embeddings = {}

    def _stats(self):
        parents_train = []
        parents_test = []
        labels_test = []
        labels_train = []
        for n in self._tree.nodes():
            if n == 0:
                continue
            parents_train.append(len(list(self._tree.predecessors(n))))
            labels_train.append(len(self.node2pairs[n]))
        for n in self.test:
            parents_test.append(len(list(self.whole.predecessors(n))))
            labels_test.append(len(self.node2pairs[n]))

        print('avg parents in seed tax:', sum(parents_train) / len(parents_train))
        print('max parents in seed tax:', max(parents_test))
        print('avg parents in test:', sum(parents_test) / len(parents_test))
        print('max parents in test:', max(parents_test))
        print('avg gt pos train:', sum(labels_train) / len(labels_train))
        print('max gt pos train:', max(labels_train))
        print('avg gt pos test:', sum(labels_test) / len(labels_test))
        print('max gt pos test:', max(labels_test))

    def _init(self):

        self.generate_node2pairs()
        # self._stats()
        self.generate_node2ancestors()
        self.generate_node2descendants()
        self.generate_edges()

    def generate_pyg_data(self, samples, q, sample_num=30):
        p_datas, c_datas = [], []
        for s in samples:
            p, c = s
            pc = list(self._tree.successors(p))
            cc = list(self._tree.successors(c)) if c != -1 else []
            if q in pc:
                pc.remove(q)
            if q in cc:
                cc.remove(q)
            if 0 < sample_num < len(pc):
                pc = random.sample(pc, sample_num)
            if 0 < sample_num < len(cc):
                cc = random.sample(cc, sample_num)
            p_features = [self.embeds[n][0].unsqueeze(0) for n in [p] + pc]
            p_edge_index = torch.Tensor([[1 + i, 0] for i in range(len(pc))] + [[0, 0]]).t().contiguous().long()
            p_edge_index = torch.cat([p_edge_index, p_edge_index.flip(0)], dim=1)
            p_data = Data(x=torch.cat(p_features), edge_index=p_edge_index)

            c_features = [self.embeds[n][0].unsqueeze(0) if n != -1 else torch.zeros_like(self.embeds[0]) for n in
                          [c] + cc]
            c_edge_index = torch.Tensor([[1 + i, 0] for i in range(len(cc))] + [[0, 0]]).t().contiguous().long()
            c_edge_index = torch.cat([c_edge_index, c_edge_index.flip(0)], dim=1)
            c_data = Data(x=torch.cat(c_features), edge_index=c_edge_index)

            p_datas.append(p_data), c_datas.append(c_data)
        return p_datas, c_datas

    def generate_edges(self):
        candidates = set(chain.from_iterable([[(n, d) for d in ds] for n, ds in self.node2descendant.items()]))
        print('Number of candidates', len(candidates))
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

    def generate_node2ancestors(self):
        for n in self._tree.nodes():
            self.node2ancestor[n] = list([node for node in nx.ancestors(self.whole, n) if node in self.train])

    def generate_node2descendants(self):
        for n in self._tree.nodes():
            self.node2descendant[n] = list(nx.descendants(self._tree, n)) + [-1]

    def _check_saved(self):
        return os.path.exists(self.d_name + '.feature.pt') and os.path.exists(self.d_name + '.pathsim.pt')

    def _process(self):
        if os.path.exists(self.d_name + '.feature.pt'):
            with open(self.d_name + '.feature.pt', 'rb') as f:
                self.embeds = torch.load(f)
            for i in self.embeds.keys():
                self.embeds[i] = torch.nn.functional.normalize(self.embeds[i].float(), p=2, dim=-1)
        else:
            tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

            for n in track(self.whole.nodes(), description='Encoding nodes'):
                description = self.names[n].split('@')[0]  # + ',' + self.descriptions[n][0]
                inputs = tokenizer(description, padding=True, return_tensors='pt', truncation=True)

                with torch.no_grad():
                    text_embedding = text_encoder(**inputs)['pooler_output']

                self.embeds[n] = text_embedding
            torch.save(self.embeds, self.d_name + '.feature.pt')

        if os.path.exists(self.d_name + '.pathsim.pt'):
            with open(self.d_name + '.pathsim.pt', 'rb') as f:
                self.path_sim_matrix = torch.load(f)
        else:
            self.path_sim_matrix = torch.zeros(len(self.names), len(self.names))
            print("generating path sim mat")

            def cal_sim(comb):
                r, c = comb
                # path_sim: a matrix containing all train pairs with whole id
                self.path_sim_matrix[r, c] = self.path_sim(r, c)
                self.path_sim_matrix[c, r] = self.path_sim_matrix[r, c]

            for comb in track(combinations(list(self._tree.nodes()), r=2)):
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

    def hard_negative_sample(self, q, num):
        # ans = list(combinations(self.node2ancestor[q], r=2))
        des = deepcopy(self.node2descendant[q])
        des.remove(-1) if -1 in des else None
        ans = self.node2ancestor[q]

        # if len(des) == 0:
        #     return []
        # if len(des) == 1:
        #     tmp = [(_, des[0]) for _ in ans]
        #     num = min(num, len(tmp))
        #     res = random.sample(tmp, num)
        #     list([res.remove(i) for i in self.node2pairs[q] if i in res])
        #     return res
        if len(des) <= 1:
            return []
        res = []
        if len(des) * (len(des) - 1) < num // 3 * 2:
            res += list(combinations(des, r=2))
        else:
            while len(res) < num // 3 * 2:
                sampled = (random.choice(des), random.choice(des))
                res.append(sampled) if sampled not in res else None
        if len(des) * len(ans) < num // 3:
            res += list([(i, j) for i in ans for j in des])
        else:
            while len(res) < num:
                a = (random.choice(ans), random.choice(des))
                res.append(a)
        res = list(set(res).difference(set(self.node2pairs[q])))
        return res

    def sample_train_pairs(self, q, num, pos_num=1):
        pos = random.sample(self.node2pairs[q], k=pos_num) if pos_num <= len(self.node2pairs[q]) else self.node2pairs[q]
        # neg = [list(e) for e in self.edges if e[0] != q and e[1] != q and e not in self.node2pairs[q]]
        remain = num - len(pos)
        if remain > 0:
            # hard negative sample
            rest = []  # self.hard_negative_sample(q, remain // 3)

            while len(rest) < remain:
                rest += [e for e in random.sample(self.edges, remain - len(rest)) if
                         e[0] != q and e[1] != q and e not in self.node2pairs[q] and e not in rest]
        else:
            rest = random.sample(pos, num)
        assert len(rest) == remain
        sims = self.path_sim_matrix[pos[0][0], [r[0] for r in rest]]
        labels = []
        reaches = []  # q in p, p in q, c in q, q in c
        rest.insert(random.randint(0, len(rest) - 1), pos[0])
        for r in rest:
            rea = [r[0] in self.node2ancestor[q], r[0] in self.node2descendant[q],
                   r[1] in self.node2descendant[q], r[1] in self.node2ancestor[q]]
            labels.append(r in self.node2pairs[q])
            reaches.append(rea)
        assert sum(labels) == 1
        return rest, torch.Tensor(labels).int(), torch.Tensor(reaches).int(), torch.Tensor(sims)

    def __getitem__(self, idx):
        if self.mode == 'eval':
            anchor = self.eva[idx]
            return self.embeds[anchor], nx.shortest_path(
                self.whole, 0, self.eva[idx]), self.node2pairs[self.eva[idx]], list(
                [1 if p[1] == -1 else 0 for p in self.node2pairs[self.eva[idx]]]), self.names[self.eva[idx]]
        elif self.mode == 'test':
            anchor = self.test[idx]
            return self.embeds[anchor], nx.shortest_path(
                self.whole, 0, self.test[idx]), self.node2pairs[self.test[idx]], list(
                [1 if p[1] == -1 else 0 for p in self.node2pairs[self.test[idx]]]), self.names[self.test[idx]]
        else:
            idx += 1
            anchor = self.train[idx]
            samples, labels, reaches, rank_sims = self.sample_train_pairs(anchor, self.sample_size)
            assert labels.sum() == 1
            sims = []
            anchor_embed = self.embeds[anchor]
            i_idx = [False] * self.sample_size
            for i, s in enumerate(samples):
                i_idx[i] = s[1] != -1
                sims.append(
                    torch.Tensor([self.path_sim_matrix[s[0], anchor],
                                  self.path_sim_matrix[anchor, s[1]]]).unsqueeze(0))
            p_datas, c_datas = self.generate_pyg_data(samples, anchor)
            return anchor_embed, p_datas, c_datas, labels, torch.cat(
                sims), reaches, torch.Tensor(i_idx).bool(), rank_sims


if __name__ == "__main__":
    from utils.mkdataset import split_tree_dataset

    G, names, descriptions, train, test, eva = split_tree_dataset(
        '/data/home10b/xw/visualCon/datasets_json/imagenet_dataset.json')
    t = TreeSet(G, names, descriptions, batch_size=200)
