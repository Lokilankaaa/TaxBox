import random

import torch
import json

from torch.utils.data import Dataset
import os.path as osp
import os
from PIL import Image
import numpy as np
from utils.utils import batch_load_img


class NodeSet(Dataset):
    def __init__(self, root, img_root, img_transform, text_tokenize, max_imgs_per_node=50, train_mask=None):
        self.transform = img_transform
        self.tokenize = text_tokenize
        self.raw_graph = None
        self.name_to_id = {}
        self.id_to_name = []
        self.id_to_imgs = []
        self.id_to_children = []
        self.id_to_text = []
        self._id = 0
        self.root = root
        self.img_root = img_root
        self.max_imgs_per_node = max_imgs_per_node
        self.train_mask = train_mask

        self._process()

        if train_mask is not None:
            assert len(train_mask) == len(self.id_to_name)
            self.valid_mask = np.logical_not(train_mask)

    def check_whether_processed(self):
        ls = [osp.join(self.root, f) for f in self.processed_files]
        return len(ls) != 0 and all(osp.exists(f) for f in ls)

    @property
    def processed_files(self):
        return ['data.pt']

    @property
    def raw_files(self):
        return ['middle_handcrafted.json']

    def _process(self):
        if self.check_whether_processed():
            d = torch.load(osp.join(self.root, self.processed_files[0]))
            # a graph containing label taxonomy which is used for sampling path during training
            self.raw_graph = d['raw_graph']
            self.id_to_name = d['id_to_name']  # list
            self.name_to_id = d['name_to_id']
            self.id_to_children = d['id_to_children']  # list
            self.id_to_imgs = d['id_to_imgs']  # list
            self.id_to_text = d['id_to_text']  # list
            return
        else:
            raw = json.load(open(osp.join(self.root, self.raw_files[0])))

            def _traverse_tree(head):
                assert len(self.id_to_name) == self._id and len(self.id_to_children) == self._id and \
                       len(self.id_to_text) == self._id and len(self.id_to_imgs) == self._id
                if len(head['children']) == 0:
                    head['id'] = self._id
                    self.name_to_id[head['name']] = self._id
                    self.name_to_id[head['name']] = self._id
                    self.id_to_name.append(head['name'])
                    self.id_to_text.append(head['name'] + ',' + head['description'])
                    self.id_to_children.append([])
                    self.id_to_imgs.append([osp.join(osp.join(self.img_root, head['name'], f)) for f in
                                            os.listdir(osp.join(self.img_root, head['name']))])
                    self._id += 1
                else:
                    head['id'] = self._id
                    self.name_to_id[head['name']] = self._id
                    self.id_to_name.append(head['name'])
                    self.id_to_text.append(head['name'] + ',' + head['description'])
                    self.id_to_imgs.append([])
                    self.id_to_children.append([])
                    self._id += 1
                    for i, child in enumerate(head['children']):
                        _traverse_tree(child)
                    self.id_to_children[head['id']] = [i['id'] for i in head['children']]

                del head['name']
                del head['description']

            _traverse_tree(raw)
            self.raw_graph = raw
            torch.save({'raw_graph': self.raw_graph, 'id_to_imgs': self.id_to_imgs, 'id_to_text': self.id_to_text,
                        'id_to_name': self.id_to_name, 'id_to_children': self.id_to_children,
                        'name_to_id': self.name_to_id}, osp.join(self.root, self.processed_files[0]))

    def __len__(self):
        return len(self.id_to_name)

    def __getitem__(self, idx):
        t_des = self.tokenize(self.id_to_text[idx], truncate=True)

        def getimgs(_i):
            if len(self.id_to_imgs[_i]) == 0:
                res = []
                for c in self.id_to_children[_i]:
                    res += getimgs(c)
                return res
            else:
                return self.id_to_imgs[_i]

        imgs_ = getimgs(idx)
        inputs = batch_load_img(imgs_, self.transform, self.max_imgs_per_node)

        return idx, self.id_to_name[idx], t_des, torch.cat(inputs, dim=1)
