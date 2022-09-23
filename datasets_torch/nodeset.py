import random

import torch
import json

from torch.utils.data import Dataset
import os.path as osp
import os
from PIL import Image

class NodeSet(Dataset):
    def __init__(self, root, img_root, img_transform, text_tokenize, max_imgs_per_node=500):
        self.transform = img_transform
        self.tokenize = text_tokenize
        self.raw_graph = None
        self.name_to_id = {}
        self.id_to_name = []
        self.id_to_imgs = []
        self.id_to_children = []
        self.id_to_text = []
        self.root = root
        self.img_root = img_root
        self.max_imgs_per_node = max_imgs_per_node

        self._process()

    def check_whether_processed(self):
        ls = [osp.join(self.root, f) for f in self.processed_files]
        return len(ls) != 0 and all(osp.exists(f) for f in ls)

    @property
    def processed_files(self):
        return ['data.pt']

    @property
    def raw_files(self):
        return ['handcrafted.json']

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

            def _traverse_tree(head, _id):
                assert len(self.id_to_name) == _id and len(self.id_to_children) == _id and \
                       len(self.id_to_text) == _id and len(self.id_to_imgs) == _id
                if 'children' not in head.keys():
                    head['id'] = _id
                    self.name_to_id[head['name']] = _id
                    self.id_to_name.append(head['name'])
                    self.id_to_text.append(head['name'] + ',' + head['description'])
                    self.id_to_children.append([])
                    self.id_to_imgs.append([osp.join(osp.join(self.img_root, head['name'], f)) for f in
                                            os.listdir(osp.join(self.img_root, head['name']))])
                    del head['name']
                    del head['description']
                else:
                    self.name_to_id[head['name']] = _id
                    self.id_to_name.append(head['name'])
                    self.id_to_text.append(head['name'] + ',' + head['description'])
                    self.id_to_imgs.append([])
                    self.id_to_children.append(list(range(_id + 1, _id + len(head['children']) + 1)))
                    for i, child in enumerate(head['children']):
                        _traverse_tree(child, _id + i + 1)

                    del head['name']
                    del head['description']

            _traverse_tree(raw, 0)
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

        imgs = getimgs(idx)
        imgs = random.choices(imgs, k=self.max_imgs_per_node) if len(imgs) > self.max_imgs_per_node else imgs
        inputs = []
        for i in imgs:
            try:
                inputs.append(self.transform(Image.open(i).convert('RGB')))
            except:
                continue

        return {
            'id': idx,
            'name': self.id_to_name[idx],
            'text': t_des,
            'imgs': inputs
        }
