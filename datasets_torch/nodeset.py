import torch
import json

from torch.utils.data import Dataset
import os.path as osp
import os
import queue

class NodeSet(Dataset):
    def __init__(self, root, img_root, img_transform, text_tokenize):
        self.transform = img_transform
        self.tokenize = text_tokenize
        self.raw_graph = None
        self.name_to_id = None
        self.id_to_name = None
        self.id_to_imgs = None
        self.descendants = None
        self.root = root
        self.img_root = img_root

        self._process()

    def check_whether_processed(self):
        ls = [osp.join(self.root, f) for f in self.processed_files]
        return len(ls) != 0 and all(osp.exists(f) for f in ls)

    @property
    def processed_files(self):
        return ['data.pt']

    @property
    def raw_files(self):
        return ['']

    def _process(self):
        if self.check_whether_processed():
            d = torch.load(osp.join(self.root, self.processed_files[0]))
            # a graph containing label taxonomy which is used for sampling path during training
            self.raw_graph = d['raw_graph']
            self.id_to_name = d['id_to_name']
            self.name_to_id = d['name_to_id']
            self.descendants = d['descendants']
            self.id_to_imgs = d['id_to_imgs']
            return
        else:
            self.raw_graph = {}
            raw = json.load(open(osp.join(self.root, self.raw_files[0])))
            q = queue.Queue()
            q.put(raw)
            while not q.empty():
                pass

