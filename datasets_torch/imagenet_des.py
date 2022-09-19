from torch.utils.data.dataset import Dataset
import json
from PIL import Image


class ImageNet_Des(Dataset):
    def __init__(self, json_root, split, transform):
        self.json_root = json_root
        self.split = split
        self.transform = transform
        datas = {}

        with open(self.json_root, 'r') as f:
            datas = json.load(f)

        assert self.split in ('train', 'val', 'test')
        self.image = []
        self.des = []
        self.name = []
        self.id = []

        for k, v in datas.items():
            self.des += [v['descriptions']] * len(v[self.split])
            self.image += v[self.split]
            self.name += [v['name'][0]] * len(v[self.split])
            self.id += [k] * len(v[self.split])

        assert len(self.image) == len(self.des) == len(self.id) == len(self.name)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        img = Image.open(self.image[idx]).convert('RGB')
        return self.transform(img), self.des[idx], self.name[idx], self.id[idx]
