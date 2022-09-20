import json

import torch
from torch_geometric.data import Dataset, InMemoryDataset, Data
import os
import numpy as np
import clip

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model, _ = clip.load('ViT-B/32', device)


def encode_description(descriptions):
    model.eval()
    descriptions = descriptions if type(descriptions) == list else [descriptions]

    text_inputs = torch.cat([clip.tokenize(d, truncate=True) for d in descriptions]).to(device)
    with torch.no_grad():
        des_embeddings = model.encode_text(text_inputs)
    return des_embeddings


class Handcrafted(InMemoryDataset):
    def __init__(self, root, pre_transforms=None, transforms=None, pre_filters=None):
        self.edges = []
        self.nodes = []
        self.raw_classes_graph = None
        super().__init__(root, transforms, pre_transforms, pre_filters)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['handcrafted.json'] + os.listdir(os.path.join(self.raw_dir, 'features'))

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        self.raw_classes_graph = json.load(open(self.raw_paths[0]))
        self.traverse_tree(self.raw_classes_graph)

        data_list = [Data(x=torch.Tensor(self.nodes), edge_index=torch.Tensor(np.array(self.edges)).type(torch.long),
                          raw_graph=self.raw_classes_graph)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def traverse_tree(self, head):
        for k, v in head.items():
            if 'children' not in v.keys():
                des = k + ',' + v['description']
                e_des = encode_description(des)[0].cpu().numpy()
                img_features = np.load(os.path.join(self.raw_dir, 'features', k + '.npy'))
                img_features_cov = np.cov(img_features.T)
                img_features_offsets = np.zeros_like(img_features)
                img_features_offsets[:, :] = 1e-6
                img_features = np.hstack((img_features, img_features_offsets))
                # may be change to use the variance of image features
                e_des_offset = np.random.multivariate_normal(np.zeros_like(e_des), img_features_cov).__abs__()
                e_des = np.hstack([e_des, e_des_offset])
                start = len(self.nodes)
                num = img_features.shape[0]
                self.nodes = img_features if len(self.nodes) == 0 else np.vstack([self.nodes, img_features])
                self.nodes = np.vstack([self.nodes, e_des])
                new_edges = np.array([list(range(start, start + num)), [start + num] * num])
                self.edges = new_edges if len(self.edges) == 0 else np.hstack([self.edges, new_edges])
                v['id'] = start + num
                return start + num
            else:
                child_node_index = []
                for child in v['children']:
                    child_node_index.append(self.traverse_tree(child))
                e_des = encode_description(v['description'])[0].cpu().numpy()
                e_des_offset = np.random.normal(size=e_des.shape).__abs__()
                e_des = np.hstack([e_des, e_des_offset])
                self.nodes = np.vstack([self.nodes, e_des])
                ind = len(self.nodes) - 1
                self.edges = np.hstack([self.edges, np.array([child_node_index, [ind] * len(child_node_index)])])
                v['id'] = ind
                return ind


if __name__ == '__main__':
    d = Handcrafted('/data/home10b/xw/visualCon/datasets_json/handcrafted')
