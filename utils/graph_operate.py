import json
import sys

sys.path.append("..")
import torch
from .utils import retrieve_model, extract_features_for_imgs, log_conditional_prob, conditional_prob
from datasets_torch.handcrafted import encode_description
import numpy as np
from torch_geometric.data import Data


def _insert_node(raw_graph, graph_embeddings, model, novel_node_des: (str, str), novel_imgs: str):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img_encoder, preprocess = retrieve_model('clip', device)
    img_features = extract_features_for_imgs(img_encoder, preprocess, novel_imgs, novel_node_des[0])
    img_features_offsets = np.zeros_like(img_features)
    img_features_offsets[:, :] = 1e-6
    img_features = np.hstack((img_features, img_features_offsets))
    e_des = encode_description(','.join(novel_node_des)).cpu().numpy()
    e_des_offset = np.random.normal(size=e_des.shape).__abs__()
    e_des = np.hstack([e_des, e_des_offset])
    nodes = np.vstack([img_features, e_des])
    edges = np.array([list(range(len(nodes) - 1)), [len(nodes) - 1] * img_features.shape[0]])
    novel_data = Data(x=torch.Tensor(nodes), edge_index=torch.Tensor(edges).type(torch.long)).to(device)
    outs = model(novel_data)
    novel_node = outs[-1]
    head = raw_graph['object']
    while 'children' in head.keys():
        max_prob = 0
        choose_child = None
        for child in head['children']:
            cur_node = graph_embeddings[child[[list(child.keys())[0]]]['id']]
            prob_cur_in_novel = conditional_prob(novel_node, cur_node)
            prob_novel_in_cur = conditional_prob(cur_node, novel_node)
            if prob_novel_in_cur > prob_cur_in_novel and prob_novel_in_cur > max_prob:
                choose_child = child
            else:
                pass

        head = choose_child
    print(head)


def test_on_insert(path_to_json, raw_graph, graph_embeddings, model):
    test_nodes = json.load(open(path_to_json))
    for k, v in test_nodes.items():
        des = v['description']
        img_lists = '/data/home10b/xw/visualCon/test_handcrafted/'
        _insert_node(raw_graph, graph_embeddings, model, (k, des), img_lists)
