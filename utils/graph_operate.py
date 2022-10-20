import json
import os
import random

import torch
from .utils import retrieve_model, extract_features_for_imgs, log_conditional_prob, conditional_prob, \
    get_graph_box_embedding, batch_load_img
from datasets_torch.handcrafted import encode_description
import numpy as np
import clip


def _insert_node(k_embeddings, dataset, model, novel_node, clip_model, trans):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    des, img_paths = novel_node
    e_des = clip.tokenize(des, truncate=True).to(device)
    imgs = batch_load_img(img_paths, trans, max_k=dataset.max_imgs_per_node)
    imgs = torch.cat(imgs).to(device)
    e_des = clip_model.encode_text(e_des).unsqueeze(0)
    imgs = clip_model.encode_image(imgs).unsqueeze(0)
    novel_embed = model.query_transformer(e_des, imgs)

    head = dataset.raw_graph
    threshold = 0.5
    while len(head['children']) > 0:
        max_prob_novel_in_c = -1
        max_prob_c_in_novel = -1
        choose_child = None
        choose_father = None
        for c in head['children']:
            c_embed = k_embeddings[c['id']]
            prob_c_in_novel = conditional_prob(novel_embed, torch.from_numpy(c_embed).unsqueeze(0).to(device))
            prob_novel_in_c = conditional_prob(torch.from_numpy(c_embed).unsqueeze(0).to(device), novel_embed)
            if prob_novel_in_c > prob_c_in_novel:
                if prob_novel_in_c > max_prob_novel_in_c and prob_novel_in_c > threshold:
                    max_prob_novel_in_c = prob_novel_in_c
                    choose_child = c
            else:
                if prob_c_in_novel > max_prob_c_in_novel and prob_c_in_novel > threshold:
                    max_prob_c_in_novel = prob_c_in_novel
                    choose_father = c

        if choose_father is None and choose_child is None:
            print('{} is son of {}'.format(des.split(',')[0], dataset.id_to_name[head['id']]))
            break
        elif max_prob_c_in_novel < max_prob_novel_in_c:
            head = choose_child
        else:
            print('{} is father of {}'.format(des.split(',')[0], dataset.id_to_name[choose_father['id']]))
            break
    if len(head['children']) == 0:
        print('{} is son of {}'.format(des.split(',')[0], dataset.id_to_name[head['id']]))


# def _insert_node(raw_graph, graph_embeddings, model, novel_node_des: (str, str), novel_imgs: str):
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     img_encoder, preprocess = retrieve_model('clip', device)
#     img_features = extract_features_for_imgs(img_encoder, preprocess, novel_imgs, novel_node_des[0])
#     img_features_offsets = np.zeros_like(img_features)
#     img_features_offsets[:, :] = 1e-6
#     img_features = np.hstack((img_features, img_features_offsets))
#     e_des = encode_description(','.join(novel_node_des)).cpu().numpy()
#     e_des_offset = np.random.normal(size=e_des.shape).__abs__()
#     e_des = np.hstack([e_des, e_des_offset])
#     nodes = np.vstack([img_features, e_des])
#     edges = np.array([list(range(len(nodes) - 1)), [len(nodes) - 1] * img_features.shape[0]])
#     novel_data = Data(x=torch.Tensor(nodes), edge_index=torch.Tensor(edges).type(torch.long)).to(device)
#     outs = model(novel_data)
#     novel_node = outs[-1]
#     cur_name = 'object'
#     head = raw_graph[cur_name]
#     choose_child = None
#     while 'children' in head.keys():
#         max_log_prob = -99999
#         choose_child = None
#         for child in head['children']:
#             cur_node = graph_embeddings[child[list(child.keys())[0]]['id']]
#             prob_cur_in_novel = log_conditional_prob(novel_node.unsqueeze(0), cur_node.unsqueeze(0))
#             prob_novel_in_cur = log_conditional_prob(cur_node.unsqueeze(0), novel_node.unsqueeze(0))
#             if prob_novel_in_cur > prob_cur_in_novel and prob_novel_in_cur > max_log_prob:
#                 choose_child = child
#                 max_log_prob = prob_novel_in_cur
#             else:
#                 pass
#         if choose_child is None:
#             break
#         else:
#             cur_name = list(choose_child.keys())[0]
#             head = choose_child[cur_name]
#     if choose_child is None:
#         print('{} is son of {}'.format(novel_node_des[0], cur_name))
#     else:
#         print('{} is sibling of {}'.format(novel_node_des[0], cur_name))


def test_on_insert(dataset, model, path_to_json, box_dim):
    test_nodes = json.load(open(path_to_json))
    # print(test_nodes)
    c, p = clip.load('ViT-B/32')
    k_embeddings = get_graph_box_embedding(dataset, model, box_dim, False, False)
    img_path = '/data/home10b/xw/visualCon/test_handcrafted/'
    for node in test_nodes:
        des = node['name'] + ',' + node['description']
        img_lists = [os.path.join(img_path, node['name'], i) for i in os.listdir(os.path.join(img_path, node['name']))]
        _insert_node(k_embeddings, dataset, model, (des, img_lists), c, p)


# def test_on_insert(path_to_json, raw_graph, graph_embeddings, model):
#     test_nodes = json.load(open(path_to_json))
#     for k, v in test_nodes.items():
#         des = v['description']
#         img_lists = '/data/home10b/xw/visualCon/test_handcrafted/'
#         _insert_node(raw_graph, graph_embeddings, model, (k, des), img_lists)
#

def test_entailment(raw_graph, graph_embeddings):
    head_name = 'object'
    head = raw_graph[head_name]
    while 'children' in head.keys():
        for child in head['children']:
            cur_name = list(child.keys())[0]
            prob = conditional_prob(graph_embeddings[head['id']], graph_embeddings[cur_name['id']])
