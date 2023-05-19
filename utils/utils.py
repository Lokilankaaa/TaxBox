import json
from typing import Union, List

import networkx
import requests
from PIL import Image
from matplotlib import pyplot as plt
# from scipy.stats import gaussian_kde, norm, kurtosis
# from scipy.cluster import vq, hierarchy
import torch
import os
import numpy as np
import clip
import random
import torch.nn.functional as F
import multiprocessing as mp
from itertools import combinations, zip_longest
import math
from typing import Union
import re
import itertools

EPS = 1e-20


def adjust_moco_momentum(epoch, m, total_epochs):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / total_epochs)) * (1. - m)
    return m


def batch_load_img(imgs_, transform, max_k=50):
    imgs = random.sample(imgs_, k=max_k) if len(imgs_) > max_k else imgs_
    inputs = []
    while len(imgs) != 0:
        i = imgs[0]
        imgs.remove(i)
        try:
            inputs.append(transform(Image.open(i).convert('RGB')).unsqueeze(0))
        except:
            imgs.append(random.choice(imgs_))
    return inputs


def get_graph_box_embedding(dataset, model, dims, save=False, load=False):
    if load and os.path.exists('k_embeddings.npy'):
        k_embeddings = np.load('k_embeddings.npy')
        return k_embeddings

    k_embeddings = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    for d in dataset:
        _, _, t, i = d
        t = t.to(device)
        i = i.to(device)
        model.eval()
        with torch.no_grad():
            k_out = model.key_transformer(t, i)
            k_embeddings.append(k_out.cpu().numpy())

    k_embeddings = np.array(k_embeddings).reshape(-1, dims)
    if save:
        np.save('k_embeddings.npy', k_embeddings)

    return k_embeddings


def checkpoint(path_to_save, model):
    print('saving model to ' + path_to_save)
    if type(model) == torch.nn.DataParallel:
        sd = model.module.state_dict()
    else:
        sd = model.state_dict()
    torch.save(sd, path_to_save)


def save_state(path_to_save, scheduler, optimizer, e):
    print("saving training state to " + path_to_save)
    torch.save({'optimizer': optimizer.state_dict(), 'e': e}, path_to_save)


def id_to_ascendants(_id, id_to_father):
    path = [_id]
    head = _id
    while id_to_father[head] != -1:
        path.insert(0, id_to_father[head])
        head = id_to_father[head]
    path.insert(0, 0)
    return path


# -1: node1 under node2, 1: node2 under node1, 0: neg
def check_pos_neg(node1, node2, id_to_father):
    if node1 == node2:
        return 2
    p1 = id_to_ascendants(node1, id_to_father)
    p2 = id_to_ascendants(node2, id_to_father)
    if node2 in p1:
        return -1
    if node1 in p2:
        return 1
    return 0


# n^2 / 2
def check_pairwise_pos_neg(node_lists, id_to_father):
    num_workers = 24
    size_list = len(node_lists)
    ma = [0] * size_list * size_list
    combs = list(combinations(node_lists, 2))

    def check_comb(cs, m):
        for c in cs:
            res = check_pos_neg(c[0], c[1], id_to_father)
            if res == -1:
                m[node_lists.index(c[1]) * size_list + node_lists.index(c[0])] = 1
            elif res == 1:
                m[node_lists.index(c[0]) * size_list + node_lists.index(c[1])] = 1
            elif res == 2:
                m[node_lists.index(c[0]) * size_list + node_lists.index(c[1])] = 1
                m[node_lists.index(c[1]) * size_list + node_lists.index(c[0])] = 1

    if len(combs) < 20 * num_workers:
        check_comb(combs, ma)
    else:
        ma = mp.Array('i', ma)
        combs = torch.Tensor(combs).chunk(num_workers, 0)
        pool = [mp.Process(target=check_comb, args=(c.to(torch.int).numpy().tolist(), ma)) for c in combs]
        list([p.start() for p in pool])
        list([p.join() for p in pool])

    return ma


def sample_n_nodes(k, id_to_father):
    N = 300
    res = random.sample(range(1, N), k=k // 2)
    fs = []
    for r in res:
        if id_to_father[r] != -1:
            a = r
            while id_to_father[r] in res or id_to_father[r] in fs:
                r = id_to_father[r]
            if id_to_father[r] != -1:
                fs.append(id_to_father[r])
            else:
                fs.append(0)
        else:
            fs.append(0)
    res += fs
    ma = check_pairwise_pos_neg(res, id_to_father)
    ma = np.array(ma).reshape((len(res), -1))
    np.fill_diagonal(ma, 1)
    return res, ma


def sample_pair(path, relation='descendant'):
    assert relation in ('son', 'descendant')
    if type(path) != tuple:
        pathx = path
        path_len = len(pathx)
        anchor = random.randint(0, path_len - 2)
        if relation == 'son':
            return pathx[anchor], pathx[anchor + 1]
        else:
            return pathx[anchor], pathx[random.randint(anchor + 1, path_len - 1)]
    else:
        pathx, pathy = path
        start = check_common_path(pathx, pathy)
        anchorx = random.randint(start, len(pathx) - 1)
        anchory = random.randint(start, len(pathy) - 1)
        return pathx[anchorx], pathy[anchory]


def check_common_path(seq_x, seq_y):
    i = 0
    while seq_x[i] == seq_y[i]:
        i += 1
    return i


def check_same_path(seq_x, seq_y):
    # if type(seq_x) == list:
    #     seq_x = np.array(seq_x, dtype=np.int32)
    #     if seq_x.ndim == 1:
    #         seq_x = np.expand_dims(seq_x, axis=0)
    if type(seq_y) == list:
        seq_y = np.array(seq_y, dtype=np.int32)
    for s in seq_x:
        s = np.array(s, dtype=np.int32)
        if len(s) == len(seq_y) and (s - seq_y).sum(-1) == 0:
            return True

    return False


def sample_triples(raw_graph, num=1, n_num=4):
    pointer = []
    paths = sample_path(raw_graph, num, pointer)
    triple = []
    for i, p in enumerate(paths):
        anchor = random.choice(range(1, len(p)))
        pos = random.choice(range(0, anchor))
        head = pointer[i][anchor - 1]
        npath = []
        negs = []
        while len(negs) < n_num:
            if len(head['children']) > 1:
                nhead = random.choice(head['children'])
                nstart = nhead['id']
                while p[anchor] == nstart:
                    nhead = random.choice(head['children'])
                    nstart = nhead['id']
                npath.append(nstart)
                while len(nhead['children']) > 0:
                    nhead = random.choice(nhead['children'])
                    npath.append(nhead['id'])
            else:
                npath = sample_path(raw_graph)[0]
                while check_same_path([p], npath):
                    npath = sample_path(raw_graph)[0]
                common_i = check_common_path(npath, p)
                npath = npath[common_i:]
            neg = random.choice(npath)
            negs.append(neg)
        triple.append([p[anchor], p[pos]] + negs)
    return triple


def sample_path(raw_graph, num=1, pointer=None):
    paths = []
    while len(paths) < num:
        path = []
        pp = []
        head = raw_graph
        path.append(head['id'])
        if pointer is not None:
            pp.append(head)
        while len(head['children']) > 0:
            head = random.choice(head['children'])
            path.append(head['id'])
            if pointer is not None:
                pp.append(head)
        if len(paths) != 0 and check_same_path(paths, path):
            continue
        paths.append(path)
        if pointer is not None:
            pointer.append(pp)
    return paths


def hard_volume(x: torch.Tensor, box_mode=False):
    assert x.dim() == 2
    z, Z = x.chunk(2, -1)
    if box_mode:
        box_length = Z - z
    else:
        box_length = Z * 2
    return box_length.prod(-1)


def softplus(x, t):
    # EPS in case of nan
    return F.softplus(x, t)


def soft_volume(x: torch.Tensor, t=10, box_mode=False):
    if x.dim() == 1:
        x = x.unsqueeze(0)
    # assert x.dim() == 2
    z, Z = x.chunk(2, -1)
    if box_mode:
        box_length = Z - z
    else:
        box_length = Z * 2
    return softplus(box_length, t).clamp(EPS)


def bessel_approx_volume(x, t=1, box_mode=False):
    EULER_GAMMA = 0.57721566490153286060
    if x.dim() == 1:
        x = x.unsqueeze(0)
    z, Z = x.chunk(2, -1)
    if box_mode:
        box_length = Z - z
    else:
        box_length = Z * 2
    return softplus(box_length - 2 * EULER_GAMMA * t, 1 / t).clamp(min=EPS)


def hard_intersection(x, y, box_mode=False):
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)
    assert x.dim() == y.dim()
    xz, xZ = x.chunk(2, -1)
    yz, yZ = y.chunk(2, -1)
    if box_mode:
        Z = torch.min(xZ, yZ)
        z = torch.max(xz, yz)
        return torch.cat([z, Z], dim=-1)
    else:
        Z = torch.min(xz + xZ, yz + yZ)
        z = torch.max(xz - xZ, yz - yZ)
        cen, off = (z + Z) / 2, (Z - z) / 2
        return torch.cat([cen, off], dim=-1)


def log_sum_exp(x, y):
    return torch.logaddexp(x, y)


def gumbel_intersection(x, y, beta=1, box_mode=False):
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)
    xz, xZ = x.chunk(2, -1)
    yz, yZ = y.chunk(2, -1)
    if box_mode:
        z = -beta * (log_sum_exp(-xZ / beta, -yZ / beta))
        Z = beta * (log_sum_exp(xz / beta, yz / beta))
        return torch.cat([z, Z], dim=-1)
    else:
        r = -beta * (log_sum_exp(-(xz + xZ) / beta, -(yz + yZ) / beta))
        l = beta * (log_sum_exp((xz - xZ) / beta, (yz - yZ) / beta))
        cen, off = (l + r) / 2, (r - l) / 2
        return torch.cat([cen, off], dim=-1)


# log p(x|y)
def log_conditional_prob(x, y, box_mode=True):
    inter = hard_intersection(x, y, box_mode=box_mode)
    log_prob = (torch.log(soft_volume(inter, box_mode=box_mode)) - torch.log(soft_volume(y, box_mode=box_mode))).sum(
        -1).clamp(max=-EPS)
    # torch.log(soft_volume(inter)
    return log_prob


# p(x | y)
def conditional_prob(x, y, box_mode=True):
    return (soft_volume(hard_intersection(x, y, box_mode=box_mode), box_mode=box_mode) / soft_volume(y,
                                                                                                     box_mode=box_mode)).prod(
        -1).clamp(min=EPS)


def center_of(x, box_mode=True):
    z, Z = x.chunk(2, -1)
    if box_mode:
        return (z + Z) / 2
    else:
        return z


def calculate_ranks_from_similarities(all_similarities, positive_relations):
    """
    all_similarities: a np array
    positive_relations: a list of array indices

    return a list
    """
    # positive_relation_similarities = all_similarities[positive_relations]
    # negative_relation_similarities = np.ma.array(all_similarities, mask=False)
    # negative_relation_similarities.mask[positive_relations] = True
    # ranks = list((negative_relation_similarities > positive_relation_similarities[:, np.newaxis]).sum(axis=1) + 1)
    # ranks = list((all_similarities > positive_relation_similarities[:, np.newaxis]).sum(axis=1) + 1)

    all_rank = np.argsort(np.argsort(-all_similarities))
    rank = list(all_rank[positive_relations] + 1)
    # first = np.where(all_rank == 0)[0]
    return rank


def obtain_ranks(outputs, targets):
    """
    outputs : tensor of size (batch_size, 1), required_grad = False, model predictions
    targets : tensor of size (batch_size, ), required_grad = False, labels
        Assume to be of format [1, 0, ..., 0, 1, 0, ..., 0, ..., 0]
    mode == 0: rank from distance (smaller is preferred)
    mode == 1: rank from similarity (larger is preferred)
    """
    calculate_ranks = calculate_ranks_from_similarities
    all_ranks = []
    prediction = outputs.cpu().detach().numpy().squeeze()
    label = targets.cpu().numpy()
    sep = np.array([0, 1], dtype=label.dtype)

    # fast way to find subarray indices in a large array, c.f. https://stackoverflow.com/questions/14890216/return-the-indexes-of-a-sub-array-in-an-array
    end_indices = [(m.start() // label.itemsize) + 1 for m in re.finditer(sep.tostring(), label.tostring())]
    end_indices.append(len(label) + 1)
    start_indices = [0] + end_indices[:-1]
    for start_idx, end_idx in zip(start_indices, end_indices):
        distances = prediction[start_idx: end_idx]
        labels = label[start_idx:end_idx]
        positive_relations = list(np.where(labels == 1)[0])
        ranks = calculate_ranks(distances, positive_relations)
        all_ranks.append(ranks)
    return list(itertools.chain(all_ranks))


def rearrange(energy_scores, candidate_position_idx, true_position_idx):
    if isinstance(true_position_idx[0], tuple):
        tmp = np.array([[tuple(x) == tuple(y) for x in candidate_position_idx] for y in true_position_idx]).any(0)
    else:
        tmp = np.array([[x == y for x in candidate_position_idx] for y in true_position_idx]).any(0)
    correct = np.where(tmp)[0]
    incorrect = np.where(~tmp)[0]
    labels = torch.cat((torch.ones(len(correct)), torch.zeros(len(incorrect)))).int()
    energy_scores = torch.cat((energy_scores[correct], energy_scores[incorrect]))
    return energy_scores, labels


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def collate(batch):
    q, pp, pc, l, s, r, i, rs = [], [], [], [], [], [], [], []
    for b in batch:
        query_embed, p_datas, c_datas, labels, sims, reaches, i_idx, rank_sims = b
        q.append(query_embed)
        pp.append(p_datas)
        pc.append(c_datas)
        l.append(labels.unsqueeze(0))
        s.append(sims.unsqueeze(0))
        r.append(reaches.unsqueeze(0))
        i.append(i_idx.unsqueeze(0))
        rs.append(rank_sims.unsqueeze(0))
    return torch.cat(q), pp, pc, torch.cat(l), torch.cat(s), torch.cat(r), torch.cat(i), torch.cat(rs)


def extract_feature(_model: torch.nn.Module, preprocess, imgs: Union[str, List[str]], label, device):
    # if label in ['flower', 'shirt', 'banana', 'watermelon', 'apple', 'cat']:
    #     return
    inputs = []
    outputs = []
    imgs = [imgs] if type(imgs) == str else imgs
    for i in imgs:
        try:
            inputs.append(preprocess(i).to(device))
        except:
            continue

    # print(label, len(inputs))

    # inputs = torch.stack(inputs).squeeze(1)
    def get_batch_feature(_features, batch_size):
        feature_num = len(_features)
        assert batch_size <= feature_num
        start = 0
        end = batch_size
        while start < feature_num:
            yield _features[start: end]
            start += batch_size
            end = end + batch_size if end <= feature_num else feature_num

    for inputs in get_batch_feature(inputs, 8):
        inputs = torch.stack(inputs).squeeze(1).to(device)
        with torch.no_grad():
            features = _model(inputs).cpu().numpy()
        outputs.append(features)
    res = np.vstack(outputs)
    np.save("datasets_json/handcrafted/raw/features/" + label + ".npy", res)
    return res


def extract_features_for_imgs(_model, preprocess, path_to_imglist, l=None):
    img_list = [os.path.join(path_to_imglist, p) for p in os.listdir(path_to_imglist)]
    f = []
    if l is not None:
        label = os.path.join(path_to_imglist, l)
        return extract_feature(_model, preprocess,
                               [os.path.join(label, p) for p in os.listdir(label)],
                               l, torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    else:
        for label in img_list:
            f.append(extract_feature(_model, preprocess, [os.path.join(label, p) for p in os.listdir(label)],
                                     label.split('/')[-1],
                                     torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))

        return f


# def visualize_distribution(path_to_npy):
#     features = np.load(path_to_npy).T
#     print(features.shape)
#     show_dims = random.sample(list(range(0, features.shape[0])), 8)
#     for dim in show_dims:
#         feature = features[dim, :]
#         norm_feature = (feature - feature.min()) / (feature.max() - feature.min())
#         density = gaussian_kde(norm_feature)
#         x = np.linspace(0, 1, 10000)
#         y = density(x)
#         plt.plot(x, y, label='feature_distribution')
#         plt.plot(x, norm.pdf(x, norm_feature.mean(), norm_feature.std()), label='normal_distribution')
#         plt.title('mean:' + str(norm_feature.mean()) + '-' + 'var:' + str(norm_feature.var()))
#         plt.legend(loc='upper right')
#         plt.show()


# def cluster_feature_dim(path_to_npy):
#     features = np.load(path_to_npy).T
#
#     def cluster_one_dim_with_kurtosis(feature):
#         norm_feature = (feature - feature.min()) / (feature.max() - feature.min())
#         k = kurtosis(norm_feature)
#         return k
#
#     kurtosis_each_dim = np.array(list(map(cluster_one_dim_with_kurtosis, features)))
#     vq.kmeans(kurtosis_each_dim, 2)
#

def retrieve_model(model_name, device):
    model_dict = {'resnet50': 'nvidia_resnet50', 'clip': 'ViT-B/32'}
    assert model_name in model_dict.keys()
    _model, _preprocess = None, None
    if model_name == 'resnet50':
        _model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', model_dict['resnet50'], pretrained=True)
        _model_utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')
        _preprocess = _model_utils.prepare_input_from_uri
        _model.fc = torch.nn.Identity()
    elif model_name == 'clip':
        _model, _pre = clip.load('ViT-B/32', device)

        def _p(uri):
            i = Image.open(uri).convert('RGB')
            return _pre(i)

        _preprocess = _p

    _model.eval().to(device)
    return _model if model_name != 'clip' else _model.encode_image, _preprocess


def partition_array(arr, k):
    if k <= 0 or len(arr) < k:
        return None

    flag = True
    try:
        arr[:k]
    except TypeError as e:
        flag = False

    result = []
    size = len(arr) // k
    remainder = len(arr) % k
    start = 0
    for i in range(k):
        end = start + size
        if remainder > 0:
            end += 1
            remainder -= 1

        if flag:
            t = arr[start:end]
        else:
            t = []
            for j in range(start, end):
                t.append(arr[j])

        result.append(t)
        start = end
    return result


def sample_neighbors(taxonomy: networkx.DiGraph, father: str, query: str, max_n: int = 10) -> [str]:
    """
    sample neighbors except query node from taxonomy given a node
    @param taxonomy:
    @param father:
    @param query:
    @param max_n:
    """
    children = list(taxonomy.successors(father))
    children.remove(query) if query in children else None
    neighbors = random.sample(children, k=max_n) if max_n < len(children) else children
    return neighbors


# instruction = """
#     The following is a transcript of a conversation between a human and a smart, helpful AI assistant. The AI assistant's responses are based only on its own pre-existing knowledge; it cannot access the internet or other data sources in any way (although it may provide the human with instructions for how to look up or access data if the human wishes to do that themselves). It will never ask the human for highly sensitive private information such as passwords, credit card numbers, social security numbers, and so on.
#     The human and the AI assistant take turns making statements. Human statements start with ¬Human¬ and AI assistant statements start with ¬AI¬. Complete the transcript in exactly that format, without commentary.
#     ¬Human¬{}
#     ¬AI¬
# """

instruction = {"role": "user", "content": "{}"}

prompt = {
    'visual': "A concept is considered to be visual if it meets both the following two conditions. "
              "First, instances of the concept have a physical entity and can be seen in vision and share some common "
              "visual features."
              "Second, it is not a place nor a city nor a village nor a brand nor a company nor an abstract concept.\n"
              "Is {}, {} a visual concept? You just answer yes or no.",
    'path': "I will give you a list of words that are arranged in a hypernym-hyponym relationship, with each word "
            "being the hypernym of the latter. They are {}. "
            "Besides, I will give another list containing words of which {} is hypernym. They are {}. "
            "Is {} placed the proper position where {} is the hypernym of it and {} are its neighbors? "
            "You only need to answer [yes or no] without any other words.",
    'candidate_i': "You're required to judge whether {} is a location of suitable granularity to place {} where {} is "
                   "the hypernym of {} and {} is hyponym of {}. "
                   "You only need to answer [yes or no] without any other words.",
    'candidate_a': "You're required to judge whether {} is a hypernym with proper granularity of {}. "
                   "You only need to answer [yes or no] without any other words.",
    'candidate_ib': "I'll give you a list of hypernym-hyponym pairs containing {} pairs which are {}. You're required "
                    "to judge whether each of them is proper to insert {} where the left is a strict hypernym of {} "
                    "and the right is a strict hyponym of {}. For each word you are only required to answer [True or "
                    "False]  and format all the answers as a list like [True,False,True......] in the same order of "
                    "given list with exactly {} boolean values. Your answers must be consistent with your "
                    "explanations and output the answer list first.",
    'candidate_ab': "I'll give you a list of words containing {} words which are {}. You're required to judge whether "
                    "each of them is a strict hypernym of [{}]. For each word you are only required to answer [True "
                    "or False]  and format all the answers as a list like [True,False,True......] in the same order "
                    "of given list with exactly {} boolean values.  Your answers must be consistent with your "
                    "explanations and output the answer list first."
}


def rescore_by_att(scores: torch.Tensor, att_idx: torch.Tensor, ins_idx: torch.Tensor,
                   edges: torch.Tensor, k: int = 5) -> torch.Tensor:
    scores_i = scores[ins_idx]
    scores_a = scores[att_idx]
    edges_i = edges[ins_idx, :]
    edges_a = edges[att_idx, :]
    scores_a_topk_idx = scores_a.topk(1)[1]

    # get the topk descendants
    topk_candidates_a = edges_a[scores_a_topk_idx, :][0][0]
    scores_i[edges_i[:, 0] == topk_candidates_a] /= scores_a[scores_a_topk_idx]
    scores[ins_idx] = scores_i
    return scores


def chatgpt_judge(prompt_type: str, query: [str], neighbors: [str] = None) -> Union[bool, List[bool]]:
    """
    function to judge whether a concept is visual and whether it is placed correctly.
    @param neighbors:
    @param prompt_type: visual|path
    @param query: [concept, desc] if visual | [path] if path
    @return: boolean value
    """
    form = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "system", "content": "You are a helpful data annotator."},
                     {"role": "user", "content": ""}],
        "max_tokens": 50,
        "temperature": 0.6,
        "top_p": 1,
        "n": 1,
        "stream": False,
        "stop": "\r\n"
    }
    url = 'http://{}/v1/chat/completions'
    if prompt_type == 'visual':
        form['messages'][1]['content'] = prompt[prompt_type].format(query[0], query[1])
    elif prompt_type == 'path':
        if neighbors is not None:
            form['messages'][1]['content'] = prompt[prompt_type].format(query, query[-2], neighbors, query[-1],
                                                                        query[-2], neighbors)
    elif prompt_type == 'candidate_i':
        form['messages'][1]['content'] = prompt[prompt_type].format('<{}, {}>'.format(query[0], query[1]), query[2],
                                                                    query[0], query[2], query[1], query[2])
    elif prompt_type == 'candidate_a':
        form['messages'][1]['content'] = prompt[prompt_type].format(query[0], query[1])

    elif prompt_type == 'candidate_ib':
        form['messages'][1]['content'] = prompt[prompt_type].format(len(query[1]), query[1], query[0], query[0],
                                                                    query[0], len(query[1]))
    elif prompt_type == 'candidate_ab':
        form['messages'][1]['content'] = prompt[prompt_type].format(len(query[1]), query[1], query[0], len(query[1]))

    cnt = 0
    while True:
        try:
            # max retries = 5
            if cnt == 5:
                return False
            resp = requests.request('POST', url, headers={'Content-Type': 'application/json'}, data=json.dumps(form))
            if resp.status_code == 200:
                res = resp.json()['choices'][0]['text']
                # print(res)
                if prompt_type == 'path':
                    return 'yes' in res.lower()
                elif 'candidate' in prompt_type:
                    pattern = r'\[.*?\]'
                    match = re.search(pattern, res)
                    if match:
                        list_str = match.group()
                        result = re.findall(r'(True|False)', list_str)
                        result = [eval(b) for b in result]
                        assert len(result) == len(query[1])
                        return result

        except AssertionError as e:
            print('AssertionError')
        except Exception as e:
            print(e)
        finally:
            cnt += 1


def rescore_by_chatgpt(scores: torch.Tensor, att_idx: torch.Tensor, ins_idx: torch.Tensor,
                       dataset: torch.utils.data.Dataset, edges: torch.Tensor, query: str,
                       k: int = 100) -> torch.Tensor:
    scores_i = scores[ins_idx]
    scores_a = scores[att_idx]
    edges_i = edges[ins_idx, :]
    edges_a = edges[att_idx, :]

    scores_i_topk_idx = scores_i.topk(k)[1]
    scores_a_topk_idx = scores_a.topk(k)[1]

    topk_candidates_i = edges_i[scores_i_topk_idx, :]
    topk_candidates_a = edges_a[scores_a_topk_idx, :]

    topk_candidates_i_names = [(dataset.names[p[0]].split('@')[0], dataset.names[p[1]].split('@')[0]) for p in
                               topk_candidates_i]
    topk_candidates_a_names = [dataset.names[p[0]].split('@')[0] for p in topk_candidates_a]

    res_i = chatgpt_judge('candidate_ib', [query, topk_candidates_i_names])
    res_a = chatgpt_judge('candidate_ab', [query, topk_candidates_a_names])

    scores[torch.where(ins_idx)[0][scores_i_topk_idx][res_i]] += 1
    scores[torch.where(att_idx)[0][scores_a_topk_idx][res_a]] += 1

    return scores


if __name__ == '__main__':
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model, preprocess = retrieve_model('resnet50', device)
    # extract_features_for_imgs(model, preprocess, 'handcrafted')
    pass