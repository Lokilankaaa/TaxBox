from typing import Union, List
from PIL import Image
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde, norm, kurtosis
from scipy.cluster import vq, hierarchy
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

EPS = 1e-13


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
    return softplus(box_length, t)


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
    prediction = outputs.cpu().numpy().squeeze()
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
    return all_ranks


def rearrange(energy_scores, candidate_position_idx, true_position_idx):
    tmp = np.array([[tuple(x) == tuple(y) for x in candidate_position_idx] for y in true_position_idx]).any(0)
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


def visualize_distribution(path_to_npy):
    features = np.load(path_to_npy).T
    print(features.shape)
    show_dims = random.sample(list(range(0, features.shape[0])), 8)
    for dim in show_dims:
        feature = features[dim, :]
        norm_feature = (feature - feature.min()) / (feature.max() - feature.min())
        density = gaussian_kde(norm_feature)
        x = np.linspace(0, 1, 10000)
        y = density(x)
        plt.plot(x, y, label='feature_distribution')
        plt.plot(x, norm.pdf(x, norm_feature.mean(), norm_feature.std()), label='normal_distribution')
        plt.title('mean:' + str(norm_feature.mean()) + '-' + 'var:' + str(norm_feature.var()))
        plt.legend(loc='upper right')
        plt.show()


def cluster_feature_dim(path_to_npy):
    features = np.load(path_to_npy).T

    def cluster_one_dim_with_kurtosis(feature):
        norm_feature = (feature - feature.min()) / (feature.max() - feature.min())
        k = kurtosis(norm_feature)
        return k

    kurtosis_each_dim = np.array(list(map(cluster_one_dim_with_kurtosis, features)))
    vq.kmeans(kurtosis_each_dim, 2)


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


if __name__ == '__main__':
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model, preprocess = retrieve_model('resnet50', device)
    # extract_features_for_imgs(model, preprocess, 'handcrafted')
    visualize_distribution('features/apple.npy')
